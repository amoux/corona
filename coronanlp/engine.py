from typing import Any, Dict, List, Optional, Tuple, Union

import faiss  # type: ignore
import numpy as np  # type: ignore
import torch
from transformers import BertForQuestionAnswering  # type: ignore
from transformers import (BertTokenizerFast, PreTrainedModel,
                          PreTrainedTokenizerBase, QuestionAnsweringPipeline,
                          SquadExample)

from .compressor import Compressor
from .core import SentenceStore
from .dataset import CORD19
from .engine_utils import (ModelOutput, QuestionAnsweringArguments,
                           QuestionAnsweringOutput)
from .retrival import frequency_summarizer
from .tokenizer import SpacySentenceTokenizer
from .ukplab import SentenceEncoder

Sid = int


POOLING_PARAMS = {
    'mean': 'do_mean_tokens',
    'sqrt': 'do_sqrt_tokens',
    'max': 'do_max_tokens',
    'cls': 'do_cls_tokens'
}


class InvalidModelNameOrPathError(ValueError):
    pass


__docstring = QuestionAnsweringPipeline.__call__.__doc__
QuestionAnsweringArguments.__doc__ = __docstring


class ScibertQuestionAnsweringConfig:
    architecture = "BertForQuestionAnswering"
    default_model_name = "amoux/scibert_nli_squad"
    do_lower_case = False
    nprobe_list = [1, 4, 16, 64, 256]

    def __init__(
            self,
            sents: Union[str, SentenceStore],
            index: Union[str, faiss.Index],
            encoder: Optional[Union[str, SentenceEncoder]] = None,
            cord19: Optional[CORD19] = None,
            model: Optional[Union[BertForQuestionAnswering, PreTrainedModel]] = None,
            tokenizer: Optional[Union[BertTokenizerFast, PreTrainedTokenizerBase]] = None,
            model_name_or_path: Optional[str] = None,
            max_seq_length: int = 256,
            do_lower_case: Optional[bool] = None,
            nlp_model: str = 'en_core_sci_sm',
            model_device: Optional[str] = None,
            encoder_device: Optional[str] = None,
            **compressor_kwargs
    ) -> None:
        """
        :param summarizer_hidden: Determines the hidden layer to use for
            embeddings. (Needs to be negative.)
        :param summarizer_reduce: Determines the reduction statistic of
            the encoding layer `(mean, median, max)`. In other words it
            determines how you want to reduce results.
        :param summarizer_kwargs: Kwargs to pass to the summarizer
            along w/input texts. Or with a `coronanlp.summarization.
            BertSummarizerArguments` instance. (These arguments can be
            overridden anytime). By either updating the properties in
            place e.g., `self.summarizer_kwargs.ratio=0.5`. Note that
            the `body` argument can be disregarded or left as None since
            it's always overridden.
        """
        self.max_seq_length = max_seq_length
        self.sents = SentenceStore.from_disk(sents) \
            if isinstance(sents, str) else sents
        assert isinstance(self.sents, SentenceStore)
        self.index = faiss.read_index(index) \
            if isinstance(index, str) else index
        assert isinstance(self.index, faiss.Index)

        sentencizer = None
        if cord19 is None and hasattr(self.sents, 'init_args'):
            cord19 = self.sents.init_cord19_dataset()
            if not cord19.sentencizer_enabled:
                cord19.init_sentencizer()
            sentencizer = cord19.sentencizer
            self.cord19 = cord19
        elif isinstance(cord19, CORD19):
            if not cord19.sentencizer_enabled:
                cord19.init_sentencizer()
            sentencizer = cord19.sentencizer
            self.cord19 = cord19
        else:
            sentencizer = SpacySentenceTokenizer(nlp_model)
        if sentencizer is not None:
            self._sentencizer = sentencizer
            self.nlp = sentencizer.nlp

        if model_name_or_path is None:
            model_name_or_path = self.default_model_name
        if do_lower_case is None:
            do_lower_case = self.do_lower_case
        if model is None or isinstance(model, str):
            self.model = BertForQuestionAnswering \
                .from_pretrained(model_name_or_path)
        elif isinstance(model, (PreTrainedModel, BertForQuestionAnswering)) \
                and self.architecture in model.config.architectures:
            self.model = model
        else:
            raise InvalidModelNameOrPathError

        if model_device is not None:
            device = torch.device(model_device)
            self.model = self.model.to(device)
        if tokenizer is None or isinstance(tokenizer, str):
            self.tokenizer = BertTokenizerFast \
                .from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        elif isinstance(tokenizer, (BertTokenizerFast, PreTrainedTokenizerBase)):
            self.tokenizer = tokenizer
        else:
            raise InvalidModelNameOrPathError

        if encoder is None:
            base: PreTrainedModel = None
            if hasattr(self.model, 'bert'):
                base = self.model.bert
            elif hasattr(self.model, 'base_model'):
                base = self.model.base_model
            base_device = base.device.type
            self.encoder = SentenceEncoder(
                transformer=base, tokenizer=self.tokenizer, device=base_device
            )
        elif isinstance(encoder, str):
            if encoder_device is None:
                encoder_device = 'cpu'
            self.encoder = SentenceEncoder \
                .from_pretrained(encoder, device=encoder_device)
        elif isinstance(encoder, SentenceEncoder):
            self.encoder = encoder
        else:
            raise InvalidModelNameOrPathError

        self.compressor = Compressor(
            model=self.model.base_model, **compressor_kwargs)
        self.compress_ratio = self.compressor.cluster.ratio

        # HF QAPipeline uses index, -1 is the default for CPU.
        device_index = -1
        if self.model.device.index is not None \
                and self.model.device.type == 'cuda':
            # Model using CUDA, set CUDA idx.
            device_index = self.model.device.index
        self.pipeline = QuestionAnsweringPipeline(
            model=self.model, tokenizer=self.tokenizer, device=device_index
        )
        self._freq_summarizer = frequency_summarizer
        self.device = self.model.device

    def _bert_summarizer(self, sentences: List[str], topk=None, max_length=None):
        max_length = self.max_seq_length if max_length is None else max_length
        inputs = self.tokenizer(
            sentences,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        inputs.update({'topk': topk})
        outputs = self.compressor(**inputs)
        summary = [sentences[k] for k in outputs['topk']]
        return " ".join(summary)


class ScibertQuestionAnswering(ScibertQuestionAnsweringConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def max_length(self) -> int:
        return self.max_seq_length

    @property
    def max_num_sents(self) -> int:
        return len(self.sents) - 1

    @property
    def all_model_devices(self) -> Dict[str, Any]:
        return {
            'summarizer_model_device': self.model.device,
            'sentence_transformer_model_device': self.encoder.device,
            'question_answering_model_device': self.model.device
        }

    def pooling_config(self, type: str = 'mean') -> None:
        """Pooling configuration for encoding sentence embeddings.
        :param type: Type of encoding: `mean`, `sqrt`, `max`, `cls`
        """
        params = POOLING_PARAMS
        assert type in params, f'Expected: `mean`, `sqrt`, `max` or `cls`, got {type}'
        for key, attr in params.items():
            if key == type:
                setattr(self.encoder.pooling, attr, True)
            else:
                setattr(self.encoder.pooling, attr, False)

    def sentencizer(self, text: Union[str, List[str]]):
        if isinstance(text, list):
            text = ' '.join(text)
        return [sent.text for sent in self._sentencizer(text)]

    def encode_features(
        self, texts: Union[str, List[str]], max_length: Optional[int] = None,
    ) -> np.ndarray:
        sentences = self.sentencizer(texts)
        if max_length is None:
            max_length = self.max_seq_length
        embedding = self.encoder.encode(
            sentences, max_length=max_length, show_progress=False)
        return embedding

    def similar(
        self, query: Union[str, List[str]], top_p: int = 15, nprobe: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        embedding = self.encode_features(query)
        self.index.nprobe = nprobe
        D, I = self.index.search(embedding, k=top_p)
        return D, I

    def compress(self, texts: List[str], mode='bert') -> str:
        summary = ''
        if mode == 'bert':
            summary = self._bert_summarizer(texts)
        if mode == 'freq':
            ratio = self.compress_ratio
            summary = self._freq_summarizer(texts, ratio, nlp=self.nlp)
        return summary

    def squad_sample(
        self, question: Union[str, List[str]], context: Union[str, List[str]],
    ) -> SquadExample:
        return self.pipeline.create_sample(question, context=context)

    def answer(
        self,
        question: str,
        topk: int = 5,
        top_p: int = 15,
        nprobe: int = 16,
        mode: str = 'bert',
        doc_stride: int = 1,
        max_answer_len: int = 128,
        max_seq_len: int = 384,
        max_question_len: int = 64,
        handle_impossible_answer: bool = True,
    ) -> QuestionAnsweringOutput:

        qa_kwargs = QuestionAnsweringArguments(
            topk=topk,
            doc_stride=doc_stride,
            max_answer_len=max_answer_len,
            max_seq_len=max_seq_len,
            max_question_len=max_question_len,
            handle_impossible_answer=handle_impossible_answer,
        )
        # Reduce the chance of getting similar "questions", we want sentences.
        q_as_query = question.replace('?', '').strip()
        dist, sids = self.similar(q_as_query, top_p, nprobe)

        texts = []
        for sid in sids.flatten():
            string = self.sents[sid.item()]
            if string == question and sid + 1 <= self.max_num_sents:
                string = self.sents[sid.item() + 1]
            texts.append(string)

        predictions = QuestionAnsweringOutput()
        context = self.compress(texts, mode=mode)
        question, context = [self.sentencizer(x) for x in (question, context)]
        inputs = (question, context, sids, dist)
        predictions.attach_(*inputs)
        pids = self.sents.decode(sids.squeeze(0))
        predictions.pids = np.array([pids])
        qa_kwargs.X = self.squad_sample(question, context)

        for x in self.pipeline(**qa_kwargs.asdict()):
            predictions.append(ModelOutput(**x))

        return predictions

    def __call__(self, *args, **kwargs):
        return self.answer(*args, **kwargs)
