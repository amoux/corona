
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import faiss
import numpy as np
import torch
from transformers import (BertForQuestionAnswering, BertTokenizer,
                          QuestionAnsweringPipeline, SquadExample)

from .core import Papers
from .dataset import CORD19Dataset
from .summarization import (BertSummarizer, BertSummarizerArguments,
                            frequency_summarizer)
from .tokenizer import SpacySentenceTokenizer
from .ukplab.sentence import SentenceTransformer


class InvalidModelNameOrPathError(ValueError):
    pass


class QuestionAnsweringArguments(NamedTuple):
    X: Optional[SquadExample] = None
    question: Optional[Union[str, List[str]]] = None
    context: Optional[Union[str, List[str]]] = None
    topk: int = 10
    doc_stride: int = 128
    max_answer_len: int = 15
    max_seq_len: int = 384
    max_question_len: int = 64
    handle_impossible_answer: bool = True

    def todict(self) -> Dict[str, Any]:
        return self._asdict()


__docstring = QuestionAnsweringPipeline.__call__.__doc__
QuestionAnsweringArguments.__doc__ = __docstring


class ModelOutput(NamedTuple):
    score: float
    start: int
    end: int
    answer: str


class QuestionAnsweringOutput(List[ModelOutput]):
    q: Optional[Union[str, List[str]]] = None
    c: Optional[Union[str, List[str]]] = None
    ids: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None

    @property
    def shape(self) -> Union[Tuple[int, int], None]:
        if self.ids is not None:
            return self.ids.shape
        return None

    def attach_(self, *inputs) -> None:
        q, c, self.ids, self.dist = inputs
        self.q = q[0] if len(q) == 1 else q
        self.c = c[0] if len(c) == 1 else c

    def popempty(self) -> Union[List[ModelOutput], None]:
        items = [self.pop(i) for i, o in enumerate(self) if not o.answer]
        if items:
            return items
        return None

    def spans(self) -> List[Tuple[int, int]]:
        return [(o.start, o.end) for o in self]

    def scores(self) -> Dict[float, int]:
        """Return a dict mapping of scores and output indices.

        - Usage example:
        ```python
        topk = self.scores()
        idx = topk[max(topk)]  # compute max k.
        output = self[idx]  # query top idx item.
        ...
        # ModelOutput(score=0.11623, start=52, end=9, ...)
        ```
        """
        return {o.score: i for i, o in enumerate(self)}

    def size(self) -> int:
        return len(self)

    def __repr__(self):
        return '{}(size: {}, shape: {})'.format(
            self.__class__.__name__, self.size(), self.shape())


class ScibertQuestionAnswering:
    default_model_name = "amoux/scibert_nli_squad"
    do_lower_case = False
    nprobe_list = [1, 4, 16, 64, 256]

    def __init__(
            self,
            papers: Union[str, Papers],
            index: Union[str, faiss.IndexIVFFlat],
            encoder: Union[str, SentenceTransformer],
            cord19: Optional[CORD19Dataset] = None,
            model: Optional[BertForQuestionAnswering] = None,
            tokenizer: Optional[BertTokenizer] = None,
            model_name_or_path: Optional[str] = None,
            do_lower_case: Optional[bool] = None,
            nlp_model: str = 'en_core_sci_sm',
            model_device: Optional[str] = None,
            encoder_device: Optional[str] = None,
            summarizer_hidden: int = -2,
            summarizer_reduce: str = 'mean',
            summarizer_kwargs: Union[Dict[str, Any], BertSummarizerArguments] = {},
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
        self.papers = Papers.from_disk(papers) \
            if isinstance(papers, str) else papers
        self.index = faiss.read_index(index) \
            if isinstance(index, str) else index
        self.encoder = SentenceTransformer(encoder, device=encoder_device) \
            if isinstance(encoder, str) else encoder
        if cord19 is None:
            if hasattr(self.papers, 'init_args'):
                self.cord19 = self.papers.init_cord19_dataset()
                if hasattr(self.cord19, 'sentencizer'):
                    self._sentencizer = self.cord19.sentencizer
        elif isinstance(cord19, CORD19Dataset):
            self.cord19 = cord19
            self._sentencizer = cord19.sentencizer
        else:
            self._sentencizer = SpacySentenceTokenizer(nlp_model)
        self.nlp = self._sentencizer.nlp
        if model_name_or_path is None:
            model_name_or_path = self.default_model_name
        if do_lower_case is None:
            do_lower_case = self.do_lower_case
        if model is None or isinstance(model, str):
            self.model = BertForQuestionAnswering.from_pretrained(
                model_name_or_path)
        elif isinstance(model, BertForQuestionAnswering):
            self.model = model
        else:
            raise InvalidModelNameOrPathError
        if model_device is not None:
            device = torch.device(model_device)
            self.model = self.model.to(device)
        if tokenizer is None or isinstance(tokenizer, str):
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name_or_path, do_lower_case=do_lower_case,
            )
        elif isinstance(tokenizer, BertTokenizer):
            self.tokenizer = tokenizer
        else:
            raise InvalidModelNameOrPathError
        # HF QAPipeline uses index, -1 is the default for CPU.
        device_index = -1
        if self.model.device.index is not None and self.model.device.type == 'cuda':
            # Model using CUDA, set CUDA idx.
            device_index = self.model.device.index
        self.pipeline = QuestionAnsweringPipeline(
            self.model, self.tokenizer, device=device_index,
        )
        if isinstance(summarizer_kwargs, dict):
            self.summarizer_kwargs = BertSummarizerArguments(
                **summarizer_kwargs)
        elif isinstance(summarizer_kwargs, BertSummarizerArguments):
            self.summarizer_kwargs = summarizer_kwargs
        self._freq_summarizer = frequency_summarizer
        self._bert_summarizer = BertSummarizer(model_name_or_path=model_name_or_path,
                                               tokenizer=self.tokenizer,
                                               hidden=summarizer_hidden,
                                               reduce_option=summarizer_reduce)

    @property
    def max_num_sents(self) -> int:
        return self.papers.num_sents - 1

    @property
    def all_model_devices(self) -> Dict[str, Any]:
        return {
            'summarizer_model_device': self._bert_summarizer.model.device,
            'sentence_transformer_model_device': self.encoder.device,
            'question_answering_model_device': self.model.device
        }

    def sentencizer(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, list):
            text = ' '.join(text)
        return [sent.text for sent in self._sentencizer(text)]

    def encode_features(self, sents: Union[str, List[str]]) -> np.ndarray:
        sentences = self.sentencizer(sents)
        embedding = self.encoder.encode(sentences, show_progress=False)
        return embedding

    def similar(self, query: Union[str, List[str]], top_p=15, nprobe=64,
                ) -> Tuple[np.ndarray, np.ndarray]:
        embedding = self.encode_features(query)
        self.index.nprobe = nprobe
        D, I = self.index.search(embedding, k=top_p)
        return D, I

    def compress(self, sents, mode='bert'):
        if mode == 'bert':
            summarizer_inputs = self.summarizer_kwargs
            if isinstance(sents, list):
                summarizer_inputs.body = ' '.join(sents)
            return self._bert_summarizer(**summarizer_inputs.todict())
        if mode == 'freq':
            return self._freq_summarizer(sents, nlp=self.nlp)

    def sample(self, question: Union[str, List[str]], context: Union[str, List[str]],
               ) -> SquadExample:
        return self.pipeline.create_sample(question=question, context=context)

    def answer(self, question: str, topk=5, top_p=15, nprobe=16, mode='bert', **kwargs):
        # Reduce the chance of getting similar "questions", we want sentences.
        kwargs.update({'topk': topk})
        q_as_query = question.replace('?', '').strip()
        dists, ids = self.similar(q_as_query, top_p, nprobe)

        sents = []
        for idx in ids.flatten():
            string = self.papers[idx.item()]
            if string == question and idx + 1 <= self.max_num_sents:
                string = self.papers[idx.item() + 1]
            sents.append(string)

        predictions = QuestionAnsweringOutput()
        context = self.compress(sents, mode=mode)
        question, context = [self.sentencizer(x) for x in (question, context)]
        inputs = (question, context, ids, dists)
        predictions.attach_(*inputs)
        squad_example = self.sample(question, context)
        params = QuestionAnsweringArguments(X=squad_example, **kwargs)
        for output in self.pipeline(**params.todict()):
            predictions.append(ModelOutput(**output))

        return predictions
