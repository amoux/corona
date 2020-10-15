
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import faiss
import numpy as np
import spacy
import torch
from transformers import (BertForQuestionAnswering, BertTokenizer,
                          QuestionAnsweringPipeline)

from .dataset import CORD19Dataset
from .datatypes import Papers
from .summarization import BertSummarizer, frequency_summarizer
from .ukplab.sentence import SentenceTransformer


class QuestionAnsweringArguments(NamedTuple):
    question: Union[str, List[str]]
    context: Union[str, List[str]]
    topk: int = 10
    doc_stride: int = 128
    max_answer_len: int = 15
    max_seq_len: int = 384
    max_question_len: int = 64
    handle_impossible_answer: bool = True

    def todict(self) -> Dict[str, Any]:
        return self._asdict()


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

    def attach_(self, *inputs) -> None:
        self.q, self.c, self.ids, self.dist = inputs

    def popempty(self) -> Union[List[ModelOutput], None]:
        items = [self.pop(i) for i, o in enumerate(self) if not o.answer]
        if items:
            return items
        return None

    def size(self) -> int:
        return len(self)

    def __repr__(self):
        return '{}(size: {}, question: {})'.format(
            self.__class__.__name__, self.size(), self.q)


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
            model_device: Optional[Any] = None,
            encoder_device: Optional[str] = None,
    ) -> None:
        self.papers = Papers.from_disk(papers) \
            if isinstance(papers, str) else papers
        self.index = faiss.read_index(index) \
            if isinstance(index, str) else index
        self.encoder = SentenceTransformer(encoder, device=encoder_device) \
            if isinstance(encoder, str) else encoder
        self.cord19 = cord19
        if hasattr(self.papers, 'init_args'):
            self.cord19 = self.papers.init_cord19_dataset()
        self.nlp = self.cord19.sentence_tokenizer.nlp() \
            if isinstance(self.cord19, CORD19Dataset) else spacy.load(nlp_model)

        if model_name_or_path is None:
            model_name_or_path = self.default_model_name
        if do_lower_case is None:
            do_lower_case = self.do_lower_case
        name = model_name_or_path

        self.model = BertForQuestionAnswering.from_pretrained(name) \
            if model is None else model
        if model_device is not None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
        self.tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=do_lower_case) \
            if tokenizer is None else tokenizer
        self.pipeline = QuestionAnsweringPipeline(self.model, self.tokenizer)

        self._freq_summarizer = frequency_summarizer
        self._bert_summarizer = BertSummarizer.load(
            name, self.encoder.tokenizer)

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
        return [sent.text for sent in self.nlp(text).sents]

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
            if isinstance(sents, list):
                sents = ' '.join(sents)
            return self._bert_summarizer(sents)
        if mode == 'freq':
            return self._freq_summarizer(sents, nlp=self.nlp)

    def answer(self, question: str, topk=5, top_p=15, nprobe=16, mode='bert', **kwargs):
        # Reduce the chance of getting similar "questions", we want sentences.
        q_as_query = question.replace('?', '').strip()
        dists, ids = self.similar(q_as_query, top_p, nprobe)
        sents = []
        for idx in ids.flatten():
            string = self.papers[idx.item()]
            if string == question and idx + 1 <= self.max_num_sents:
                string = self.papers[idx.item() + 1]
            sents.append(string)

        output = QuestionAnsweringOutput()
        context = self.compress(sents, mode=mode)
        inputs = (question, context, ids, dists)
        output.attach_(*inputs)
        question, context = [self.sentencizer(x) for x in (question, context)]
        params = QuestionAnsweringArguments(question, context, topk, **kwargs)
        for prediction in self.pipeline(**params.todict()):
            output.append(ModelOutput(**prediction))
        return output
