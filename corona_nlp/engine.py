from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch

from transformers import BertForQuestionAnswering

from .dataset import CORD19Dataset
from .datatypes import Papers
from .retrival import frequency_summarizer
from .tokenizer import SpacySentenceTokenizer
from .transformer import BertSummarizer, SentenceTransformer
from .utils import clean_tokenization, normalize_whitespace


class QuestionAnsweringEngine:
    nprobe_list = [1, 4, 16, 64, 256]

    def __init__(
            self,
            papers: Union[str, Path, Papers],
            index: Union[str, faiss.IndexIVFFlat],
            encoder: Union[str, Path, SentenceTransformer],
            cord19: Optional[CORD19Dataset] = None,
            model_name: str = "amoux/scibert_nli_squad",
            nlp_model: str = "en_core_web_sm",
            model_device: str = "cpu",
            **kwargs,
    ) -> None:
        self.papers = papers if isinstance(papers, Papers) \
            else Papers.from_disk(papers)
        if cord19 is None:
            if hasattr(self.papers, "init_args"):
                cord19 = self.papers.init_cord19_dataset()
            else:
                try:
                    cord19 = CORD19Dataset(**kwargs)
                except Exception:
                    pass
        if cord19 is not None:
            for name, attr in cord19.__dict__.items():
                setattr(self, name, attr)

        self.index = index if isinstance(index, faiss.IndexIVFFlat) \
            else faiss.read_index(index)
        self.encoder = encoder if isinstance(encoder, SentenceTransformer) \
            else SentenceTransformer(encoder)
        self.tokenizer = self.encoder.tokenizer
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        if model_device == "cuda":
            self.model = self.model.to(self.encoder.device)
        self.nlp = self.sentence_tokenizer.nlp() if cord19 is not None \
            else SpacySentenceTokenizer(nlp_model).nlp()
        self._freq_summarizer = frequency_summarizer
        self._bert_summarizer = BertSummarizer.load(model_name, self.tokenizer)

    @property
    def max_num_sents(self) -> int:
        return len(self.papers) - 1

    @property
    def engine_devices(self) -> Dict[str, torch.device]:
        return dict(
            summarizer_model_device=self._bert_summarizer.model.device,
            sentence_transformer_model_device=self.encoder.device,
            question_answering_model_device=self.model.device,
        )

    def compress(self, sents: Union[str, List[str]], mode='bert') -> str:
        if mode == 'freq':
            return self._freq_summarizer(sents, nlp=self.nlp)
        elif mode == 'bert':
            if isinstance(sents, list):
                sents = ' '.join(sents)
            return self._bert_summarizer(sents)

    def encode(self, sents: List[str]) -> np.array:
        embedding = self.encoder.encode(sents, show_progress=False)
        return embedding

    def similar(self, query: str, k=5, nprobe=1) -> Tuple[np.array, np.array]:
        query = normalize_whitespace(query.replace('?', ' '))
        embed = self.encode([query])
        self.index.nprobe = nprobe
        return self.index.search(embed, k)

    def decode(self, question: str,
               context: str) -> Tuple[str, str, Tuple[int, int]]:
        inputs = self.tokenizer.encode_plus(text=question.strip(),
                                            text_pair=context,
                                            add_special_tokens=True,
                                            return_tensors='pt')
        if self.model.device.type == 'cuda':
            inputs = inputs.to(self.model.device)

        top_k = self.model(**inputs)
        start, end = (torch.argmax(top_k[0]),
                      torch.argmax(top_k[1]) + 1)
        input_ids = inputs['input_ids'].tolist()
        answer = self.tokenizer.decode(input_ids[0][start:end],
                                       skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)
        if len(answer) > 0:
            context = self.tokenizer.decode(input_ids[0],
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
        span = tuple((start, end))
        return answer, context, span

    def answer(self, question: str, k=15,
               nprobe=1, mode: str = None) -> Dict[str, Any]:
        question = question.strip()
        dists, indices = self.similar(question, k, nprobe)

        sentences = []
        for index in indices.flatten():
            string = self.papers[index]
            if string == question and index + 1 <= self.max_num_sents:
                string = self.papers[index + 1]

            doc = self.nlp(string)
            for sent in doc.sents:
                string = clean_tokenization(sent.text)
                if len(sent) > 1 and sent[0].is_title:
                    if (not sent[-1].like_num
                        and not sent[-1].is_bracket
                        and not sent[-1].is_quote
                        and not sent[-1].is_stop
                        and not sent[-1].is_punct):
                        string = f'{string}.'
                if string in sentences:
                    continue
                sentences.append(string)

        context = ' '.join(sentences)
        if mode is not None and mode in ('freq', 'bert'):
            context = self.compress(context, mode=mode)
        context = normalize_whitespace(context)

        answer, context, span = self.decode(question, context)
        context = clean_tokenization(context)
        dists, indices = dists.tolist()[0], indices.tolist()[0]

        return {'answer': answer, 'context': context,
                'dist': dists, 'ids': indices, 'span': span}
