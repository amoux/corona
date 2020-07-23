from typing import Any, Dict, List, Tuple, Union

import faiss
import numpy as np
import torch
from transformers import BertForQuestionAnswering

from .dataset import CORD19Dataset
from .datatypes import Papers
from .retrival import frequency_summarizer
from .transformer import BertSummarizer, SentenceTransformer
from .utils import clean_tokenization, normalize_whitespace


class QAEngine(CORD19Dataset):
    nprobe_list = [1, 4, 16, 64, 256]

    def __init__(self, source: Union[str, List[str]], papers: str,
                 index: str, encoder: str, model: str, **kwargs):
        super(QAEngine, self).__init__(source, **kwargs)
        self.papers = Papers.from_disk(papers)
        self.index = faiss.read_index(index)
        self.encoder = SentenceTransformer(encoder)
        self.tokenizer = self.encoder.tokenizer
        self.model = BertForQuestionAnswering.from_pretrained(model)
        self.nlp = self.sentence_tokenizer.nlp()
        self._freq_summarizer = frequency_summarizer
        self._bert_summarizer = BertSummarizer.load(model, self.tokenizer)

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

    def decode(self, question: str, context: str) -> Tuple[str, str]:
        inputs = self.tokenizer.encode_plus(text=question.strip(),
                                            text_pair=context,
                                            add_special_tokens=True,
                                            return_tensors='pt')
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
        return answer, context

    def answer(self, question: str, k=15,
               nprobe=1, mode: str = None) -> Dict[str, Any]:
        question = question.strip()
        dists, indices = self.similar(question, k, nprobe)

        sentences = []
        for index in indices.flatten():
            string = self.papers[index]
            if string == question:
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

        answer, context = self.decode(question, context)
        context = clean_tokenization(context)
        dists, indices = dists.tolist()[0], indices.tolist()[0]

        return {'answer': answer,
                'context': context, 'dist': dists, 'ids': indices}
