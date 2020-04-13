# cython: infer_types=True
# coding: utf8
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Union

from tqdm.auto import tqdm

from .datatypes import Papers, Sentences
from .indexing import PaperIndexer
from .preprocessing import normalize_whitespace
from .tokenizer import SpacySentenceTokenizer


class CORD19Dataset(PaperIndexer):
    def __init__(
            self,
            source_dir: Union[str, Path],
            text_keys: Tuple[str] = ("abstract", "body_text"),
    ):
        super().__init__(source_dir=source_dir)
        self.text_keys = text_keys
        self._tokenizer = SpacySentenceTokenizer()

    def samples(self, k: int = None) -> List[int]:
        """Return all or k iterable of paper-id to index mappings.

        `k`: For all available papers use `k=-1`. Otherwise, pass `k=n`
            number of samples to load from the available dataset papers.
        """
        indices = list(self.index_paper.keys())
        if k == -1:
            return indices
        assert k <= self.num_papers
        return random.sample(indices, k=k)

    def tokenize(self, string: str) -> List[str]:
        return self._tokenizer.tokenize(string)

    def title(self, index: int = None, paper_id: str = None) -> str:
        return self.load_paper(index, paper_id)["metadata"]["title"]

    def titles(self, indices: List[int] = None, paper_ids: List[str] = None):
        for paper in self.load_papers(indices, paper_ids):
            yield paper["metadata"]["title"]

    def texts(self, indices: List[int] = None, paper_ids: List[str] = None):
        for paper in self.load_papers(indices, paper_ids):
            for key in self.text_keys:
                for string in paper[key]:
                    yield string["text"]

    def sents(self, indices: List[int], minlen=20, batch_size=10) -> Papers:
        """Return instance of papers with texts transformed to sentences."""
        rem = batch_size % 2
        if rem:
            batch_size - rem
        samples = len(indices)

        with tqdm(total=samples, desc="papers") as pbar:
            index = Sentences(indices)
            cluster = index.init_cluster()
            for i in range(0, samples, batch_size):
                split = indices[i: min(i + batch_size, samples)]
                queue = deque(split)
                node = 0
                while len(queue) > 0:
                    node = queue.popleft()
                    batch = self.texts(split)
                    for line in tqdm(batch, desc="lines", leave=False):
                        sentences = self.tokenize(line)
                        for token in sentences:
                            string = normalize_whitespace(token.text)
                            length = len(string)
                            if length <= minlen:
                                continue
                            if string not in cluster[node]:
                                index.strlen += length
                                index.counts += 1
                                index.maxlen = max(length, index.maxlen)
                                cluster[node].append(string)

                pbar.update(len(split))
        return Papers(index, cluster=cluster)

    def __repr__(self):
        return f"<CORD19Dataset({self.source_name}, papers={self.num_papers})>"