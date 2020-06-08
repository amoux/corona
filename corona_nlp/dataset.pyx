# cython: infer_types=True
# coding: utf8
import random
from collections import deque
from typing import List, Tuple, Union

from tqdm.auto import tqdm

from .datatypes import Papers, Sentences
from .indexing import PaperIndexer
from .tokenizer import SpacySentenceTokenizer
from .utils import clean_tokenization, normalize_whitespace


class CORD19Dataset(PaperIndexer):
    def __init__(
            self,
            source: Union[str, List[str]],
            text_keys: Tuple[str] = ("abstract", "body_text"),
            index_start=1,
    ):
        super().__init__(source, index_start)
        self.text_keys = text_keys
        self.sentence_tokenizer = SpacySentenceTokenizer()

    def sample(self, k: int = None, seed: int = None) -> List[int]:
        """Return all or k iterable of paper-id to index mappings.

        `k`: A sample from all available papers use `k=-1`. Otherwise, pass
            `k=n` number of indices to load from the available dataset files.
        """
        random.seed(seed)
        indices = list(self.index_paper.keys())
        if k == -1:
            return indices
        assert k <= self.num_papers
        return random.sample(indices, k=k)

    def title(self, index: int = None, paper_id: str = None) -> str:
        return self.load_paper(index, paper_id)["metadata"]["title"]

    def titles(self, indices: List[int] = None, paper_ids: List[str] = None):
        for paper in self.load_papers(indices, paper_ids):
            yield paper["metadata"]["title"]

    def docs(self, indices=None, paper_ids=None, affix="\n"):
        for paper in self.load_papers(indices, paper_ids):
            doc = []
            for key in self.text_keys:
                for line in paper[key]:
                    doc.append(line["text"])
            yield affix.join(doc)

    def lines(self, indices: List[int] = None, paper_ids: List[str] = None):
        for paper in self.load_papers(indices, paper_ids):
            for key in self.text_keys:
                for line in paper[key]:
                    yield line["text"]

    def build(self, indices: List[int], minlen=20, batch_size=10) -> Papers:
        """Return instance of papers with texts transformed to sentences."""
        sample = len(indices)
        with tqdm(total=sample, desc="papers") as pbar:
            index = Sentences(indices)
            cluster = index.init_cluster()
            for i in range(0, sample, batch_size):
                split = indices[i: min(i + batch_size, sample)]
                queue, docs = deque(split), self.docs(split)
                node = 0
                while len(queue) > 0:
                    node = queue.popleft()
                    for sent in self.sentence_tokenizer.tokenize(next(docs)):
                        string = normalize_whitespace(sent.text)
                        string = clean_tokenization(string)
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
        return "CORD19Dataset(papers={}, source={})".format(
            self.num_papers, self.source_name)
