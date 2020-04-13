import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm.auto import tqdm

from .indexing import PaperIndexer
from .preprocessing import normalize_whitespace
from .tokenizer import SpacySentenceTokenizer


@dataclass
class Sentences:
    indices: List[str] = field(repr=False, default=list)
    counts: int = 0
    maxlen: int = 0
    strlen: int = 0

    def init_cluster(self) -> Dict[int, List[str]]:
        return dict([(index, []) for index in self.indices])

    def __len__(self):
        return len(self.counts)


@dataclass
class Papers:
    sentences: Sentences = field(repr=False)
    cluster: Dict[int, List[str]] = field(repr=False)
    avglen: float = field(init=False)
    num_papers: int = field(init=False)
    num_sents: int = field(init=False)

    def __post_init__(self):
        if isinstance(self.sentences, Sentences):
            for key, val in self.sentences.__dict__.items():
                setattr(self, key, val)
        self.avglen = round(self.strlen / self.counts, 2)
        self.num_papers = len(self.indices)
        self.num_sents = self.counts

    def __len__(self):
        return self.num_papers

    def __getitem__(self, index):
        if isinstance(index, int):
            return iter(self.cluster[index])

    def __iter__(self):
        for index in self.cluster:
            for sentence in self.cluster[index]:
                yield sentence


class CORD19Dataset(PaperIndexer):
    def __init__(self, source_dir: Path, text_keys=("abstract", "body_text")):
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

    def title(self, index: int = None, paper_id: str = None) -> None:
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
            sents_i = Sentences(indices)
            cluster = sents_i.init_cluster()
            for index in range(0, samples, batch_size):
                split = indices[index: min(index + batch_size, samples)]
                queue, node = (deque(split), 0)
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
                                sents_i.strlen += length
                                sents_i.counts += 1
                                sents_i.maxlen = max(length, sents_i.maxlen)
                                cluster[node].append(string)
                pbar.update(len(split))
        return Papers(sents_i, cluster)

    def __repr__(self):
        return f"<CORD19Dataset({self.source_name}, papers={self.num_papers})>"
