import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

from .indexing import PaperIndexer
from .preprocessing import load_papers_with_text, normalize_whitespace
from .tokenizer import SpacySentenceTokenizer


@dataclass
class Sentences:
    indices: List[str] = field(repr=False, default=list)
    counts: int = 0
    maxlen: int = 0
    strlen: int = 0

    def init_cluster(self) -> Dict[int, List[str]]:
        cluster_shape = zip(self.indices, range(len(self.indices)))
        return dict([(index, []) for index, _ in cluster_shape])

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


class Cord19Dataset(PaperIndexer):
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

    def sents(self, indices: List[int], minlen=20, maxlen=2000) -> Papers:
        """Return instance of papers with texts transformed to sentences."""
        sentences = Sentences(indices)
        cluster = sentences.init_cluster()
        for i, line in tqdm(zip(indices, self.texts(indices)), desc="papers"):
            line = normalize_whitespace(line)
            if len(line) <= maxlen:
                tokens = self.tokenize(line)
                for token in tokens:
                    string = normalize_whitespace(token.text)
                    if len(string) >= minlen:
                        if string not in cluster[i]:
                            length = len(string)
                            sentences.strlen += length
                            sentences.counts += 1
                            sentences.maxlen = max(length, sentences.maxlen)
                            cluster[i].append(string)

        return Papers(sentences, cluster=cluster)
