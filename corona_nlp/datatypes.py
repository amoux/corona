from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Sentences:
    indices: List[int] = field(default_factory=list, repr=False)
    counts: int = 0
    maxlen: int = 0
    strlen: int = 0

    def init_cluster(self) -> Dict[int, List[str]]:
        return dict([(index, []) for index in self.indices])

    def __len__(self):
        return self.counts


@dataclass
class Papers:
    sentences: Sentences = field(repr=False)
    cluster: Dict[int, List[str]] = field(repr=False)
    avg_strlen: float = field(init=False)
    num_papers: int = field(init=False)
    num_sents: int = field(init=False)

    def __post_init__(self):
        if isinstance(self.sentences, Sentences):
            for key, val in self.sentences.__dict__.items():
                setattr(self, key, val)
        self.avg_strlen = round(self.strlen / self.counts, 2)
        self.num_papers = len(self.indices)
        self.num_sents = self.counts

    def __len__(self):
        return self.num_papers

    def __getitem__(self, index: int):
        if isinstance(index, int):
            return iter(self.cluster[index])

    def __iter__(self):
        for index in self.cluster:
            for sentence in self.cluster[index]:
                yield sentence