import collections
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from .utils import DataIO


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
    _meta: List[Tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.sentences, Sentences):
            for key, val in self.sentences.__dict__.items():
                setattr(self, key, val)
        self.avg_strlen = round(self.strlen / self.counts, 2)
        self.num_papers = len(self.indices)
        self.num_sents = self.counts
        self._meta = list(self._edges())

    def _edges(self):
        for i in self.indices:
            for j in range(0, len(self.cluster[i])):
                yield (i, j)

    def string(self, sent_id: int) -> str:
        """Retrive a single string from a sentence ID.

        * Same as `self[sent_id]`
        """
        return self[sent_id]

    def largest(self, topk: int = 10) -> List[Tuple[int, int]]:
        """Return an iterable of paper ids and their number of sentences
        for that paper id; from biggest to the smallest.
        """
        count = collections.Counter([m[0] for m in self._meta])
        if topk == -1:
            return count.most_common()
        else:
            return count.most_common(topk)

    def lookup(self, sent_ids: Union[int, List[int]], mode="inline"):
        """Lookup paper-ids by a single or list of sentence ids.

        :param mode: Data format. `inline` has index keys: `pid, sid, loc`
         and `table` has paper-ids as keys and sentence-ids as values.
        - modes:
          - `inline`: List[Dict[str, Union[int, Tuple[int, int]]]]
          - `table` : Dict[int, List[int]]
        """
        is_single_input = False
        if isinstance(sent_ids, int):
            sent_ids = [sent_ids]
            is_single_input = True

        inline: List[Dict[str, Union[int, Tuple[int, int]]]] = None
        table: Dict[int, List[int]] = None
        if mode == "inline":
            inline = []
        else:
            table = {}
        for sent_id in sent_ids:
            pid, item = self._meta[sent_id]
            if table is None:
                inline.append({
                    "pid": pid,
                    "sid": sent_id,
                    "loc": (pid, item),
                })
            elif inline is None:
                if pid not in table:
                    table[pid] = [sent_id]
                else:
                    table[pid].append(sent_id)
        if table is None:
            if is_single_input:
                return inline[0]
            return inline
        else:
            return table

    def sents(self, paper_id: int) -> List[str]:
        """Retrive all sentences belonging to the given paper ID."""
        return self.cluster[paper_id]

    def to_disk(self, path: str):
        """Save the current state to a directory."""
        DataIO.save_data(path, self)

    @staticmethod
    def from_disk(path: str):
        """Load the state from a directory."""
        return DataIO.load_data(path)

    def attach_init_args(self, cord19: 'CORD19Dataset') -> None:
        """Attach the initialization **kwargs used in CORD19Dataset.__init__.

        * An attribute `init_args` of type `Dict[str, Any]` added with the
        initialization keyword arguments, which can later be used to load
        the state of parameters responsible for the origin of this "self".
        """
        setattr(self, 'init_args', dict(
            source=cord19.paths,
            text_keys=cord19.text_keys,
            index_start=cord19.index_start,
            sort_first=cord19.is_files_sorted,
            nlp_model=cord19.sentence_tokenizer.nlp_model,
        ))

    def init_cord19_dataset(self) -> 'CORD19Dataset':
        if not hasattr(self, 'init_args'):
            raise AttributeError(
                "Current `self` was not saved w/or self.attach_init_args() "
                "hasn't been called to attach `init_args` attr this `self`."
            )
        from .dataset import CORD19Dataset
        return CORD19Dataset(**self.init_args)

    def __len__(self):
        return self.num_sents

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self.cluster[i[0]][i[1]] for i in self._meta[item]]
        p_id, item = self._meta[item]
        return self.cluster[p_id][item]

    def __iter__(self):
        for p_id in self.cluster:
            for sent in self.cluster[p_id]:
                yield sent


def merge_papers(papers: List[Papers]) -> Papers:
    """Merge a list of instances of Papers into one."""
    if isinstance(papers, list):
        if not isinstance(papers[0], Papers):
            raise TypeError("Expected a List[Papers], but found "
                            f"a List[{type(papers[0])}] instead.")
    i = Sentences()
    c = i.init_cluster()
    for p in papers:
        i.strlen += p.strlen
        i.counts += p.counts
        i.maxlen = max(i.maxlen, p.maxlen)
        i.indices.extend(p.indices)
        c.update(p.cluster)
    return Papers(i, c)
