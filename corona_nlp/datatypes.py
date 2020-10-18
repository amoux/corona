import collections
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Tuple, Union

from nltk.tokenize import word_tokenize

from .utils import DataIO


@dataclass
class Sentences:
    indices: List[int] = field(default_factory=list, repr=False)
    counts: int = 0
    maxlen: int = 0
    seqlen: int = 0

    def init_cluster(self) -> Dict[int, List[Any]]:
        return dict([(index, []) for index in self.indices])

    def __len__(self):
        return self.counts


@dataclass
class Papers:
    sentences: Sentences = field(repr=False)
    cluster: Dict[int, List[str]] = field(repr=False)
    avg_seqlen: float = field(init=False)
    num_papers: int = field(init=False)
    num_sents: int = field(init=False)
    num_tokens: int = field(init=False)
    _meta: List[Tuple[int, int]] = field(init=False, repr=False)
    init_args: Optional[Dict[str, Any]] = field(
        init=False, repr=False, default=None,
    )

    def __post_init__(self):
        if isinstance(self.sentences, Sentences):
            for key, val in self.sentences.__dict__.items():
                setattr(self, key, val)
        self.avg_seqlen = round(self.seqlen / self.counts, 2)
        self.num_papers = len(self.indices)
        self.num_sents = self.counts
        self.num_tokens = self.seqlen
        self._meta = list(self._edges())

    def _edges(self):
        for i in self.indices:
            for j in range(0, len(self.cluster[i])):
                yield (i, j)

    def string(self, sent_id: int) -> str:
        """Retrive a single string from a sentence ID.
        * Same as `self[sent_id]`
        """
        pid, item = self._meta[sent_id]
        return self.cluster[pid][item]

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
         and `table` has paper-ids as keys and sentence-ids as values (note
         that table is the only format that adds by id - no-repeats). And
         `index` is a list of integers of paper-ids (for all sentence-ids).

        - modes:
          - `inline`: List[Dict[str, Union[int, Tuple[int, ...]]]]
          - `table` : Dict[int, List[int]]
          - `index` : List[int]

        """
        is_single_input = False
        if isinstance(sent_ids, int):
            sent_ids = [sent_ids]
            is_single_input = True

        inline: Optional[List[Dict[str, Union[int, Tuple[int, ...]]]]] = None
        table: Optional[Dict[int, List[int]]] = None
        index: Optional[List[int]] = None
        if mode == "inline":
            inline = []
        elif mode == "index":
            index = []
        else:
            table = {}

        for sent_id in sent_ids:
            pid, item = self._meta[sent_id]
            if inline is not None:
                inline.append({
                    "pid": pid,
                    "sid": sent_id,
                    "loc": (pid, item),
                })
            elif index is not None:
                index.append(pid)
            elif table is not None:
                if pid not in table:
                    table[pid] = [sent_id]
                else:
                    table[pid].append(sent_id)

        if inline is not None:
            if is_single_input:
                return inline[0]
            return inline
        elif index is not None:
            return index
        else:
            return table

    def sents(self, paper_id: int) -> List[str]:
        """Retrive all sentences belonging to the given paper ID."""
        return self.cluster[paper_id]

    def to_disk(self, path: str):
        """Save the current state to a directory."""
        DataIO.save_data(path, self)

    @staticmethod
    def from_disk(path: str) -> 'Papers':
        """Load the state from a directory."""
        return DataIO.load_data(path)

    def attach_init_args(self, cord19) -> None:
        """Attach initialization keyword arguments to self (Papers instance).

        - An attribute `init_args` of type `Dict[str, Any]` added with the
        initialization keyword arguments, which can later be used to load
        the state of parameters responsible for the origin of this "self".
        """
        source = cord19.paths
        if isinstance(source[0], Path):
            # detach posix from paths (for compatibility with other os env)
            source = [p.as_posix() for p in source]

        self.init_args = {
            'source': source,
            'text_keys': cord19.text_keys,
            'index_start': cord19.index_start,
            'sort_first': cord19.is_files_sorted,
            'nlp_model': cord19.sentence_tokenizer.nlp_model,
        }

    def init_cord19_dataset(self):
        if not hasattr(self, 'init_args') or self.init_args is None:
            raise AttributeError(
                "Current `self` was not saved w/or self.attach_init_args() "
                "hasn't been called to attach `init_args` attr this `self`."
            )
        from .dataset import CORD19Dataset
        return CORD19Dataset(**self.init_args)

    def index_select(self, ids, reverse=False, shuffle=False):
        if hasattr(ids, 'sample'):
            ids = list(ids.sample())

        child_ids = [pid for pid, _ in Counter(ids).most_common()]
        if reverse:
            child_ids.sort(reverse=True)
        elif shuffle:
            random.shuffle(child_ids)

        select = Sentences(child_ids)
        children = select.init_cluster()
        for parent in select.indices:
            for sentence in self.cluster[parent]:
                seqlen = len(word_tokenize(sentence))
                select.seqlen += seqlen
                select.counts += 1
                select.maxlen = max(select.maxlen, seqlen)
                children[parent].append(sentence)

        papers = Papers(select, children)
        if self.init_args is not None:
            papers.init_args = self.init_args
        return papers

    def __len__(self) -> int:
        return self.num_sents

    def __contains__(self, item: int) -> bool:
        if isinstance(item, int):
            return item in self.cluster

    def __getitem__(self, item: int) -> Union[List[str], str]:
        if isinstance(item, slice):
            return [self.cluster[i[0]][i[1]] for i in self._meta[item]]
        if isinstance(item, int):
            pid, item = self._meta[item]
            return self.cluster[pid][item]

    def __iter__(self):
        for pid in self.cluster:
            for sent in self.cluster[pid]:
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
        i.seqlen += p.seqlen
        i.counts += p.counts
        i.maxlen = max(i.maxlen, p.maxlen)
        i.indices.extend(p.indices)
        c.update(p.cluster)
    return Papers(i, c)
