import collections
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Tuple, Union)

import numpy as np  # type: ignore

from .utils import DataIO, get_all_store_paths, load_store, save_stores

Pid = int
Sid = int


class GoldPids(NamedTuple):
    task_id: int
    pids: np.ndarray
    dist: np.ndarray

    def size(self) -> int:
        return self.pids.size

    def mindist(self) -> float:
        return self.dist.min().item()

    def maxdist(self) -> float:
        return self.dist.max().item()

    def __repr__(self):
        return '{}(task_id: {}, size: {}, mindist: {}, maxdist: {})'.format(
            self.__class__.__name__, self.task_id, self.size(),
            round(self.mindist(), 4), round(self.maxdist(), 4),
        )


class GoldPidsOutput(List[GoldPids]):
    pids: Optional[List[Pid]] = None

    @property
    def num_tasks(self) -> int:
        return len(self)

    def common(self, topk=10) -> List[Tuple[Pid, int]]:
        counts = Counter(self.sample())
        if topk == -1:
            return counts.most_common()
        return counts.most_common()[:topk]

    def sizes(self) -> List[int]:
        return [gold.size() for gold in self]

    def sample(self) -> Iterable[Pid]:
        for gold in self:
            for pid in gold.pids:
                yield pid.item()

    def iterall(self) -> Iterable[Tuple[int, np.int64, np.float32]]:
        for gold in self:
            for pid, dist in zip(gold.pids, gold.dist):
                yield gold.task_id, pid, dist

    def __repr__(self):
        return '{}(num_tasks: {}, size: {})'.format(
            self.__class__.__name__, len(self), tuple(self.sizes()),
        )


@dataclass
class MetaData:
    sid: int
    pid: int
    loc: int
    strlen: int
    seqlen: int


@dataclass
class Sampler:
    pids: List[int] = field(default_factory=list, repr=False)
    counts: int = 0
    maxlen: int = 0
    seqlen: int = 0
    _store: Dict[Pid, List] = field(default_factory=dict, repr=False)
    _meta: List[MetaData] = field(default_factory=list, repr=False)

    def init(self):
        self._store = dict([(pid, []) for pid in self.pids])
        return self._store

    def sents(self, pid: Pid) -> List[str]:
        return self._store[pid]

    def map(self, function: Callable[[str], str], inplace: bool = False):
        """Apply a function to all the sentences in store."""
        assert callable(function)
        if not inplace:
            clone = Sampler()
            clone.merge_(deepcopy(self))
            for pid in clone.pids:
                clone._store[pid][:] = map(function, clone._store[pid])
            return clone
        else:
            for pid in self.pids:
                self._store[pid][:] = map(function, self._store[pid])
            return

    def include(self, pid: Pid, seqlen: int, text: str) -> None:
        sid = self.counts
        args = (len(self._store[pid]), len(text), seqlen)
        self._meta.append(MetaData(sid, pid, *args))
        self._store[pid].append(text)
        self.maxlen = max(self.maxlen, seqlen)
        self.seqlen += seqlen
        self.counts += 1

    def merge(self, other: 'Sampler') -> 'Sampler':
        copy = deepcopy(self)
        copy.merge_(other)
        return copy

    def merge_(self, other: 'Sampler') -> None:
        assert isinstance(other, Sampler), TypeError
        intersection = set(self.pids).intersection(set(other.pids))
        if intersection:
            raise Exception(
                'Merging intersecting Pid(s) from one or more sampler(s) is'
                f' currently not supported. Tried merging:\n{intersection}'
            )
        self.seqlen += other.seqlen
        self.counts += other.counts
        self.maxlen = max(self.maxlen, other.maxlen)
        self.pids.extend(other.pids)
        self._meta.extend(other._meta)
        self._store.update(other._store)

    def get(self, item: Union[Union[int, slice], Iterable[int]]):
        if isinstance(item, (int, slice)):
            return self.__getitem__(item)
        if isinstance(item, (set, tuple, list)):
            return list(map(self.__getitem__, item))

    def __getitem__(self, item: Union[int, slice]):
        meta = self._meta
        if isinstance(item, int):
            m = meta[item]
            return self._store[m.pid][m.loc]
        if isinstance(item, slice):
            return [self._store[m.pid][m.loc] for m in meta[item]]

    def __iter__(self):
        store = self._store
        for pid in store:
            for sent in store[pid]:
                yield sent

    def __add__(self, other: 'Sampler') -> 'Sampler':
        return self.merge(other)

    def __len__(self):
        return self.counts


def merge_samplers(samplers: List[Sampler]) -> Sampler:
    """Merge a list of instances of Sampler objects into one."""
    if isinstance(samplers, list):
        if not isinstance(samplers[0], Sampler):
            raise TypeError("Expected a List[Sampler], but found "
                            f"a List[{type(samplers[0])}] instead.")
    root = Sampler()
    for sampler in samplers:
        root.merge_(sampler)
    return root


class SentenceStore:
    def __init__(
        self,
        pids: List[Pid],
        counts: int,
        maxlen: int,
        seqlen: int,
        store: Dict[Pid, List[str]],
        meta: List[MetaData],
    ) -> None:
        self.pids = pids
        self.counts = counts
        self.seqlen = seqlen
        self.maxlen = maxlen
        self._store = store
        self._meta = meta
        self.avg_seqlen = round(self.seqlen / self.counts, 2)
        self.num_papers = len(self.pids)
        self.num_sents = self.counts
        self.num_tokens = self.seqlen
        self.init_args = None

    def decode(self, sids: Union[Sid, List[Sid]]) -> Union[Pid, List[Pid]]:
        """Decode an single or an iterable of sentence-ids to paper-ids."""
        if isinstance(sids, Sid):
            m = self._meta[sids]
            return m.pid
        else:
            pids = []
            for sid in sids:
                m = self._meta[sid]
                pids.append(m.pid)
            return pids

    def string(self, sid: Sid) -> str:
        """Retrive a single string from a sentence ID.
        * Same as `self[sid]`
        """
        m = self._meta[sid]
        return self._store[m.pid][m.loc]

    def largest(self, topk: int = 10) -> List[Tuple[Pid, int]]:
        """Return an iterable of paper ids and their number of sentences
        for that paper id; from biggest to the smallest.
        """
        count = collections.Counter([m.pid for m in self._meta])
        if topk == -1:
            return count.most_common()
        else:
            return count.most_common(topk)

    def lookup(self, sids: Union[Sid, List[Sid]], mode="inline"):
        """Lookup paper-ids by a single or list of sentence ids.

        :param mode: Data format. `inline` has index keys: `pid, sid, loc`
         and `table` has paper-ids as keys and sentence-ids as values (note
         that table is the only format that adds by id - no-repeats). And
         `index` is a list of integers of paper-ids (for all sentence-ids).

        - modes:
          - `inline`: List[Dict[str, Union[Pid, Tuple[Pid, Sid]]]]
          - `table` : Dict[Pid, List[Sid]]
          - `index` : List[Pid]

        """
        is_single_input = False
        if isinstance(sids, Sid):
            sids = [sids]
            is_single_input = True

        inline: Optional[List[Dict[str, Union[Pid, Tuple[Pid, Sid]]]]] = None
        table: Optional[Dict[Pid, List[Sid]]] = None
        index: Optional[List[Pid]] = None

        if mode == "inline":
            inline = []
        elif mode == "index":
            index = []
        else:
            table = {}

        for sid in sids:
            m = self._meta[sid]
            if inline is not None:
                inline.append({
                    "pid": m.pid,
                    "sid": m.sid,
                    "loc": (m.pid, m.loc),
                })
            elif index is not None:
                index.append(m.pid)
            elif table is not None:
                if m.pid not in table:
                    table[m.pid] = [m.loc]
                else:
                    table[m.pid].append(m.loc)

        if inline is not None:
            if is_single_input:
                return inline[0]
            return inline
        elif index is not None:
            return index
        else:
            return table

    def sents(self, pid: Pid) -> List[str]:
        """Retrive all sentences belonging to the given paper ID."""
        return self._store[pid]

    def to_disk(self, path: Optional[str] = None, store_name: Optional[str] = None):
        """Save the current state to a directory."""
        if store_name is not None:
            if isinstance(store_name, str):
                save_stores(sents=self, store_name=store_name)
            else:
                save_stores(sents=self)
        elif path is not None:
            DataIO.save_data(path, self)

    @staticmethod
    def from_sampler(sampler: Union[Sampler, List[Sampler]]) -> 'SentenceStore':
        if isinstance(sampler, list):
            sampler = merge_samplers(samplers=sampler)
        if isinstance(sampler, Sampler):
            return SentenceStore(
                sampler.pids, sampler.counts, sampler.maxlen,
                sampler.seqlen, sampler._store, sampler._meta,
            )

    @staticmethod
    def from_disk(path_or_store_name: Optional[str] = None):
        """Load the state from a directory.

        :param path_or_store_name: A custom path to the file saved using
            the method; `self.to_disk("my/path/to/filename.pkl")` or a
            store-name with the format; `YY-MM-DD` | `20-10-12` - if saved
            without adding a custom-name to the store. Otherwise, pass the
            name of the store you assigned, e.g., `my_store.` If None, then
            the last saved will be loaded automatically if possible (whether
            it was saved automatically with a date or given a custom name).
        """
        if path_or_store_name is None:
            return load_store('sents')
        try:
            store_names = get_all_store_paths()
        except FileNotFoundError:
            return DataIO.load_data(path_or_store_name)
        if path_or_store_name in store_names:
            return load_store('sents', path_or_store_name)
        return None

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
            'text_key': cord19.text_key,
            'index_start': cord19.index_start,
            'sort_first': cord19.is_files_sorted,
            'nlp_model': cord19.sentencizer.nlp_model,
        }

    def init_cord19_dataset(self):
        if not hasattr(self, 'init_args') or self.init_args is None:
            raise AttributeError(
                "Current `self` was not saved w/or self.attach_init_args() "
                "hasn't been called to attach `init_args` attr this `self`."
            )
        from .dataset import CORD19
        return CORD19(**self.init_args)

    def index_select(self, x, reverse=False, shuffle=False) -> Iterable[str]:
        child_ids = None
        # filter unique ids (as we dont want to iterate a pid more than once).
        if isinstance(x, GoldPidsOutput):
            if x.pids is not None:
                child_ids = x.pids
            else:
                child_ids = [i for i, _ in x.common(topk=-1)]
        else:
            common = []
            for pid in x:
                if pid in common:
                    continue
                common.append(pid)
            child_ids = common

        def selected(cache_ids: List[Pid]):
            for pid in cache_ids:
                for sent in self._store[pid]:
                    yield sent

        if child_ids is not None:
            if reverse:
                child_ids.sort(reverse=True)
            if shuffle:
                random.shuffle(child_ids)
            iter_selected = selected(child_ids)
            return iter_selected
        return None

    def map(self, function: Callable[[str], str], inplace: bool = False):
        """Apply a function to all the sentences in store."""
        assert callable(function)
        if not inplace:
            clone = deepcopy(self)
            for pid in clone.pids:
                clone._store[pid][:] = map(function, clone._store[pid])
            return clone
        else:
            for pid in self.pids:
                self._store[pid][:] = map(function, self._store[pid])
            return

    def get(self, item: Union[Union[int, slice], Iterable[int]]):
        if isinstance(item, (int, slice)):
            return self.__getitem__(item)
        if isinstance(item, (set, tuple, list)):
            return list(map(self.__getitem__, item))

    def __len__(self) -> int:
        return self.num_sents

    def __contains__(self, pid: Pid) -> bool:
        if isinstance(pid, Pid):
            return pid in self._store

    def __getitem__(self, item: Union[Sid, slice]) -> Union[str, List[str]]:
        meta, store = self._meta, self._store
        if isinstance(item, int):
            m = meta[item]
            return store[m.pid][m.loc]
        if isinstance(item, slice):
            return [store[m.pid][m.loc] for m in meta[item]]

    def __iter__(self):
        store = self._store
        for pid in store:
            for sent in store[pid]:
                yield sent

    def __repr__(self):
        s = '{}(avg_seqlen: {} | num_papers: {:,} | num_sents: {:,} | num_tokens: {:,})'
        return s.format(self.__class__.__name__, self.avg_seqlen,
                        self.num_papers, self.num_sents, self.num_tokens)
