import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore

Pid = int
Uid = str


def fit_index_ivf_fpq(embedding, k=8, nlist=4096, m=32):
    import faiss
    _, d = embedding.shape
    innerprod = faiss.METRIC_INNER_PRODUCT
    quantizer = faiss.IndexHNSWFlat(d, m, innerprod)
    index_fpq = faiss.IndexIVFPQ(quantizer, d, nlist, m*2, k, innerprod)
    index_fpq.verbose = True
    index_fpq.train(embedding)
    index_fpq.add(embedding)
    assert index_fpq.ntotal == embedding.shape[0]
    return index_fpq


def fit_index_ivf_hnsw(
    embedding: np.array,
    metric: Union[str, int, "faiss.METRIC"] = "l2",
    nlist: Optional[int] = None,
    m: int = 32,
) -> "faiss.IndexIVFFlat":
    import faiss  # type: ignore
    if isinstance(metric, str):
        if metric.lower().strip() == "l1":
            metric = faiss.METRIC_L1
        elif metric.lower().strip() == "l2":
            metric = faiss.METRIC_L2
    assert isinstance(metric, int)

    n, d = embedding.shape
    if nlist is None:
        nlist = int(np.sqrt(n))
    quantizer = faiss.IndexHNSWFlat(d, m)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    index_ivf.verbose = True
    index_ivf.train(embedding)
    index_ivf.add(embedding)
    assert index_ivf.ntotal == embedding.shape[0]
    return index_ivf


class PaperIndexer:
    def __init__(self, source: Union[str, List[str]],
                 index_start=1, sort_first=False, extension=".json"):
        assert index_start > 0, 'Expected index value to be greater than zero.'
        self.index_start = index_start
        self.extension = extension
        self.is_files_sorted = sort_first
        self.first_index: int = 0
        self.last_index: int = 0
        self.bins: List[int] = []
        self.splits: List[int] = []
        self.paths: List[Path] = []
        self.uid2pid: Dict[Uid, Pid] = {}
        self.pid2uid: Dict[Pid, Uid] = {}
        if not isinstance(source, list):
            source = [source]
        file_paths = []
        for path in source:
            path = Path(path)
            if path.is_dir():
                files = [file for file in path.glob(f"*{extension}")]
                if sort_first:
                    files.sort()
                file_paths.extend(files)
                self.paths.append(path)
                self.bins.append(len(files))
            else:
                raise ValueError(f"Path, {path} directory not found.")

        self._map_files_to_ids(file_paths)
        if len(self.bins) > 1:
            b = self.bins
            b[0] += self.index_start - 1
            self.splits = np.array(b).cumsum(0).tolist()
        else:
            self.splits = self.bins
        del file_paths

    def _map_files_to_ids(self, json_files: List[str]) -> None:
        for pid, file in enumerate(json_files, self.index_start):
            uid = file.name.replace(self.extension, "")
            try:
                uid = self.pid2uid[pid]
            except KeyError:
                self.pid2uid[pid] = uid
            finally:
                self.uid2pid[uid] = pid

        self.first_index = min(self.pid2uid)
        self.last_index = max(self.pid2uid)

    def _index_dirpath(self, pid: Pid) -> Union[Path, None]:
        # Lower bound if one source or pid is less or equal to the first item.
        if len(self.bins) == 1 or pid <= self.splits[0]:
            return self.paths[0]
        # Interpolation search - search the correct path for a given index by
        # returning the id to path closest to its maximum split size. A split
        # is based on the cumulative sum of each bin (bin: number of files in
        # a directory), after the first item value.

        def nearest_mid(start: int, end: int, x: List[int], y: int) -> int:
            m = start + ((end - start) // (x[end] - x[start])) * (y - x[start])
            return m

        splits = self.splits
        size = len(splits) - 1
        first = 0
        last = size
        while first <= last:
            mid = nearest_mid(first, last, x=splits, y=pid)
            if mid > last or mid < first:
                return None
            if splits[mid] >= pid:
                return self.paths[mid]
            elif pid > splits[last - 1] and pid <= self.last_index:
                return self.paths[splits.index(self.last_index)]
            if pid > splits[mid]:
                first = mid + 1
            else:
                last = mid - 1
        if first > last:
            return None

    def _load_data(self, uid: Uid) -> Dict[str, Any]:
        path = self._index_dirpath(self.uid2pid[uid])
        fp = path.joinpath(f"{uid}{self.extension}")
        with fp.open("rb") as file:
            return json.load(file)

    def _encode(self, uids: List[Uid]) -> List[Pid]:
        return [self.uid2pid[uid] for uid in uids if uid in self.uid2pid]

    def _decode(self, pids: List[Pid]) -> List[Uid]:
        return [self.pid2uid[pid] for pid in pids if pid in self.pid2uid]

    @property
    def num_papers(self) -> int:
        return sum(self.bins)

    @property
    def source_name(self) -> Union[str, List[str]]:
        if len(self.paths) == 1:
            return self.paths[0].name
        return [p.name for p in self.paths]

    def file_path(self, id: Union[Pid, Uid]) -> Path:
        """Return the path to file from an integer or string ID."""
        if isinstance(id, Uid):
            id = self.uid2pid[id]
        assert isinstance(id, Pid)
        file = self.pid2uid[id]
        path = self._index_dirpath(id)
        return path.joinpath(file + self.extension)

    def sample(self, k: int = None, s: int = None, seed: int = None):
        """Return a sample (all|random k) or split of paper ID's.

        :param k: number of ids to return from all samples, if `k=-1` then all
            ids are returned sorted. Otherwise, if `k < max ids` -> shuffled.
        :param s: return a split of all ids @ `s` e.g., if s=1 then all ids@1.
        """
        if k is not None:
            random.seed(seed)
            ids = list(self.pid2uid.keys())
            if k == -1:
                return ids
            assert k <= self.num_papers
            return random.sample(ids, k=k)

        if s is not None:
            splits = self.splits
            assert s <= len(splits), f'Expected `s` between: [0,{len(splits)}]'
            if s == 0:
                return list(range(self.index_start, splits[s] + 1))
            if s > 0:
                return list(range(splits[s - 1] + 1, splits[s] + 1))

    def load_paper(self, pid: Optional[Pid] = None, uid: Optional[Uid] = None,
                   ) -> Dict[str, Any]:
        """Load a single paper and data by either index or paper ID."""
        if pid is not None and isinstance(pid, Pid):
            uid = self.pid2uid[pid]
        assert isinstance(uid, Uid)
        return self._load_data(uid)

    def load_papers(self, pids: Optional[List[Pid]] = None, uids: Optional[List[Uid]] = None,
                    ) -> List[Dict[str, Any]]:
        """Load many papers and data by either pids or paper ID's."""
        if pids is not None:
            if isinstance(pids, list) and isinstance(pids[0], Pid):
                uids = self._decode(pids)
                return [self._load_data(pid) for pid in uids]
            else:
                raise ValueError("Indices not of type List[int].")

        elif uids is not None:
            if isinstance(uids, list) and isinstance(uids[0], Uid):
                return [self._load_data(pid) for pid in uids]
            else:
                raise ValueError("Paper ID's not of type List[str].")

    def __getitem__(self, id: Union[Pid, Uid]) -> Union[Uid, Pid, None]:
        if isinstance(id, Pid):
            return self.pid2uid[id]
        if isinstance(id, Uid):
            return self.uid2pid[id]
        return None

    def __len__(self) -> int:
        return self.num_papers

    def __repr__(self):
        multi_src = "[\n  {},\n]"  # Template for a list of sources.
        src = self.source_name if isinstance(self.source_name, str) \
            else multi_src.format(', '.join(self.source_name))
        return "{}(papers: {}, files_sorted: {}, source: {})".format(
            self.__class__.__name__, self.num_papers, self.is_files_sorted, src,
        )
