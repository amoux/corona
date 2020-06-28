import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class PaperIndexer:
    def __init__(self, source: Union[str, List[str]],
                 index_start=1, sort_first=False, extension=".json"):
        assert index_start > 0, 'Expected index value to be greater than zero.'
        self.index_start = index_start
        self.extension = extension
        self.is_files_sorted = sort_first
        self._bins: List[int] = []
        self._splits: Optional[List[int]] = None
        self.paths: List[Path] = []
        self.paper_index: Dict[str, int] = {}
        self.index_paper: Dict[int, str] = {}
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
                self._bins.append(len(files))
            else:
                raise ValueError(f"Path, {path} directory not found.")

        self._map_files_to_ids(file_paths)
        if len(self._bins) > 1:
            x = np.array(self._bins).cumsum(0)
            x[-1] += self.index_start
            self._splits = x.tolist()
        del file_paths

    @property
    def num_papers(self) -> int:
        return sum(self._bins)

    @property
    def source_name(self) -> Union[str, List[str]]:
        if len(self.paths) == 1:
            return self.paths[0].name
        return [p.name for p in self.paths]

    def _map_files_to_ids(self, json_files: List[str]) -> None:
        for index, file in enumerate(json_files, self.index_start):
            paper_id = file.name.replace(self.extension, "")
            if paper_id not in self.paper_index:
                self.paper_index[paper_id] = index
                self.index_paper[index] = paper_id

    def _index_dirpath(self, index: int) -> Path:
        # return lower bound if index is less or equal item in first item
        if len(self._bins) == 1 or index < self._bins[0]:
            return self.paths[0]

        # Interpolation search the correct path for a given index by
        # returning the id to path closest to its maximum split size. A
        # split is based on the cumulative sum of each item (a bin is the
        # number of files in n directory). It also adjusts to the index
        # starting position. That is if the number of directories (sources)
        # is higher than one.

        def nearest_mid(start: int, end: int, x: List[int], y: int) -> int:
            m = start + ((end - start) // (x[end] - x[start])) * (y - x[start])
            return m

        bins = self._splits
        size = len(bins) - 1
        maxid = bins[-1]
        first = 0
        last = size

        while first <= last:
            mid = nearest_mid(first, last, x=bins, y=index)
            if mid > last or mid < first:
                return None
            if bins[mid] >= index:
                return self.paths[mid]
            elif index > bins[last - 1] and index <= maxid:
                return self.paths[bins.index(maxid)]
            if index > bins[mid]:
                first = mid + 1
            else:
                last = mid - 1
        if first > last:
            return None

    def _load_data(self, paper_id: str) -> Dict[str, Any]:
        path = self._index_dirpath(self.paper_index[paper_id])
        file_path = path.joinpath(f"{paper_id}{self.extension}")
        with file_path.open("rb") as file:
            return json.load(file)

    def _encode(self, paper_ids: List[str]) -> List[int]:
        pid2idx = self.paper_index
        return [pid2idx[pid] for pid in paper_ids if pid in pid2idx]

    def _decode(self, indices: List[int]) -> List[str]:
        idx2pid = self.index_paper
        return [idx2pid[idx] for idx in indices if idx in idx2pid]

    def load_paper(self, index: int = None,
                   paper_id: str = None) -> Dict[str, Any]:
        """Load a single paper and data by either index or paper ID."""
        if index is not None and isinstance(index, int):
            paper_id = self.index_paper[index]
        return self._load_data(paper_id)

    def load_papers(self, indices: List[int] = None,
                    paper_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Load many papers and data by either indices or paper ID's."""
        if indices is not None:
            if isinstance(indices, list) and isinstance(indices[0], int):
                paper_ids = self._decode(indices)
                return [self._load_data(pid) for pid in paper_ids]
            else:
                raise ValueError("Indices not of type List[int].")

        elif paper_ids is not None:
            if isinstance(paper_ids, list) and isinstance(paper_ids[0], str):
                return [self._load_data(pid) for pid in paper_ids]
            else:
                raise ValueError("Paper ID's not of type List[str].")

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.index_paper[item]
        elif isinstance(item, str):
            return self.paper_index[item]

    def __len__(self):
        return self.num_papers

    def __repr__(self):
        return "PaperIndexer(papers={}, files_sorted={}, source={})".format(
            self.num_papers, self.is_files_sorted, self.source_name)
