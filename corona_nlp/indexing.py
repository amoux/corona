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
            x = self._bins.copy()
            x[0] += self.index_start - 1
            self._splits = np.array(x).cumsum(0).tolist()
        del file_paths

    @property
    def num_papers(self) -> int:
        return sum(self._bins)

    @property
    def source_name(self) -> Union[str, List[str]]:
        if len(self.paths) == 1:
            return self.paths[0].name
        return [p.name for p in self.paths]

    def file_path(self, id: Union[int, str]) -> Path:
        """Return the path to file from an integer or string ID."""
        if isinstance(id, str):
            id = self.paper_index[id]
        file = self.index_paper[id]
        path = self._index_dirpath(id)
        return path.joinpath(file + self.extension)

    def _map_files_to_ids(self, json_files: List[str]) -> None:
        for index, file in enumerate(json_files, self.index_start):
            paper_id = file.name.replace(self.extension, "")
            if paper_id not in self.paper_index:
                self.paper_index[paper_id] = index
                self.index_paper[index] = paper_id

        ids = self.index_paper.keys()
        setattr(self, "first_index", min(ids))
        setattr(self, 'last_index', max(ids))
        del ids

    def _index_dirpath(self, index: int) -> Path:
        # return lower bound if one source or index is less or equal to the
        # first item.
        if len(self._bins) == 1 or index <= self._splits[0]:
            return self.paths[0]

        # Interpolation search - searches the correct path for a given index
        # by returning the id to path closest to its maximum split size. A
        # split is based on the cumulative sum of each item (a bin is the
        # number of files in n directory), after the first item value.

        def nearest_mid(start: int, end: int, x: List[int], y: int) -> int:
            m = start + ((end - start) // (x[end] - x[start])) * (y - x[start])
            return m

        splits = self._splits
        size = len(splits) - 1
        first = 0
        last = size

        while first <= last:
            mid = nearest_mid(first, last, x=splits, y=index)
            if mid > last or mid < first:
                return None
            if splits[mid] >= index:
                return self.paths[mid]
            elif index > splits[last - 1] and index <= self.last_index:
                return self.paths[splits.index(self.last_index)]
            if index > splits[mid]:
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
