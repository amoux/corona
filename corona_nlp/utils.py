import pickle
from collections import namedtuple
from pathlib import Path
from typing import IO, Any, List, NamedTuple, Tuple

import numpy as np

Cord19Paths = NamedTuple(
    'Cord19Paths', [
        ('readme', Path), ('metadata', Path), ('dirs', List[Path]),
        ('pmc_custom_license', Path),
        ('biorxiv_medrxiv', Path),
        ('comm_use_subset', Path),
        ('noncomm_use_subset', Path), ])


def split_dataset(dataset: List[Any],
                  subset: float = 0.8,
                  samples: int = None,
                  seed: int = 12345) -> Tuple[List[Any], List[Any]]:
    """Split an iterable dataset into a train and evaluation sets."""
    np.random.seed(seed)
    np.random.shuffle(dataset)
    maxlen = len(dataset)
    if not samples or samples > maxlen:
        samples = maxlen
    split = int(subset * samples)
    train_data = dataset[:split]
    test_data = dataset[split:samples]
    return train_data, test_data


def save_train_test(texts: List[str],
                    out_dir="data",
                    subset=0.8,
                    files=["train.txt", "test.txt"]) -> IO:
    """Split an iterable of string sequences and save train and test to file"""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    x, y = split_dataset(texts, subset=subset)
    for file, dataset in zip(files, [x, y]):
        filepath = out_dir.joinpath(file)
        with filepath.open("w") as file:
            for line in file:
                file.write(f"{line}\n")


class DataIO:
    @staticmethod
    def save_data(file_path: str, data_obj: Any) -> IO:
        file_path = Path(file_path)
        if file_path.is_dir():
            if not file_path.exists():
                file_path.mkdir(parents=True)
        with file_path.open("wb") as pkl:
            pickle.dump(data_obj, pkl, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(file_path: str) -> Any:
        file_path = Path(file_path)
        with file_path.open("rb") as pkl:
            return pickle.load(pkl)


def calc_chunksize_info(workers: int, n_samples: int, factor=4):
    """Calculate chunksize numbers."""
    Chunkinfo = namedtuple(
        'Chunkinfo', ['workers', 'n_samples', 'n_chunks',
                      'chunksize', 'last_chunk'])
    chunksize, extra = divmod(n_samples, workers * factor)
    if extra:
        chunksize += 1
    n_chunks = n_samples // chunksize + (n_samples % chunksize > 0)
    last_chunk = n_samples % chunksize or chunksize
    return Chunkinfo(workers, n_samples,
                     n_chunks, chunksize, last_chunk)


def load_dataset_paths(basedir: str) -> Cord19Paths:
    """Return an organized representation of all paths in the dataset.

    ```python
    basedir = "path/to/CORD-19-research-challenge/2020-03-13/"
    load_dataset_paths(basedir)._fields
    ...
        ('readme', 'metadata', 'dirs',
        'pmc_custom_license',
        'biorxiv_medrxiv',
        'comm_use_subset',
        'noncomm_use_subset')
    ```
    """
    basedir = Path(basedir)
    paths, filesdir = {}, []
    for p in basedir.iterdir():
        if p.suffix == '.csv':
            paths['metadata'] = p
        elif p.suffix == '.readme':
            paths['readme'] = p
        elif p.is_dir():
            dirdir = p.joinpath(p.name)
            if dirdir.is_dir():
                filesdir.append(dirdir)

    paths['dirs'] = filesdir
    for p in filesdir:
        paths[p.name] = p
    return Cord19Paths(**paths)
