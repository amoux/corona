import pickle
from collections import namedtuple
from pathlib import Path
from typing import IO, Any, List, Tuple

import numpy as np


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
    def save_data(file_name: str, data_obj: Any, dirname="data") -> IO:
        path = Path(dirname).absolute()
        if not path.exists():
            path.mkdir(parents=True)
        file = path.joinpath(file_name)
        with file.open("wb") as pkl:
            pickle.dump(data_obj, pkl, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(file_name: str, dirname="data") -> Any:
        path = Path(dirname).absolute()
        file = path.joinpath(file_name)
        with file.open("rb") as pkl:
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
