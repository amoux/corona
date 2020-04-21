import pickle
import re
from pathlib import Path
from typing import IO, Any, Dict, List, NamedTuple, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .indexing import PaperIndexer
from .jsonformatter import generate_clean_df

Cord19Paths = NamedTuple(
    'Cord19Paths', [
        ('readme', Path), ('metadata', Path), ('dirs', List[Path]),
        ('pmc_custom_license', Path),
        ('biorxiv_medrxiv', Path),
        ('comm_use_subset', Path),
        ('noncomm_use_subset', Path), ])


def normalize_whitespace(string: str) -> str:
    """Normalize excessive whitespace."""
    linebreak = re.compile(r"(\r\n|[\n\v])+")
    nonebreaking_space = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)
    return nonebreaking_space.sub(" ", linebreak.sub(r"\n", string)).strip()


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


def papers_to_csv(sources: Union[str, Cord19Paths],
                  dirs: Tuple[Sequence[str]] = ('all',),
                  out_dir='data') -> None:
    """Convert one or more directories with json files into a csv file(s).

    `sources`: Path to the `CORD-19-research-challenge/2020-03-13/` dataset
        directory, or an instance of `Cord19Paths`.

    `dirs`: Use `all` for all available directories or a sequence of the names.
        You can pass the full name or the first three characters e.g., `('pmc
        ', 'bio', 'com', 'non')`

    `out_dir`: Directory where the csv files will be saved.
    """
    if isinstance(sources, str):
        if not Path(sources).exists():
            raise ValueError("Invalid path, got {sources}")
        sources = load_dataset_paths(sources)
    assert isinstance(sources, Cord19Paths)

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    metadata = pd.read_csv(sources.metadata)
    has_full_text = metadata.loc[metadata["has_full_text"] == True, ["sha"]]
    has_full_text = list(set(has_full_text["sha"].to_list()))

    def with_full_text(index: PaperIndexer) -> List[int]:
        # filter only paper-id's if full-text is available in the metadata
        indices = []
        for paper_id in has_full_text:
            if paper_id in index.paper_index:
                paper_id = index[paper_id]
                if paper_id in indices:
                    continue
                indices.append(paper_id)
        return indices

    if len(dirs) == 4 or 'all' in dirs:
        sources = sources.dirs
    else:
        sources = [d for d in sources.dirs if d.name[:3] in dirs]

    for path in sources:
        index = PaperIndexer(path)
        papers = index.load_papers(with_full_text(index))
        df = generate_clean_df(papers)

        filepath = out_dir.joinpath(f"{index.source_name}.csv")
        df.to_csv(filepath, index=False)
        print("All {} files from directory {} saved in: {}".format(
            index.num_papers, index.source_name, filepath))


def concat_csv_files(
    source_dir: str,
    file_name="covid-lg.csv",
    out_dir="data",
    drop_cols=["raw_authors", "raw_bibliography"],
    return_df=False,
):
    """Concat all CSV files into one single file.

    return_df: If True, saving to file is ignored and the pandas
        DataFrame instance holding the data is returned.

    Usage:
        >>> concat_csv_files('path/to/csv-files-dir/', out_dir='data')
    """
    dataframes = []
    for csv_file in Path(source_dir).glob("*.csv"):
        df = pd.read_csv(csv_file, index_col=None, header=0)
        df.drop(columns=drop_cols, inplace=True)
        dataframes.append(df)

    master_df = pd.concat(dataframes, axis=0, ignore_index=True)
    if not return_df:
        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        file_path = out_dir.joinpath(file_name)
        master_df.to_csv(file_path, index=False)
    else:
        return master_df
