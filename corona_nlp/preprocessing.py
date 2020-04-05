import re
from pathlib import Path
from typing import Iterable, List, NamedTuple, Sequence, Tuple, Union

import pandas as pd

from .indexing import PaperIndexing
from .jsonformatter import generate_clean_df
from .utils import Cord19Paths, load_dataset_paths


def normalize_whitespace(string: str):
    """Normalize excessive whitespace."""
    LINEBREAK = re.compile(r"(\r\n|[\n\v])+")
    NONBREAKING_SPACE = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)

    return NONBREAKING_SPACE.sub(" ", LINEBREAK.sub(r"\n", string)).strip()


def load_papers_with_text(
    covid: PaperIndexing,
    indices: List[int],
    keys: Iterable[str] = ("abstract", "body_text", "back_matter"),
):
    """For every paper grab its title and all the available texts.

    keys: Iterable of string sequences, if None, the default keys
        will be used for obtaining texts: ('abstract', 'body_text',
        'back_matter')
    """
    if not isinstance(covid, PaperIndexing):
        raise ValueError(f"{type(covid)} is not an instance of PaperIndexing.")

    batch = []
    papers = covid.load_papers(indices=indices)
    for i, paper in zip(indices, papers):
        title = paper["metadata"]["title"]
        texts = []
        for key in keys:
            sequences = [x["text"] for x in paper[key]]
            for string in sequences:
                if len(string) == 0 and string in texts:
                    continue
                texts.append(string)
        batch.append({"id": i, "title": title, "texts": texts})

    return batch


def papers_to_csv(sources: Union[str, Cord19Paths],
                  dirs: Tuple[Sequence[str]] = ('all',),
                  out_dir: str = "data"):
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

    def with_full_text(index: PaperIndexing) -> List[int]:
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
        index = PaperIndexing(path)
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
