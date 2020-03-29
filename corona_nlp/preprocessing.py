import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .indexing import PaperIndexing, all_dataset_sources, all_sources_metadata
from .jsonformatter import generate_clean_df


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


def convert_sources_to_csv(out_dir="data", source_paths: List[Path] = None):
    """Convert all json-files from all or some directories.

    source_paths: A list of paths to the directory holding the json files.
        If None, All papers sources will be converted and saved as csv.
    """
    if source_paths is None:
        source_paths = all_dataset_sources
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    metadata = pd.read_csv(all_sources_metadata)
    has_full_text = metadata.loc[metadata["has_full_text"] == True, ["sha"]]
    has_full_text = list(set(has_full_text["sha"].to_list()))

    def indices_with_fulltext(covid: PaperIndexing) -> List[int]:
        """Filter only paper-id's if full-text is available in the metadata."""
        text_indices = []
        for paper_id in has_full_text:
            if paper_id in covid.paper_index:
                index = covid.paper_index[paper_id]
                if index in text_indices:
                    continue
                text_indices.append(index)
        return text_indices

    for source_path in source_paths:
        covid = PaperIndexing(source_path)
        indices = indices_with_fulltext(covid)
        batches = covid.load_papers(indices)
        dataframe = generate_clean_df(batches)
        file_path = out_dir.joinpath(f"{covid.source_name}.csv")
        dataframe.to_csv(file_path, index=False)
        print(
            "All {} files for {} saved in {}".format(
                covid.num_papers, covid.source_name, file_path
            )
        )


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
