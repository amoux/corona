import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .jsonformatter import generate_clean_df
from .papers_indexing import (CovidPapers, all_dataset_sources,
                              all_sources_metadata)


def normalize_whitespace(string: str):
    """Normalize excessive whitespace from a string without formating the original form."""
    LINEBREAK = re.compile(r"(\r\n|[\n\v])+")
    NONBREAKING_SPACE = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)

    return NONBREAKING_SPACE.sub(" ", LINEBREAK.sub(r"\n", string)).strip()


def load_papers_with_text(
    conv: CovidPapers, indices: List[int], keys: Iterable[str] = None
):
    """For every paper grab its title and all the available texts.

    keys: Iterable of string sequences, if None, the default keys
        will be used for obtaining texts: ('abstract', 'body_text', 'back_matter')
    """
    if not isinstance(conv, CovidPapers):
        raise ValueError(f"{type(conv)} is not an instance of CovidPapers.")

    if keys is None:
        keys = ("abstract", "body_text", "back_matter")

    batch = []
    papers = conv.load_papers(indices=indices)
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

    def indices_with_fulltext(conv: CovidPapers) -> List[int]:
        """Filter only the paper-id's with True in has_full_text."""
        text_indices = []
        for paper_id in has_full_text:
            if paper_id in conv.paper_index:
                index = conv.paper_index[paper_id]
                if index in text_indices:
                    continue
                text_indices.append(index)
        return text_indices

    for source_path in source_paths:
        conv = CovidPapers(source_path)
        indices = indices_with_fulltext(conv)
        batches = conv.load_papers(indices)

        dataframe = generate_clean_df(batches)
        file_path = out_dir.joinpath(f"{conv.source_name}.csv")
        dataframe.to_csv(file_path, index=False)

        print(
            "All {} files for {} saved in {}".format(
                conv.num_papers, conv.source_name, file_path
            )
        )


def concat_csv_files(source_dir: str, out_dir="data", return_df=False):
    """Concat all CSV files into one single file.

    return_df: If True, saving to file is ignored the concat Dataframe returned.

    Usage:
        >>> concat_csv_files('path/to/csv-files-dir/', out_dir='data')
    """
    dataframes = []
    for csv_file in Path(source_dir).glob("*.csv"):
        df = pd.read_csv(csv_file, index_col=None, header=0)
        df.drop(columns=["raw_authors", "raw_bibliography"], inplace=True)
        dataframes.append(df)

    master_df = pd.concat(dataframes, axis=0, ignore_index=True)
    if return_df:
        return master_df
    else:
        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        file_path = out_dir.joinpath("covid-master-lg.csv")
        master_df.to_csv(file_path, index=False)
