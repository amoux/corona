import pickle
import re
from pathlib import Path
from string import punctuation
from typing import IO, Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from spacy import displacy

from .indexing import PaperIndexer
from .jsonformatter import generate_clean_df


def clean_punctuation(text: str) -> str:
    punct = re.compile("[{}]".format(re.escape(punctuation)))
    tokens = word_tokenize(text)
    text = " ".join(filter(lambda t: punct.sub("", t), tokens))
    return normalize_whitespace(text)


def normalize_whitespace(string: str) -> str:
    """Normalize excessive whitespace."""
    linebreak = re.compile(r"(\r\n|[\n\v])+")
    nonebreaking_space = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)
    return nonebreaking_space.sub(" ", linebreak.sub(r"\n", string)).strip()


def clean_tokenization(sequence: str) -> str:
    """Clean up spaces before punctuations and abbreviated forms."""
    return (
        sequence.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
        .replace(" / ", "/")
        .replace(" )", ")")
        .replace("( ", "(")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace(" ;", ";")
        .replace(" - ", "-")
    )


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


def render(question: str, prediction: Dict[str, str], jupyter=True,
           return_html=False, style="ent", manual=True, label='ANSWER'):
    """Spacy displaCy visualization util for the question answering model."""
    options = {"compact": True, "bg": "#ed7118", "color": '#000000'}
    display_data = {}
    start, end = 0, 0
    match = re.search(prediction["answer"], prediction["context"])
    if match and match.span() is not None:
        start, end = match.span()

    display_data["ents"] = [{'start': start, 'end': end, 'label': label}]
    options['ents'] = [label]
    options['colors'] = {label: "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    if len(prediction['context']) > 1:
        display_data['text'] = prediction['context']

    display_data['title'] = f'Q : {question}\n'
    if return_html:
        return displacy.render([display_data], style=style,
                               jupyter=False, options=options, manual=manual)
    else:
        displacy.render([display_data], style=style,
                        page=False, minify=True,
                        jupyter=jupyter, options=options, manual=manual)


def papers_to_csv(sources: Union[str, Path, List[Union[str, Path]]],
                  out_dir: str = "data") -> None:
    """Convert one or more directories with json files into a csv file(s).

    :param sources: Path or iterable of paths to the root directory of
        the CORD19 Dataset e.g., `CORD-19-research-challenge/2020-03-13/`.
    """
    def sample(i: int, splits: List[int], index_start: int) -> List[int]:
        if i == 0:
            return list(range(index_start, splits[i] + 1))
        if i > 0:
            return list(range(splits[i - 1] + 1, splits[i] + 1))

    out_dir = Path(out_dir) if not isinstance(out_dir, Path) else out_dir
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    indexer = PaperIndexer(sources)
    index_start = indexer.index_start
    splits = indexer._splits
    for i in range(len(splits)):
        try:
            ids = sample(i, splits, index_start)
            papers = indexer.load_papers(ids)
        except Exception as err:
            raise Exception(
                f"{indexer.source_name[i]} generated an exception; {err}"
            )
        else:
            df = generate_clean_df(papers)
            df.drop_duplicates(subset=["paper_id"], inplace=True)
            df.dropna(inplace=True)

            file_name = f"{indexer.source_name[i]}_{splits[i]}_papers.csv"
            file_path = out_dir.joinpath(file_name)
            df.to_csv(file_path, index=False)

            print("All {} files from directory {} saved in: {}".format(
                splits[i], indexer.source_name[i], file_path)
            )


def concat_csv_files(source_dir: str,
                     file_name="covid-lg.csv",
                     out_dir="data",
                     drop_cols=["raw_authors", "raw_bibliography"],
                     return_df=False):
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
