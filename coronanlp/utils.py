import os
import pickle
import re
import shelve
import sys
from datetime import datetime
from pathlib import Path
from string import punctuation
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
from nltk.tokenize import word_tokenize
from spacy import displacy

QuestionAnsweringOutput = TypeVar('QuestionAnsweringOutput')

GRADIENTS = {
    'blue': ['#2193b0', '#6dd5ed'],
    'virgin': ['#7b4397', '#dc2430'],
    'feels': ['#4568dc', '#b06ab3'],
    'frost': ['#000428', '#004e92'],
    'kashmir': ['#614385', '#516395'],
    'mauve': ['#42275a', '#734b6d'],
    'lush': ['#56ab2f', '#a8e063'],
    'river': ['#43cea2', '#185a9d'],
    'celestial': ['#c33764', '#1d2671'],
    'royal': ['#141e30', '#243b55']
}

CACHE_APPNAME = 'coronanlp'
CACHE_STORE_NAME = 'store'
STORE_FILE_NAMES = {'sents': 'sents',
                    'embed': 'embed.npy',
                    'index': 'index.bin'}
STORE_SHELVE_SUFFIXES = ['.bak', '.dat', '.dir']
STORE_ALL_FILE_NAMES = list(STORE_FILE_NAMES.values())
STORE_ALL_FILE_NAMES.append('sents.pkl')
STORE_ALL_FILE_NAMES.extend([
    f'sents{suffix}' for suffix in STORE_SHELVE_SUFFIXES])

FLAGS = re.MULTILINE | re.DOTALL


def re_sub(pattern, repl, text, flags=None):
    if flags is None:
        return re.sub(pattern, repl, text, flags=FLAGS)
    else:
        return re.sub(pattern, repl, text, flags=(FLAGS | flags))


def clean_string(text: str) -> str:
    text = re.sub(r"[a-zA-Z]+\/[a-zA-Z]+", " ", text)
    text = re.sub(r"&#160;", "", text)
    # Remove URL
    text = re_sub(r"(http)\S+", "", text)
    text = re_sub(r"(www)\S+", "", text)
    text = re_sub(r"(href)\S+", "", text)
    # remove repetition
    text = re_sub(r"([!?.]){2,}", r"\1", text)
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", text)
    return text.strip()


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


def split_on_char(string, char: str = '?', new: str = None, reverse=False):
    """Split on a character and concanate the character relative
    to its prior sequence.

    ```python
    string = "First question? Second question?"
    split_on_char(string, char="?")
    ...
    # ['First question?', 'Second question?']

    # optionally replace with a new char or/and return reversed
    split_on_char(string, char="?",  new=" <|Q|>", reverse=True)
    ...
    # ['Second question <|Q|>', 'First question <|Q|>']
    ```
    """
    reverse = 0 if not reverse else -1
    count = string.count(char)
    if count == 0 or not char:
        return string
    splits = string.split(char)
    if isinstance(new, str):
        char = new
    output = []
    while 0 < count:
        seq = splits.pop(reverse)
        if not seq or len(seq.strip()) == 0:
            continue
        output.append(seq.strip() + char)
        count -= 1
    return output


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
    def save_data(file_path: Union[str, Path], data_obj: Any) -> None:
        file_path = Path(file_path)
        if file_path.is_dir():
            if not file_path.exists():
                file_path.mkdir(parents=True)
        with file_path.open("wb") as pkl:
            pickle.dump(data_obj, pkl, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(file_path: Union[str, Path]) -> Any:
        file_path = Path(file_path)
        with file_path.open("rb") as pkl:
            return pickle.load(pkl)


def render_output(
    output: Optional[Union[Dict[str, str], QuestionAnsweringOutput]] = None,
    answer: Optional[str] = None,
    context: Optional[str] = None,
    question: Optional[str] = None,
    label: str = 'ANSWER',
    title: str = 'Question',
    grad_deg: str = '90deg',
    grad_pair: List[str] = ['#aa9cfc', '#fc9ce7'],
    span: Optional[Tuple[int, int]] = None,
    style: str = "ent",
    manual: bool = True,
    jupyter: bool = True,
    page: bool = False,
    minify: bool = True,
    return_html: bool = False,
    manual_data: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    """DisplaCy Visualizer for QA-Model Outputs.

    :param output: An output from the question-answering model. The output
        can be a dictionary with any or all keys: `question, answer, context`.
        Or a `QuestionAnsweringOutput` type object - If answer param is None,
        then the first `top-scored` answer will be chosen automatically.
    :param answer: (optional) A string sequence to represent as the answer.
    :param context: (optional) A list of string sequences or a single string
        to represet as the context (if `List[str]` - sequences will be joined).
    :param span: Span for highlighting the answer within the context. If
        None, its detected automatically.
    :param options: Visualizer options; visit the link for official DOCS:
        `https://spacy.io/api/top-level#displacy_options`
    :param manual_data: Defaults to ENT, keys; `'text', 'ents', 'titles'`
        DOCS: `https://spacy.io/usage/visualizers#manual-usage`
    """
    if output is not None:
        if isinstance(output, dict):
            if 'question' in output:
                question = output['question']
            if 'answer' in output:
                answer = output['answer']
            if 'context' in output:
                context = output['context']

        elif all(hasattr(output, attr) for attr in ('q', 'c', 'sids')):
            question, context = output.q, output.c
            # select the first top answer, if none provided.
            if answer is None:
                answer = output[0].answer

    if context is not None:
        if isinstance(context, list):
            context = ' '.join(context)
            e = f'Found item in List[{type(context[0])}], but expected List[str]'
            assert isinstance(context[0], str), e

    start, end = span if span is not None else (0, 0)
    if span is None:
        match = re.search(answer, context)
        if match and match.span() is not None:
            start, end = match.span()

    docs = dict() if manual_data is None else manual_data
    if manual_data is None:
        if style == "ent":
            docs["ents"] = [dict(start=start, end=end, label=label)]
            if len(context.strip()) > 1:
                docs['text'] = context
            if question is not None:
                docs['title'] = f"\n{title}: {question}\n"

    if options is None:
        if style == "dep":
            options = dict(compact=True, bg="#ed7118", color="#000000")
        else:
            options = dict(ents=None, colors=None)
            gradient = ", ".join([grad_deg] + grad_pair)
            colors = f"linear-gradient({gradient})"
            options.update({'ents': [label], 'colors': {label: colors}})

    if return_html:
        return displacy.render([docs], style=style, jupyter=False,
                               options=options, manual=manual)

    displacy.render([docs], style=style, page=page, minify=minify,
                    jupyter=jupyter, options=options, manual=manual)


def user_cache_dir(appname: Optional[str] = None, version: Optional[str] = None):
    """Return full path to the user-specific cache directory.
    User cache directories for MacOS, and Linux (current OS's only):
        MacOS:  ~/Library/Caches/<AppName>
        Linux:  ~/.cache/<AppName> (XDG default)
    """
    dirpath = ''
    system = sys.platform
    if system == 'darwin':
        dirpath = os.path.expanduser('~/Library/Caches/')
        if appname:
            dirpath = os.path.join(dirpath, appname)
    elif system == 'linux':
        dirpath = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        if appname:
            dirpath = os.path.join(dirpath, appname)
    if appname and version:
        dirpath = os.path.join(dirpath, version)
    return dirpath


def get_store_dir():
    cache_dir = Path(user_cache_dir(CACHE_APPNAME))
    store_path = cache_dir.joinpath(CACHE_STORE_NAME)
    return store_path


def get_all_store_paths() -> Dict[str, Path]:
    store_path = get_store_dir()
    return {p.name: p for p in store_path.iterdir() if p.is_dir()}


def delete_store(store_name: str) -> None:
    store_map = get_all_store_paths()
    if store_name in store_map:
        store_dir = store_map[store_name]
        for filename in STORE_ALL_FILE_NAMES:
            fp = store_dir.joinpath(filename)
            if not fp.is_file():
                continue
            fp.unlink()
        # Remove directory after deleting all files.
        store_dir.rmdir()
    else:
        raise ValueError(
            "The store-name does not exists or name is invalid"
            f" Found the following stores {store_map.items()}"
        )


def setup_user_dir(appname='coronanlp', version: str = None) -> str:
    path = Path(user_cache_dir(appname, version=version))
    store_dir = path.joinpath(CACHE_STORE_NAME)
    if not store_dir.exists() or not store_dir.is_dir():
        store_dir.mkdir(parents=True)
    return store_dir.as_posix()


def cache_user_data(filename: str, dirname=None, override=False):
    cache_dir = Path(setup_user_dir(CACHE_APPNAME))
    dirpath = cache_dir.joinpath(dirname)
    if not dirpath.exists() or not dirpath.is_dir():
        dirpath.mkdir(parents=True, exist_ok=override)
    filepath = dirpath.joinpath(filename).as_posix()
    return filepath


def save_stores(sents=None, embed=None, index=None, store_name=None, override=False):
    if store_name is None:
        store_name = datetime.now().strftime('%Y-%m-%d')

    if sents is not None:
        assert hasattr(sents, 'to_disk')
        path = cache_user_data(STORE_FILE_NAMES['sents'], store_name, override)
        with shelve.open(path) as db:
            db['data'] = sents._store
            db['meta'] = sents._meta
            db['init'] = sents.init_args
            db['kwargs'] = {
                'counts': sents.counts,
                'maxlen': sents.maxlen,
                'seqlen': sents.seqlen
            }

    if embed is not None:
        assert isinstance(embed, np.ndarray)
        path = cache_user_data(STORE_FILE_NAMES['embed'], store_name, override)
        np.save(path, embed)

    if index is not None:
        import faiss
        assert isinstance(index, faiss.Index)
        path = cache_user_data(STORE_FILE_NAMES['index'], store_name, override)
        faiss.write_index(index, path)


def load_store(type_store: str, store_name: str = None) -> Any:
    store_path = get_store_dir()
    if store_name is None:
        accessed_logs = {
            p.lstat().st_atime: p.name for p in store_path.iterdir()
            if p.is_dir()
        }
        last_accessed_dir = accessed_logs[max(accessed_logs.keys())]
        store_name = last_accessed_dir

    cache_dir = store_path.joinpath(store_name)

    if type_store == 'sents':
        path = cache_dir.joinpath(STORE_FILE_NAMES['sents'])
        # Validate shelve if it contains the expected three files:
        # Here we obtain the name of the "file.suffix" in order to
        # show the user the files found vs the expected suffixes (if any).
        db_suffixes = STORE_SHELVE_SUFFIXES
        shelvefiles = [p.name for p in path.parent.iterdir() if p.is_file()
                       and p.suffix in db_suffixes]
        if not len(db_suffixes) == len(shelvefiles):
            raise ValueError(
                'Expected files with extensions {}, found only: {}'.format(
                    db_suffixes, shelvefiles,
                ))
        from .core import SentenceStore
        with shelve.open(path.as_posix()) as db:
            data = db['data']
            pids = list(data.keys())
            sentence_store = SentenceStore(
                pids, store=data, meta=db['meta'], **db['kwargs'])
            sentence_store.init_args = db['init']
        return sentence_store

    if type_store == 'embed':
        path = cache_dir.joinpath(STORE_FILE_NAMES['embed'])
        assert path.is_file()
        return np.load(path.as_posix())

    if type_store == 'index':
        path = cache_dir.joinpath(STORE_FILE_NAMES['index'])
        assert path.is_file()
        import faiss
        return faiss.read_index(path.as_posix())
