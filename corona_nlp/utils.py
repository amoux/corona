import pickle
import re
from pathlib import Path
from string import punctuation
from typing import IO, Any, Dict, List, Optional, Tuple, TypeVar, Union

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

        elif all(hasattr(output, attr) for attr in ('q', 'c', 'ids')):
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
