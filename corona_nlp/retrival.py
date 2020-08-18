import sys
import concurrent.futures
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Union

import spacy
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

from .dataset import CORD19Dataset
from .datatypes import Papers
from .utils import clean_punctuation, normalize_whitespace

try:
    _faiss_lib = __import__("faiss")
except ModuleNotFoundError:
    print(sys.exc_info())
else:
    globals()["faiss"] = _faiss_lib


def frequency_summarizer(text: Union[str, List[str]],
                         topk=7, min_tokens=30, nlp=None) -> str:
    """Frequency Based Summarization.

    :param text: sequences of strings or an iterable of string sequences.
    :param topk: number of topmost leading scored sentences.
    :param min_tokens: minimum number of tokens to consider in a sentence.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(" ".join(text) if isinstance(text, list) else text)

    vocab = {}
    for token in doc:
        if not token.is_stop and not token.is_punct:
            if token.text not in vocab:
                vocab[token.text] = 1
            else:
                vocab[token.text] += 1

    for word in vocab:
        vocab[word] = vocab[word] / max(vocab.values())

    score = {}
    for sent in doc.sents:
        for token in sent:
            if len(sent) > min_tokens:
                continue
            if token.text in vocab:
                if sent not in score:
                    score[sent] = vocab[token.text]
                else:
                    score[sent] += vocab[token.text]

    nlargest = sorted(score, key=score.get, reverse=True)[:topk]
    summary = " ".join([sent.text for sent in nlargest])
    return summary


def common_tokens(texts: List[str], minlen=3, nlp=None,
                  pos_tags=("NOUN", "ADJ", "VERB", "ADV",)):
    """Top Common Tokens (removes stopwords and punctuation).

    :param texts: iterable of string sequences.
    :param minlen: dismiss tokens with a minimum length.
    :param nlp: use an existing spacy language instance.
    :param pos_tags: lemmatize tokens based on part-of-speech tags.
    """
    common = {}
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    for doc in nlp.pipe(texts):
        tokens = []
        for token in doc:
            if token.is_stop:
                continue
            if token.pos_ in pos_tags:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        text = " ".join(tokens)
        text = clean_punctuation(text)
        for token in word_tokenize(text):
            if len(token) < minlen:
                continue
            if token not in common:
                common[token] = 1
            else:
                common[token] += 1

    common = sorted(common.items(),
                    key=lambda k: k[1], reverse=True)
    return common


def extract_questions(papers: Papers, min_length=30, sentence_ids=False):
    """Extract questions from an instance of papers.

    :param min_length: minimum length of a question to consider.
    :param sentence_ids: whether to return the decoded ids `paper[index]`.
    """
    interrogative = ['how', 'why', 'when',
                     'where', 'what', 'whom', 'whose']
    sents = []
    ids = []
    for index in tqdm(range(len(papers)), desc='sentences'):
        string = papers[index]
        if len(string) < min_length:
            continue
        toks = string.lower().split()
        if toks[0] in interrogative and toks[-1].endswith("?"):
            sents.append(string)
            ids.append(index)

    questions = list(set(sents))
    print(f'found {len(questions)} interrogative questions.')

    if not sentence_ids:
        return questions
    return questions, ids


def extract_titles_slow(sample: List[int],
                        minlen: int = 10,
                        cord19: CORD19Dataset = None,
                        show_progress: bool = False) -> Dict[int, str]:
    """Extract titles from the CORD-19-Dataset (Slow).

    :param sample: (optional) An existing list of integer ids (paper indices).
        Otherwise if None; ids extracted from the `COR19Dataset` instance.
    :param minlen: Minimum title string length (filtered without punctuation).
    :param maxids: Number of ids to slice, if `-1` then all ids are used.

    Returns Dict[int, str] a dict of mapped paper indices to titles.
    """
    if show_progress:
        sample = tqdm(sample, desc="titles")

    mapped = {}
    for index in sample:
        title = cord19.title(index)
        title = normalize_whitespace(title)
        if len(clean_punctuation(title)) <= minlen:
            continue
        if index not in mapped:
            mapped[index] = title

    return mapped


def extract_titles_fast(sample: Optional[List[int]] = None,
                        minlen: int = 10,
                        cord19: CORD19Dataset = None,
                        maxids: int = -1) -> Dict[int, str]:
    """Extract titles from the CORD-19-Dataset (Fast).

    :param sample: (optional) An existing list of integer ids (paper indices).
        Otherwise if None; ids extracted from the `COR19Dataset` instance.
    :param minlen: Minimum title string length (filtered without punctuation).
    :param maxids: Number of ids to slice, if `-1` then all ids are used.

    Returns Dict[int, str] a dict of mapped paper indices to titles.
    """
    if sample is None:
        sample = cord19.sample(-1)
    else:
        assert isinstance(sample, list)
        assert isinstance(sample[0], int)
    if maxids > -1:
        sample = sample[:maxids]

    jobs = []
    maxsize = len(sample)
    workers = cpu_count()
    for i in range(0, maxsize, workers):
        job_ids = sample[i: min(i + workers, maxsize)]
        jobs.append(job_ids)

    with tqdm(total=maxsize, desc="titles") as pbar:
        batch = {}
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            future_to_ids = {
                pool.submit(extract_titles_slow, job, **dict(
                    minlen=minlen, cord19=cord19)): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_ids):
                ids = future_to_ids[future]
                try:
                    mapped = future.result()
                except Exception as err:
                    print(f"{ids} generated an exception; {err}")
                    break
                else:
                    batch.update(mapped)
                    pbar.update(len(ids))

    mapping = {}  # Sort the dictionary from keys (paper_ids).
    indices = iter(sorted(batch.keys()))
    for index in indices:
        title = batch[index]
        mapping[index] = title

    return mapping


def tune_ids_to_tasks(
        tasks: Union[Dict[str, List[str]], List[Dict[str, List[str]]],
                     List[str], List[List[str]]],
        encoder: 'SentenceTransformer',
        minlen: Optional[int] = 10,
        maxids: Optional[int] = -1,
        cord19: Optional[CORD19Dataset] = None,
        ids_titles: Optional[Dict[int, str]] = None,
        target_size: Optional[int] = None,
        k_nn: Optional[int] = None,
        show_progress: bool = False) -> Union[List[int], List[List[int]]]:
    """Tune a sample of ids to a single or multiple task(s).

    param: tasks (Union[List[str], List[List[str]]]):
        An iterable of string sequences or a list of iterables of string
        sequences. Tasks are expected to be in form of text queries.
        Multiple tasks available in the `cord_nlp.tasks` module.
    param: minlen (Optional, int):
        Minimum title length, ignored if ids_titles is not None.
    param: maxids (Optional, int):
        Sample size for obtaining the titles, ignored if ids_titles
        is not None.
    param: ids_titles (Optional, Dict[int, str]):
        A mapping of paper ids to its titles. If None, then a
        ``CORD19Dataset`` instance is expected.
    param: target_size (Optional, int):
        Expected size of a sample. If the number of unique IDs is less
        than the target size; additional ID's from the sample will be added
        (these are not similar to the tasks) in order to meet the target
        sample size. Otherwise, no additional ids are added.
    param: k_nn (Optional, int):
        Number of k nearest neighbors to query against the titles.
    param: show_progress (bool):
        Whether to display the progress of encoding.
    """
    if isinstance(tasks, dict) or isinstance(tasks[0], str):
        tasks = [tasks]
    if isinstance(tasks[0], dict):
        tasks_ = []
        for i in range(len(tasks)):
            if "tasks" not in tasks[i]:
                raise ValueError("Missing key `<tasks>` in dictionary.")
            else:
                tasks_.append(tasks[i]["tasks"])
        tasks = tasks_

    if isinstance(tasks[0], list) and isinstance(tasks[0][0], str):
        assert sum([len(n) for n in tasks]) > 1, \
            "Total number of string sequences (tasks) < 1."
    else:
        raise ValueError(
            "Expected a single or iterable of task(s) with type "
            f"Dict[str, List[str]] | List[str] got, {type(tasks[0])}")

    if ids_titles is None:
        if cord19 is not None:
            ids_titles = extract_titles_fast(
                sample=None, minlen=minlen, cord19=cord19, maxids=maxids)
        else:
            raise Exception('Expected an ``CORD19Dataset`` instance or '
                            'a Dict[int, str] ``ids_titles`` mapping.')

    titles = list(ids_titles.values())
    sample = list(ids_titles.keys())
    decode = dict(enumerate(sample))

    k_iter = []
    if k_nn is None and target_size is not None:
        for task in tasks:
            ntasks = len(task)
            k_nn = round(target_size / ntasks) - ntasks % 2
            maxk = len(sample) - target_size
            assert (k_nn * ntasks) <= maxk, (
                'target_size is larger than n queries possible '
                'given the sample size and number of tasks, pick '
                'a smaller ``target_size`` or add more tasks.')
            k_iter.append(k_nn)

    embedded_titles = encoder.encode(titles, 8, show_progress)
    ndim = embedded_titles.shape[1]
    index = faiss.IndexFlat(ndim)
    index.add(embedded_titles)

    gold_ids = []
    for i in range(len(k_iter)):
        topk = k_iter[i]
        task = encoder.encode(tasks[i], 8, show_progress)
        k_nn = index.search(task, topk)[1].flatten().tolist()
        ids = sorted(set([decode[k] for k in k_nn]))
        if target_size is None:
            gold_ids.append(ids)
        else:
            gold_ids.extend(ids)

    if target_size is None:
        return gold_ids

    gold_ids = list(set(gold_ids))
    ntotal = len(gold_ids)

    extra_ids = []
    if ntotal < target_size:
        target = target_size - ntotal
        count = 0
        for id in sample:
            if id in gold_ids:
                continue
            if count < target:
                extra_ids.append(id)
                count += 1
        assert len(extra_ids) + ntotal == target_size

    gold_ids.extend(extra_ids)
    gold_ids.sort()
    return gold_ids
