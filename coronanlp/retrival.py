import concurrent.futures
import sys
from multiprocessing import cpu_count
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

from .core import GoldPids, GoldPidsOutput, Sampler, SentenceStore
from .dataset import CORD19
from .tasks import TaskList
from .utils import clean_punctuation, normalize_whitespace

try:
    _faiss_lib = __import__("faiss")
except ModuleNotFoundError:
    print(sys.exc_info())
else:
    globals()["faiss"] = _faiss_lib


Pid = int


def common_tokens(texts: List[str], minlen=3, nlp=None,
                  pos_tags=("NOUN", "ADJ", "VERB", "ADV",)):
    """Top Common Tokens (removes stopwords and punctuation).

    :param texts: iterable of string titles.
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


def extract_questions(sentstore: Union[Sampler, SentenceStore],
                      minlen: int = 10,
                      remove_empty: bool = True,
                      sample: Optional[List[Pid]] = None) -> Union[Sampler, None]:
    """Extract questions from a Sampler or SentenceStore instance.

    :param sentstore: An instance of Sampler or SentenceStore holding sentences.
    :param minlen: Minimum sequence length of a title to consider as valid.
        The length is computed via token units using `str.split()`.
    :param remove_empty: Whether to remove empty (key, value) pairs if no questions
        where found for that key `(pid)`.
    :param sample: (Optional) An iterable of paper ids (pids) to use for retriving
        for questions. If None, all paper ids from the papers instance is used. 
    :return: A newly constructed Sampler instance with question(s) mapped to
        their respective paper-id (pid). Keeping only ids with one item (question)
        or more. If no questions where found for all pid(s); return None.
    """
    interrogative = ['how', 'why', 'when', 'where', 'what', 'whom', 'whose']
    type_err = f'Expected an instance of SentenceStore, instead got, {type(sentstore)}'
    assert isinstance(sentstore, (SentenceStore, Sampler)), type_err
    sample = sentstore.pids if sample is None else sample
    store = None
    if isinstance(sentstore, Sampler):
        store = sentstore.store
    elif isinstance(sentstore, SentenceStore):
        store = sentstore._store

    X = Sampler(sample)
    questions = X.init()
    for pid in tqdm(sample, desc='pids'):
        for sent in store[pid]:
            if sent in questions[pid]:
                continue
            words = sent.lower().split()
            seqlen = len(words)
            if seqlen < minlen:
                continue
            if words[0] in interrogative and words[-1].endswith("?"):
                args = (len(questions[pid]), len(sent), seqlen)
                X.addmeta(pid, *args)
                questions[pid].append(sent)
                X.maxlen = max(X.maxlen, seqlen)
                X.seqlen += seqlen
                X.counts += 1
        if remove_empty and len(questions[pid]) == 0:
            questions.pop(pid)
    if X.counts == 0:
        return None
    return X


def extract_titles_slow(cord19: CORD19,
                        sample: Optional[Iterable[Pid]] = None,
                        minlen: int = 10,
                        show_progress: bool = False) -> Dict[Pid, str]:
    """Extract titles from the CORD-19-Dataset (Slow).

    :param cord19: A CORD19 instance with the method `corona.title()`.
    :param sample: An existing iterable of paper ids (Assumes the ids belong
        to the CORD19 instance).
    :param minlen: Minimum title string length (filtered without punctuation).
    :param maxids: Number of ids to slice, if `-1` then all ids are used.

    Returns Dict[int, str] a dict of mapped paper indices to titles.
    """
    if sample is None:
        sample = cord19.sample(-1)
    if show_progress:
        sample = tqdm(sample, desc="titles")

    mapped = {}
    for pid in sample:
        title = cord19.title(pid)
        title = normalize_whitespace(title)
        if len(clean_punctuation(title)) <= minlen:
            continue
        if pid not in mapped:
            mapped[pid] = title

    return mapped


def extract_titles_fast(cord19: CORD19,
                        sample: Optional[Iterable[Pid]] = None,
                        minlen: int = 10,
                        maxids: int = -1) -> Dict[Pid, str]:
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
        assert isinstance(sample[0], Pid)
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
                pool.submit(extract_titles_slow, **dict(
                    cord19=cord19, sample=job, minlen=minlen,
                )): job for job in jobs
            }
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
    pids = iter(sorted(batch.keys()))
    for pid in pids:
        title = batch[pid]
        mapping[pid] = title

    return mapping


def tune_ids(encoder,
             title_map: Dict[Pid, str],
             task_list: Optional[TaskList] = None,
             target_size: int = 1000,
             batch_size: int = 16,
             skip_false_trg: bool = False,
             show_progress: bool = True) -> GoldPidsOutput:
    """Tune paper ids of titles to a single or multiple task(s)."""
    if task_list is None:
        task_list = TaskList()

    titles = list(title_map.values())
    sample = list(title_map.keys())
    decode = dict(enumerate(sample))

    def get_k_targets() -> List[int]:
        targets = []
        for task in task_list:
            ntasks = len(task.all())
            goal = round(target_size / ntasks) - ntasks % 2
            maxk = len(sample) - target_size
            is_spec = (goal * ntasks) <= maxk
            if is_spec:
                targets.append(goal)
            elif skip_false_trg:
                continue
            else:
                raise ValueError(
                    'Target size is larger than k queries possible '
                    'given the sample size and number of tasks, pick '
                    'a smaller ``target_size`` or add more tasks.')
        return targets

    targets = get_k_targets()  # Eval targets before embedding and indexing.
    embedded_titles = encoder.encode(titles, batch_size, show_progress)
    ndim = embedded_titles.shape[1]
    index_flat = faiss.IndexFlat(ndim)
    index_flat.add(embedded_titles)

    output = GoldPidsOutput()
    for i, task in enumerate(task_list):
        task_embed = encoder.encode(task.all(), batch_size=16,
                                    show_progress=False)
        D, I = index_flat.search(task_embed, targets[i])
        pids = np.array([decode[k] for k in I.flatten()])
        output.append(GoldPids(task.id, pids=pids, dist=D.flatten()))

    output.pids = [pid for pid, _ in output.common(topk=-1)]
    return output
