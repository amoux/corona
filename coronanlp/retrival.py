import concurrent.futures
import sys
from multiprocessing import cpu_count
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.lib.utils import source  # type: ignore
import spacy  # type: ignore
from spacy.language import Language  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from .core import GoldPids, GoldPidsOutput, Sampler, SentenceStore
from .dataset import CORD19
from .tasks import TaskList
from .ukplab import SentenceEncoder
from .utils import clean_punctuation, normalize_whitespace

try:
    _faiss_lib = __import__("faiss")
except ModuleNotFoundError:
    print(sys.exc_info())
else:
    globals()["faiss"] = _faiss_lib


Pid = int


def common_tokens(
        texts: Iterable[str],
        minlen: int = 3,
        lowercase: bool = False,
        nlp: Optional[Language] = None
) -> List[Tuple[str, int]]:
    """Extract common tokens from a collection of strings.

    * Preprocess: remove stop-words, punctuation and lemmatize.

    :param texts: Iterable of string sequences.
    :param minlen: Dismiss tokens with a minimum length.
    :param nlp: An existing spacy language instance.
    """
    nlp = nlp if nlp is not None and isinstance(nlp, Language) \
        else spacy.load('en_core_web_sm', disable=['tagger'])

    counter = {}
    for text in texts:
        text = clean_punctuation(text)
        for token in nlp(text):
            if token.is_stop:
                continue
            token = token.lemma_ if not lowercase \
                else token.lemma_.lower()
            if len(token) < minlen:
                continue
            if token not in counter:
                counter[token] = 1
            else:
                counter[token] += 1

    common = sorted(
        counter.items(), key=lambda k: k[1], reverse=True)
    return common


def frequency_summarizer(texts: List[str], ratio: float = 0.3, nlp=None) -> str:
    """Frequency Based Summarization.

    - Words are lowercased.
    - Stopwords and punctuation ignored.

    :param text: Sequences of strings or an iterable of string sequences.
    :param ratio: Summary size from the topmost leading scored sentences.
        The expected output size is based on: `topk = len(texts) * ratio`.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    words = {}
    for doc in nlp.pipe(texts):
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            word = token.text.lower()
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1

    features = {}
    max_freq = max(words.values())
    for word, frequency in words.items():
        features[word] = frequency / max_freq

    scores = {}
    for doc in nlp.pipe(texts):
        for sent in doc.sents:
            for token in sent:
                word = token.text.lower()
                if word not in words:
                    continue
                if sent not in scores:
                    scores[sent] = features[word]
                else:
                    scores[sent] += features[word]

    topk = max(int(len(scores) * ratio), 1)
    nlargest = sorted(scores, key=scores.get, reverse=True)[:topk]
    summary = " ".join([sent.text for sent in nlargest])
    return summary


def extract_questions(
    sent_store: Union[Sampler, SentenceStore],
    minlen: int = 10,
    remove_empty: bool = True,
    sample: Optional[List[Pid]] = None,
) -> Union[Sampler, None]:
    """Extract questions from a Sampler or SentenceStore instance.

    :param sent_store: An instance of Sampler or SentenceStore holding sentences.
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

    type_err = ('Expected an instance of SentenceStore'
                f' or Sampler, instead got, {type(sent_store)}')
    assert isinstance(sent_store, (SentenceStore, Sampler)), type_err

    interrogative = [
        'how', 'why', 'when', 'where', 'what', 'whom', 'whose']

    store = sent_store._store
    if sample is None:
        sample = sent_store.pids

    X = Sampler(sample)
    questions = X.init()
    for pid in tqdm(sample, desc='sentences'):
        for sent in store[pid]:
            if sent in questions[pid]:
                continue
            words = sent.lower().split()
            seqlen = len(words)
            if seqlen < minlen:
                continue
            if words[0] in interrogative \
                    and words[-1].endswith("?"):
                X.include(pid, seqlen, text=sent)

        if remove_empty and len(questions[pid]) == 0:
            questions.pop(pid)

    if X.counts == 0:
        return None
    else:
        return X


def extract_titles_slow(
    cord19: CORD19,
    sample: Optional[Iterable[Pid]] = None,
    minlen: int = 10,
    show_progress: bool = False,
) -> Dict[Pid, str]:
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


def extract_titles_fast(
    cord19: CORD19,
    sample: Optional[Iterable[Pid]] = None,
    minlen: int = 10,
    maxids: int = -1,
) -> Dict[Pid, str]:
    """Extract titles from the CORD-19-Dataset (Fast).

    :param sample: (optional) An existing list of integer ids (`PIDs`).
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


def tune_ids(
    encoder: SentenceEncoder,
    title_map: Dict[Pid, str],
    tasklist: Optional[TaskList] = None,
    target_size: int = 1000,
    batch_size: int = 16,
    skip_false_trg: bool = False,
    show_progress: bool = True
) -> GoldPidsOutput:
    """Tune paper ids of titles to a single or multiple task(s).

    * Example usage: First, get the paper-ids you'll like to tune
        and a title_map. Which is a `Dict[int, str]` where `int` is
        (PID), and `str` is its respective title. The following code
        is one way to get the needed inputs.

    ```python
    from coronanlp import CORD19, TaskList, extract_titles_fast

    cord19 = CORD19(...)
    pids = cord19.sample(5000)
    title_map = extract_titles_fast(cord19, sample=pids)

    # Depending on the sample size you can tune for all tasks
    # or a few - in this example we only do the first five:
    num_tasks = 5
    tasklist = TaskList()[:num_tasks]

    # target size can be set to a specific value, e.g `500` or:
    target_size = len(title_map) // num_tasks
    tuned = tune_ids(encoder, title_map, tasklist, target_size)
    ...
    # GoldPidsOutput(num_tasks: 5, size: (475, 480, 490, 496, 496))

    tuned[::]
    ...
    # [GoldPids(task_id: 1, size: 475, mindist: 11.9891, maxdist: 37.1647),
    #  GoldPids(task_id: 2, size: 480, mindist: 15.0586, maxdist: 51.9199),
    #  GoldPids(task_id: 3, size: 490, mindist: 12.5479, maxdist: 29.0433),
    #  ... ]
    ```
    """
    task_list: TaskList
    if tasklist is None:
        task_list = TaskList()
    else:
        task_list = tasklist

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
                    'a smaller ``target_size`` or add more tasks.'
                )
        return targets

    # Eval targets before embedding and indexing.
    eval_targets = get_k_targets()
    embedded_titles = encoder.encode(
        titles, batch_size=batch_size, show_progress=show_progress,
    )
    ndim = embedded_titles.shape[1]
    index_flat = faiss.IndexFlat(ndim)
    index_flat.add(embedded_titles)

    output = GoldPidsOutput()
    for i, task in enumerate(task_list):
        task_embed = encoder.encode(
            task.all(), batch_size=16, show_progress=False,
        )
        D, I = index_flat.search(task_embed, eval_targets[i])
        pids = np.array([decode[k] for k in I.flatten()])
        output.append(GoldPids(task.id, pids=pids, dist=D.flatten()))

    output.pids = [pid for pid, _ in output.common(topk=-1)]
    return output
