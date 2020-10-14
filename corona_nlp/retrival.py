import concurrent.futures
import sys
from multiprocessing import cpu_count
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

from .dataset import CORD19Dataset
from .datatypes import Papers, Sentences
from .tasks import TaskList
from .utils import clean_punctuation, normalize_whitespace

try:
    _faiss_lib = __import__("faiss")
except ModuleNotFoundError:
    print(sys.exc_info())
else:
    globals()["faiss"] = _faiss_lib


class GoldIds(NamedTuple):
    task_id: int
    ids: np.ndarray
    dist: np.ndarray

    def size(self) -> int:
        return self.ids.size

    def mindist(self) -> float:
        return self.dist.min().item()

    def maxdist(self) -> float:
        return self.dist.max().item()

    def __repr__(self):
        return '{}(task_id: {}, size: {}, mindist: {}, maxdist: {})'.format(
            self.__class__.__name__, self.task_id, self.size(),
            round(self.mindist(), 4), round(self.maxdist(), 4),
        )


class GoldIdsOutput(List[GoldIds]):
    @property
    def num_tasks(self) -> int:
        return len(self)

    def all_sizes(self) -> List[int]:
        return [gold.size() for gold in self]

    def sample(self) -> Iterable[int]:
        for gold in self:
            for pid in gold.ids:
                yield pid.item()

    def iterall(self) -> Iterable[Tuple[int, np.int64, np.float32]]:
        for gold in self:
            for pid, dist in zip(gold.ids, gold.dist):
                yield gold.task_id, pid, dist

    def __repr__(self):
        return '{}(num_tasks: {}, size: {})'.format(
            self.__class__.__name__, len(self), tuple(self.all_sizes()),
        )


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


def extract_questions(papers: Papers,
                      minlen: int = 30,
                      sample: Optional[Iterable[int]] = None) -> Union[None, Papers]:
    """Extract questions from an instance of papers.

    :param papers: An instance of Papers.
    :param minlen: Minimum length of a question to add, e.g., by using `len(str)`.
    :param sample: (Optional) An iterable of paper ids to use for retriving for
        questions. If None, all paper ids from the papers instance is used. 
    :return: A newly constructed Papers instance with questions mapped to
        their respective paper-id. Keeping only ids with one item (question) or more.
        If no questions where found (count == 0); return None.
    """
    interrogative = ['how', 'why', 'when', 'where', 'what', 'whom', 'whose']

    type_err = f'Expected an instance of Papers, instead got, {type(papers)}'
    assert isinstance(papers, Papers), type_err

    if sample is None:
        sample = papers.indices

    questions: Dict = {}
    idx = Sentences()
    for pid in tqdm(sample, desc='paper_ids'):
        for sent in papers.sents(pid):
            length = len(sent)
            if length < minlen:
                continue
            words = sent.lower().split()
            if words[0] in interrogative and words[-1].endswith("?"):
                if pid not in questions:
                    questions[pid] = []
                if sent in questions[pid]:
                    continue
                idx.strlen += length
                idx.counts += 1
                idx.maxlen = max(idx.maxlen, length)
                questions[pid].append(sent)

    if idx.counts == 0:
        return None
    idx.indices = list(questions.keys())
    return Papers(idx, questions)


def extract_titles_slow(cord19: CORD19Dataset,
                        sample: Optional[Iterable[int]] = None,
                        minlen: int = 10,
                        show_progress: bool = False) -> Dict[int, str]:
    """Extract titles from the CORD-19-Dataset (Slow).

    :param cord19: A CORD19Dataset instance with the method `corona.title()`.
    :param sample: An existing iterable of paper ids (Assumes the ids belong
        to the CORD19Dataset instance).
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


def extract_titles_fast(cord19: CORD19Dataset,
                        sample: Optional[Iterable[int]] = None,
                        minlen: int = 10,
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
    indices = iter(sorted(batch.keys()))
    for index in indices:
        title = batch[index]
        mapping[index] = title

    return mapping


def tune_ids(encoder,
             title_map: Dict[int, str],
             task_list: Optional[TaskList] = None,
             target_size: int = 1000,
             batch_size: int = 16,
             skip_false_trg: bool = False,
             show_progress: bool = True) -> GoldIdsOutput:
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

    output = GoldIdsOutput()
    for i, task in enumerate(task_list):
        task_embed = encoder.encode(task.all(), batch_size=16,
                                    show_progress=False)
        D, I = index_flat.search(task_embed, targets[i])
        ids = np.array([decode[k] for k in I.flatten()])
        output.append(GoldIds(task.id, ids=ids, dist=D.flatten()))

    return output
