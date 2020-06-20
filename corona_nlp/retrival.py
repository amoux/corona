from typing import Dict, List, Optional, Union

import faiss
import spacy
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

from .dataset import CORD19Dataset
from .datatypes import Papers
from .utils import clean_punctuation, normalize_whitespace


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


def extract_paper_titles(cord19: CORD19Dataset,
                         minlen=10, size=-1) -> Dict[int, str]:
    """Return a map of paper ids to titles extracted from the dataset."""
    sample = cord19.sample(-1)
    if size > -1:
        sample = sample[:size]

    paper_titles = {}
    for pid in tqdm(sample, desc='titles'):
        title = normalize_whitespace(cord19.title(pid))
        if len(clean_punctuation(title)) <= minlen:
            continue
        if pid not in paper_titles:
            paper_titles[pid] = title

    return paper_titles


def tune_dataset_to_tasks(
        tasks: Union[List[str], List[List[str]]],
        encoder: 'SentenceTransformer',
        minlen: Optional[int] = 10,
        n_size: Optional[int] = -1,
        cord19: Optional[CORD19Dataset] = None,
        paper_titles: Optional[Dict[int, str]] = None,
        target_size: Optional[int] = None,
        k_nn: Optional[int] = None,
        show_progress: bool = False) -> Union[List[int], List[List[int]]]:
    """Return a sample of ids tuned to an iterable of tasks.

    param: tasks (Union[List[str], List[List[str]]]):
        An iterable of string sequences or a list of iterables of string
        sequences. Tasks are expected to be in form of text queries.
        Multiple tasks available in the `cord_nlp.tasks` module.
    param: minlen (Optional, int):
        Minimum title length, ignored if paper_titles is not None.
    param: n_size (Optional, int):
        Sample size for obtaining the titles, ignored if paper_titles
        is not None.
    param: paper_titles (Optional, Dict[int, str]):
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
    if not isinstance(tasks[0], list):
        tasks = [tasks]
    if paper_titles is None:
        if cord19 is not None:
            paper_titles = extract_paper_titles(cord19, minlen, n_size)
        else:
            raise Exception('Expected an ``CORD19Dataset`` instance or '
                            'a Dict[int, str] ``paper_titles`` mapping.')

    decode = dict(enumerate(paper_titles.keys()))
    sample = list(paper_titles.keys())
    titles = list(paper_titles.values())

    k_iter = []
    if k_nn is None and target_size is not None:
        # find best fit to top k, given number of queries per
        # centroid (n tasks) in relation to the expected return
        # size and the total samples available (n paper IDs)
        for task in tasks:
            ntasks = len(task)
            k_nn = round(target_size / ntasks) - ntasks % 2
            maxk = len(sample) - target_size
            assert (k_nn * ntasks) <= maxk, (
                'goal_size is larger than n queries possible given the '
                'sample size and number of tasks, choose a smaller goal.'
            )
            k_iter.append(k_nn)

    db = encoder.encode(titles, 8, show_progress)
    ndim = db.shape[1]
    index = faiss.IndexFlat(ndim)
    index.add(db)

    gold_ids = []
    for id in range(len(k_iter)):
        topk = k_iter[id]
        w_x = encoder.encode(tasks[id], 8, show_progress)
        knn = index.search(w_x, topk)[1].flatten().tolist()
        ids = sorted(set([decode[k] for k in knn]))
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
