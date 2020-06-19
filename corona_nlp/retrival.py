from typing import List, Optional, Union

import faiss
import spacy
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

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


def tune_dataset_to_tasks(cord19: 'CORD19Dataset',
                          encoder: 'SentenceTransformer',
                          tasks: List[str],
                          minlen: int = 10,
                          size: int = -1,
                          goal_size: Optional[int] = None,
                          k_nn: Optional[int] = None,
                          show_progress: bool = False) -> List[int]:
    """Return a sample of ids tuned to an iterable of tasks.

    param: minlen: filter, minimum length of titles.
    param: size: sample size to use for obtaining the titles.
    param: goal_size: expected size of a sample. if the expected size
        is larger; random selected ids will be added to meet fulfillment
    param: k_nn: k number of neighbors to query against the titles.
    param: show_progress: wheather to display the progress of encoding.
    """
    sample = cord19.sample(-1)
    if size > -1:
        sample = sample[:size]

    if k_nn is None and goal_size is not None:
        # find best fit to top k, given number of queries per
        # centroid (n tasks) in relation to the expected return
        # size and the total samples available (n paper IDs)
        ntasks = len(tasks)
        k_nn = round(goal_size / ntasks) - ntasks % 2
        maxk = len(sample) - goal_size
        assert (k_nn * ntasks) <= maxk, (
            'goal_size is larger than the number of queries possible given'
            ' the sample size and number of tasks, choose a smaller goal.')
    else:
        raise ValueError(
            'Expected either ``k_nn`` or ``goal_size`` values.')

    paper_titles = {}
    for pid in tqdm(sample, desc='titles'):
        title = normalize_whitespace(cord19.title(pid))
        if len(clean_punctuation(title)) <= minlen:
            continue
        if pid not in paper_titles:
            paper_titles[pid] = title

    paper_index = dict(enumerate(paper_titles.keys()))
    titles = list(paper_titles.values())

    titles_embed = encoder.encode(titles, show_progress=show_progress)
    tasks_embed = encoder.encode(tasks, show_progress=show_progress)

    ndim = titles_embed.shape[1]
    index = faiss.IndexFlat(ndim)
    index.add(titles_embed)
    _, ids = index.search(tasks_embed, k_nn)

    k_nn_ids = ids.flatten().tolist()
    gold_ids = sorted(set([paper_index[i] for i in k_nn_ids]))
    if goal_size is None:
        return gold_ids

    ntotal = len(gold_ids)
    extra_ids = []
    if ntotal < goal_size:
        needs = goal_size - ntotal
        count = 0
        for need_id in sample:
            if need_id in gold_ids:
                continue
            if count < needs:
                extra_ids.append(need_id)
                count += 1
        assert len(extra_ids) + len(gold_ids) == goal_size

    gold_ids.extend(extra_ids)
    return gold_ids
