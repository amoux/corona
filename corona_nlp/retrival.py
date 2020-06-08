from typing import List, Union

import spacy
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

from .datatypes import Papers
from .utils import clean_punctuation


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
