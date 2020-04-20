from typing import List, Set, Union

import spacy
from spacy.lang.en import STOP_WORDS


def summarize(texts: Union[str, List[str]],
              nlargest=7,
              min_token_freq=30,
              spacy_model="en_core_web_sm",
              spacy_nlp: object = None,
              stop_words: Union[List[str], Set[str]] = None) -> str:
    """Text summarizer built on top of spaCy.

    - You want fast or quality:
        - If quality, please look at the Transformers summarizer pipeline
            I highly recommend using it as opposed to this summarizer.
        - If speed, try it.
    - This method is an alternative to using Transformers's summarizer
        pipeline - the outputs are state-of-the-art but extremely slow!
        This summarization algorithm is simple, good enough and fast!

    `texts`: An iterable list of string sequences.
    `spacy_model`: If `spacy_nlp=None`, the default nlp model will be
        used. The summarization quality improves the larger the model.
        For accuracy I recommend using `en_core_web_md`.
    `spacy_nlp`: Use an existing `spacy.lang.en.English` instance.
        The object is usually referred as `nlp`. Otherwise, a new
        instance will be loaded (which can take some time!).
    """
    nlp, doc, stop_words = (None, None, stop_words)
    if spacy_nlp is not None:
        nlp = spacy_nlp
    else:
        nlp = spacy.load(spacy_model)
    if isinstance(texts, list):
        doc = nlp(" ".join(texts))
    elif isinstance(texts, str):
        doc = nlp(texts)
    if stop_words is None:
        stop_words = STOP_WORDS

    k_words = {}  # word level frequency
    for token in doc:
        token = token.text
        if token not in stop_words:
            if token not in k_words:
                k_words[token] = 1
            else:
                k_words[token] += 1

    # normalize word frequency distributions
    for w in k_words:
        k_words[w] = k_words[w] / max(k_words.values())

    scores = {}  # sentence level scores.
    for sent in [i for i in doc.sents]:
        for word in sent:
            word = word.text.lower()
            if word in k_words:
                if len(sent.text.split()) < min_token_freq:
                    if sent not in scores:
                        scores[sent] = k_words[word]
                    else:
                        scores[sent] += k_words[word]

    # find the n[:n] largest sentences from the scores.
    sents = sorted(scores, key=scores.get, reverse=True)[:nlargest]
    summary = " ".join([i.text for i in sents])
    return summary
