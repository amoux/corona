from typing import List, Union

import spacy
from summarizer import Summarizer
from transformers import BertConfig, BertModel, BertTokenizer


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


class BertSummarizer:
    @staticmethod
    def load(model: str, tokenizer: BertTokenizer, device=None) -> Summarizer:
        config = BertConfig.from_pretrained(model)
        config.output_hidden_states = True
        bert_model = BertModel.from_pretrained(model, config=config)
        if device is not None:
            bert_model = bert_model.to(device)
        return Summarizer(custom_model=bert_model, custom_tokenizer=tokenizer)
