from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import spacy
from summarizer import SingleModel
from summarizer.sentence_handler import SentenceHandler
from transformers import AutoModel, AutoTokenizer


@dataclass
class BertSummarizerConfig:
    model: Optional[str] = None
    custom_model: Optional[AutoModel] = None
    custom_tokenizer: Optional[AutoTokenizer] = None
    hidden: int = -2
    reduce_option: str = 'mean'
    sentence_handler: Optional[SentenceHandler] = None
    random_state: int = 12345

    def todict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BertSummarizerArguments:
    """Default arguments for Bert Extractive Summarizer.
    https://github.com/dmmiller612/bert-extractive-summarizer

    :param body: Text of strings / input to summarize.
    :param ratio: Ratio of sentences to summarize to from the original body.
    :param min_length: Minimum sequence length to accept as a sentence.
    :param max_length: Maximum sequence length to accept as a sentence.
    :param algorithm: Clustering algorithm: `kmeans` (default) or `gmm`.
    """
    body: Optional[str] = None
    ratio: float = 0.2
    min_length: int = 40
    max_length: int = 600
    use_first: bool = True
    algorithm: str = 'kmeans'

    def todict(self) -> Dict[str, Any]:
        return asdict(self)


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


class BertSummarizer(SingleModel):
    def __init__(
        self,
        model: Optional[AutoModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_name_or_path: Optional[str] = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: Optional[SentenceHandler] = None,
        random_state: int = 12345,
    ) -> None:
        if model_name_or_path is not None:
            model = AutoModel.from_pretrained(model_name_or_path,
                                              output_hidden_states=True)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        elif model is not None and hasattr(model, 'config') \
                and model.config is not None:
            model.config.output_hidden_states = True
        else:
            raise ValueError(
                'Expected a `model and tokenizer` as initialized instances '
                'or a `model_name_or_path` string for constructing new '
                'AutoModel and AutoTokenizer transformer\'s instances.'
            )
        if sentence_handler is None:
            sentence_handler = SentenceHandler()

        kwargs = BertSummarizerConfig(
            # The API enforced this (We simply ignore it)
            model='bert-base-uncased',
            custom_model=model,
            custom_tokenizer=tokenizer,
            hidden=hidden,
            reduce_option=reduce_option,
            sentence_handler=sentence_handler,
            random_state=random_state,
        )
        super(BertSummarizer, self).__init__(**kwargs.todict())
