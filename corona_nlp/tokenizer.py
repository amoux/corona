import functools
from typing import List, NamedTuple, Optional

import spacy
from spacy.lang.en import English
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span


class TextScore(NamedTuple):
    word_ratio: float
    char_ratio: float
    num_tokens: int


class SpacySentenceTokenizer:
    def __init__(
        self,
        nlp_model: str = "en_core_sci_sm",
        word_ratio: float = 0.40,
        char_ratio: float = 0.60,
        min_tokens: int = 5,
        disable: List[str] = ["ner", "tagger"],
        max_length: int = 2_000_000,
    ) -> None:
        """Spacy Sentence Tokenizer.

        :params nlp_model: spaCy model to use for the tokenizer.
        :params disable: name of spaCy's pipeline components to disable.
        """
        self.nlp_model = nlp_model
        self.word_ratio = word_ratio
        self.char_ratio = char_ratio
        self.min_tokens = min_tokens
        self.disable = disable
        self.max_length = max_length

    @property
    def cache(self):
        info = self.nlp.cache_info()
        if info.hits:
            return info.hits

    @functools.lru_cache()
    def nlp(self) -> List[English]:
        nlp_ = spacy.load(self.nlp_model, disable=self.disable)
        nlp_.max_length = self.max_length
        return nlp_

    def tokenize(self, doc: str) -> List[Span]:
        """Tokenize to sentences from a string of sequences to sentences."""
        doc = self.nlp()(doc)
        return list(doc.sents)

    def score(self, doc: Doc) -> TextScore:
        num_tokens = len(doc)
        return TextScore(
            word_ratio=sum([x.is_alpha for x in doc]) / num_tokens,
            char_ratio=sum([x.isalpha() for x in doc.text]) / num_tokens,
            num_tokens=num_tokens,
        )

    def is_sentence(self, doc: Optional[Doc] = None, k: Optional[TextScore] = None) -> bool:
        """Check whether a sequence is a valid english sentence."""
        k = k if k is not None else self.score(doc)
        if k.num_tokens < self.min_tokens:
            return False
        if k.word_ratio < self.word_ratio:
            return False
        if k.char_ratio < self.char_ratio:
            return False
        return True

    def __repr__(self):
        return "{}(model: {}, pipe: {}, word_ratio: {}, char_ratio: {})".format(
            self.__class__.__name__, self.nlp_model,
            tuple(self.disable), self.word_ratio, self.char_ratio
        )
