from typing import Iterable, List, NamedTuple, Optional

import spacy
from spacy.tokens import Doc, Span


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
        nlp = spacy.load(nlp_model, disable=disable)
        nlp.max_length = max_length
        self.nlp = nlp
        self.nlp_model = nlp_model
        self.word_ratio = word_ratio
        self.char_ratio = char_ratio
        self.min_tokens = min_tokens
        self.disabled_pipes = disable
        self.max_length = max_length

    def tokenize(self, text: str) -> List[Span]:
        """Tokenize to sentences from a string of sequences to sentences."""
        doc = self.nlp(text)
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
            tuple(self.disabled_pipes), self.word_ratio, self.char_ratio
        )

    def __call__(self, text: str) -> Iterable[Span]:
        return self.nlp(text).sents
