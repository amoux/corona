import functools
from typing import List

import spacy
from spacy.lang.en import English
from spacy.tokens.span import Span


class SpacySentenceTokenizer:

    def __init__(
        self,
        nlp_model="en_core_web_sm",
        disable=["ner", "tagger"],
        max_length=2_000_000,
    ):
        """Spacy Sentence Tokenizer.

        :params nlp_model: spaCy model to use for the tokenizer.
        :params disable: name of spaCy's pipeline components to disable.
        """
        self.nlp_model = nlp_model
        self.disable = disable
        self.max_length = max_length

    @property
    def cache(self):
        info = self.nlp.cache_info()
        if info.hits:
            return info.hits

    @functools.lru_cache()
    def nlp(self) -> List[English]:
        en = spacy.load(self.nlp_model, disable=self.disable)
        en.max_length = self.max_length
        return en

    def tokenize(self, doc: str) -> List[Span]:
        """Tokenize to sentences from a string of sequences to sentences."""
        doc = self.nlp()(doc)
        return list(doc.sents)

    def __repr__(self):
        model, pipe = self.nlp_model, self.disable
        return f"<SpacySentenceTokenizer({model}, disable={pipe})>"
