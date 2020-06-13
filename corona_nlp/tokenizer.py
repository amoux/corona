import functools
from typing import List, Union

import spacy
from spacy.lang.en import English
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span


class SpacySentenceTokenizer:

    def __init__(
        self,
        nlp_model="en_core_sci_sm",
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
        nlp_ = spacy.load(self.nlp_model, disable=self.disable)
        nlp_.max_length = self.max_length
        return nlp_

    def tokenize(self, doc: str) -> List[Span]:
        """Tokenize to sentences from a string of sequences to sentences."""
        doc = self.nlp()(doc)
        return list(doc.sents)

    def is_sentence(self, doc: Union[str, Doc],
                    token_count=5, word_ratio=0.40, char_ratio=0.60) -> bool:
        """Check whether a sequence is a valid english sentence."""
        if not isinstance(doc, Doc) and isinstance(doc, str):
            doc = self.nlp()(doc)

        tokens = [token for token in doc.doc]
        if len(tokens) < token_count:
            return False

        num_words = sum([token.is_alpha for token in tokens])
        if num_words / len(tokens) < word_ratio:
            return False

        num_chars = sum([char.isalpha() for char in doc.text])
        if num_chars / len(tokens) < char_ratio:
            return False

        return True

    def __repr__(self):
        model, pipe = self.nlp_model, self.disable
        return f"<SpacySentenceTokenizer({model}, disable={pipe})>"
