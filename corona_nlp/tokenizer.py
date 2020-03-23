import functools
import re
from pathlib import Path
from typing import IO, List, Sequence

import spacy
from spacy.lang.en import English
from tqdm import tqdm

from .preprocessing import normalize_whitespace


class SpacySentenceTokenizer:

    def __init__(
        self,
        out_file="covid-texts.txt",
        nlp_model="en_core_web_sm",
        disable=["ner", "tagger"],
    ):
        """Spacy Sentence Tokenizer.

        `out_file`: name of the file if tokenizenizing to file.
        `nlp_model`: spaCy model to use for the tokenizer.
        `disable`: name of spaCy's pipeline components to disable.
        """
        self.out_file = out_file
        self.nlp_model = nlp_model
        self.disable = disable

    @property
    def num_sents(self):
        info = self.nlp.cache_info()
        if info.hits:
            return info.hits

    @property
    def max_strlen(self):
        info = self.nlp.cache_info()
        if info.maxsize:
            return info.maxsize

    @functools.lru_cache()
    def nlp(self) -> List[English]:
        return spacy.load(self.nlp_model, disable=self.disable)

    def tokenize(self, doc: str) -> List[spacy.tokens.span.Span]:
        """Tokenize to sentences from a string of sequences to sentences."""
        doc = self.nlp()(doc)
        return list(doc.sents)

    def tokenize_batch(self, batch: List[str]) -> List[str]:
        """Tokenize an iterable list of string sequences to sentences."""
        sentences = []
        for string in self.tokenize_generator(batch):
            sentences.append(string)
        return sentences

    def tokenize_tofile(self, document: List[str], out_file: str = None) -> IO:
        """Tokenize an iterable list of string sequences, and save to file."""
        out_file = self.out_file if out_file is None else out_file
        with Path(out_file).open("w") as writer:
            for string in self.tokenize_generator(document):
                writer.write(f"{string}\n")

    def tokenize_generator(self, batch: List[str]) -> Sequence[str]:
        """Tokenize an iterable list of string sequences (Yields strings)."""
        for doc in tqdm(batch, desc="documents", unit=""):
            doc = re.sub(r"\s+", " ", doc).strip()
            sentences = self.tokenize(doc)
            for string in sentences:
                string = normalize_whitespace(string.text)
                if string == "":
                    continue
                else:
                    yield string

    def __repr__(self):
        model, pipe = self.nlp_model, self.disable
        return f"<SpacySentenceTokenizer({model}, disable={pipe})>"
