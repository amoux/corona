import json
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from sentence_transformers import __version__
from sentence_transformers.util import import_from_string
from sklearn.model_selection import train_test_split
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer)

BERT_MODELS_UNCASED = {
    'squad': "bert-large-uncased-whole-word-masking-finetuned-squad",
    'scibert': "allenai/scibert_scivocab_uncased"
}


class SentenceTransformer(nn.Sequential):

    def __init__(
        self,
        model_path: str = None,
        modules: Iterable[nn.Module] = None,
        device: str = None,
    ):
        """Sentence Transformer Class.

        NOTE: This class is slightly modified from the original version with
        minor performance enhancements. Like eliminating constant calls of the
        len function on all the sentences in a loop - the consequences of this
        alone; are noticeable when the number of samples is > 30k thousand.

        Original source: https://github.com/UKPLab/sentence-transformers

        model_path: path to the directory with the config and modules file.
        modules: iterable object of nn.Module instances.
        device: computation device to choose.
        """
        if modules is not None:
            if not isinstance(modules, OrderedDict):
                modules = OrderedDict([(str(i), m) for i, m in modules])

        if model_path is not None:
            model_path = Path(model_path)
            assert model_path.is_dir()
            logging.info(f"loading model from: {model_path}")

            config_file = model_path.joinpath("config.json")
            if config_file.is_file():
                with config_file.open("r") as file:
                    cfg = json.load(file)
                    if cfg['__version__'] > __version__:
                        logging.warning(
                            "You try to use a model that was created with "
                            "version {}, however, your version is {}. This "
                            "might cause unexpected behavior or errors. In "
                            "that case, try to update to the latest version"
                            ".\n\n\n".format(cfg['__version__'], __version__),
                        )
            modules = OrderedDict()
            modules_file = model_path.joinpath("modules.json")
            if modules_file.is_file():
                with modules_file.open("r") as file:
                    contained_modules = json.load(file)
                    for config in contained_modules:
                        class_ = import_from_string(config["type"])
                        module = model_path.joinpath(config["path"])
                        module_class = class_.load(module.as_posix())
                        modules[config["name"]] = module_class

        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"using pytorch device {device}")
        self.device = torch.device(device)
        self.to(device)
        self._tokenize_module = self._first_module().tokenize
        self._features_module = self._first_module().get_sentence_features

    def encode(self, sentences: List[str], batch_size: int = 8, as_numpy=True):
        """Encode an iterable of string sequences to a embedding matrix."""
        self.eval()
        sorted_lengths = np.argsort([len(sent) for sent in sentences])
        n_samples = sorted_lengths.size

        embeddings = []
        for i in tqdm(range(0, n_samples, batch_size), desc="batches"):
            start = i
            end = min(start + batch_size, n_samples)
            batch_tokens, maxlen = [], 0
            for j in sorted_lengths[start:end]:
                string = sentences[j]
                tokens = self.tokenize(string)
                maxlen = max(maxlen, len(tokens))
                batch_tokens.append(tokens)

            sentence_features = {}
            for sentence in batch_tokens:
                features = self.get_sentence_features(sentence, maxlen)
                for name in features:
                    if name not in sentence_features:
                        sentence_features[name] = []
                    sentence_features[name].append(features[name])

            for feature in sentence_features:
                features = np.asarray(sentence_features[feature])
                sentence_features[feature] = torch.tensor(
                    features).to(self.device)

            with torch.no_grad():
                output = self.forward(sentence_features)
                matrix = output["sentence_embedding"]
            if as_numpy:
                matrix = matrix.to("cpu").numpy()
            embeddings.extend(matrix)

        embeddings = [embeddings[i] for i in np.argsort(sorted_lengths)]
        return np.array(embeddings)

    def tokenize(self, string: str):
        return self._tokenize_module(string)

    def get_sentence_features(self, *features):
        return self._features_module(*features)

    def _first_module(self):
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        return self._modules[next(reversed(self._modules))]


class TextDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        model="scibert",
        max_length=512,
        overwrite_cache=False,
    ):
        file_path = Path(file_path)
        assert file_path.is_file()
        self.model_type = BERT_MODELS_UNCASED[model]
        self.samples = []

        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        block_size = (tokenizer.max_len - tokenizer.max_len_single_sentence)
        max_length = max_length - block_size

        outdir = file_path.parent
        file = file_path.name.replace(file_path.suffix, "")
        name = self.model_type.replace("allenai/", "")
        cached_features = outdir.joinpath(
            f"{name}_cached_lm_{max_length}_{file}")

        if cached_features.exists() and not overwrite_cache:
            print(f"loading features from cashed files: {cached_features}")
            with cached_features.open("rb") as handle:
                self.samples = pickle.load(handle)
        else:
            print(f"creating features from papers file at: {outdir}")
            with file_path.open(encoding="utf-8") as file:
                texts = file.read()

            tokenized = tokenizer.tokenize(texts)
            sequences = tokenizer.convert_tokens_to_ids(tokenized)
            truncated_range = range(
                0, len(sequences) - max_length + 1, max_length)
            for i in tqdm(truncated_range, desc="sequences", unit=""):
                inputs = sequences[i: i + max_length]
                tokens = tokenizer.build_inputs_with_special_tokens(inputs)
                self.samples.append(tokens)

            with cached_features.open("wb") as handle:
                pickle.dump(self.samples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
                del texts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return torch.tensor(self.samples[item], dtype=torch.long)

    def load_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_type)


class LineByLineTextDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        model: str = "scibert",
        max_length: int = 512,
        add_special_tokens: bool = True,
        overwrite_cache=False,
    ):
        file_path = Path(file_path)
        assert file_path.is_file()

        self.model_type = BERT_MODELS_UNCASED[model]
        self.samples = []

        outdir = file_path.parent
        file = file_path.name.replace(file_path.suffix, "")
        name = self.model_type.replace("allenai/", "")
        cached_features = outdir.joinpath(f"{model}_cashed_lm_{max_length}")

        if cached_features.exists() and not overwrite_cache:
            print(f"loading features from cashed file: {cached_features}")
            with cached_features.open("rb") as handle:
                self.samples = pickle.load(handle)
        else:
            print(f"creating features from cashed files: {outdir}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            with file_path.open("r", encoding="utf-8") as file:
                for line in tqdm(file, desc="tokenized-lines"):
                    strings = line.splitlines()
                    for string in strings:
                        token_ids = tokenizer.encode(
                            text=string,
                            max_length=max_length,
                            add_special_tokens=add_special_tokens,
                        )
                    self.samples.append(token_ids)

            with cached_features.open("wb") as handle:
                pickle.dump(self.samples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return torch.tensor(self.samples[item], dtype=torch.long)

    def load_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_type)

    def load_model(self) -> AutoModel:
        return AutoModel.from_pretrained(self.model_type)
