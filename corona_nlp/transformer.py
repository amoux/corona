import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from sentence_transformers import __version__
from sentence_transformers.util import import_from_string
from torch import nn
from tqdm import tqdm


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
