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
from tqdm.auto import tqdm


class SentenceTransformer(nn.Sequential):
    input_attrs = ('input_ids', 'token_type_ids',
                   'input_mask', 'sentence_lengths')

    def __init__(
        self,
        model_path: str = None,
        modules: Iterable[nn.Module] = None,
        device: str = None,
    ):
        """Sentence Transformer Class.

        This class is slightly modified from the original version.
        Original source: https://github.com/UKPLab/sentence-transformers

        `model_path`: Path to the directory with the config and modules file.
        `modules`: Iterable object of `nn.Module` instances.
        `device`: Computation device to choose.
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

    def encode(self, sentences: List[str], batch_size=8, with_tqdm=True):
        """Encode an iterable of string sequences to a embedding matrix."""
        self.eval()
        lengths = np.argsort([len(sent) for sent in sentences])
        maxsize = lengths.size
        batch = range(0, maxsize, batch_size)
        if with_tqdm:
            batch = tqdm(batch, desc="batch")

        embedding = []
        for i in batch:
            maxlen = 0
            tokens = []
            for j in lengths[i: min(i + batch_size, maxsize)]:
                ids = self.tokenize(sentences[j])
                maxlen = max(maxlen, len(ids))
                tokens.append(ids)

            attrs = dict([(x, []) for x in self.input_attrs])
            for ids in tokens:
                inputs = self.sentence_features(ids, maxlen)
                for input in inputs:
                    tensor = torch.tensor(inputs[input].tolist())
                    attrs[input].append(tensor.unsqueeze(0))

            for input in self.input_attrs[:3]:
                attrs[input] = torch.cat(attrs[input]).to(self.device)

            with torch.no_grad():
                output = self.forward(attrs)
                embedd = output["sentence_embedding"]
                embedding.extend(embedd.to("cpu").numpy())

        embedding = [embedding[i] for i in np.argsort(lengths)]
        return np.array(embedding)

    def tokenize(self, string: str):
        return self._tokenize_module(string)

    def sentence_features(self, *features):
        return self._features_module(*features)

    def _first_module(self):
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        return self._modules[next(reversed(self._modules))]
