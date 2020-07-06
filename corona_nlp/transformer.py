import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import torch
from sentence_transformers import __version__
from sentence_transformers.util import import_from_string
from summarizer import Summarizer
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from .modules import BertModule, PoolingModule

_ST_INCOMPATIBLE_VERSION_TO_LATEST_HF = '0.2.6'


class BertSummarizer:
    @staticmethod
    def load(model: str, tokenizer: BertTokenizer, device=None) -> Summarizer:
        config = BertConfig.from_pretrained(model)
        config.output_hidden_states = True
        bert_model = BertModel.from_pretrained(model, config=config)
        if device is not None:
            bert_model = bert_model.to(device)
        return Summarizer(custom_model=bert_model, custom_tokenizer=tokenizer)


class SentenceTransformer(nn.Sequential):
    input_attrs = ["input_ids", "token_type_ids", "attention_mask"]

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
        if model_path is not None:
            model_path = Path(model_path)
            modules_file = model_path.joinpath("modules.json")
            assert model_path.is_dir()
            assert modules_file.is_file()
            logging.info(f"Loading model from path: {model_path}")

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
        if modules is not None:
            if not isinstance(modules, OrderedDict):
                modules = OrderedDict([(str(i), m) for i, m in modules])
        else:
            modules = OrderedDict()
            with modules_file.open("r") as file:
                contained_modules = json.load(file)

            if __version__ <= _ST_INCOMPATIBLE_VERSION_TO_LATEST_HF:
                models = [BertModule, PoolingModule]
                for model, config in zip(models, contained_modules):
                    path = model_path.joinpath(config["path"])
                    modules[config["name"]] = model(path.as_posix())
            else:
                for config in contained_modules:
                    model = import_from_string(config["type"])
                    path = model_path.joinpath(config["path"])
                    modules[config["name"]] = model.load(path.as_posix())

        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"using pytorch device {device}")
        self.device = torch.device(device)
        self.to(device)
        # methods from the first module e.g., ``0_BERT```
        self.basic_tokenize = self._first_module().tokenize
        self.get_sentence_features = self._first_module().get_sentence_features
        self.tokenizer = self._first_module().tokenizer

    def encode(self, sentences: List[str],
               batch_size: int = 8, show_progress: bool = True) -> np.array:
        """Encode an iterable of string sequences to a embedding matrix."""
        self.eval()
        lengths = np.argsort([len(sent) for sent in sentences])
        maxsize = lengths.size
        batch = range(0, maxsize, batch_size)
        if show_progress:
            batch = tqdm(batch, desc="batch")

        embeddings = []
        for i in batch:
            splits = []
            for j in lengths[i: min(i + batch_size, maxsize)]:
                tokens = self.tokenizer.tokenize(sentences[j])
                splits.append(tokens)

            batch = self.tokenizer(text=splits,
                                   is_pretokenized=True,
                                   padding='longest',
                                   return_tensors='pt').to(self.device)
            with torch.no_grad():
                output = self.forward(batch)
                embedding = output["sentence_embedding"]
                embeddings.extend(embedding.to("cpu").numpy())

        embeddings = [embeddings[i] for i in np.argsort(lengths)]
        return np.array(embeddings)

    def embed(self, inputs: Dict[str, Tensor], codes: str = "sentence",
              astype: str = "torch") -> Union[Tensor, np.array]:
        """Transform inputs to embeddings.

        :param inputs: Dict[str, Tensor], inputs with tensors set to device.
        :param codes: ``sentence`` for sentence embedding | ``token`` for
            token embeddings outputs.
        :param astype: return the embedding as ``torch`` or ``numpy`` type.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(inputs)
            codes = "token_embeddings" if codes == "token" \
                else "sentence_embedding"

            embedding = output[codes]
            if codes == "token_embeddings":
                attn_mask = output["attention_mask"]
                attn_mask = attn_mask.unsqueeze(-1).expand(
                    embedding.size()).float()
                embedding = embedding * attn_mask

            if astype == "numpy":
                embedding = embedding.to("cpu").numpy()

        return embedding

    def _first_module(self):
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        return self._modules[next(reversed(self._modules))]
