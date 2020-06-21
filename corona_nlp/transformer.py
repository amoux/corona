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


class SummarizerTransformer:
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
            maxlen = 0
            tokens = []
            for j in lengths[i: min(i + batch_size, maxsize)]:
                ids = self.string_to_token_ids(sentences[j])
                maxlen = max(maxlen, len(ids))
                tokens.append(ids)

            inputs = dict([(key, []) for key in self.input_attrs])
            for ids in tokens:
                features = self.encode_features(ids, max_seqlen=maxlen)
                for input in features:
                    inputs[input].append(features[input])

            for input in inputs:
                inputs[input] = torch.cat(inputs[input]).to(self.device)

            with torch.no_grad():
                output = self.forward(inputs)
                embedding = output["sentence_embedding"]
                embeddings.extend(embedding.to("cpu").numpy())

        embeddings = [embeddings[i] for i in np.argsort(lengths)]
        return np.array(embeddings)

    def string_to_token_ids(self, string: str) -> List[int]:
        """Encode a string sequence into a sequence of token ids."""
        return self._tokenize_module(string)

    def encode_features(self, token_ids: List[int],
                        max_seqlen: int) -> Dict[str, Tensor]:
        """Encode a sequence of token ids to sentence features.

        :param token_ids: List[int], sequence of token ids.
        :param max_seqlen: int, maximum size of a single or all sequence(s)
            of token ids.
        """
        return self._features_module(token_ids, max_seqlen)

    def encode_features_plus(self,
                             token_ids: Union[List[int], List[List[int]]],
                             max_seqlen: int = None) -> Dict[str, Tensor]:
        """"Encode sequence(s) of ids to sentence features before model.

        :param token_ids: List[int] | List[List[int]], sequence(s) of ids.
        :param max_seqlen: Optional[int], maximum size of a single or all
            sequences, if None; max_seqlen is computed automatically.

        Returns Dict[str, Tensor], with tensors added to ``self.device``.
        """
        if isinstance(token_ids[0], list):
            if max_seqlen is None:
                max_seqlen = len(max(token_ids, key=len))
        elif isinstance(token_ids[0], int):
            if max_seqlen is None:
                max_seqlen = len(token_ids)
            token_ids = [token_ids]
        else:
            raise TypeError(
                "Expected sequence(s) of ids, List[int] or List[List[int]]")

        inputs = dict([(key, []) for key in self.input_attrs])
        for ids in token_ids:
            features = self.encode_features(ids, max_seqlen)
            for input in features:
                inputs[input].append(features[input])

        for input in inputs:
            features = torch.cat(inputs[input])
            inputs[input] = features.to(self.device)

        return inputs

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
