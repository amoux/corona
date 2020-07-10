import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

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
    input_attrs = ['input_ids', 'token_type_ids', 'attention_mask']

    def __init__(self, model_path: str = None,
                 modules: Iterable[nn.Module] = None, device: str = None):
        """Sentence Transformer Class.

        This class is slightly modified from the original version.
        Original source: https://github.com/UKPLab/sentence-transformers

        :param model_path: Path to the config and modules json file.
        :param modules: Iterable object of `nn.Module` instances.
        :param device: Computation device to choose.
        """
        if model_path is not None:
            model_path = Path(model_path)
            modules_file = model_path.joinpath('modules.json')
            assert model_path.is_dir()
            assert modules_file.is_file()
            logging.info(f'Loading model from path: {model_path}')

            config_file = model_path.joinpath('config.json')
            if config_file.is_file():
                with config_file.open('r') as file:
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
            with modules_file.open('r') as file:
                contained_modules = json.load(file)

            if __version__ <= _ST_INCOMPATIBLE_VERSION_TO_LATEST_HF:
                models = [BertModule, PoolingModule]
                for model, config in zip(models, contained_modules):
                    path = model_path.joinpath(config['path'])
                    modules[config['name']] = model(path.as_posix())
            else:
                for config in contained_modules:
                    model = import_from_string(config['type'])
                    path = model_path.joinpath(config['path'])
                    modules[config['name']] = model.load(path.as_posix())

        super().__init__(modules)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f'using pytorch device {device}')
        self.device = torch.device(device)
        self.to(device)
        # methods from the first module e.g., ``0_BERT```
        self.get_sentence_features = self._first_module().get_sentence_features
        self.max_seq_length = self._first_module().max_seq_length
        self.tokenize = self._first_module().tokenize
        self.tokenizer = self._first_module().tokenizer

    def _first_module(self):
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        return self._modules[next(reversed(self._modules))]

    def encode(self, sentences: List[str],
               batch_size: int = 8, show_progress: bool = True) -> np.array:
        """Encode an iterable of string sequences to a embedding matrix."""
        self.eval()
        lengths = np.argsort([len(sent) for sent in sentences])
        maxsize = lengths.size
        batches = range(0, maxsize, batch_size)
        if show_progress:
            batches = tqdm(batches, desc='batches')

        embeddings = []
        for i in batches:
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
                embedding = output['sentence_embedding']
                embeddings.extend(embedding.to('cpu').numpy())

        embeddings = [embeddings[i] for i in np.argsort(lengths)]
        return np.array(embeddings)

    def embed(self, inputs: Dict[str, Tensor], coding: str = 'sentence',
              astype: str = 'torch') -> Union[Tensor, np.array]:
        """Transform inputs to embeddings.

        :param inputs: Dict[str, Tensor], inputs with tensors set to device.
        :param coding: ``sentence`` for sentence embedding | ``token`` for
            token embeddings outputs.
        :param astype: return the embedding as ``torch`` or ``numpy`` type.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(inputs)
            coding = 'token_embeddings' if coding == 'token' \
                else 'sentence_embedding'

            embedding = output[coding]
            if coding == 'token_embeddings':
                attn_mask = output['attention_mask']
                attn_mask.unsqueeze_(-1).expand(embedding.size()).float()
                embedding = embedding * attn_mask

            if astype == 'numpy':
                embedding = embedding.to('cpu').numpy()

        return embedding

    def encode_sentence(
            self, text: Union[str, List[str], List[int]],
            max_length: int = None,
            padding: Union[bool, str] = False,
            truncation: Union[bool, str] = False,
            is_pretokenized: bool = False,
            return_tensors: str = None,
    ) -> Dict[str, Union[List[int], np.array, Tensor]]:
        """Encode a sequence to inputs of token-ids, segment-ids and mask.

        Note: The `text` parameter expects values (a string or tokens) without
            added special tokens, e.g,. `[CLS]` and `[SEP]` or in integer form.

        :param text: Sequence to be encoded. This can be a string, a list of
            strings (tokenized string using the `tokenizer.tokenize` method)
            or a list of integers (tokenized string ids using the `tokenize`
            or `tokenizer.convert_tokens_to_ids` method.
        :param max_length: If None, `max_length` is computed automatically
            for either; a string or list of strings (pretokenized). Since
            - `max_length` is obtained from either int|str - `padding` and
            `is_pretokenized` parameters are set to True.
        """
        if max_length is None:
            if isinstance(text, str):
                text: List[str] = self.tokenizer.tokenize(text)
                max_length = len(text)
            elif isinstance(text[0], int) or isinstance(text[0], str) \
                    and isinstance(text[:1], list):
                max_length = len(text)
            # Any text variation is pre-tokenized within this < if > block.
            padding = is_pretokenized = True
        # Prepend two spaces for [CLS] and [SEP] special tokens.
        max_length = min(max_length, self.max_seq_length) + 2
        inputs = self.tokenizer.encode_plus(text, padding=padding,
                                            truncation=truncation,
                                            max_length=max_length,
                                            is_pretokenized=is_pretokenized,
                                            return_tensors=return_tensors)
        if return_tensors == 'pt':
            return inputs.to(self.device)
        return inputs

    def encode_sentences(
        self, text: Union[str, List[str], List[List[str]]],
        max_length: int = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        is_pretokenized: bool = False,
        return_tensors: str = None,
    ) -> Dict[str, Union[List[List[int]], np.array, Tensor]]:
        """Encode sequence(s) to inputs of token-ids, segments, and mask.

        NOTE: The `text` param expects a sequence or list of sequences
            of strings without added special tokens, e.g,. `[CLS]` and
            `[SEP]`. This method should be used only with sequences of
            `type=str` and not of `type=int`.

        * Padding and truncation strategy: `padding to specific length`

        - Encoding a list of sequences of strings (sentences) List[str]

        ```python
        encode_sentences(batch_sentences,     # or tokenized batch
                         padding='max_length',
                         truncation=True,
                         max_seq_length=None, # computed automatically
                         is_pretokenized=False)
        ```

        - The following arg values cause a fallback to model's default
            max_length (meaning, a custom max_length value is ignored).

        ```python
        ...
        max_length = len(max(batch_pretokenized, key=len))
        encode_batch(batch_pretokenized,        # or batch_sentences
                     padding='longest',         < causes fallback >
                     truncation=False,          < causes fallback >
                     max_seq_length=max_length, < will be ignored >
                     is_pretokenized=False) # applies to true|false
        ```
        :param text: A sequence or batch of sequences to be encoded.
            Each sequence can be a string or a list of strings (pre-
            tokenized string). If the sequences are provided as list
            of strings (pretokenized), and `max_seq_length` is given
            then, you must set `is_pretokenized=True` (to lift the
            ambiguity with a batch of sequences).
        """
        if max_length is None:
            # If a string sequence.
            if isinstance(text, str):
                text: List[str] = self.tokenizer.tokenize(text)
                max_length = len(text)
            elif isinstance(text, list):
                # If list of string sequences.
                if isinstance(text[0][:1], str):
                    batch_tokenized: List[List[str]] = []
                    maxlen = 0
                    for string in text:
                        tokens = self.tokenizer.tokenize(string)
                        maxlen = max(maxlen, len(tokens))
                        batch_tokenized.append(tokens)
                    text, max_length = batch_tokenized, maxlen
                # If list of list of token sequences.
                elif isinstance(text[0][:1], list):
                    max_length = len(max(text, key=len))
            # Any text variation is pre-tokenized within this < if > block.
            padding = True if padding is None else padding
            is_pretokenized = True
        # Prepend two spaces for [CLS] and [SEP] special tokens.
        max_length = min(max_length, self.max_seq_length) + 2
        batch = self.tokenizer(text, padding=padding,
                               truncation=truncation,
                               max_length=max_length,
                               is_pretokenized=is_pretokenized,
                               return_tensors=return_tensors)
        if return_tensors == 'pt':
            return batch.to(self.device)
        return batch
