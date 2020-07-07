import json
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

import torch.nn as nn
from sentence_transformers.models import Pooling as PoolingModule
from torch import Tensor
from transformers import BertModel, BertTokenizer


class BertModule(nn.Module):
    config_keys = ['max_seq_length', 'do_lower_case']

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        do_lower_case: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = {},
        tokenizer_kwargs: Optional[Dict[str, Any]] = {},
    ):
        super(BertModule, self).__init__()
        self.max_seq_length = max_seq_length = 510 \
            if max_seq_length > 510 else max_seq_length
        self.do_lower_case = do_lower_case
        tokenizer_kwargs['do_lower_case'] = self.do_lower_case
        self.model = BertModel \
            .from_pretrained(model_name_or_path, **model_kwargs)
        self.tokenizer = BertTokenizer \
            .from_pretrained(model_name_or_path, **tokenizer_kwargs)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = self.model(**inputs)
        embeddings = outputs[0]
        cls_tokens = embeddings[:, 0, :]  # [CLS] token
        inputs.update({'token_embeddings': embeddings,
                       'cls_token_embeddings': cls_tokens,
                       'attention_mask': inputs['attention_mask']})

        if len(outputs) > 2:
            inputs.update({'all_layer_embeddings': outputs[2]})
        return inputs

    def get_word_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def tokenize(self, text: str, add_special_tokens=False) -> List[int]:
        token_ids = self.tokenizer.encode(
            text=text, add_special_tokens=add_special_tokens)
        return token_ids

    def get_sentence_features(self, text: Union[str, List[str], List[int]],
                              max_seq_length: int = None) -> Dict[str, Tensor]:
        """Convert a sequence to inputs of token-ids, segment-ids and mask.

        Note: The `text` parameter expects values (a string or tokens) without
            added special tokens, e.g,. `[CLS]` and `[SEP]` or in integer form.

        :param text: Sequence to be encoded. This can be a string, a list of
            strings (tokenized string using the `tokenizer.tokenize` method)
            or a list of integers (tokenized string ids using the `tokenize`
            or `tokenizer.convert_tokens_to_ids` method.
        """
        if max_seq_length is None:
            if isinstance(text, str):
                text = self.tokenize(text, add_special_tokens=False)
                max_seq_length = len(text)
            elif isinstance(text[0], int) or isinstance(text[0], str) \
                    and isinstance(text[:1], list):  # is valid string token
                max_seq_length = len(text)

        # Add space for [CLS] and [SEP] special tokens
        max_length = min(max_seq_length, self.max_seq_length) + 2
        inputs = self.tokenizer.encode_plus(text=text,
                                            max_length=max_length,
                                            padding=True,
                                            return_tensors='pt')
        return inputs

    def get_config_dict(self):
        return dict([(k, self.__dict__[k]) for k in self.config_keys])

    def save(self, out_dir: str) -> IO:
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        config = out_dir.joinpath('sentence_bert_config.json')
        with config.open('w') as file:
            json.dump(self.get_config_dict(), file, indent=2)

    @staticmethod
    def load(path: str) -> 'BertModule':
        config = Path(path).joinpath('sentence_bert_config.json')
        with config.open('r') as file:
            config = json.load(file)
        return BertModule(model_name_or_path=path, **config)
