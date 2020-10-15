from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .models import Pooling, Transformer


class SentenceTransformer(nn.Sequential):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 128,
        do_lower_case: bool = False,
        do_class_token: bool = False,
        do_sqrt_tokens: bool = False,
        do_mean_tokens: bool = True,
        device: Optional[str] = None,
        model_kwargs: Dict = {},
        tokenizer_kwargs: Dict = {},
    ) -> None:
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        transformer = Transformer(model_name_or_path, **model_kwargs)
        pooling = Pooling(
            hidden_size=transformer.word_dim,
            do_class_token=do_class_token,
            do_sqrt_tokens=do_sqrt_tokens,
            do_mean_tokens=do_mean_tokens,
        )
        modules = OrderedDict()
        for idx, module in enumerate([transformer, pooling]):
            modules[str(idx)] = module.to(device)
        super(SentenceTransformer, self).__init__(modules)
        self.device = device
        tokenizer_kwargs.update({'do_lower_case': do_lower_case})
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_kwargs,
        )
        self.max_length = max_length
        self.do_lower_case = do_lower_case

    @property
    def word_dim(self) -> int:
        return self._modules['0'].word_dim

    @property
    def sent_dim(self) -> int:
        return self._modules['1'].sent_dim

    def encode(self, sentences: List[str], batch_size=9, show_progress=True):
        if hasattr(sentences, '_meta'):
            sentences = list(sentences)

        self.eval()
        lengths = np.argsort([len(sent) for sent in sentences])
        maxsize = lengths.size
        batches = range(0, maxsize, batch_size)
        if show_progress:
            batches = tqdm(batches, desc='batch')

        encode = self.tokenizer.encode
        batch_encode_plus = self.tokenizer.batch_encode_plus

        embeddings = []
        for i in batches:
            splits: List[List[int]] = []
            maxlen = 0
            for j in lengths[i: min(i + batch_size, maxsize)]:
                string = sentences[j]
                tokens = encode(string, add_special_tokens=False)
                maxlen = max(maxlen, len(tokens))
                splits.append(tokens)

            max_length = min(maxlen, self.max_length) + 2
            batch = batch_encode_plus(batch_text_or_text_pairs=splits,
                                      padding=True,
                                      truncation=True,
                                      max_length=max_length,
                                      is_split_into_words=True,
                                      return_tensors='pt').to(self.device)
            with torch.no_grad():
                output = self.forward(batch)['sentence_embed']
                embeddings.extend(output.to('cpu').numpy())

        embeddings = [embeddings[i] for i in np.argsort(lengths)]

        return np.array(embeddings)
