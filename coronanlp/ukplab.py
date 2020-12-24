"""Sentence Transformers
Multilingual Sentence Embeddings using BERT / RoBERTa / XLM-RoBERTa & Co. with PyTorch

* This module's code has been slightly modified from the original
implementation specific to the project's criteria. Please refer to the
author's GitHub-repo: https://github.com/UKPLab/sentence-transformers
for documentation and its actual implementation.

@article{thakur-2020-AugSBERT,
    title = "Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks",
    author = "Thakur, Nandan and Reimers, Nils and Daxenberger, Johannes and  Gurevych, Iryna", 
    journal= "arXiv preprint arXiv:2010.08240",
    month = "10",
    year = "2020",
    url = "https://arxiv.org/abs/2010.08240",
}
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore
import torch
from tqdm.auto import tqdm  # type: ignore
from transformers import (AutoModel, AutoTokenizer,  # type: ignore
                          PreTrainedModel, PreTrainedTokenizer)

from .core import Sampler, SentenceStore


def semantic_search(
    xq: Union[torch.Tensor, np.ndarray],
    xb: Union[torch.Tensor, np.ndarray],
    top_k: int = 10,
    q_chunk: int = 100,
    d_chunk: int = 100000,
) -> List[List[Dict[str, Union[int, float]]]]:

    if isinstance(xq, np.ndarray):
        xq = torch.from_numpy(xq)
    if isinstance(xb, np.ndarray):
        xb = torch.from_numpy(xb)
    if len(xq.shape) == 1:
        xq = xq.unsqueeze(0)
    xq = xq / xq.norm(dim=1)[:, None]
    xb = xb / xb.norm(dim=1)[:, None]
    if xb.device != xq.device:
        xb = xb.to(xq.device)

    output: List[List[Any]] = [[] for q in range(len(xq))]
    for query_start in range(0, len(xq), q_chunk):
        query_end = min(query_start + q_chunk, len(xq))

        for db_start in range(0, len(xb), d_chunk):
            db_end = min(db_start + d_chunk, len(xb))
            query = xq[query_start: query_end]
            db = xb[db_start: db_end].transpose(0, 1)
            cosine = torch.mm(query, db).cpu().numpy()
            cosine = np.nan_to_num(cosine)
            max_k = len(cosine[0]) - 1
            top_k_args = np.argpartition(-cosine, min(top_k, max_k))
            top_k_args = top_k_args[:, :top_k]

            for x in range(len(cosine)):
                for k in top_k_args[x]:
                    index = {'xb': db_start + k, 'score': cosine[x][k]}
                    output[query_start + x].append(index)

    for i in range(len(output)):
        output[i] = sorted(output[i], key=lambda j: j['score'], reverse=True)
        output[i] = output[i][:top_k]
    return output


class SentenceEncoderPooling(torch.nn.Module):
    def __init__(
        self,
        do_cls_tokens=False,
        do_max_tokens=False,
        do_sqrt_tokens=False,
        do_mean_tokens=True,
        hidden_size=768,
    ):
        super(SentenceEncoderPooling, self).__init__()
        self.do_cls_tokens = do_cls_tokens
        self.do_max_tokens = do_max_tokens
        self.do_sqrt_tokens = do_sqrt_tokens
        self.do_mean_tokens = do_mean_tokens
        self.hidden_size = hidden_size

    @property
    def multiplier(self) -> int:
        return sum((self.do_cls_tokens, self.do_max_tokens,
                    self.do_sqrt_tokens, self.do_mean_tokens))

    @property
    def output_dim(self) -> int:
        return self.multiplier * self.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output: List[torch.Tensor] = []
        token_embeddings = hidden_states[0]
        extended_attn = attention_mask.unsqueeze(-1) \
            .expand(token_embeddings.size()).float()

        if self.do_cls_tokens:
            cls_tokens = token_embeddings[:, 0, :]
            output.append(cls_tokens)

        if self.do_max_tokens:
            token_embeddings[extended_attn == 0] = -1e9
            max_tokens = torch.max(token_embeddings, 1)[0]
            output.append(max_tokens)

        if self.do_mean_tokens or self.do_sqrt_tokens:
            embedding_sum = torch.sum(token_embeddings * extended_attn, 1)
            attn_mask_sum = torch.clamp(extended_attn.sum(1), min=1e-9)
            if self.do_mean_tokens:
                mean_tokens = embedding_sum / attn_mask_sum
                output.append(mean_tokens)
            else:
                sqrt_tokens = embedding_sum / torch.sqrt(attn_mask_sum)
                output.append(sqrt_tokens)

        sentence_embeddings = torch.cat(output, dim=1)
        return sentence_embeddings


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        transformer: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        do_cls_tokens: bool = False,
        do_max_tokens: bool = False,
        do_sqrt_tokens: bool = False,
        do_mean_tokens: bool = True,
        device: str = 'cpu',
    ):
        super(SentenceEncoder, self).__init__()
        self.device = torch.device(device)
        self.transformer = transformer.to(self.device)
        self.pooling = SentenceEncoderPooling(
            do_cls_tokens=do_cls_tokens,
            do_max_tokens=do_max_tokens,
            do_sqrt_tokens=do_sqrt_tokens,
            do_mean_tokens=do_mean_tokens,
            hidden_size=transformer.config.hidden_size,
        )
        self.tokenizer = tokenizer

    def encode(
        self,
        sentences: Union[List[str], Sampler, SentenceStore],
        max_length: int = 256,
        batch_size: int = 8,
        show_progress: bool = True,
        return_tensors: str = 'np',
    ) -> Union[np.ndarray, torch.Tensor]:
        lengths: np.ndarray
        if isinstance(sentences, (Sampler, SentenceStore)):
            lengths = np.argsort([x.seqlen for x in sentences._meta])
        else:
            lengths = np.argsort([len(x) for x in sentences])

        maxsize = lengths.size
        batches = range(0, maxsize, batch_size)
        if show_progress:
            batches = tqdm(batches, desc='batch')
        tokenize = self.tokenizer.tokenize
        max_seq_length = max_length if max_length is not None \
            else self.tokenizer.max_seq_length
        if max_seq_length >= 512:
            max_seq_length = 500

        self.transformer.eval()
        embeddings = []
        for i in batches:
            maxlen = 0
            splits: List[List[str]] = []
            for j in lengths[i: min(i + batch_size, maxsize)]:
                string = sentences[j.item()]
                tokens = tokenize(string)
                maxlen = max(maxlen, len(tokens))
                splits.append(tokens)

            max_length = min(maxlen, max_seq_length) + 2
            batch_inputs = self.tokenizer(
                text=splits,
                padding=True,
                truncation=True,
                max_length=max_length,
                is_split_into_words=True,
                return_tensors='pt',
            )
            batch_inputs = batch_inputs.to(self.device)
            with torch.no_grad():
                output = self.forward(**batch_inputs)
                sent_embed = output['sentence_embeddings']
                embeddings.extend(sent_embed.to('cpu').numpy())

        embeddings = [embeddings[i] for i in np.argsort(lengths)]
        embeddings = np.array(embeddings)
        if return_tensors == 'np':
            return embeddings
        else:
            return torch.from_numpy(embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:

        if output_hidden_states is None:
            if self.transformer.config.output_hidden_states:
                output_hidden_states = True

        hidden_states = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        output = {}
        if output_hidden_states:
            layer_id = 1 if len(hidden_states) < 3 else 2
            output["layer_embeddings"] = hidden_states[layer_id]

        sentence_embeddings = self.pooling(hidden_states, attention_mask)

        output.update({"token_embeddings": hidden_states[0],
                       "sentence_embeddings": sentence_embeddings})
        return output

    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path,
        do_cls_tokens=False,
        do_max_tokens=False,
        do_sqrt_tokens=False,
        do_mean_tokens=True,
        device='cpu',
        use_fast=True,
        *model_args,
        **model_kwargs,
    ) -> 'SentenceEncoder':

        sentence_encoder = SentenceEncoder(
            transformer=AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **model_kwargs
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                use_fast=use_fast
            ),
            do_cls_tokens=do_cls_tokens,
            do_max_tokens=do_max_tokens,
            do_sqrt_tokens=do_sqrt_tokens,
            do_mean_tokens=do_mean_tokens,
            device=device,
        )

        return sentence_encoder
