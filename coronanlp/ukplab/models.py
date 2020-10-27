
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel


class Transformer(nn.Module):
    def __init__(self, model_name_or_path: str, **kwargs):
        super(Transformer, self).__init__()
        cfg = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=cfg)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def word_dim(self) -> int:
        return self.hidden_size

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = self.model(**inputs)
        token_embed = output[0]
        class_token = token_embed[:, 0, :]
        inputs.update(dict(token_embed=token_embed, class_token=class_token))
        if len(output) > 2:
            inputs.update(dict(layer_embed=output[2]))

        return inputs


class Pooling(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        do_class_token: bool = False,
        do_sqrt_tokens: bool = False,
        do_mean_tokens: bool = True,
    ) -> None:
        super(Pooling, self).__init__()
        self.hidden_size = hidden_size
        self.do_class_token = do_class_token
        self.do_sqrt_tokens = do_sqrt_tokens
        self.do_mean_tokens = do_mean_tokens
        multi = sum([do_class_token, do_sqrt_tokens, do_mean_tokens])
        self.out_features = multi * hidden_size

    @property
    def sent_dim(self) -> int:
        return self.out_features

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = []
        token_embed, class_token, attn_mask = (
            inputs['token_embed'],
            inputs['class_token'],
            inputs['attention_mask'],
        )
        if self.do_class_token:
            output.append(class_token)
        if self.do_mean_tokens or self.do_sqrt_tokens:
            ext_attn_mask = attn_mask.unsqueeze(-1).expand(token_embed.size())
            sum_embedding = torch.sum(token_embed * ext_attn_mask.float(), 1)
            sum_attn_mask = torch.clamp(ext_attn_mask.sum(1), min=1e-9)
            if self.do_mean_tokens:
                mean = sum_embedding / sum_attn_mask
                output.append(mean)
            if self.do_sqrt_tokens:
                mean_sqrt = sum_embedding / torch.sqrt(sum_attn_mask)
                output.append(mean_sqrt)

        output = torch.cat(output, dim=1)
        inputs.update({'sentence_embed': output})

        return inputs
