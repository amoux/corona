from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class SoftmaxLoss(nn.Module):
    def __init__(
        self,
        model: nn.Sequential,
        in_features: int,
        out_features: int,
        do_dot_prod: bool = False,
        do_cat_pair: bool = True,
        do_abs_diff: bool = True,
    ) -> None:
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.do_dot_prod = do_dot_prod
        self.do_cat_pair = do_cat_pair
        self.do_abs_diff = do_abs_diff
        self.criterion = sum([do_dot_prod, do_cat_pair, do_abs_diff])
        in_features = self.criterion * in_features
        self.classifier = nn.Linear(in_features, out_features)
        self.loss = nn.CrossEntropyLoss()

    @property
    def in_features(self) -> int:
        return self.classifier.in_features

    @property
    def out_features(self) -> int:
        return self.classifier.out_features

    def forward(
        self,
        input_batch: List[Dict[str, Tensor]],
        labels: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[List[Tensor], Tensor]]:
        sentence_pair = []
        for x in input_batch:
            sentence_pair.append(self.model(x)['sentence_embed'])

        features = []
        sent_a, sent_b = sentence_pair
        if self.do_cat_pair:
            features.append([sent_a, sent_b])
        if self.do_abs_diff:
            features.append(torch.abs(sent_a - sent_b))
        if self.do_dot_prod:
            features.append(sent_a * sent_b)

        features = torch.cat(features, dim=1)
        output = self.classifier(features)

        if labels is not None:
            losses = self.loss(output, labels.view(-1))
            return losses
        return sentence_pair, output
