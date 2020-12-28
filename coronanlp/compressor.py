from typing import Dict, Optional, Union

import numpy as np  # type: ignore
import torch
from sklearn import cluster, mixture  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from transformers import PreTrainedModel  # type: ignore


class ClusterBase:
    def __init__(self, use_first=True, ratio=0.2, n_pca=None, state=12345):
        self.use_first = use_first
        self.ratio = ratio
        self.state = state
        self.n_pca = n_pca

    def search(self, x: np.ndarray, nlist: int, state: int) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        topk: Optional[int] = None,
    ) -> torch.Tensor:

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.numpy()
        if self.n_pca is not None:
            inputs = PCA(self.n_pca).fit_transform(inputs)

        nlist = 0
        if topk is not None:
            nlist = min(topk, len(inputs))
        else:
            nlist = max(int(len(inputs) * self.ratio), 1)

        centroids = self.search(inputs, nlist, self.state)

        centroid_min = 1e10
        cur_arg = -1
        args = {}
        used_idx = []

        for j, c in enumerate(centroids):
            for i, x in enumerate(inputs):
                dist = np.linalg.norm(x - c)
                if dist < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min = dist

            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e10
            cur_arg = -1

        hidden_args = sorted(args.values())
        if self.use_first and hidden_args[0] != 0:
            hidden_args.insert(0, 0)

        return torch.tensor(hidden_args)


class KMeansCluster(ClusterBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search(self, x: np.ndarray, nlist: int, state: int) -> np.ndarray:
        x = cluster.KMeans(nlist, random_state=state).fit(x)
        x = x.cluster_centers_
        return x


class GaussianMixtureCluster(ClusterBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search(self, x: np.ndarray, nlist: int, state: int) -> np.ndarray:
        x = mixture.GaussianMixture(nlist, random_state=state).fit(x)
        x = x.means_
        return x


class Compressor(torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        cluster: str = 'kmeans',
        pooling: str = 'mean',
        hidden_layer: int = -2,
        **kwargs
    ):
        super(Compressor, self).__init__()
        self.model = model
        self.pooling = pooling
        self.hidden_layer = hidden_layer
        self.cluster = KMeansCluster(**kwargs) \
            if cluster == 'kmeans' else GaussianMixtureCluster(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        topk: Optional[int] = None,
        return_clustered_ids: bool = False,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        _, hidden_states = outputs[-2:]
        last_hidden = hidden_states[self.hidden_layer]

        hidden = None
        if self.pooling == 'max':
            hidden = last_hidden.max(dim=1)[0]
        elif self.pooling == 'median':
            hidden = last_hidden.median(dim=1)[0]
        else:
            hidden = last_hidden.mean(dim=1)

        embeddings = hidden.squeeze(0).data.cpu()
        topk_values = self.cluster(embeddings, topk=topk)

        output = {'topk': topk_values,
                  'embeddings': embeddings[topk_values]}
        if return_clustered_ids:
            output['input_ids'] = input_ids[topk_values]
        return output
