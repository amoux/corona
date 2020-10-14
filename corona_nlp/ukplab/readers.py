import csv
import gzip
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple, Union
from urllib import request
from zipfile import ZipFile

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

UKP_SERVER = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/"
DATA_FILES = ["AllNLI.zip", "stsbenchmark.zip"]


def clean(sequence: str) -> str:
    return sequence.strip().replace(' .', '.').replace(' ,', ',')


def download_nli_sources(out_dir: str = 'data'):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    for dataset in DATA_FILES:
        url = UKP_SERVER + dataset
        dataset_path = out_dir.joinpath(dataset)
        request.urlretrieve(url, dataset_path)
        with ZipFile(dataset_path) as zipfile:
            zipfile.extractall(out_dir)
        os.remove(dataset_path)


class Input(NamedTuple):
    idx: int
    label: int
    pair: List[str]
    token_ids: List[List[int]] = []

    @property
    def is_tokenized(self) -> bool:
        return 0 < len(self.token_ids)


def load_sts(fp: str, norm=True, score_col=4, s1_col=5, s2_col=6):
    with open(fp, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
        data: Dict[str, List[Any]] = dict(score=[], s1=[], s2=[])
        for row in reader:
            score = float(row[score_col])
            data['score'].append(score)
            data['s1'].append(row[s1_col])
            data['s2'].append(row[s2_col])

    if norm:
        scores = data['score']
        maxscore = max(scores)
        data['score'] = [k / maxscore for k in scores]

    return pd.DataFrame(data)


class NLIReader:
    labels = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }

    def __init__(self, out_dir: str, download: bool = False):
        if download:
            download_nli_sources(out_dir=out_dir)
        self.out_dir = Path(out_dir)
        self.filemap = dict(train=[], dev=[])
        for p in sorted(self.out_dir.iterdir()):
            endswith = p.name.endswith
            if not endswith('.gz'):
                continue
            if endswith('train.gz'):
                self.filemap['train'].append(p)
            else:
                self.filemap['dev'].append(p)

    @property
    def num_labels(self):
        return len(self.labels)

    def sample(self, train_or_dev: str = 'train') -> Iterable[Input]:
        it = [iter(gzip.open(fp, 'rt', encoding='utf-8').readlines())
              for fp in self.filemap[train_or_dev]]

        for i, (label, s1, s2) in enumerate(zip(*it)):
            label_id = self.labels[clean(label.lower())]
            sentence_pair = [clean(s1), clean(s2)]
            yield Input(i, label=label_id, pair=sentence_pair)

    def __call__(self, train_or_dev: str) -> Iterable[Input]:
        return self.sample(train_or_dev)


class SentenceDataset(Dataset):
    def __init__(
        self,
        sample: Union[Iterable[Input], List[Input]],
        tokenizer: Union[str, AutoTokenizer],
    ) -> None:
        self.sample = sample
        self.tokenizer = tokenizer
        self.num_inputs: int
        if not isinstance(sample, list):
            sample = list(sample)
            self.sample = sample
        assert isinstance(self.sample[0], Input)
        if not isinstance(tokenizer, AutoTokenizer) \
                and isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.num_inputs = len(self.sample)

    def encode(self, text_pair: List[str]) -> List[List[int]]:
        token_ids = []
        for text in text_pair:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids.append(ids)

        return token_ids

    def __getitem__(self, item: int) -> Tuple[List[List[int]], torch.Tensor]:
        input = self.sample[item]
        label = torch.tensor(input.label, dtype=torch.long)
        token_ids = input.token_ids
        if not input.is_tokenized:
            token_ids = self.encode(input.pair)

        return token_ids, label

    def __len__(self):
        return self.num_inputs
