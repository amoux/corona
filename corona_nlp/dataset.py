
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer)

BERT_MODELS_UNCASED = {
    'squad': "bert-large-uncased-whole-word-masking-finetuned-squad",
    'scibert': "allenai/scibert_scivocab_uncased"
}


def split_dataset(dataset: List[Any],
                  subset: float = 0.5,
                  samples: int = None,
                  seed: int = 12345) -> Tuple[List[Any], List[Any]]:
    """Split an iterable dataset into a train and evaluation sets."""
    np.random.seed(seed)
    np.random.shuffle(dataset)
    maxlen = len(dataset)
    if not samples or samples > maxlen:
        samples = maxlen
    split = int(subset * samples)
    train_data = dataset[:split]
    eval_data = dataset[split:samples]
    return train_data, eval_data


class TextDataset(Dataset):

    def __init__(self, file_path: str, model="scibert", max_length=512, overwrite_cache=False):
        file_path = Path(file_path)
        assert file_path.is_file()

        self.model_type = BERT_MODELS_UNCASED[model]
        self.samples = []

        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        block_size = (tokenizer.max_len - tokenizer.max_len_single_sentence)
        max_length = max_length - block_size

        outdir = file_path.parent
        file = file_path.name.replace(file_path.suffix, "")
        name = self.model_type.replace("allenai/", "")
        cached_features = outdir.joinpath(
            f"{name}_cached_lm_{max_length}_{file}")

        if cached_features.exists() and not overwrite_cache:
            print(f"loading features from cashed files: {cached_features}")
            with cached_features.open("rb") as handle:
                self.samples = pickle.load(handle)
        else:
            print(f"creating features from papers file at: {outdir}")
            with file_path.open(encoding="utf-8") as file:
                texts = file.read()

            tokenized = tokenizer.tokenize(texts)
            sequences = tokenizer.convert_tokens_to_ids(tokenized)
            truncated_range = range(0, len(sequences)-max_length+1, max_length)
            for i in tqdm(truncated_range, desc="sequences", unit=""):
                inputs = sequences[i: i + max_length]
                tokens = tokenizer.build_inputs_with_special_tokens(inputs)
                self.samples.append(tokens)

            with cached_features.open("wb") as handle:
                pickle.dump(self.samples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
                del texts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return torch.tensor(self.samples[item], dtype=torch.long)

    def load_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_type)


class LineByLineTextDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        model: str = "scibert",
        max_length: int = 512,
        add_special_tokens: bool = True,
        overwrite_cache=False,
    ):
        file_path = Path(file_path)
        assert file_path.is_file()

        self.model_type = BERT_MODELS_UNCASED[model]
        self.samples = []

        outdir = file_path.parent
        file = file_path.name.replace(file_path.suffix, "")
        name = self.model_type.replace("allenai/", "")
        cached_features = outdir.joinpath(f"{model}_cashed_lm_{max_length}")

        if cached_features.exists() and not overwrite_cache:
            print(f"loading features from cashed file: {cached_features}")
            with cached_features.open("rb") as handle:
                self.samples = pickle.load(handle)
        else:
            print(f"creating features from cashed files: {outdir}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            with file_path.open("r", encoding="utf-8") as file:
                for line in tqdm(file, desc="tokenized-lines"):
                    strings = line.splitlines()
                    for string in strings:
                        token_ids = tokenizer.encode(
                            text=string,
                            max_length=max_length,
                            add_special_tokens=add_special_tokens,
                        )
                    self.samples.append(token_ids)

            with cached_features.open("wb") as handle:
                pickle.dump(self.samples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return torch.tensor(self.samples[item], dtype=torch.long)

    def load_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_type)

    def load_model(self) -> AutoModel:
        return AutoModel.from_pretrained(self.model_type)


# class LineByLineTextDataset(Dataset):

#     def __init__(self, file_path: str, model="scibert", max_length=512):
#         file_path = Path(file_path)
#         assert file_path.is_file()

#         self.model_type = BERT_MODELS_UNCASED[model]
#         self.samples = []

#         with file_path.open("r", encoding="utf-8") as file:
#             for line in tqdm(file, desc="lines", unit=""):
#                 strings = line.splitlines()
#                 for string in strings:
#                     self.samples.append(string)

#         tokenizer = AutoTokenizer.from_pretrained(self.model_type)
#         self.samples = tokenizer.batch_encode_plus(self.samples,
#                                                    add_special_tokens=True,
#                                                    max_length=max_length)["input_ids"]

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, item):
#         return torch.tensor(self.samples[item], dtype=torch.long)

#     def load_tokenizer(self) -> AutoTokenizer:
#         return AutoTokenizer.from_pretrained(self.model_type)
