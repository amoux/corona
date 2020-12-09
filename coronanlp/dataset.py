import concurrent.futures
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import torch
from filelock import FileLock
from torch.utils.data import Dataset
from tqdm.auto import tqdm  # type: ignore
from transformers import PreTrainedTokenizer

from .core import Sampler, SentenceStore
from .indexing import PaperIndexer
from .tokenizer import SpacySentenceTokenizer
from .utils import DataIO, clean_tokenization, load_store, normalize_whitespace

Pid = int
Uid = str


class CORD19(PaperIndexer):
    all_text_keys = ["abstract", "body_text"]

    def __init__(
            self,
            source: Union[str, List[str]],
            text_key: str = "body_text",
            index_start: int = 1,
            sort_first: bool = False,
            nlp_model: str = "en_core_sci_sm",
            **kwargs,
    ):
        """CORD19 represents a data reader/loader (IO) over the source files.

        :param text_key: A string key to extract texts from each file
            in the dataset following: `body_text, abstract`.
        """
        super(CORD19, self).__init__(source, index_start, sort_first)
        self.text_key = text_key
        self.sentencizer = SpacySentenceTokenizer(nlp_model, **kwargs)
        self.nlp = self.sentencizer.nlp
        self.doc_suffix = " "

    @property
    def all_dataset_keys(self):
        return self.load_paper(self.splits[0]).keys()

    @staticmethod
    def prep(text: str) -> str:
        text = normalize_whitespace(text)
        text = clean_tokenization(text)
        return text

    def doc(self, pid=None, uid=None) -> str:
        """Return the full text document for a single ID.

        * This method is intended to be used for a single doc/id, for
          more than one; use `self.docs()` or `self.lines()` methods.

        :param pid: An integer or a list of integers.
        :param uid: A string or list of string sequences.
        """
        if pid is not None:
            pid = [pid] if isinstance(pid, int) else pid
        elif uid is not None:
            pid = [self[uid]] if isinstance(uid, str) else uid
        assert isinstance(pid, list), f'Expected a `List`, not {type(pid)}'
        return next(self.docs(pid))

    def title(self, pid: Pid = None, uid: Uid = None) -> str:
        return self.load_paper(pid, uid)["metadata"]["title"]

    def titles(self, pids: List[Pid] = None, uids: List[Uid] = None,
               ) -> Iterator[str]:
        for paper in self.load_papers(pids, uids):
            yield paper["metadata"]["title"]

    def docs(self, pids: List[Pid] = None, uids: List[Uid] = None,
             ) -> Iterator[str]:
        """Chain a list of pids and return a full doc per id, until exhausted."""
        suffix = self.doc_suffix.join
        for paper in self.load_papers(pids, uids):
            lines = []
            for line in paper[self.text_key]:
                text = self.prep(line["text"])
                lines.append(text)
            yield suffix(lines)

    def lines(self, pids: List[Pid] = None, uids: List[Uid] = None,
              ) -> Iterator[str]:
        """Chain a list of pids and return texts line-by-line, until exhausted."""
        for paper in self.load_papers(pids, uids):
            for line in paper[self.text_key]:
                text = self.prep(line["text"])
                yield text

    def build(self, pids: List[Pid], minlen: int = 15) -> Sampler:
        """Return a Sampler object of papers with texts transformed to sentences."""
        X = Sampler(pids)
        docs = self.docs(pids, uids=None)
        tokenize = self.sentencizer
        is_sentence = self.sentencizer.is_sentence

        store = X.init()
        for pid in store:
            for sent in tokenize(next(docs)):
                if sent.text in store[pid]:
                    continue
                seqlen = len(sent)  # number of tokens.
                if seqlen < minlen:
                    continue
                if not is_sentence(sent):
                    continue
                X.include(pid, seqlen, sent.text)

        return X

    def batch(self, pids: List[Pid], minlen=15, workers=4, build=None) -> SentenceStore:
        maxsize = len(pids)
        workers = cpu_count() if workers is None else workers
        build = self.build if build is None else build

        jobs = []
        for i in range(0, maxsize, workers):
            tasks = pids[i: min(i + workers, maxsize)]
            jobs.append(tasks)

        with tqdm(total=maxsize, desc="files/papers") as pbar:
            batches: List[Sampler] = []
            with concurrent.futures.ThreadPoolExecutor(workers) as pool:
                future_to_pids = {
                    pool.submit(build, job, minlen): job for job in jobs
                }
                for future in concurrent.futures.as_completed(future_to_pids):
                    pids = future_to_pids[future]
                    try:
                        sampler = future.result()
                    except Exception as e:
                        print(f"{pids} generated an exception: {e}")
                        break
                    else:
                        batches.append(sampler)
                        pbar.update(len(pids))

        sentence_store = SentenceStore.from_sampler(batches)
        sentence_store.attach_init_args(self)
        return sentence_store

    def __call__(self, id: Union[Pid, Uid, List[Pid], List[Uid]],
                 ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """Retrive a paper from disk and return a dict obj of (key,value) pairs.

        * Example usage:
        ```python
        cord19(100).keys()
        #  dict_keys(['paper_id', 'metadata', 'abstract', ...,])
        cord19([100, 200, 300])[0]['paper_id']
        cord19('0031e47b76374e05a18c266bd1a1140e5eacb54f')
        cord19(['0031e47b76374e05a18c266bd1a1140e5eacb54f'])
        ```
        """
        if isinstance(id, (Pid, Uid)):
            args = (id, None) if isinstance(id, Pid) else (None, id)
            return self.load_paper(*args)
        if isinstance(id, list) and isinstance(id[0], (Pid, Uid)):
            args = (id, None) if isinstance(id[0], Pid) else (None, id)
            return self.load_papers(*args)
        return None

    @staticmethod
    def from_store(name: str) -> 'CORD19':
        return load_store('cord', store_name=name)

    @staticmethod
    def from_init_args(dict_obj) -> 'CORD19':
        return CORD19(**dict_obj)


def _encode(splits, sent_store, tokenizer, block_size, batch_size):
    lengths = np.argsort([sent_store._meta[i].strlen for i in splits])
    max_len = lengths.size
    batches = range(0, max_len, batch_size)
    sentences = [sent_store.string(x) for x in splits]
    encoded = []
    for i in batches:
        lines = []
        for j in lengths[i: min(i + batch_size, max_len)]:
            string = sentences[j]
            lines.append(string)
        batch = tokenizer(lines, truncation=True,
                          max_length=block_size,
                          add_special_tokens=True)
        encoded.extend(batch['input_ids'])

    encoded = [encoded[i] for i in np.argsort(lengths)]
    assert len(encoded) == max_len
    return encoded


def _cfg_cache(block_size, tokenizer, file_name=None, cache_dir=None) -> Tuple[Path, str]:
    if file_name is None:
        raise ValueError("Setting do_cache requires the a file name.")
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
    file_name = file_name if file_name.endswith('.txt') \
        else f'{file_name}.txt'
    cache_file_name = 'cached_lm_{}_{}_{}'.format(
        tokenizer.__class__.__name__, block_size, file_name)
    cache_file_path = Path(cache_file_name) if cache_dir is None \
        else cache_dir.joinpath(cache_file_name)
    cache_lock_path = cache_file_path.absolute().as_posix() + ".lock"
    return cache_file_path, cache_lock_path


class SentenceDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        splits: Optional[List[int]] = None,
        sent_store: Optional[SentenceStore] = None,
        block_size: Optional[int] = None,
        batch_size: int = 8,
        do_cache: bool = False,
        overwrite: bool = False,
        file_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        if sent_store is not None:
            assert isinstance(sent_store, SentenceStore)
        self.encoded: List[torch.Tensor] = []
        self.max_len: int = 0

        if block_size is None:
            block_size = tokenizer.model_max_length
            block_size = block_size - \
                tokenizer.num_special_tokens_to_add(pair=False)

        if do_cache:
            file_path, lock_path = _cfg_cache(
                block_size, tokenizer, file_name, cache_dir)

            with FileLock(lock_path):
                if file_path.exists() and not overwrite:
                    start = time.time()
                    self.encoded = DataIO.load_data(file_path)
                    self.max_len = len(self.encoded)
                    print("Loading from cached file {} [took {:.3f} s]".format(
                        file_path.name, time.time() - start))
                else:
                    start = time.time()
                    self.encoded = _encode(
                        splits, sent_store, tokenizer, block_size, batch_size)
                    self.max_len = len(self.encoded)
                    DataIO.save_data(file_path, self.encoded)
                    print("Saving into cached file {} [took {:.3f} s]".format(
                        file_path.name, time.time() - start))
        else:
            self.encoded = _encode(
                splits, sent_store, tokenizer, block_size, batch_size)
            self.max_len = len(self.encoded)

    def __len__(self) -> int:
        return self.max_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return torch.tensor(self.encoded[idx], dtype=torch.long)
