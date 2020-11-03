import concurrent.futures
from multiprocessing import cpu_count
from typing import Any, Dict, Iterator, List, Union

from tqdm.auto import tqdm  # type: ignore

from .core import Sampler, SentenceStore
from .indexing import PaperIndexer
from .tokenizer import SpacySentenceTokenizer
from .utils import clean_tokenization, normalize_whitespace

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

    def batch(self, pids: List[Pid], minlen=15, workers=None, build=None) -> SentenceStore:
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
