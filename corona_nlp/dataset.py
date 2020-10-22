import concurrent.futures
from multiprocessing import cpu_count
from typing import Iterable, Iterator, List, Tuple, Union

from tqdm.auto import tqdm

from .core import Papers, Sentences, merge_papers
from .indexing import PaperIndexer
from .tokenizer import SpacySentenceTokenizer
from .utils import clean_tokenization, normalize_whitespace


class CORD19Dataset(PaperIndexer):
    all_text_keys = ["abstract", "body_text"]

    def __init__(
            self,
            source: Union[str, List[str]],
            text_key: str = "body_text",
            index_start: int = 1,
            sort_first: bool = False,
            nlp_model: str = "en_core_web_sm",
            **kwargs,
    ):
        """CORD19-Dataset object.

        :param text_key: A string key to extract texts from each file
            in the dataset following: `body_text, abstract`.
        """
        super(CORD19Dataset, self).__init__(source, index_start, sort_first)
        self.text_key = text_key
        self.sentencizer = SpacySentenceTokenizer(nlp_model, **kwargs)
        self.nlp = self.sentencizer.nlp
        self.doc_suffix = " "

    @property
    def all_dataset_keys(self):
        return self.load_paper(self._splits[0]).keys()

    @staticmethod
    def prep(text: str) -> str:
        text = normalize_whitespace(text)
        text = clean_tokenization(text)
        return text

    def doc(self, index=None, paper_id=None) -> str:
        """Return the full text document for a single ID.

        * This method is intended to be used for a single doc/id, for
          more than one; use `self.docs()` or `self.lines()` methods.

        :param index: An integer or a list of integers.
        :param paper_id: A string or list of string sequences.
        """
        if index is not None:
            index = [index] if isinstance(index, int) else index
        elif paper_id is not None:
            index = [self[paper_id]] if isinstance(paper_id, str) else paper_id
        assert isinstance(index, list), f'Expected a `List`, not {type(index)}'
        return next(self.docs(index))

    def title(self, index: int = None, paper_id: str = None) -> str:
        return self.load_paper(index, paper_id)["metadata"]["title"]

    def titles(self, indices: List[int] = None, paper_ids: List[str] = None,
               ) -> Iterator[str]:
        for paper in self.load_papers(indices, paper_ids):
            yield paper["metadata"]["title"]

    def docs(self, indices: List[int] = None, paper_ids: List[str] = None,
             ) -> Iterator[str]:
        """Chain a list of ids and return a full doc per id, until exhausted."""
        suffix = self.doc_suffix.join
        for paper in self.load_papers(indices, paper_ids):
            lines = []
            for line in paper[self.text_key]:
                text = self.prep(line["text"])
                lines.append(text)
            yield suffix(lines)

    def lines(self, indices: List[int] = None, paper_ids: List[str] = None,
              ) -> Iterator[str]:
        """Chain a list of ids and return texts line-by-line, until exhausted."""
        for paper in self.load_papers(indices, paper_ids):
            for line in paper[self.text_key]:
                text = self.prep(line["text"])
                yield text

    def build(self, indices: List[int], minlen: int = 15) -> Papers:
        """Return an instance of papers with texts transformed to sentences."""
        idx = Sentences(indices)
        docs = self.docs(indices, paper_ids=None)
        tokenize = self.sentencizer
        is_sentence = self.sentencizer.is_sentence

        papers = idx.init_cluster()
        for pid in papers:
            for sent in tokenize(next(docs)):
                if sent.text in papers[pid]:
                    continue
                seqlen = len(sent)
                if seqlen < minlen:
                    continue
                if not is_sentence(sent):
                    continue
                idx.seqlen += seqlen
                idx.counts += 1
                idx.maxlen = max(idx.maxlen, seqlen)
                papers[pid].append(sent.text)

        return Papers(idx, cluster=papers)

    def batch(self, indices: List[int], minlen=15, workers: int = None) -> Papers:
        maxsize = len(indices)
        workers = cpu_count() if workers is None else workers

        jobs = []
        for i in range(0, maxsize, workers):
            tasks = indices[i: min(i + workers, maxsize)]
            jobs.append(tasks)

        with tqdm(total=maxsize, desc="papers") as pbar:
            batch_: List[Papers] = []
            with concurrent.futures.ThreadPoolExecutor(workers) as pool:
                future_to_ids = {
                    pool.submit(self.build, pids, minlen): pids for pids in jobs
                }
                for future in concurrent.futures.as_completed(future_to_ids):
                    ids = future_to_ids[future]
                    try:
                        papers = future.result()
                    except Exception as e:
                        print(f"{ids} generated an exception: {e}")
                        break
                    else:
                        batch_.append(papers)
                        pbar.update(len(ids))

        papers = merge_papers(batch_)
        papers.attach_init_args(self)
        return papers
