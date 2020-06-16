import concurrent.futures
import random
from multiprocessing import cpu_count
from typing import Callable, Iterator, List, Tuple, Union

from tqdm.auto import tqdm

from .datatypes import Papers, Sentences, merge_papers
from .indexing import PaperIndexer
from .tokenizer import SpacySentenceTokenizer
from .utils import clean_tokenization, normalize_whitespace


class CORD19Dataset(PaperIndexer):
    def __init__(
            self,
            source: Union[str, List[str]],
            text_keys: Tuple[str] = ("abstract", "body_text",),
            index_start: int = 1,
            sort_first: bool = False,
            nlp_model: str = "en_core_web_sm",
            sentence_tokenizer: Callable = None,
    ):
        super(CORD19Dataset, self).__init__(source, index_start, sort_first)
        self.text_keys = text_keys
        self.sentence_tokenizer = sentence_tokenizer
        if sentence_tokenizer is not None:
            if not hasattr(sentence_tokenizer, 'tokenize'):
                raise AttributeError(f'Callable[{sentence_tokenizer.__name__}]'
                                     ' missing ``self.tokenize()`` attribute.')
        else:
            self.sentence_tokenizer = SpacySentenceTokenizer(nlp_model)

    def sample(self, k: int = None, seed: int = None) -> List[int]:
        """Return all or k iterable of paper-id to index mappings.

        `k`: A sample from all available papers use `k=-1`. Otherwise, pass
            `k=n` number of indices to load from the available dataset files.
        """
        random.seed(seed)
        indices = list(self.index_paper.keys())
        if k == -1:
            return indices
        assert k <= self.num_papers
        return random.sample(indices, k=k)

    def title(self, index: int = None, paper_id: str = None) -> str:
        return self.load_paper(index, paper_id)["metadata"]["title"]

    def titles(self, indices: List[int] = None,
               paper_ids: List[str] = None) -> Iterator:
        for paper in self.load_papers(indices, paper_ids):
            yield paper["metadata"]["title"]

    def docs(self, indices: List[int] = None,
             paper_ids: List[str] = None, suffix="\n") -> Iterator:
        for paper in self.load_papers(indices, paper_ids):
            doc = []
            for key in self.text_keys:
                for line in paper[key]:
                    doc.append(line["text"])
            yield suffix.join(doc)

    def lines(self, indices: List[int] = None,
              paper_ids: List[str] = None) -> Iterator:
        for paper in self.load_papers(indices, paper_ids):
            for key in self.text_keys:
                for line in paper[key]:
                    yield line["text"]

    def build(self, indices: List[int], minlen: int = 20) -> Papers:
        """Return an instance of papers with texts transformed to sentences."""
        index = Sentences(indices)
        cluster = index.init_cluster()
        docs = self.docs(indices)

        for paper in cluster:
            for line in self.sentence_tokenizer.tokenize(next(docs)):
                string = normalize_whitespace(line.text)
                string = clean_tokenization(string)
                length = len(string)
                if length <= minlen:
                    continue
                if string not in cluster[paper]:
                    index.strlen += length
                    index.counts += 1
                    index.maxlen = max(index.maxlen, length)
                    cluster[paper].append(string)

        return Papers(index, cluster=cluster)

    def batch(self, indices: List[int], minlen=20, workers=None) -> Papers:
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
                    pool.submit(self.build, job, minlen): job for job in jobs
                }
                for future in concurrent.futures.as_completed(future_to_ids):
                    ids = future_to_ids[future]
                    try:
                        papers = future.result()
                    except Exception as e:
                        print(f"{ids} generated an exception: {e}")
                        raise
                    else:
                        batch_.append(papers)
                        pbar.update(len(ids))

        return merge_papers(batch_)

    def __repr__(self):
        return "CORD19Dataset(papers={}, files_sorted={}, source={})".format(
            self.num_papers, self.is_files_sorted, self.source_name)
