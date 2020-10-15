import concurrent.futures
import random
from multiprocessing import cpu_count
from typing import Callable, Iterator, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from .datatypes import Papers, Sentences, merge_papers
from .indexing import PaperIndexer
from .tokenizer import SpacySentenceTokenizer
from .utils import clean_tokenization, normalize_whitespace


class CORD19Dataset(PaperIndexer):
    def __init__(
            self,
            source: Union[str, List[str]],
            text_keys: Tuple[str, ...] = ("abstract", "body_text",),
            index_start: int = 1,
            sort_first: bool = False,
            nlp_model: str = "en_core_web_sm",
            sentence_tokenizer=None,
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

    def sample(
        self, k: Optional[int] = None, s: Optional[int] = None, seed=None,
    ) -> Union[List[int], None]:
        """Return a sample (all|random k) or split of paper ID's.

        :param k: number of ids to return from all samples, if `k=-1` then all
            ids are returned sorted. Otherwise, if `k < max ids` -> shuffled.
        :param s: return a split of all ids @ `s` e.g., if s=1 then all ids@1.
        """
        if k is not None:
            random.seed(seed)
            ids = list(self.index_paper.keys())
            if k == -1:
                return ids
            assert k <= self.num_papers
            return random.sample(ids, k=k)

        if s is not None:
            splits = self._splits
            assert s <= len(splits), f'Expected `s` between: [0,{len(splits)}]'
            if s == 0:
                return list(range(self.index_start, splits[s] + 1))
            if s > 0:
                return list(range(splits[s - 1] + 1, splits[s] + 1))

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

    def build(self, indices: List[int], minlen: int = 15) -> Papers:
        """Return an instance of papers with texts transformed to sentences."""
        index = Sentences(indices)
        cluster = index.init_cluster()
        docs = self.docs(indices)

        tokenize = self.sentence_tokenizer.tokenize
        is_sentence = self.sentence_tokenizer.is_sentence

        for paper in cluster:
            for sent in tokenize(next(docs)):
                length = len(sent)  # Token/word level length (not chars).
                if length <= minlen:
                    continue
                if not is_sentence(sent):
                    continue
                string = normalize_whitespace(sent.text)
                string = clean_tokenization(string)
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
                        break
                    else:
                        batch_.append(papers)
                        pbar.update(len(ids))

        papers = merge_papers(batch_)
        papers.attach_init_args(self)
        return papers

    def __repr__(self):
        multi_src = "[\n  {},\n]"  # Template for a list of sources.
        src = self.source_name if isinstance(self.source_name, str) \
            else multi_src.format(', '.join(self.source_name))
        return "{}(papers: {}, files_sorted: {}, source: {})".format(
            self.__class__.__name__, self.num_papers, self.is_files_sorted, src,
        )
