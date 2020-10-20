import concurrent.futures
from multiprocessing import cpu_count
from typing import Iterable, Iterator, List, Tuple, Union

from tqdm.auto import tqdm

from .core import Papers, Sentences, merge_papers
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
            **kwargs,
    ):
        super(CORD19Dataset, self).__init__(source, index_start, sort_first)
        self.text_keys = text_keys
        self.sentencizer = SpacySentenceTokenizer(nlp_model, **kwargs)

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

    def titles(self, indices: List[int] = None,
               paper_ids: List[str] = None) -> Iterator[str]:
        for paper in self.load_papers(indices, paper_ids):
            yield paper["metadata"]["title"]

    def docs(self, indices: List[int] = None,
             paper_ids: List[str] = None, suffix="\n") -> Iterator[str]:
        for paper in self.load_papers(indices, paper_ids):
            doc = []
            for key in self.text_keys:
                for line in paper[key]:
                    doc.append(line["text"])
            yield suffix.join(doc)

    def lines(self, indices: List[int] = None,
              paper_ids: List[str] = None) -> Iterator[str]:
        for paper in self.load_papers(indices, paper_ids):
            for key in self.text_keys:
                for line in paper[key]:
                    yield line["text"]

    def build(self, indices: List[int], minlen: int = 15) -> Papers:
        """Return an instance of papers with texts transformed to sentences."""
        idx = Sentences(indices)
        docs = self.docs(indices)
        tokenize = self.sentencizer
        is_sentence = self.sentencizer.is_sentence

        def pipeline() -> Iterable:
            text = next(docs)
            text = normalize_whitespace(text)
            text = clean_tokenization(text)
            sent_spans = tokenize(text)
            return sent_spans

        papers = idx.init_cluster()
        for pid in papers:
            for sent in pipeline():
                if sent.text in papers[pid]:
                    continue
                length = len(sent)
                if length < minlen:
                    continue
                if not is_sentence(sent):
                    continue
                idx.seqlen += length
                idx.counts += 1
                idx.maxlen = max(idx.maxlen, length)
                papers[pid].append(sent.text)

        return Papers(idx, cluster=papers)

    def batch(self, indices: List[int], minlen=15, workers=None) -> Papers:
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
