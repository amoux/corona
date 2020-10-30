import concurrent.futures
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from tqdm.auto import tqdm  # type: ignore

from .core import Sampler, SentenceStore
from .indexing import PaperIndexer
from .parser import parse
from .tokenizer import SpacySentenceTokenizer
from .utils import clean_tokenization, normalize_whitespace

Pid = int
Uid = str

WIKI_LIKE_TEMPLATE = {
    'body': "\n {} \n",
    'header':  "\n = = {} = = \n",
    'section': "\n = = = {} = = = \n\n {}\n",
}


def cache_for_tokenizer_vocab(
        outdir: str,
        papers: SentenceStore,
        pids: Optional[List[Pid]] = None,
        suffix: str = '\n') -> List[str]:
    paths = []
    if pids is None:
        pids = papers.indices
    outdir = Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    for pid in tqdm(pids, desc='train-files'):
        fp = outdir.joinpath(f'{pid}.txt')
        with fp.open('w', encoding='utf-8') as f:
            for sent in papers.sents(pid):
                f.write(f'{sent}{suffix}')
        paths.append(fp.as_posix())
    return paths


def cache_for_language_modeling(
        outdir: str,
        papers: SentenceStore,
        train_ids: List[Pid],
        test_ids: Optional[List[Pid]] = None,
        suffix: str = '\n') -> Union[Tuple[Path, Path], Path, None]:
    def is_list_of_ids(obj): return (
        isinstance(obj, list) and isinstance(obj[0], Pid))

    def cache(fp, iterable):
        with fp.open('w', encoding='utf-8') as f:
            for sent in iterable:
                f.write(f'{sent}{suffix}')

    outdir = Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    train_fp, test_fp = None, None

    if is_list_of_ids(train_ids):
        train_fp = outdir.joinpath('train.txt')
        train_it = papers.index_select(train_ids)
        cache(train_fp, train_it)

    if test_ids is not None and is_list_of_ids(test_ids):
        test_fp = outdir.joinpath('test.txt')
        test_it = papers.index_select(test_ids)
        cache(test_fp, test_it)

    if all((train_fp, test_fp)):
        return train_fp, test_fp
    return train_fp


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
        papers = X.init()
        for pid in papers:
            for sent in tokenize(next(docs)):
                if sent.text in papers[pid]:
                    continue
                seqlen = len(sent)
                if seqlen < minlen:
                    continue
                if not is_sentence(sent):
                    continue
                text = sent.text
                args = (len(papers[pid]), len(text), seqlen)
                X.addmeta(pid, *args)
                papers[pid].append(text)
                X.maxlen = max(X.maxlen, seqlen)
                X.seqlen += seqlen
                X.counts += 1
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


def build_wiki_like_dataset(sample: List[Pid],
                            cord19: CORD19,
                            fn: str = 'train.txt',
                            outdir: str = 'data',
                            wiki_template: Optional[Dict[str, str]] = None) -> None:
    """Build a Wiki like dataset for language modeling, e.g., GTP/GPT2.

    NOTE: This function should be used for fine-tuning and not for training
    a new tokenizer. New tokenizer require file splits not single files.

    - Removes (The following tags are removed, since generative models like
    GPT2 will << generate >> similar for every output, which we dont want 🤗):
      - `cite-spans`, e.g., `"[1]"`
      - `ref-spans`, e.g., `"Figure 3C"`

    Example usage:
    ```python
    sample = cord19.sample(-1)
    train_ids, test_ids = coronanlp.split_dataset(sample, subset=0.9)
    build_wiki_like_dataset(train_ids, cord19, fn='train.txt')
    build_wiki_like_dataset(test_ids, cord19, fn='test.txt')
    ```
    """
    if wiki_template is None:
        wiki_template = WIKI_LIKE_TEMPLATE
    BODY = wiki_template['body']
    HEADER = wiki_template['header']
    SECTION_BODY = wiki_template['section']

    outdir = Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    fp = outdir.joinpath(fn)
    with fp.open('w', encoding='utf-8') as file:
        done = 0
        size = len(sample)
        pids = iter(sample)
        while done < size:
            pid = next(pids)
            paper = parse(pid, cord19.load_paper(pid))
            title = paper.title
            if title:
                header = HEADER.format(title)
                file.write(header)
            for body in paper:
                text = body.text
                for cite in body.cite_spans:
                    cite_text = cite.text
                    if cite_text:
                        text = text.replace(cite_text, '')
                for ref in body.ref_spans:
                    ref_text = ref.text
                    if ref_text:
                        text = text.replace(ref_text, '')
                text = normalize_whitespace(text)
                text = clean_tokenization(text)
                section = body.section
                if section:
                    text = SECTION_BODY.format(section, text)
                else:
                    text = BODY.format(text)
                file.write(text)
            done += 1
