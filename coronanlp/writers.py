from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from .core import Sampler, SentenceStore
from .dataset import CORD19
from .parser import parse

Pid = int
Uid = str

# e.g., NEWLINE*2+INPUT+NEWLINE  # easier on the eyes :)
EOT, NEWLINE, INPUT = "<|endoftext|>", "\n", "{}"

WIKI_TEMPLATE = {
    'header': "\n = = {} = =\n\n",
    'section': " = = = {} = = =\n\n {}\n",
    'body_a': " {}\n",
    'body_b': "{}\n"
}


def files_for_tokenizer(
        outdir: str,
        sampler_or_store: Union[Sampler, SentenceStore],
        pids: Optional[List[Pid]] = None,
        suffix: str = '\n') -> List[str]:
    paths = []
    if pids is None:
        pids = sampler_or_store.pids
    outdir = Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    for pid in tqdm(pids, desc='train-files'):
        fp = outdir.joinpath(f'{pid}.txt')
        with fp.open('w', encoding='utf-8') as f:
            for sent in sampler_or_store.sents(pid):
                f.write(f'{sent}{suffix}')
        paths.append(fp.as_posix())
    return paths


def files_for_model(
        outdir: str,
        sent_store: SentenceStore,
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
        train_it = sent_store.index_select(train_ids)
        cache(train_fp, train_it)

    if test_ids is not None and is_list_of_ids(test_ids):
        test_fp = outdir.joinpath('test.txt')
        test_it = sent_store.index_select(test_ids)
        cache(test_fp, test_it)

    if all((train_fp, test_fp)):
        return train_fp, test_fp
    return train_fp


def wiki_like_file(sample: List[Pid],
                   cord19: CORD19,
                   fn: str = 'train.txt',
                   outdir: str = 'data',
                   wiki_template: Optional[Dict[str, str]] = None) -> None:
    """Build a Wiki like dataset as a single text file.

    - Removes (The following tags are removed, since generative models like
    GPT2 will << generate >> similar for every output, which we dont want 🤗):
      - `cite-spans`, e.g., `"[1]"`
      - `ref-spans`, e.g., `"Figure 3C"`

    Example usage:
    ```python
    sample = cord19.sample(-1)
    train_ids, test_ids = coronanlp.split_dataset(sample, subset=0.9)
    wiki_like_file(train_ids, cord19, fn='train.txt')
    wiki_like_file(test_ids, cord19, fn='test.txt')
    ```
    """
    def astitle(text: str) -> str:
        # Normalize and enforce title style format.
        return text.strip().lower().title()

    if wiki_template is None:
        wiki_template = WIKI_TEMPLATE
    BODY_A = wiki_template['body_a']
    BODY_B = wiki_template['body_b']
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
                header = HEADER.format(astitle(title))
                file.write(header)
            # Cache the last/previous section (title) string and use it
            # to check wheather the next string is the same as previous.
            # This way we write ONE section per paragraph(s) in sequence.
            previous_section = ''
            # Indent only the first paragraph (BODY_A) and for any
            # subsequent paragraphs we wont apply indentation (BODY_B).
            BODY = ''
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
                text = cord19.prep(text)
                section = body.section
                if section and section.strip().lower() != previous_section:
                    text = SECTION_BODY.format(astitle(section), text)
                    previous_section = section.strip().lower()
                    BODY = BODY_A  # apply indentation.
                else:
                    BODY = BODY_B
                    text = BODY.format(text)  # body: paragraph
                file.write(f'{text}\n')
            done += 1


def wiki_like_splits(
    sample: List[Pid],
    cord19: CORD19,
    num_splits: int = 10,
    fn: str = 'train',
    outdir: str = 'data',
    wiki_template: Optional[Dict[str, str]] = None,
) -> None:
    """Build a Wiki like dataset and split into multiple text files.

    This method is specially usefull when a single file is too
    large to fit into memory.

    :param: num_splits: move last remaining list of ids to prev-last.

    * Example usage:
    ```python
    sample = cord19.sample(-1)
    train_ids, test_ids = coronanlp.split_dataset(sample, subset=0.9)
    # split the training data into 10 files (instead of a single file).
    wiki_like_splits(train_ids, cord19, num_splits=10, outdir='splits/')
    # and we keep the test/evaluation dataset as a single file since subset=0.9.
    wiki_like_file(test_ids, cord19, fn='test.txt', outdir='data/')

    ```
    """
    tasklist = []
    length = len(sample)
    splits = int(length / num_splits)

    for i in range(0, length, splits):
        sliced_sample = sample[i:min(i+splits, length)]
        tasklist.append(sliced_sample)

    if len(tasklist[-1]) != len(tasklist[0]):
        tasklist[-2].extend(tasklist.pop(-1))

    fn = fn[:fn.index('.')] if '.' in fn else fn
    for idx, pids in enumerate(tasklist):
        wiki_like_file(
            sample=pids,
            cord19=cord19,
            fn=f'{fn}_{idx}.txt',
            outdir=outdir,
            wiki_template=wiki_template,
        )
