from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from coronanlp.core import SentenceStore
from coronanlp.dataset import CORD19
from tqdm.auto import tqdm


def init_model_directory(model_name: str, model_path: str,
                         papers_dir: Optional[str] = None) -> Tuple[Sequence[Path]]:
    hf_model_root = Path(model_name) \
        if model_path is None else Path(model_path).joinpath(model_name)
    if papers_dir is not None and isinstance(papers_dir, str):
        papers_dir = Path(papers_dir)
    training_files = hf_model_root.joinpath("training_files/")
    training_dataset = hf_model_root.joinpath("training_dataset/")
    checkpoints = hf_model_root.joinpath("checkpoints/")
    directories = [hf_model_root, training_files,
                   training_dataset, checkpoints]
    if papers_dir is not None:
        directories.append(papers_dir)
    [
        p.mkdir(parents=True, exist_ok=True) for p in directories
    ]
    return directories


def papers_to_training_files(papers: SentenceStore, out_dir: Path,
                             dataset: Optional[CORD19] = None,
                             ) -> List[str]:
    joinpath = out_dir.joinpath
    if dataset is None:
        dataset = papers.init_cord19_dataset()
    paths = []
    paper_ids, sentences = papers.indices, papers.sents
    for pid in tqdm(paper_ids, desc='paper-to-file'):
        fp = joinpath(f'{dataset[pid]}.txt')
        paths.append(fp.as_posix())
        with fp.open('w', encoding='utf-8') as file:
            for sent in sentences(pid):
                file.write(f'{sent}\n')
    pids, sids = papers.num_papers, papers.num_sents
    print(f'{sids} sentences and {pids} papers saved as files.')
    return paths


def papers_to_training_dataset(papers: SentenceStore, fp: Path) -> None:
    paper_ids, sentences = papers.indices, papers.sents
    with fp.open('w', encoding='utf-8') as file:
        for pid in tqdm(paper_ids, desc='line-by-line-dataset'):
            for sent in sentences(pid):
                file.write(f'{sent}\n')
    print(f'Done! building line-by-line, lines: {len(papers)}')


def build_new_cord19_dataset(cord19_dir: str,
                             num_papers: int,
                             minlen: int = 15,
                             store_name: Optional[str] = None,
                             path: Optional[str] = None
                             ) -> Tuple[CORD19, SentenceStore]:
    root = Path(cord19_dir)
    source = [p.joinpath(p.name) for p in root.iterdir()
              if not p.name.endswith("4_17") and p.is_dir()]
    dataset = CORD19(source=source, sort_first=True)
    sample = dataset.sample(num_papers)
    papers = dataset.batch(sample, minlen=minlen)
    if store_name is not None and isinstance(store_name, str):
        papers.to_disk(store_name=store_name)
    elif path is not None and isinstance(path, str):
        papers.to_disk(path=path)
    else:
        raise ValueError
    return dataset, papers
