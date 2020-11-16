from pathlib import Path

import faiss
import numpy as np
import plac
from coronanlp.dataset import CORD19
from coronanlp.indexing import fit_index_ivf_hnsw
from sentence_transformers import SentenceTransformer

DEFAULT_SOURCE = [
    '/home/carlos/Datasets/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json',
    '/home/carlos/Datasets/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json',
    '/home/carlos/Datasets/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json'
]


@plac.annotations(
    num_papers=("Number of papers. '-1' for all papers", "option", "n", int),
    minlen=("Minimum length of a string to consider", "option", "minlen", int),
    centroids=("Number of centroids, if None; sqrt(n).", "option", "c", int),
    nlp_model=("spaCy model name", "option", "m", str),
    data_dir=("Path to the directory for outputs", "option", "data_dir", str),
    source=("Path to cord19 dir of json files", "option", "source", str),
    encoder=("Path to sentence encoder model", "option", "encoder", str),
)
def main(num_papers=-1, minlen=20, centroids=None, nlp_model="en_core_sci_sm",
         data_dir="data/", source=None, encoder="model/scibert-nli"):
    """Build and encode CORD-19 dataset texts to sentences and embeddings."""
    if source is None:
        source = DEFAULT_SOURCE

    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    dataset = CORD19(source=source,
                            text_key="body_text",
                            index_start=1,
                            sort_first=True,
                            nlp_model=nlp_model)

    sample = dataset.sample(num_papers)
    papers = dataset.batch(sample, minlen=minlen)

    # save the instance of papers to file
    data_info = (papers.num_papers, papers.num_sents, papers.num_tokens)
    prep_info = (minlen, papers.maxlen)
    sents_file = '{}_{}_{}_[min={},max={}].cord'.format(*data_info, *prep_info)
    papers.to_disk(data_dir.joinpath(sents_file))

    encoder = SentenceTransformer(encoder)
    embedding = np.array(encoder.encode(papers, show_progress_bar=True))
    shape = embedding.shape
    assert shape[0] == len(papers)

    # save the encoded embeddings to file
    pids = data_info[0]
    embed_file = '{}_embed_[n={},d={}]'.format(pids, *shape)
    np.save(data_dir.joinpath(embed_file), embedding)

    index_ivf = fit_index_ivf_hnsw(embedding, metric='l2', nlist=centroids)
    # save the indexer of embeddings to file
    index_file = '{}_ivf_hnsw_[n={},d={}].index'.format(pids, *shape)
    index_file = data_dir.joinpath(index_file).as_posix()
    faiss.write_index(index_ivf, index_file)

    print(f'Done: index and papers saved in path: {data_dir}')


if __name__ == '__main__':
    plac.call(main)
