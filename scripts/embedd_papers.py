import sys
sys.path.append("../")

from pathlib import Path
import numpy as np
from tqdm import tqdm

from corona_nlp.indexing import PaperIndexing
from corona_nlp.preprocessing import (load_papers_with_text,
                                      normalize_whitespace)
from corona_nlp.tokenizer import SpacySentenceTokenizer
from corona_nlp.transformer import SentenceTransformer
from corona_nlp.utils import DataIO, load_dataset_paths

# datasets available: ('pmc', 'bio', 'com', 'non')
dataset = "pmc"
max_papers = -1
min_strlen = 20
max_strlen = 2000
batch_size = 9
text_keys = ("abstract", "body_text")

out_dir = "../data/cluster_data/"
scibert_nli_model = "/home/carlos/transformer_models/scibert-nli/"
cord_dataset_path = "CORD-19-research-challenge/2020-03-13/"

out_dir = Path(out_dir)
if not out_dir.is_dir():
    out_dir.mkdir(parents=True)

if __name__ == '__main__':
    source = load_dataset_paths(cord_dataset_path)
    dataset = [p for p in source.dirs if p.name[:3] == dataset]
    index = PaperIndexing(dataset[0])
    print(f"loaded source: \n{index}\n")

    max_papers = index.num_papers if max_papers == -1 else max_papers
    assert max_papers == index.num_papers
    indices = list(range(1, max_papers + 1))
    papers = iter(load_papers_with_text(index, indices, text_keys))

    # preprocessing steps to normalize, clean, reduce noise,
    # and overhead when encoding sequences to embeddings
    sentences = []
    tokenizer = SpacySentenceTokenizer()
    for paper in tqdm(papers, desc="papers"):
        for line in sorted(set(paper["texts"])):
            line = normalize_whitespace(line)
            if len(line) <= max_strlen:
                tokens = tokenizer.tokenize(line)
                for string in tokens:
                    string = normalize_whitespace(string.text)
                    if len(string) >= min_strlen:
                        sentences.append(string)

    sentences = list(set(sentences))
    print(f"number of sentences: {len(sentences)}")

    io = DataIO()
    fp = out_dir.joinpath(f"sents_{index.source_name}.pkl")
    io.save_data(fp, sentences)
    print(f"saved sentences in path: {fp}")
    del sentences

    # encode all sentences to the transformer model
    transformer = SentenceTransformer(scibert_nli_model)
    embedding = transformer.encode(sentences=io.load_data(fp),
                                   batch_size=batch_size)
    print(f"done embedding sentences: {embedding.shape}")

    fp = out_dir.joinpath(f"embedd_{index.source_name}")
    np.savez_compressed(fp, embedding=embedding)
    print(f"saved embedding matrix in path: {fp}")
