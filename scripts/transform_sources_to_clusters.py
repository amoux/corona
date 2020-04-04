import sys
sys.path.append("../")
from pathlib import Path

import numpy as np
from tqdm import tqdm

from corona_nlp.indexing import PaperIndexing, all_dataset_sources
from corona_nlp.preprocessing import (load_papers_with_text,
                                      normalize_whitespace)
from corona_nlp.tokenizer import SpacySentenceTokenizer
from corona_nlp.transformer import SentenceTransformer
from corona_nlp.utils import DataIO

source_id = 1  # source ids: 0 to 4
text_keys = ("abstract", "body_text")
min_strlen = 10
max_strlen = 1000
batch_size = 1024
scibert_nli_model = "/home/carlos/transformer_models/scibert-nli/"
output_dir = "data/cluster-data"

cord19 = PaperIndexing(all_dataset_sources[source_id])
print(f"loaded source:\n{cord19}\n")
indices = list(range(1, cord19.num_papers))
papers_with_text = iter(load_papers_with_text(covid=cord19,
                                              indices=indices,
                                              keys=text_keys))
# transform all texts to sentences.
sentence_tokenizer = SpacySentenceTokenizer()
print(f"tokenizing texts from: {cord19.num_papers} papers.")

sentences = []
for paper in tqdm(papers_with_text, desc="papers"):
    for line in sorted(set(paper["texts"])):
        line = normalize_whitespace(line)
        if len(line) <= max_strlen:
            sents = sentence_tokenizer.tokenize(line)
            for string in sents:
                string = normalize_whitespace(string.text)
                if len(string) >= min_strlen:
                    sentences.append(string)

sentences = list(set(sentences))
print(f"number of sentences: {len(sentences)}")

# encode all sentences to the transformer model
transformer = SentenceTransformer(scibert_nli_model)
embeddings = transformer.encode(sentences, batch_size=batch_size)
embeddings = np.array(embeddings)
print(f"done embedding sentences: {embeddings.shape}")

# save the embeddings and sentences as a pickle files
output_dir = Path(output_dir)
if not output_dir.exists():
    output_dir.mkdir(parents=True)

file_name = f"embedd_{cord19.source_name}"
DataIO.save_data(file_name, embeddings, output_dir)
print(f"saved {file_name} in path: {output_dir}")

file_name = f"sents_{cord19.source_name}"
DataIO.save_data(file_name, sentences, output_dir)
print(f"saved {file_name} in path: {output_dir}")
