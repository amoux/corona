import sys
sys.path.append("../")
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from corona_nlp.utils import DataIO
from corona_nlp.indexing import PaperIndexing, all_dataset_sources
from corona_nlp.preprocessing import (load_papers_with_text,
                                      normalize_whitespace)
from corona_nlp.tokenizer import SpacySentenceTokenizer


source_id = 1  # source ids: 0 to 4
text_keys = ("abstract", "body_text")
min_strlen = 10
scibert_nli_model = "/home/carlos/transformer_models/scibert-nli/"
output_dir = "data/cluster-data"


cord19 = PaperIndexing(all_dataset_sources[source_id])
print(f"loaded source:\n{cord19}\n")
indices = list(range(1, cord19.num_papers))
papers_with_text = load_papers_with_text(covid=cord19,
                                         indices=indices,
                                         keys=text_keys)

# extract texts from all papers indices:
texts = []
for paper in tqdm(papers_with_text, desc="strings"):
    for line in sorted(set(paper["texts"])):
        string = normalize_whitespace(line)
        if len(string) >= min_strlen:
            texts.append(string)
print(f"number of strings: {len(texts)}")

# tranform all texts to sentences:
sentence_tokenizer = SpacySentenceTokenizer()
sentences = sentence_tokenizer.tokenize_batch(texts)
sentences = list(set(sentences))
print(f"number of sentences: {len(sentences)}")

# encode all sentences to the transformer model
transformer = SentenceTransformer(scibert_nli_model)
embeddings = transformer.encode(sentences)
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
