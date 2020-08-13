# corona


## CORD19Dataset

```python
import corona_nlp as nlp

rootdir = nlp.Path("path/to/CORD-19-research-challenge/2020-xx-xx/")
sources = [p.joinpath(p.name) for p in rootdir.iterdir() if p.is_dir()]

dataset = nlp.CORD19Dataset(
    source=source,
    index_start=1,
    sort_first=True,
    nlp_model="en_core_sci_sm",
    text_keys=("body_text",),
)
print(dataset)
...
# CORD19Dataset(papers=45941, files_sorted=True, source=['pdf_json', 'pdf_json', 'pdf_json', 'pdf_json'])
```

## Papers

```python
sample = dataset.sample(1000)
papers = dataset.batch(sample, minlen=25)
...
# papers: 100% ██████████████ | 1000/1000 [02:53<00:00, 5.59it/s]
```

```python
print(papers)
...
# Papers(avg_strlen=169.4, num_papers=1000, num_sents=305337)
```

## SentenceTransformer

```python
encoder = nlp.SentenceTransformer("path/to/scibert-nli/")
embedding = encoder.encode(papers)
...
# batches: 96% ██████████████ | 36585/38168 [17:10<01:17, 20.38it/s]
```

## faiss.IndexIVFFlat

```python
index_ivf = nlp.fit_index_ivf_hnsw(embedding, metric="L2")
print(index_ivf.is_trained)
...
# True
```

## QuestionAnsweringEngine

```python
from corona_nlp.engine import QuestionAnsweringEngine

engine = QuestionAnsweringEngine(
    papers=papers,
    index=index_ivf,
    encoder=encoder,
    cord19=dataset,
    model_name="amoux/scibert_nli_squad",
)
```

```python
question = "What has been published details ethical and social science considerations?"
output = engine.answer(question, k=25, nprobe=16, mode="bert")
output.keys()
...
# dict_keys(['answer', 'context', 'dist', 'ids', 'span'])
```

```python
print(output["answer"])
...
# Articles that discussed prepublication release of data,
# social media release of research data, the use of prepublication
# data, social media publicization of conferences and journals
```
