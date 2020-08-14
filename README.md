# corona_nlp

Experimental

## CORD19Dataset class

> `CORD19Dataset.__init__` ***method***

Construct a `CORD19Dataset` object.  Initialize with a single path or a list paths pointing to the directory with JSON files, e.g., `/../dir/*.json`

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
# CORD19Dataset(papers=45941, files_sorted=True, source=['biorxiv_medrxiv', 'custom_license', ...,])
```

> `CORD19Dataset.sample` ***method***

| name    | type                        | description                                                                     |
|---------|-----------------------------|---------------------------------------------------------------------------------|
| *k*     | int                         | Sample size, if `k < max_ids`; random sample. Otherwise, `-1` for all ids.      |
| *s*     | int                         | Split of ids, `id` is based on the order of path(s) passed and `n` directories. |
| *seed*  | Optional[Union[int, float]] | Random seed if not None.                                                        |
| returns | List[int]                   | Returns a list of paper ids.                                                    |

```python
sample = dataset.sample(s=0)  # e.g., All ids from the biorxiv_medrxiv directory
sample = dataset.sample(k=-1) # Returns all ids available in order.
sample = dataset.sample(k=1000) # 1K randomly selected ids, e.g., from all 45941
```

> `CORD19Dataset.batch` ***method***

| name      | type       | description                                                   |
|-----------|------------|---------------------------------------------------------------|
| *sample*  | List[int]  | A list of integer sequences.                                  |
| *minlen*  | int        | Minimum length of a string to consider valid.                 |
| *workers* | int        | Number of cores to use, if None; obtained from `cpu_count()`. |
| returns   | **Papers** | The newly constructed object.                                 |

```python
papers = dataset.batch(sample,  minlen=25, workers=None)
...
# papers: 100% ██████████████ | 1000/1000 [02:53<00:00, 5.59it/s]
```

```python
print(papers)
...
# Papers(avg_strlen=169.4, num_papers=1000, num_sents=305337)
```

> `SentenceTransformer.__init__` **method**

```python
encoder = nlp.SentenceTransformer("path/to/scibert-nli/")
embedding = encoder.encode(papers)
...
# batches: 96% ██████████████ | 36585/38168 [17:10<01:17, 20.38it/s]
```

> `corona_nlp.indexing.fit_index_ivf_hnsw` **method**

```python
index_ivf = nlp.fit_index_ivf_hnsw(embedding,  metric="L2")
print(index_ivf.is_trained)
...
# True
```

> `QuestionAnsweringEngine.__init__` **method**

```python
from corona_nlp.engine import QuestionAnsweringEngine
engine =  QuestionAnsweringEngine(
    papers=papers,
    index=index_ivf,
    encoder=encoder,
    cord19=dataset,
    model_name="amoux/scibert_nli_squad",
)
```

> `QuestionAnsweringEngine.answer` **method**

```python
question =  "What has been published details ethical and social science considerations?"
output = engine.answer(question,  k=25,  nprobe=16,  mode="bert")
output.keys()
...
# dict_keys(['answer', 'context', 'dist', 'ids', 'span'])
```

```python
print(output["answer"])
...
# Articles that discussed prepublication release of data, social media
# release of research data, the use of prepublication data, social media
# publicization of conferences and journals
```
