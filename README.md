# ⚕ coronanlp

The current status of the project is experimental 🔬

## Docs

> `CORD19.__init__` ***method***

Construct a `CORD19` object.  Initialize with a single path or a list paths pointing to the directory with JSON files, e.g., `/../dir/*.json`

```python
import coronanlp
root = coronanlp.Path('/path/to/Datasets/AllenAI/CORD-19/2020-03-13/')
source = [p for p in root.iterdir() if p.is_dir()]
cord19 = coronanlp.CORD19(source, sort_first=True)
print(cord19)
...
# CORD19(papers: 13202, files_sorted: True, source: [
#   comm_use_subset, noncomm_use_subset, pmc_custom_license, biorxiv_medrxiv,
# ])
```

> `CORD19.sample` ***method***

| name    | type                        | description                                                                     |
|---------|-----------------------------|---------------------------------------------------------------------------------|
| *k*     | int                         | Sample size, if `k < max_ids`; random sample. Otherwise, `-1` for all ids.      |
| *s*     | int                         | Split of ids, `id` is based on the order of path(s) passed and `n` directories. |
| *seed*  | Optional[Union[int, float]] | Random seed if not None.                                                        |
| returns | List[int]                   | Returns a list of paper ids.                                                    |

```python
sample = cord19.sample(s=0)  # e.g., All ids from the biorxiv_medrxiv directory
sample = cord19.sample(k=-1) # Returns all ids available in order.
sample = cord19.sample(k=1000) # 1K randomly selected ids, e.g., from all 45941
```

> `CORD19.batch` ***method***

| name      | type              | description                                                   |
|-----------|-------------------|---------------------------------------------------------------|
| *sample*  | List[int]         | A list of integer sequences.                                  |
| *minlen*  | int               | Minimum length of a string to consider valid.                 |
| *workers* | int               | Number of cores to use, if None; obtained from `cpu_count()`. |
| returns   | **SentenceStore** | The newly constructed object.                                 |

```python
sample = cord19.sample(-1)
sentence_store = cord19.batch(sample,  minlen=15)
...
# files/papers: 24% ████     | 3112/13202 [09:49<19:51, 8.47it/s]
```

After extracting the text from the JSON files, pre-processing, cleaning, and sentence-tokenization we get a newly constructed `SentenceStore` object with a container `Dict[int, List[str]]`. holding all the sentences.

> Using the spaCy model `en_core_sci_sm` is recommended for building high quality sentences.

- Below we can see the contrast in how the models recognize sentence boundaries; `en-core-web-sm` starts splitting at `" ( "` ? that doesn't make sense!? if this is done here; where else it will do it? Given the fact the text **is** scientific literature, we can expect equations, many abbreviations, and symbols - all destroyed because we used the wrong tokenizer. Meanwhile, `en_core_sci_sm` has no issues 🥇 [scispacy - SpaCy models for biomedical text processing](https://allenai.github.io/scispacy/)

```bash
# en_core_web_sm
2: We analyzed ... pulmonary artery endothelial cells (hPAECs).
3: The effect of ... electric resistance, molecule trafficking, calcium (
4: Ca 2+ ) homeostasis, gene expression and proliferation studies.

# en_core_sci_sm
2: We analyzed ... pulmonary artery endothelial cells (hPAECs).
3: The effect of ... electric resistance, molecule trafficking, calcium (Ca 2+ ) homeostasis, gene expression and proliferation studies.
```

> `coronanlp.core.SentenceStore.__str__` **method**

- Wow 🤔 `13,202` papers produce `1,890,230` million sentences and `60,847,005` million tokens! It takes about ~30 minutes with *SSD* and around ~40 on *HDD*, but it really depends on hardware.

```python
print(sentence_store)
...
# SentenceStore(avg_seqlen=32.19, num_papers=13202, num_sents=1890230, num_tokens=60847005)
```

> `SentenceTransformer.__init__` **method**

```python
encoder = coronanlp.SentenceTransformer('model_name_or_path')
sentence_embeddings = encoder.encode(list(sentence_store), batch_size=8)
...
# batches: 96% ██████████████ | 36585/38168 [17:10<01:17, 20.38it/s]
```

> `coronanlp.indexing.fit_index_ivf_hnsw` **method**

### NOTE

---

The following method requires `faiss,` which should be easy to install for both *Linux* and *MacOS*. Except for *Windows* which needs to be built from source, read more here: [windows support? | issue<1437>](https://github.com/facebookresearch/faiss/issues/1437)

Install faiss with `CUDA or CPU` (I have only used the *CPU* version and haven't experienced any issues even at `2 million` dense vectors runs extremely fast. See official repo: [faiss github](https://github.com/facebookresearch/faiss)

- **cpu-version**:
  - `conda install faiss-cpu -c pytorch`

- **gpu-version**:
  - [`8.0, 9.0, 10.0`] replace `X` with your version.
    - `conda install faiss-gpu cudatoolkit=X.0 -c pytorch`

```python
index_ivf = coronanlp.fit_index_ivf_hnsw(sentence_embeddings,  metric='l2')
print(index_ivf.is_trained)
...
# True
```

> `ScibertQuestionAnswering.__init__` **method**

The following table defines the minimum arguments required to construct a new `ScibertQuestionAnswering` object.

| name      | type                            | description              |
|-----------|---------------------------------|--------------------------|
| *papers*  | Union[str, SentenceStore]       | A path or papers object. |
| *index*   | Union[str, faiss.IndexIVFFlat]  | A path or index object   |
| *encoder* | Union[str, SentenceTransformer] | A path or model object   |

```python
from coronanlp.engine import ScibertQuestionAnswering
qa =  ScibertQuestionAnswering(sentence_store, index_ivf, encoder)
print(qa.all_model_devices)
...
# {'summarizer_model_device': device(type='cuda'),
#  'sentence_transformer_model_device': device(type='cuda'),
#  'question_answering_model_device': device(type='cpu')}
```

> `QuestionAnsweringEngine.answer` **method**

| name       | type                      | description                                                        |
|------------|---------------------------|--------------------------------------------------------------------|
| *question* | str                       | Question(s) to query the QA model (split by sentence tokenization) |
| *topk*     | int                       | Number of answers to return (chosen by likelihood)                 |
| *top_p*    | int                       | A path or model object                                             |
| *nprobe*   | int                       | index search probe value                                           |
| *mode*     | str                       | Compressor mode to use on context; `'bert', 'freq'`                |
| returns    | *QuestionAnsweringOutput* | Object holding the predictions of the model(s)                     |

---

The subsequent keyword arguments can be used in `QuestionAnsweringEngine.answer` method, which then gets passed to `transformers.QuestionAnsweringPipeline.__call__`. Click in [here](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.QuestionAnsweringPipeline.__call__) to read the official  HuggingFace documentation.

| **kwargs                   | type | description                                                                             |
|----------------------------|------|-----------------------------------------------------------------------------------------|
| *doc_stride*               | int  | (default=`128`) If input length overlaps split by chunks.                               |
| *max_answer_len*           | int  | (default=`15`) Max length of predicted answers.                                         |
| *max_seq_len*              | int  | (default=`384`) Max length of the total sentence (context+question) after tokenization. |
| *max_question_len*         | int  | (default=`64`) Max length of the question after tokenization.                           |
| *handle_impossible_answer* | bool | (default=`False`) Whether or not we accept impossible as an answer.                     |

---

### Question Answering

Open book question-answering on CORD-19 literature

- base-model: `scibert-scivocab-cased`
- finetuned: CORD-19 dataset `nli-stsb-mean-tokens` (using the `sentence-transformers` library).
- downstream-task: question-answering `SQUAD 2.0`

```python
question =  ("What has been published concerning systematic, holistic approach to"
            " diagnostics (from the public health surveillance perspective to being"
            " able to predict clinical outcomes)?")
preds = qa.answer(question,  topk=5, top_p=25, nprobe=64, mode='bert')
preds.popempty()  # quickly pop any empty answers from the list.
print(preds.ids, preds.dist)
...
```

- Each sentence id with its respective distance (nearest-neighbor to query/question):

```bash
(array([[179943,  38779,  48340, 171641,  11026,  16090, 132451,  10551,
         231547,  32627, 203359, 123822, 231157,  59945, 333167, 203328,
          37302,  74584,   1534, 425932, 261597, 268659, 397260,  27072,
         117127]]),
 array([[114.0309 , 119.05531, 127.31623, 128.10754, 134.01633, 135.51642,
         137.91711, 138.64822, 139.26003, 141.16678, 142.14966, 144.36464,
         147.06775, 147.06958, 147.22888, 148.9502 , 149.55885, 149.9512 ,
         150.12485, 150.1834 , 150.33147, 151.04956, 151.89029, 152.3189 ,
         152.47305]], dtype=float32))
```

- In the following output we can see all the predicted answers and get a nice view of each answer's information.

```python
list(preds)
...
[
  ModelOutput(score=0.02010919339954853, start=11, end=35, answer='laboratory confirmation,'),
  ModelOutput(score=0.00906957034021616, start=133, end=143, answer='sequencing.'),
  ModelOutput(score=0.00279330019839108, start=40, end=143, answer='national reference, ...'),
  ModelOutput(score=0.002625082153826952, start=11, end=21, answer='laboratory')
]
```

- Get all spans `[(start, end)]` index of each answer. Both styles produce the same result.

```python
for span in preds: print(span.start, span.end)
# or
preds.spans()
...
  [(11, 35), (133, 143), (40, 143), (11, 21)]
```

- Highlighting the predicted answer:

```python
from coronanlp import render_output as render

output = preds[2]  # e.g., iter all answer: [o.answer for o in preds]
answer = output.answer
render(answer=answer, context=preds.c, question=preds.q)
```

```markdown
* Question

> What has been published concerning systematic, holistic approach to 
diagnostics (from the public health surveillance perspective to being 
able to predict clinical outcomes)?


* Context:

In case of laboratory confirmation, `<< the national reference laboratory 
aims to obtain material from regional laboratories for further sequencing. ANSWER >>` 
An alternate measure of program success is the extent to which screening delays the 
first importation of cases into the community, possibly providing additional time 
to train medical staff, deploy public health responders or refine travel policies 
(Cowling et al., Unlike the UK national strategy documents and plans, the US National 
Health Information Infrastructure Strategy document (also known as "Information for Health") 
refers explicitly to GIS and real-time health and disease monitoring and states that "public 
health will need to include in its toolkit integrated data systems; high-quality 
community-level data; tools to identify significant health trends in real-time data streams; 
and geographic information systems" [48]. This thinktank is used to address matters related 
to combating new issues and to expand on the Saudi reach to the international scientific 
community. For the purpose of this paper, the following de�nition is used, "Public health 
surveillance is the ongoing systematic collection, analysis, interpretation and 
dissemination of health data for the planning, implementation and evaluation of 
public health action" (see Section 2.3 below). Finally, we discuss policy and 
logistical and technological obstacles to achieving a potential transformation 
of public health microbiology.
```

- Example of the `render_output()` method shown above in a jupyter notebook:

![question-answering-jupyter-demo](src/img/question-answering.gif)
