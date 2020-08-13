from .dataset import CORD19Dataset
from .datatypes import Papers, Sentences, merge_papers
from .indexing import PaperIndexer, Path
from .retrival import (common_tokens, extract_questions, extract_titles,
                       frequency_summarizer)
from .tasks import *
from .tokenizer import SpacySentenceTokenizer
from .transformer import BertSummarizer, SentenceTransformer
from .utils import (DataIO, clean_punctuation, clean_tokenization,
                    concat_csv_files, normalize_whitespace, papers_to_csv,
                    render_output)

try:
    _faiss_lib = __import__("faiss")
except ModuleNotFoundError:
    print(sys.exc_info())
else:
    globals()["faiss"] = _faiss_lib
    from .indexing import fit_index_ivf_hnsw
    from .retrival import tune_ids_to_tasks
