import sys

from . import parser
from .core import Sampler, SentenceStore, merge_samplers
from .dataset import CORD19, SentenceDataset
from .indexing import PaperIndexer, Path
from .retrival import (common_tokens, extract_questions, extract_titles_fast,
                       extract_titles_slow)
from .summarization import BertSummarizer, frequency_summarizer
from .tasks import TaskList
from .tokenizer import SpacySentenceTokenizer
from .ukplab import SentenceEncoder, semantic_search
from .utils import (DataIO, clean_punctuation, clean_string,
                    clean_tokenization, load_store, normalize_whitespace,
                    render_output, save_stores, split_dataset, split_on_char)
from .writers import (WIKI_TEMPLATE, files_for_model, files_for_tokenizer,
                      wiki_like_file, wiki_like_splits)

try:
    _faiss_lib = __import__("faiss")
except ModuleNotFoundError:
    print(sys.exc_info())
else:
    globals()["faiss"] = _faiss_lib
    from .indexing import fit_index_ivf_hnsw
    from .indexing import fit_index_ivf_fpq
    from .retrival import tune_ids
