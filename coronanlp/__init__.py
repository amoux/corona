import sys

from . import parser
from .core import SentenceStore, Sampler, merge_samplers
from .dataset import (CORD19, build_wiki_like_dataset,
                      cache_for_language_modeling, cache_for_tokenizer_vocab)
from .indexing import PaperIndexer, Path
from .retrival import (common_tokens, extract_questions, extract_titles_fast,
                       extract_titles_slow)
from .summarization import BertSummarizer, frequency_summarizer
from .tasks import TaskList
from .tokenizer import SpacySentenceTokenizer
from .utils import (DataIO, clean_punctuation, clean_tokenization, load_store,
                    normalize_whitespace, render_output, save_stores,
                    split_dataset)

try:
    _faiss_lib = __import__("faiss")
except ModuleNotFoundError:
    print(sys.exc_info())
else:
    globals()["faiss"] = _faiss_lib
    from .indexing import fit_index_ivf_hnsw
    from .retrival import tune_ids
