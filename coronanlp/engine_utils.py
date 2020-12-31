from dataclasses import asdict as _asdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np  # type: ignore
from pandas import DataFrame  # type: ignore


@dataclass
class QuestionAnsweringArguments:
    X: Optional[Any] = field(default=None)
    question: Optional[Union[str, List[str]]] = field(default=None)
    context: Optional[Union[str, List[str]]] = field(default=None)
    topk: int = field(default=10)
    doc_stride: int = field(default=128)
    max_answer_len: int = field(default=15)
    max_seq_len: int = field(default=384)
    max_question_len: int = field(default=64)
    handle_impossible_answer: bool = field(default=True)

    def asdict(self):
        return _asdict(self)


class ModelOutput(NamedTuple):
    score: float
    start: int
    end: int
    answer: str


class QuestionAnsweringOutput(List[ModelOutput]):
    q: Optional[Union[str, List[str]]] = None
    c: Optional[Union[str, List[str]]] = None
    sids: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None
    pids: Optional[np.ndarray] = None
    _actual_size = 0

    @property
    def a(self) -> List[str]:
        return self.answers

    @property
    def context(self) -> str:
        """Return the model's output context as a single string."""
        c_str = ""
        if isinstance(self.c, list):
            c_str = " ".join(self.c)
        elif isinstance(self.c, str):
            c_str = self.c
        return c_str

    @property
    def shape(self) -> Union[Tuple[int, int], None]:
        if self.sids is not None:
            return self.sids.shape
        return None

    def attach_(self, *inputs) -> None:
        q, c, self.sids, self.dist = inputs
        self.q = q[0] if len(q) == 1 else q
        self.c = c[0] if len(c) == 1 else c

    def popempty(self) -> Union[ModelOutput, List[ModelOutput], None]:
        prev_length = len(self)
        items = [self.pop(i) for i, o in enumerate(self) if not o.answer]
        if items:
            self._actual_size = prev_length - 1
            return items[0] if len(items) == 1 else items
        return None

    @property
    def answers(self) -> List[str]:
        return [o.answer for o in self]

    @property
    def spans(self) -> List[Tuple[int, int]]:
        return [(o.start, o.end) for o in self]

    @property
    def lengths(self) -> List[int]:
        return list(map(len, self.answers))

    def scores(self) -> Dict[float, int]:
        """Return a dict mapping of scores and output indices.

        - Usage example:
        ```python
        topk = self.scores()
        idx = topk[max(topk)]  # compute max k.
        output = self[idx]  # query top idx item.
        ...
        # ModelOutput(score=0.11623, start=52, end=9, ...)
        ```
        """
        # Include answer length as a feature for best score.
        lengths, size = self.lengths, self.size()
        return {(o.score + lengths[i]) / size: i for i, o in enumerate(self)}

    def topk(self, n: Optional[Union[int, slice, Tuple[int, ...]]] = None):
        scores = self.scores()
        if n is None or isinstance(n, int) and n in (0, 1):
            return scores[max(scores)]
        elif isinstance(n, (int, slice, tuple)):
            n = slice(*n) if isinstance(n, tuple) and len(n) > 1 else n
            lengths = self.lengths
            argsort = [lengths.index(l) for l in sorted(lengths, reverse=True)]
            if isinstance(n, int):
                if n == -1:
                    return argsort
                if n > 1:
                    return argsort[:n]
            if isinstance(n, slice):
                sliced = n
                return argsort[sliced]
        return None

    def size(self) -> int:
        return len(self)

    def __repr__(self):
        return '{}(size: {}, shape: {})'.format(
            self.__class__.__name__, self.size(), self.shape)


def preds_to_df(pred, engine=None, sents=None, cord19=None) -> DataFrame:

    data: Dict[str, List] = {
        'sid': [], 'pid': [], 'dist': [], 'in_ctx': [], 'query': [],
        'answer': [], 'score': [], 'title': [], 'sent': []}

    if engine is not None:
        sents = engine.sents
        if cord19 is None and hasattr(engine, 'cord19'):
            cord19 = engine.cord19

    (n, d), k = pred.shape, len(pred)
    queries = [pred.q] * k if n == 1 else pred.q * k
    annexes = list(zip(queries, pred.answers, pred.scores()))
    D, I = pred.dist.squeeze(0), pred.sids.squeeze(0)
    for qas in annexes:
        for x in range(d):
            query, answer, score = qas
            dist, sid = D[x].item(), I[x].item()
            pid = sents.decode(sid)
            title = cord19.title(pid)
            sent = sents[sid]
            in_ctx = True if sent in pred.c else False
            rows = [
                sid, pid, dist, in_ctx, query,
                answer, score, title, sent
            ]
            for col, row in zip(data, rows):
                data[col].append(row)

    return DataFrame(data=data)
