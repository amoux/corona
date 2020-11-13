import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union

# matches+ e.g., [34] | [2]
regex_bracket = re.compile(r'\[\d*?\]{1,2}')

# matches+ e.g., (Figure 6e2) | Fig.1e | Figure 3. | (Fig. 3B and 3C)
regex_figure = re.compile(
    r'((\(F|F)ig((\s|\.)|(gure|[\w\d]){1,2}){1,9}(\)|\.))'
)

DEFAULT_ENTRY_KEYS = {
    'ref': ['id', 'text', 'type', 'latex', 'ref_id'],
    'bib': ['id', 'title', 'authors', 'year', 'venue',
            'volume', 'issn', 'pages', 'other_ids', 'ref_id']
}


class SpanField(NamedTuple):
    start: int
    end: int
    text: str
    ref_id: str


class CiteSpan(SpanField):
    """Cite Span Field.

    Example:
      - text: "[3]"
      - ref_id: "BIBREF2"
    """
    ...


class RefSpan(SpanField):
    """Reference Span Field.

    Example:
      - text: "Figure 3C"
      - ref_id: "FIGREF2"
    """
    ...


@dataclass
class RefEntry:
    id: str
    text: Optional[str] = None
    type: Optional[str] = None
    latex: Optional[Any] = None
    ref_id: Optional[str] = None  # Original key name found in the file.

    def dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BibEntry:
    id: str
    title: str
    authors: List[Dict[str, Union[str, List[str]]]]
    year: Union[str, int]
    venue: str
    volume: str
    issn: str
    pages: str
    other_ids: Dict[str, Any]
    ref_id: Optional[str] = None  # Original key name found in the file.

    def dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetaData:
    title: str
    authors: List[Dict[str, Any]]

    def dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Body:
    text: str = field(default_factory=str, repr=False)
    section: str = field(default_factory=str)
    cite_spans: List[CiteSpan] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    ref_spans: List[RefSpan] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    bib_entries: Dict[str, BibEntry] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    ref_entries: Dict[str, RefEntry] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

    def _add_spans(self, key: str, spans: List[Dict[str, Any]]):
        if key == 'cite':
            self.cite_spans = [
                CiteSpan(**field) for field in spans]
        if key == 'ref':
            self.ref_spans = [
                RefSpan(**field) for field in spans]


class Paper(List[Body]):
    uid: Optional[str] = None
    pid: Optional[int] = None
    meta: Optional[MetaData] = None

    @property
    def title(self) -> Union[str, None]:
        if self.meta is not None:
            return self.meta.title
        return None

    @property
    def authors(self) -> Union[List[Dict[str, Any]], None]:
        if self.meta is not None:
            return self.meta.authors
        return None

    @property
    def texts(self) -> Iterable[str]:
        for body in self:
            yield body.text

    @property
    def sections(self) -> Iterable[str]:
        for body in self:
            yield body.section

    @property
    def bib_entries(self) -> Iterable[BibEntry]:
        for body in self:
            entries = body.bib_entries
            for entry in entries:
                if not entries[entry]:
                    continue
                yield entries[entry]

    @property
    def ref_entries(self) -> Iterable[RefEntry]:
        for body in self:
            entries = body.ref_entries
            for entry in entries:
                if not entries[entry]:
                    continue
                yield entries[entry]

    @property
    def ref_spans(self) -> Iterable[RefSpan]:
        return (span for body in self
                for span in body.ref_spans)

    @property
    def cite_spans(self) -> Iterable[CiteSpan]:
        return (span for body in self
                for span in body.cite_spans)

    def size(self) -> int:
        return len(self)

    def __repr__(self):
        ids = (self.pid, self.uid)
        return '{}((pid, uid): {}, size: {})'.format(
            self.__class__.__name__, ids, len(self))


def _verify_keys(verify, entry) -> None:
    entrylist = DEFAULT_ENTRY_KEYS[verify]
    for key in list(entry.keys()):
        if key in entrylist:
            continue
        entry.pop(key)
    return


def parse(pid, data: Dict[str, Any]) -> Paper:
    """Parser for CORD-19 papers."""
    paper = Paper()
    paper.pid = pid
    paper.uid = data['paper_id']
    for item in data['body_text']:
        body = Body(item['text'], section=item['section'])
        body._add_spans('cite', spans=item['cite_spans'])
        body._add_spans('ref', spans=item['ref_spans'])
        # Add cite and ref spans in separate iterations
        # due to size disambiguation from either entries.
        for cite_span in body.cite_spans:
            bib_key = cite_span.ref_id
            if bib_key is None:
                continue
            bib_map = data['bib_entries'][bib_key]
            bib_map.update({'id': bib_key})
            _verify_keys('bib', entry=bib_map)
            body.bib_entries[bib_key] = BibEntry(**bib_map)
        for ref_span in body.ref_spans:
            ref_key = ref_span.ref_id
            if ref_key is None:
                continue
            ref_map = data['ref_entries'][ref_key]
            ref_map.update({'id': ref_key})
            _verify_keys('ref', entry=ref_map)
            body.ref_entries[ref_key] = RefEntry(**ref_map)
        paper.append(body)
    paper.meta = MetaData(
        title=data['metadata']['title'],
        authors=data['metadata']['authors'],
    )
    return paper
