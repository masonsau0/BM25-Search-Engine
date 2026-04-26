"""Self-contained BM25 retrieval engine.

A tidy reimplementation of the LA Times pipeline (`IndexEngine.py`,
`BM25Retrieval.py`, `SearchEngine.py`, `GetDoc.py`) that works on **any
corpus of plain-text files**, with a Python API suitable for embedding in a
notebook or a Streamlit app. Same BM25 scoring formula, same tokeniser,
same inverted-index data structure — but the document format is decoupled
from the LA Times SGML, so you can index a folder of `.txt` files in one
call.

Usage
-----
>>> from bm25 import Corpus
>>> corpus = Corpus.from_directory("demo_corpus", k1=1.2, b=0.75)
>>> hits = corpus.search("electric vehicle battery", top_k=5)
>>> for h in hits:
...     print(h.docno, h.score, h.headline)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens — same scheme as the LA Times pipeline."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Document:
    docno: str
    headline: str
    body: str
    length: int = 0
    term_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class Hit:
    docno: str
    headline: str
    score: float
    snippet: str
    matched_terms: list[str]


# ---------------------------------------------------------------------------
# Corpus + BM25
# ---------------------------------------------------------------------------


class Corpus:
    """Indexes a collection of text documents and answers BM25 queries.

    Each document is identified by a `docno` (its filename stem). The first
    non-empty line is treated as the headline; the rest is body text.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.lexicon: dict[str, int] = {}
        self.inverted_index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.docs: list[Document] = []
        self.docno_to_id: dict[str, int] = {}

    # ----------------------------------------------------------------- build

    @classmethod
    def from_directory(
        cls, path: str | Path, k1: float = 1.2, b: float = 0.75
    ) -> "Corpus":
        c = cls(k1=k1, b=b)
        c.add_files(Path(path).glob("*.txt"))
        return c

    def add_files(self, paths: Iterable[Path]) -> None:
        for p in sorted(paths, key=lambda x: x.name):
            text = p.read_text(encoding="utf-8")
            lines = [l for l in text.splitlines() if l.strip()]
            if not lines:
                continue
            headline = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            self.add_document(p.stem, headline, body)

    def add_document(self, docno: str, headline: str, body: str) -> None:
        if docno in self.docno_to_id:
            raise ValueError(f"Duplicate docno: {docno}")
        text = f"{headline}\n{body}"
        tokens = tokenize(text)
        term_counts: dict[int, int] = {}
        for tok in tokens:
            tid = self.lexicon.get(tok)
            if tid is None:
                tid = len(self.lexicon) + 1
                self.lexicon[tok] = tid
            term_counts[tid] = term_counts.get(tid, 0) + 1

        internal_id = len(self.docs)
        doc = Document(
            docno=docno, headline=headline, body=body,
            length=len(tokens), term_counts=term_counts,
        )
        self.docs.append(doc)
        self.docno_to_id[docno] = internal_id

        for tid, cnt in term_counts.items():
            self.inverted_index[tid].append((internal_id, cnt))

    # ----------------------------------------------------------------- query

    @property
    def avg_doc_length(self) -> float:
        if not self.docs:
            return 0.0
        return sum(d.length for d in self.docs) / len(self.docs)

    def idf(self, term_id: int) -> float:
        """Robertson/Spärck-Jones IDF — same form used in the LA Times engine."""
        N = len(self.docs)
        df = len(self.inverted_index.get(term_id, []))
        if df == 0:
            return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, term_ids: list[int], doc_id: int) -> float:
        doc = self.docs[doc_id]
        avgdl = self.avg_doc_length
        s = 0.0
        for tid in term_ids:
            tf = doc.term_counts.get(tid, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * doc.length / avgdl)
            s += self.idf(tid) * (tf * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        terms = tokenize(query)
        term_ids = [self.lexicon[t] for t in terms if t in self.lexicon]
        if not term_ids:
            return []
        # Candidate set = union of postings for the query terms (avoids scoring everything)
        candidates: set[int] = set()
        for tid in term_ids:
            candidates.update(d for d, _ in self.inverted_index.get(tid, []))
        scored = [(d, self.score(term_ids, d)) for d in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        hits = []
        for d, s in scored[:top_k]:
            doc = self.docs[d]
            hits.append(Hit(
                docno=doc.docno,
                headline=doc.headline,
                score=s,
                snippet=self._snippet(doc, terms),
                matched_terms=[t for t in terms if self.lexicon.get(t) in doc.term_counts],
            ))
        return hits

    def explain(self, query: str, doc_id: int) -> list[dict]:
        """Per-term BM25 contribution table — useful for debugging relevance."""
        terms = tokenize(query)
        doc = self.docs[doc_id]
        avgdl = self.avg_doc_length
        rows = []
        for t in terms:
            tid = self.lexicon.get(t)
            tf = doc.term_counts.get(tid, 0) if tid else 0
            idf_val = self.idf(tid) if tid else 0
            if tid is None or tf == 0:
                rows.append({"term": t, "tf": 0, "idf": idf_val,
                             "norm": 1.0, "score": 0.0})
                continue
            norm = 1 - self.b + self.b * doc.length / avgdl
            sc = idf_val * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            rows.append({"term": t, "tf": tf, "idf": round(idf_val, 4),
                         "norm": round(norm, 4), "score": round(sc, 4)})
        return rows

    # ----------------------------------------------------------- presentation

    def _snippet(self, doc: Document, terms: list[str], window: int = 25) -> str:
        body = doc.body
        body_tokens = tokenize(body)
        first = -1
        for i, tok in enumerate(body_tokens):
            if tok in terms:
                first = i
                break
        if first == -1:
            return body[:200] + ("..." if len(body) > 200 else "")
        start = max(0, first - window // 2)
        end = min(len(body_tokens), start + window)
        # Recover original-cased substring via approximate index search
        approx_chars = sum(len(t) + 1 for t in body_tokens[:start])
        snippet = body[approx_chars:approx_chars + 280]
        return ("..." if start > 0 else "") + snippet.strip() + ("..." if end < len(body_tokens) else "")

    def stats(self) -> dict:
        return {
            "documents": len(self.docs),
            "vocabulary": len(self.lexicon),
            "total_tokens": sum(d.length for d in self.docs),
            "avg_doc_length": round(self.avg_doc_length, 1),
            "min_doc_length": min((d.length for d in self.docs), default=0),
            "max_doc_length": max((d.length for d in self.docs), default=0),
        }
