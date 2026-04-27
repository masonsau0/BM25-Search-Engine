# BM25 Search Engine

**[Live demo](https://mason-bm25-search-engine.streamlit.app/)** : runs in the browser, no install required.

A from-scratch information-retrieval system implementing the **Okapi BM25**
ranking model : no Lucene, no Whoosh, no third-party search library. The
tokenizer, inverted index, and scoring formula are all built from first
principles in pure Python, exposed as both a tidy library API and an
interactive search dashboard.

The bundled **30-article demo corpus** lets the project run out of the box;
the same `Corpus` class will index any folder of `.txt` files with no code
changes.

## What BM25 actually does

For a document `D` and a query `Q = {q₁, …, qₙ}`, BM25 scores

```
score(D, Q) = Σᵢ IDF(qᵢ) · [ tf(qᵢ, D) · (k₁ + 1) ] / [ tf(qᵢ, D) + k₁ · (1 − b + b · |D| / avgdl) ]
```

where

| Symbol | Meaning |
|---|---|
| `tf(qᵢ, D)` | times term `qᵢ` appears in document `D` |
| `IDF(qᵢ)` | inverse document frequency: `log((N − df + 0.5) / (df + 0.5) + 1)` |
| `\|D\|` | length of document `D` in tokens |
| `avgdl` | average document length across the corpus |
| `k₁` | term-frequency saturation (default 1.2) |
| `b` | length-normalisation strength (default 0.75) |

The dashboard exposes `k₁` and `b` as sliders so you can see how the
ranking responds to tuning.

## Repository layout

```
.
├── bm25.py                  ← self-contained Corpus + BM25 implementation
├── search_engine_app.py     ← Streamlit search dashboard
├── build_demo_corpus.py     ← generates the 30-article demo corpus
├── demo_corpus/             ← 30 short news-style articles (one .txt per doc)
├── requirements.txt
└── README.md
```

## Run it

```bash
pip install -r requirements.txt
python build_demo_corpus.py            # writes demo_corpus/*.txt
streamlit run search_engine_app.py
```

The dashboard:

- **Index stats** : number of documents, vocabulary size, average document length, total tokens.
- **BM25 tunables** : sliders for `k₁` and `b` so you can see the ranking shift in real time.
- **Quick-start example queries** : buttons that pre-populate the search box with sensible queries against the demo corpus.
- **Per-result expandable view** : full document text plus a per-term BM25 score breakdown showing exactly why this document scored higher than the next.
- **Score distribution chart** : bar chart of the top-K scores so you can spot a steep cutoff vs. a flat tail.

### Programmatic use

```python
from bm25 import Corpus

corpus = Corpus.from_directory("demo_corpus", k1=1.2, b=0.75)
hits = corpus.search("electric vehicle battery", top_k=5)
for h in hits:
    print(f"{h.score:.2f}  {h.headline}")
    print(f"  {h.snippet}")
```

To index a different document set, drop your own `.txt` files in a folder
and point `Corpus.from_directory()` at it : the rest of the pipeline (index
build, query, score breakdown) works unchanged.

## Stack

Python (standard library + `pandas` for tabular display) · **Streamlit**
(dashboard). No third-party search/IR library : the engine is built from
first principles.
