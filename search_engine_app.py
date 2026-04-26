"""Interactive BM25 search dashboard.

Run with::

    streamlit run search_engine_app.py

Indexes the bundled `demo_corpus/` on launch (cached). Type a query, get
ranked hits with snippets, click into any hit to see the full document and
the per-term BM25 score breakdown that produced its rank.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from bm25 import Corpus

st.set_page_config(page_title="BM25 Search", layout="wide", page_icon="🔍")


# ---------------------------------------------------------------------------
# Cached index
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Indexing corpus...")
def load_corpus(path: str, k1: float, b: float) -> Corpus:
    return Corpus.from_directory(path, k1=k1, b=b)


# ---------------------------------------------------------------------------
# Sidebar — corpus + tunables
# ---------------------------------------------------------------------------


st.sidebar.header("Index")

corpus_path = st.sidebar.text_input("Corpus directory", "demo_corpus",
                                     help="Folder of .txt files. Each file's first line is the headline.")
if not Path(corpus_path).exists():
    st.sidebar.error(f"`{corpus_path}` not found. Run `python build_demo_corpus.py` to create the bundled corpus.")
    st.stop()

st.sidebar.subheader("BM25 parameters")
k1 = st.sidebar.slider("k₁ (term-frequency saturation)", 0.5, 3.0, 1.2, 0.1,
                        help="Higher = give more weight to repeated occurrences of a term.")
b = st.sidebar.slider("b (length normalisation)", 0.0, 1.0, 0.75, 0.05,
                       help="0 = no length normalisation; 1 = full length normalisation.")

corpus = load_corpus(corpus_path, k1, b)

st.sidebar.subheader("Index stats")
stats = corpus.stats()
st.sidebar.metric("Documents", stats["documents"])
st.sidebar.metric("Vocabulary", f"{stats['vocabulary']:,}")
st.sidebar.metric("Avg doc length", stats["avg_doc_length"])
st.sidebar.caption(f"Total tokens: {stats['total_tokens']:,}  ·  longest doc: {stats['max_doc_length']} tokens")


# ---------------------------------------------------------------------------
# Main panel — query + results
# ---------------------------------------------------------------------------


st.title("BM25 Search Engine")
st.caption("Full-text retrieval over a 30-article demo corpus, built from scratch in pure Python — no Lucene, no Whoosh, no third-party search library.")

with st.expander("How to use this app", expanded=False):
    st.markdown("""
**What this app does in plain English.**
This is a tiny search engine, like a private mini-Google over 30 short
news-style articles. Type a query, the engine ranks documents by how
relevant they are. The interesting part is *how* it ranks them — using
a formula called **BM25** that's been the gold standard of keyword
search since the 1990s and still powers most of Google, Elasticsearch,
and the search bar inside your apps.

**Quick start (30 seconds).**
1. Type a query into the search box, or click one of the **example
   queries** below it.
2. Read the ranked results. Each result shows a relevance score and a
   snippet from the document.
3. Expand any result to see the **per-term score breakdown** — exactly
   *why* this document scored higher than the next one.

**What the sliders mean (in the sidebar).**
- **k1** — how much extra credit to give for a word appearing many
  times. Higher k1 = a doc that mentions "battery" 10 times scores
  much higher than one that mentions it once. Default 1.2 is a
  long-standing sweet spot.
- **b** — how much to penalise long documents. b = 1.0 fully
  normalises by length (long docs get hit hardest); b = 0 ignores
  length. Default 0.75 is the standard.
- **Top-K results** — how many results to show.

**What the per-term breakdown shows.**
Each query word gets its own score for each document, based on:
- **tf** — how often that word appears in this document.
- **idf** — how rare the word is across ALL documents (rare words = more
  informative).
- **norm** — adjustment for document length.

The final BM25 score is the sum of all term scores. The breakdown lets
you audit the ranking — useful when a result feels surprising.

**Try this.** Search "electric vehicle battery". Then push **b** all the
way down to 0 and watch long documents climb up the ranking — that's
length normalisation in action.
""")

query = st.text_input("Query", placeholder="e.g. electric vehicle battery, fusion energy, supreme court")

if not query.strip():
    st.info("Try one of these example queries:")
    examples = [
        "electric vehicle battery breakthrough",
        "supreme court platform liability",
        "olympic swimming records",
        "carbon border adjustment europe",
        "AI ransomware hospital",
        "tesla price cuts",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["q"] = ex
            st.rerun()
    if "q" in st.session_state:
        query = st.session_state["q"]

if query.strip():
    top_k = st.slider("Show top K results", 1, 20, 10)
    hits = corpus.search(query, top_k=top_k)

    if not hits:
        st.warning("No documents matched any term in your query. Try simpler keywords.")
        st.stop()

    st.markdown(f"**{len(hits)} results** · {sum(corpus.lexicon.get(t, 0) is not None for t in query.lower().split())} of "
                f"{len(query.split())} query terms found in vocabulary")

    for rank, hit in enumerate(hits, 1):
        with st.expander(f"**{rank}. {hit.headline}**  ·  score {hit.score:.2f}  ·  matched {', '.join(hit.matched_terms) or '—'}"):
            st.markdown(f"**docno:** `{hit.docno}`")
            st.markdown(f"**snippet:** {hit.snippet}")
            internal_id = corpus.docno_to_id[hit.docno]
            doc = corpus.docs[internal_id]
            with st.container():
                col_doc, col_break = st.columns([1.5, 1])
                with col_doc:
                    st.markdown("**Full document**")
                    st.text(doc.body)
                with col_break:
                    st.markdown("**BM25 score breakdown**")
                    breakdown = corpus.explain(query, internal_id)
                    df = pd.DataFrame(breakdown)
                    st.dataframe(df, hide_index=True, use_container_width=True)
                    st.caption(
                        f"score = Σ idf · (tf · (k₁+1)) / (tf + k₁ · (1 − b + b · |D|/avgdl))  \n"
                        f"|D| = {doc.length} tokens, avgdl = {corpus.avg_doc_length:.1f}"
                    )

    # Aggregate visualisation
    st.divider()
    st.markdown("### Score distribution")
    chart = pd.DataFrame({
        "rank": range(1, len(hits) + 1),
        "score": [h.score for h in hits],
        "docno": [h.docno for h in hits],
    })
    st.bar_chart(chart, x="rank", y="score", x_label="rank", y_label="BM25 score")
