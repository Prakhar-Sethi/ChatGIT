"""
Baseline Retrieval Systems for ChatGIT Comparison.

Implements the following baselines ChatGIT is evaluated against:

  1. BM25                — classic lexical baseline (Robertson & Zaragoza, 2009)
  2. VanillaRAG          — BGE embeddings + cosine similarity, no novelties
  3. BM25-SlidingWindow  — BM25 with one-round query augmentation from top snippet
                           (NOTE: this is NOT RepoCoder from Zhang et al. 2023;
                            RepoCoder is an iterative retrieval-generation loop
                            for code *completion*, a different task. This baseline
                            tests simple query augmentation on top of BM25.)
  4. ConvAwareRAG        — VanillaRAG + previous query appended to current query;
                           the simplest multi-turn-aware dense baseline; used to
                           isolate the contribution of N3 (session memory)
  5. GraphRAG-Code       — PageRank-weighted dense retrieval without QC-attention
                           or session memory (ablated N2)

Each baseline exposes:
    retrieve(query, chunks, k) -> List[str]  (ranked chunk IDs)

All implementations are self-contained (no external model calls) so they
can run deterministically in unit tests and ablation studies.
For a production comparison, swap the embedding functions for real models.
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np


# ===========================================================================
# BM25 Baseline
# ===========================================================================

class BM25:
    """
    Okapi BM25 lexical retrieval.
    Robertson & Zaragoza (2009) — The Probabilistic Relevance Framework.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self._corpus: List[List[str]] = []
        self._chunk_ids: List[str]    = []
        self._idf: Dict[str, float]   = {}
        self._avgdl: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def fit(self, chunks: List[Dict[str, Any]]) -> "BM25":
        """
        chunks: list of {"id": str, "text": str, ...}
        """
        self._corpus   = [self._tokenize(c["text"]) for c in chunks]
        self._chunk_ids = [c["id"] for c in chunks]
        self._avgdl     = np.mean([len(d) for d in self._corpus]) if self._corpus else 1.0

        N = len(self._corpus)
        df: Counter = Counter()
        for doc in self._corpus:
            for term in set(doc):
                df[term] += 1

        self._idf = {
            term: math.log((N - f + 0.5) / (f + 0.5) + 1)
            for term, f in df.items()
        }
        return self

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Returns list of (chunk_id, score) sorted descending."""
        q_terms = self._tokenize(query)
        scores = []
        for doc_idx, doc in enumerate(self._corpus):
            tf_map = Counter(doc)
            dl = len(doc)
            score = 0.0
            for term in q_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self._idf.get(term, 0.0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * num / den
            scores.append((self._chunk_ids[doc_idx], score))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def retrieve_ids(self, query: str, k: int = 10) -> List[str]:
        return [cid for cid, _ in self.retrieve(query, k)]

    def score_all(self, query: str) -> np.ndarray:
        """Return BM25 scores for all corpus documents as a numpy array."""
        q_terms = self._tokenize(query)
        scores = np.zeros(len(self._corpus), dtype=np.float32)
        for doc_idx, doc in enumerate(self._corpus):
            tf_map = Counter(doc)
            dl = len(doc)
            score = 0.0
            for term in q_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self._idf.get(term, 0.0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * num / den
            scores[doc_idx] = score
        return scores


# ===========================================================================
# TF-IDF Vector Baseline (lightweight stand-in for BGE when not available)
# ===========================================================================

class TFIDFRetriever:
    """
    TF-IDF cosine similarity retrieval.
    Stand-in VanillaRAG when embedding model is unavailable.
    """
    def __init__(self):
        self._chunk_ids: List[str] = []
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.array([])

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def fit(self, chunks: List[Dict[str, Any]]) -> "TFIDFRetriever":
        self._chunk_ids = [c["id"] for c in chunks]
        corpus = [self._tokenize(c["text"]) for c in chunks]
        N = len(corpus)

        # Build vocabulary
        all_terms = sorted({t for doc in corpus for t in doc})
        self._vocab = {t: i for i, t in enumerate(all_terms)}
        V = len(self._vocab)

        # TF matrix
        tf = np.zeros((N, V), dtype=np.float32)
        for di, doc in enumerate(corpus):
            cnt = Counter(doc)
            total = max(len(doc), 1)
            for term, freq in cnt.items():
                if term in self._vocab:
                    tf[di, self._vocab[term]] = freq / total

        # IDF vector
        df = (tf > 0).sum(axis=0)
        self._idf = np.log((N + 1) / (df + 1)) + 1.0

        self._tfidf_matrix = tf * self._idf
        # L2 normalise rows
        norms = np.linalg.norm(self._tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._tfidf_matrix /= norms
        return self

    def _query_vec(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        cnt = Counter(tokens)
        total = max(len(tokens), 1)
        V = len(self._vocab)
        vec = np.zeros(V, dtype=np.float32)
        for term, freq in cnt.items():
            if term in self._vocab:
                vec[self._vocab[term]] = (freq / total) * self._idf[self._vocab[term]]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q = self._query_vec(query)
        sims = self._tfidf_matrix @ q
        top_idx = np.argsort(-sims)[:k]
        return [(self._chunk_ids[i], float(sims[i])) for i in top_idx]

    def retrieve_ids(self, query: str, k: int = 10) -> List[str]:
        return [cid for cid, _ in self.retrieve(query, k)]


# ===========================================================================
# VanillaRAG  (dense retrieval only, no reranking, no novelties)
# ===========================================================================

class VanillaRAG:
    """
    Dense retrieval baseline using a provided embedding function.
    Corresponds to a plain RAG pipeline without any ChatGIT novelties.
    """
    def __init__(self, embed_fn: Optional[Callable[[str], np.ndarray]] = None):
        """
        embed_fn: callable str -> np.ndarray (unit-normalised).
        If None, falls back to TF-IDF.
        """
        self._embed_fn = embed_fn
        self._tfidf = TFIDFRetriever()
        self._chunk_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def fit(self, chunks: List[Dict[str, Any]]) -> "VanillaRAG":
        self._chunk_ids = [c["id"] for c in chunks]
        if self._embed_fn is not None:
            embs = np.stack([self._embed_fn(c["text"]) for c in chunks])
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embeddings = embs / norms
        else:
            self._tfidf.fit(chunks)
        return self

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if self._embed_fn is not None and self._embeddings is not None:
            q_emb = self._embed_fn(query)
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            sims  = self._embeddings @ q_emb
            top_idx = np.argsort(-sims)[:k]
            return [(self._chunk_ids[i], float(sims[i])) for i in top_idx]
        else:
            return self._tfidf.retrieve(query, k)

    def retrieve_ids(self, query: str, k: int = 10) -> List[str]:
        return [cid for cid, _ in self.retrieve(query, k)]


# ===========================================================================
# BM25-SlidingWindow Baseline
# ===========================================================================

class BM25SlidingWindow:
    """
    BM25 with one round of query augmentation from the top retrieved snippet.

    This tests the hypothesis that appending the top snippet to the query
    (a simple form of iterative retrieval) improves lexical retrieval.
    It is NOT an implementation of RepoCoder (Zhang et al., NeurIPS 2023),
    which is an iterative retrieval-generation loop for code *completion*,
    a fundamentally different task from conversational QA.

    We name this baseline honestly to avoid misattribution.
    """
    def __init__(self, base_retriever: Optional[Any] = None):
        self._retriever = base_retriever or BM25()

    def fit(self, chunks: List[Dict[str, Any]]) -> "BM25SlidingWindow":
        self._chunks_by_id = {c["id"]: c["text"] for c in chunks}
        self._retriever.fit(chunks)
        return self

    def retrieve_ids(self, query: str, k: int = 10) -> List[str]:
        # Round 1: raw BM25
        round1 = self._retriever.retrieve_ids(query, k=max(k, 5))
        if not round1:
            return []
        # Augment: append top snippet (first 300 chars) to query
        top_snippet = self._chunks_by_id.get(round1[0], "")[:300]
        augmented_query = query + " " + top_snippet
        # Round 2: re-retrieve with augmented query
        round2 = self._retriever.retrieve_ids(augmented_query, k=k)
        # Merge: prefer round2, fill gaps with round1
        seen: set = set()
        merged: List[str] = []
        for cid in round2 + round1:
            if cid not in seen:
                seen.add(cid)
                merged.append(cid)
        return merged[:k]


# Keep old name as alias for backward compatibility
RepoCoderStyle = BM25SlidingWindow


# ===========================================================================
# ConvAwareRAG Baseline  (novel — used to isolate N3 contribution)
# ===========================================================================

class ConvAwareRAG:
    """
    Conversationally-aware VanillaRAG: appends the previous turn's query
    to the current query before embedding.

    This is the simplest multi-turn-aware dense baseline.  It answers the
    question: "How much of ChatGIT's multi-turn gain comes from just
    remembering the last question?"

    Used to isolate the contribution of N3 (session-aware retrieval memory
    with redundancy penalties and zone coherence bonuses) over and above
    naive query concatenation.
    """
    def __init__(self, embed_fn: Optional[Callable[[str], np.ndarray]] = None):
        self._embed_fn  = embed_fn
        self._tfidf     = TFIDFRetriever()
        self._chunk_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def fit(self, chunks: List[Dict[str, Any]]) -> "ConvAwareRAG":
        self._chunk_ids = [c["id"] for c in chunks]
        if self._embed_fn is not None:
            embs  = np.stack([self._embed_fn(c["text"]) for c in chunks])
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embeddings = embs / norms
        else:
            self._tfidf.fit(chunks)
        return self

    def retrieve_ids(
        self,
        query: str,
        k: int = 10,
        prev_query: str = "",
    ) -> List[str]:
        """
        Retrieve with query optionally augmented by previous turn's query.
        Call with prev_query="" for turn 0, prev_query=last_query for turn >0.
        """
        augmented = f"{query} {prev_query}".strip() if prev_query else query
        if self._embed_fn is not None and self._embeddings is not None:
            q_emb = self._embed_fn(augmented)
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            sims  = self._embeddings @ q_emb
            top_idx = np.argsort(-sims)[:k]
            return [self._chunk_ids[i] for i in top_idx]
        else:
            return self._tfidf.retrieve_ids(augmented, k)


# ===========================================================================
# GraphRAG-Code Baseline  (PageRank only, no QC-attention or session memory)
# ===========================================================================

class GraphRAGCode:
    """
    Graph-augmented RAG using static PageRank reranking only.
    This is the ablated version of ChatGIT's N2 (without QC-attention)
    and represents the closest prior work (e.g. GraphCodeBERT-style retrieval).

    Reference: Guo et al. "GraphCodeBERT: Pre-Training Code Representations
    with Data Flow." ICLR 2021.
    """
    def __init__(
        self,
        base_retriever: Optional[Any] = None,
        pagerank_scores: Optional[Dict[str, float]] = None,
        pr_alpha: float = 0.3,
    ):
        self._retriever = base_retriever or TFIDFRetriever()
        self._pagerank  = pagerank_scores or {}
        self._pr_alpha  = pr_alpha

    def fit(self, chunks: List[Dict[str, Any]]) -> "GraphRAGCode":
        self._chunk_meta = {c["id"]: c for c in chunks}
        self._retriever.fit(chunks)
        return self

    def retrieve_ids(self, query: str, k: int = 10) -> List[str]:
        # Dense retrieval
        candidates = self._retriever.retrieve(query, k=k * 3)

        # Re-score with PageRank boost
        rescored = []
        for cid, sim in candidates:
            chunk = self._chunk_meta.get(cid, {})
            node_name = chunk.get("node_name", "")
            pr_score = self._pagerank.get(node_name, 0.0)
            combined = (1 - self._pr_alpha) * sim + self._pr_alpha * pr_score
            rescored.append((cid, combined))

        rescored.sort(key=lambda x: -x[1])
        return [cid for cid, _ in rescored[:k]]


# ===========================================================================
# Baseline registry  (for use by run_benchmark.py)
# ===========================================================================

BASELINE_REGISTRY: Dict[str, Any] = {
    "BM25":                BM25,
    "TF-IDF":              TFIDFRetriever,
    "VanillaRAG":          VanillaRAG,
    "ConvAwareRAG":        ConvAwareRAG,
    "BM25-SlidingWindow":  BM25SlidingWindow,
    "GraphRAG-Code":       GraphRAGCode,
    # kept for backward compatibility
    "RepoCoder":           BM25SlidingWindow,
}


def build_all_baselines(
    chunks: List[Dict[str, Any]],
    pagerank_scores: Optional[Dict[str, float]] = None,
    embed_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Instantiate and fit all baselines on a given chunk set.
    Returns dict of {name: fitted_retriever}.
    """
    bm25         = BM25().fit(chunks)
    tfidf        = TFIDFRetriever().fit(chunks)
    vanilla      = VanillaRAG(embed_fn=embed_fn).fit(chunks)
    conv_aware   = ConvAwareRAG(embed_fn=embed_fn).fit(chunks)
    bm25_sw      = BM25SlidingWindow(base_retriever=BM25()).fit(chunks)
    graphrag     = GraphRAGCode(
        base_retriever=TFIDFRetriever(),
        pagerank_scores=pagerank_scores or {},
    ).fit(chunks)

    return {
        "BM25":               bm25,
        "TF-IDF":             tfidf,
        "VanillaRAG":         vanilla,
        "ConvAwareRAG":       conv_aware,
        "BM25-SlidingWindow": bm25_sw,
        "GraphRAG-Code":      graphrag,
    }


def describe_baselines() -> str:
    """Return a formatted description of all baselines for paper writing."""
    return """
Baselines Used in ChatGIT Evaluation
=====================================

1. BM25 (Robertson & Zaragoza, 2009)
   - Classic Okapi BM25 lexical retrieval; k1=2.0, b=0.75
   - No semantic understanding; strong identifier-match baseline for code
   - Stateless: no session memory, same k for every query

2. VanillaRAG (BGE-small-en-v1.5)
   - BGE-small-en-v1.5 dense embeddings + cosine similarity top-k
   - No cross-encoder reranking, no graph signals, no session state
   - Standard off-the-shelf RAG pipeline; the closest prior-work comparison

3. ConvAwareRAG  [NEW — isolates N3 contribution]
   - VanillaRAG but with the previous turn's query appended to the current query
   - Simplest multi-turn-aware dense baseline
   - Tests whether N3's gains come from naive query augmentation vs.
     true session-aware scoring (redundancy penalties + zone coherence)

4. BM25-SlidingWindow
   - BM25 with one round of query augmentation: appends the top retrieved
     snippet (first 300 chars) to the query, then re-retrieves
   - Tests simple iterative augmentation over a lexical retriever
   - NOTE: distinct from RepoCoder (Zhang et al., NeurIPS 2023), which is
     an iterative retrieval-generation loop for code *completion*

5. GraphRAG-Code (static PageRank reranking)
   - Dense retrieval + multiplicative PageRank boost (alpha=0.3)
   - Ablated version of ChatGIT N2 (no query-conditioned attention)
   - Represents graph-augmented RAG without session state or intent routing

6. ChatGIT (Ours — Full System)
   - All 5 novelties: N1 (git volatility), N2 (hybrid PageRank+QC-attention),
     N3 (session-aware retrieval memory), N4 (intent-adaptive granularity),
     N5 (bidirectional call-context neighbourhood)
   - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2), diversity capping,
     intent-specific token-budget management
"""
