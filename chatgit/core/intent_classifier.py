"""
Query Intent Classifier for Granularity-Adaptive Retrieval — Novelty 4

Classifies each user query into one of four retrieval intents and returns
a RetrievalConfig that drives all downstream retrieval parameters:

  LOCATE   — find exact code location / definition
             → higher top-k, tight per-file cap, prefer small/statement chunks

  EXPLAIN  — understand a function or class
             → balanced top-k, function-level chunks, call neighbourhood ON

  SUMMARIZE — understand a module or the whole system architecture
             → lower top-k, larger per-file cap, prefer module_summary chunks

  DEBUG    — find bug / error source
             → highest top-k, call neighbourhood ON, relaxed diversity cap

Each intent maps to a RetrievalConfig dataclass that the chat endpoint
reads to configure vector search, reranking, and context assembly.
"""

import os
import re
from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    intent: str                   # locate | explain | summarize | debug
    top_k: int                    # vector search similarity_top_k
    rerank_n: int                 # cross-encoder output size
    max_per_file: int             # file-diversity cap
    include_neighborhood: bool    # attach call-graph neighborhood
    token_budget: int             # max tokens for context block
    granularity: str              # statement | function | module | mixed
    granularity_boost: float      # score multiplier for preferred chunk type


# ---------------------------------------------------------------------------
# Intent → retrieval configuration
# ---------------------------------------------------------------------------

_CONFIGS = {
    "locate": RetrievalConfig(
        intent="locate",
        top_k=25,
        rerank_n=6,
        max_per_file=2,
        include_neighborhood=False,
        token_budget=4000,
        granularity="statement",
        granularity_boost=1.40,
    ),
    "explain": RetrievalConfig(
        intent="explain",
        top_k=20,
        rerank_n=8,
        max_per_file=3,
        include_neighborhood=True,
        token_budget=6000,
        granularity="function",
        granularity_boost=1.20,
    ),
    "summarize": RetrievalConfig(
        intent="summarize",
        top_k=12,
        rerank_n=5,
        max_per_file=5,
        include_neighborhood=False,
        token_budget=7000,
        granularity="module",
        granularity_boost=1.50,
    ),
    "debug": RetrievalConfig(
        intent="debug",
        top_k=30,
        rerank_n=10,
        max_per_file=4,
        include_neighborhood=True,
        token_budget=6500,
        granularity="mixed",
        granularity_boost=1.10,
    ),
}

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_LOCATE_KW = [
    "find", "locate", "where is", "which file", "which line", "show me",
    "exact", "definition of", "declared", "defined", "implemented in",
    "what line", "where", "find the", "search for",
]
_EXPLAIN_KW = [
    "how does", "what does", "explain", "describe", "understand",
    "what is", "how is", "purpose of", "what are", "tell me about",
    "walk me through", "break down", "how do i", "what happens",
]
_SUMMARIZE_KW = [
    "overview", "architecture", "structure", "summarize", "summary",
    "high level", "big picture", "overall", "entire", "whole",
    "system design", "module", "components", "how do .* work together",
    "general", "codebase",
]
_DEBUG_KW = [
    "bug", "error", "exception", "crash", "fail", "wrong", "issue",
    "problem", "traceback", "fix", "broken", "not working", "debug",
    "incorrect", "why does .* fail", "why is .* not",
]

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> RetrievalConfig:
    """
    Classify a natural-language query into a retrieval intent.

    Strategy: try the trained TF-IDF + LinearSVC model first (97.35% CV
    accuracy on 604 labeled benchmark turns). Falls back to the keyword /
    regex classifier if the model file is unavailable.
    """
    neural = classify_intent_neural(query)
    if neural is not None:
        return neural

    # Keyword fallback
    q = query.lower()

    scores: dict[str, int] = {
        "locate":    _score(q, _LOCATE_KW),
        "explain":   _score(q, _EXPLAIN_KW),
        "summarize": _score(q, _SUMMARIZE_KW),
        "debug":     _score(q, _DEBUG_KW),
    }

    # Extra weight for regex patterns
    if re.search(r"how do .+ work(s)? together", q):
        scores["summarize"] += 2
    if re.search(r"why does .+ fail", q):
        scores["debug"] += 2
    if re.search(r"where is .+ (defined|declared|implemented)", q):
        scores["locate"] += 2

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "explain"  # safe default

    return _CONFIGS[best]


# ---------------------------------------------------------------------------
# Neural classifier — loaded once at module import, fallback to keywords
# ---------------------------------------------------------------------------

_CLF_PATH = os.path.join(os.path.dirname(__file__), "intent_clf.pkl")
_neural_clf = None

def _load_neural_clf():
    global _neural_clf
    if _neural_clf is not None:
        return _neural_clf
    if os.path.exists(_CLF_PATH):
        try:
            import pickle
            with open(_CLF_PATH, "rb") as f:
                _neural_clf = pickle.load(f)
        except Exception:
            _neural_clf = None
    return _neural_clf


def classify_intent_neural(query: str):
    """
    Classify intent using trained TF-IDF + LinearSVC model (97.35% CV accuracy).
    Returns None if model unavailable; caller should fall back to keyword classifier.
    """
    clf = _load_neural_clf()
    if clf is None:
        return None
    try:
        pred = clf.predict([query])[0]
        if pred in _CONFIGS:
            return _CONFIGS[pred]
    except Exception:
        pass
    return None


def _score(query: str, keywords: list) -> int:
    total = 0
    for kw in keywords:
        try:
            if re.search(kw, query):
                total += 1
        except re.error:
            if kw in query:
                total += 1
    return total
