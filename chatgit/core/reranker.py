"""
Cross-encoder reranker for ChatGIT.

Uses sentence-transformers CrossEncoder to re-score query-document pairs
after initial top-k vector retrieval. This is the component described in the
report as "cross-encoder reranker" and is the main precision improvement.
"""

from typing import List, Any, Optional

_MODEL = None
_LOADED = False
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model():
    global _MODEL, _LOADED
    if _LOADED:
        return _MODEL
    try:
        from sentence_transformers import CrossEncoder
        print(f"[Reranker] Loading {_MODEL_NAME}...")
        _MODEL = CrossEncoder(_MODEL_NAME, max_length=512)
        print("[Reranker] Cross-encoder ready.")
    except Exception as e:
        print(f"[Reranker] Could not load cross-encoder ({e}). Reranking disabled.")
        _MODEL = None
    _LOADED = True
    return _MODEL


def rerank(query: str, candidates: List[Any], top_n: int = 8) -> List[Any]:
    """
    Re-rank retrieval candidates using cross-encoder scores.

    Args:
        query: The user query string.
        candidates: List of dicts, each with key 'snippet' (a LlamaIndex NodeWithScore).
        top_n: Number of results to return after reranking.

    Returns:
        Reranked list of candidates (length <= top_n).
    """
    model = _get_model()
    if model is None or not candidates:
        return candidates[:top_n]

    try:
        # Truncate document text to 800 chars for speed
        pairs = [(query, c['snippet'].text[:800]) for c in candidates]
        scores = model.predict(pairs)

        for i, c in enumerate(candidates):
            c['cross_encoder_score'] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x.get('cross_encoder_score', 0.0), reverse=True)
        print(f"[Reranker] Reranked {len(candidates)} -> returning top {top_n}")
        return reranked[:top_n]

    except Exception as e:
        print(f"[Reranker] Reranking failed ({e}), using original order.")
        return candidates[:top_n]
