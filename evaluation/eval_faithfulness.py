"""
Faithfulness / Hallucination Evaluation for ChatGIT.

Inspired by RAGAS (Es et al., 2023) and TruLens.

Metrics:
  - Faithfulness Score: fraction of claims in the answer that are
    grounded in the retrieved context
  - Context Precision: fraction of context chunks that are relevant
    to the ground-truth answer
  - Context Recall: fraction of ground-truth information present
    in the retrieved context
  - Answer Groundedness: token-level overlap of answer with context
  - Hallucination Rate: fraction of answer tokens NOT in context

Usage:
    from evaluation.eval_faithfulness import evaluate_faithfulness
    results = evaluate_faithfulness(predictions)
"""

import re
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np

from evaluation.statistical_tests import bootstrap_ci


# ---------------------------------------------------------------------------
# Claim extraction (simple sentence-level)
# ---------------------------------------------------------------------------

def _split_claims(text: str) -> List[str]:
    """Split a response into atomic claims (sentences)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _token_set(text: str) -> set:
    return set(re.findall(r'\w+', text.lower()))


def _token_overlap_score(text: str, context: str) -> float:
    """Fraction of text tokens present in context."""
    t_tokens = _token_set(text)
    c_tokens = _token_set(context)
    if not t_tokens:
        return 0.0
    return len(t_tokens & c_tokens) / len(t_tokens)


# ---------------------------------------------------------------------------
# Faithfulness Score
# ---------------------------------------------------------------------------

def faithfulness_score(
    answer: str,
    retrieved_context: str,
    threshold: float = 0.4,
) -> float:
    """
    Faithfulness = (# claims grounded in context) / (# total claims).

    A claim is considered grounded if its token overlap with the
    retrieved context exceeds `threshold`.

    Returns score in [0, 1]. 1 = fully faithful.
    """
    claims = _split_claims(answer)
    if not claims:
        return 1.0  # No claims to check

    grounded = sum(
        1 for claim in claims
        if _token_overlap_score(claim, retrieved_context) >= threshold
    )
    return grounded / len(claims)


# ---------------------------------------------------------------------------
# Context Precision
# ---------------------------------------------------------------------------

def context_precision(
    context_chunks: List[str],
    ground_truth: str,
    threshold: float = 0.25,
) -> float:
    """
    Fraction of retrieved chunks that contain information relevant to
    the ground-truth answer.

    Relevance is measured by token overlap > threshold.
    """
    if not context_chunks:
        return 0.0
    relevant = sum(
        1 for chunk in context_chunks
        if _token_overlap_score(ground_truth, chunk) >= threshold
    )
    return relevant / len(context_chunks)


# ---------------------------------------------------------------------------
# Context Recall
# ---------------------------------------------------------------------------

def context_recall(
    context_chunks: List[str],
    ground_truth: str,
    threshold: float = 0.3,
) -> float:
    """
    Fraction of ground-truth claims that can be attributed to
    at least one retrieved chunk.

    Returns score in [0, 1]. 1 = all ground-truth info is present.
    """
    gt_claims = _split_claims(ground_truth)
    if not gt_claims:
        return 1.0

    combined_context = " ".join(context_chunks)
    covered = sum(
        1 for claim in gt_claims
        if _token_overlap_score(claim, combined_context) >= threshold
    )
    return covered / len(gt_claims)


# ---------------------------------------------------------------------------
# Answer Groundedness
# ---------------------------------------------------------------------------

def answer_groundedness(answer: str, context: str) -> float:
    """
    Token-level fraction of the answer that is grounded in context.
    More lenient than faithfulness (no claim segmentation).
    """
    return _token_overlap_score(answer, context)


# ---------------------------------------------------------------------------
# Hallucination Rate
# ---------------------------------------------------------------------------

def hallucination_rate(answer: str, context: str) -> float:
    """
    Fraction of answer tokens NOT present anywhere in the context.
    Lower is better.
    """
    a_tokens = _token_set(answer)
    c_tokens = _token_set(context)
    if not a_tokens:
        return 0.0
    hallucinated = a_tokens - c_tokens
    return len(hallucinated) / len(a_tokens)


# ---------------------------------------------------------------------------
# Answer Relevance (question → answer)
# ---------------------------------------------------------------------------

def answer_relevance(question: str, answer: str) -> float:
    """
    Token overlap between question and answer as a proxy for relevance.
    Higher is better.
    """
    return _token_overlap_score(question, answer)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_faithfulness(
    predictions: List[Dict[str, Any]],
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """
    Evaluate faithfulness and grounding of model answers.

    Input format:
      {
        "query_id":         "q001",
        "question":         str,
        "answer":           str,
        "ground_truth":     str,        # gold answer
        "context_chunks":   [str, ...], # list of retrieved context strings
        "intent":           "explain"   # optional
      }

    Returns per_query metrics + aggregate summary with bootstrap CIs.
    """
    per_query = []
    intent_scores: Dict[str, Dict[str, list]] = {}

    for pred in predictions:
        qid      = pred["query_id"]
        question = pred.get("question", "")
        answer   = pred.get("answer", "")
        gt       = pred.get("ground_truth", "")
        chunks   = pred.get("context_chunks", [])
        intent   = pred.get("intent", "unknown")
        context  = " ".join(chunks)

        row: Dict[str, Any] = {"query_id": qid, "intent": intent}

        row["faithfulness"]       = faithfulness_score(answer, context)
        row["context_precision"]  = context_precision(chunks, gt)
        row["context_recall"]     = context_recall(chunks, gt)
        row["answer_groundedness"] = answer_groundedness(answer, context)
        row["hallucination_rate"] = hallucination_rate(answer, context)
        row["answer_relevance"]   = answer_relevance(question, answer)

        # RAGAS-style combined score
        row["ragas_score"] = float(np.mean([
            row["faithfulness"],
            row["context_precision"],
            row["context_recall"],
            row["answer_relevance"],
        ]))

        per_query.append(row)

        if intent not in intent_scores:
            intent_scores[intent] = {}
        for k, v in row.items():
            if k not in ("query_id", "intent"):
                intent_scores[intent].setdefault(k, []).append(v)

    # Aggregate
    all_scores: Dict[str, list] = {}
    for row in per_query:
        for k, v in row.items():
            if k not in ("query_id", "intent"):
                all_scores.setdefault(k, []).append(v)

    summary = {}
    for metric, vals in all_scores.items():
        summary[metric] = bootstrap_ci(np.array(vals), n_resamples=n_bootstrap)

    per_intent = {
        intent: {m: float(np.mean(vals)) for m, vals in metrics.items()}
        for intent, metrics in intent_scores.items()
    }

    return {
        "per_query":  per_query,
        "summary":    summary,
        "per_intent": per_intent,
        "n_queries":  len(predictions),
    }


def print_faithfulness_report(results: Dict[str, Any],
                               title: str = "Faithfulness Results") -> None:
    print(f"\n{'='*60}")
    print(f"  {title}  (n={results['n_queries']})")
    print(f"{'='*60}")
    key = ["faithfulness", "context_precision", "context_recall",
           "answer_groundedness", "hallucination_rate", "answer_relevance",
           "ragas_score"]
    summary = results["summary"]
    for m in key:
        if m in summary:
            s = summary[m]
            arrow = "↓" if "hallucination" in m else "↑"
            print(f"  {m:<26} {s['mean']:.4f}  {arrow}  95% CI [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]")
    print()
