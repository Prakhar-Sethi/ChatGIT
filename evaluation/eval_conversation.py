"""
Multi-Turn Conversation Quality Evaluation for ChatGIT (ConvCodeBench).

Metrics specific to conversational code Q&A:
  - Context Carry-Over Score (CCS): does the model leverage prior turns?
  - Coreference Accuracy: are pronouns/references resolved correctly?
  - Redundancy Rate: fraction of retrieved chunks already shown in prior turns
  - Turn Consistency Score: do answers across turns remain non-contradictory?
  - Session Coherence: overall semantic coherence across a conversation
  - Response Relevance: cosine similarity between question and answer tokens
"""

import math
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np

from evaluation.statistical_tests import bootstrap_ci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union)


def _token_overlap(text_a: str, text_b: str) -> float:
    toks_a = set(text_a.lower().split())
    toks_b = set(text_b.lower().split())
    return _jaccard(toks_a, toks_b)


# ---------------------------------------------------------------------------
# Context Carry-Over Score
# ---------------------------------------------------------------------------

def context_carry_over(
    conversation: List[Dict[str, Any]],
) -> float:
    """
    Measures whether answers in turn t reference entities/identifiers
    introduced in turns < t.

    conversation: list of turns, each with:
      - "query":   user question
      - "answer":  model answer
      - "retrieved_chunks": list of chunk IDs used (optional)

    Returns: CCS in [0, 1]. 1 = perfect carry-over.
    """
    if len(conversation) <= 1:
        return 1.0  # Single turn — trivially carries over

    # Collect all entity tokens from prior turns (simple: all capitalised words)
    cumulative_entities: set = set()
    scores = []

    for i, turn in enumerate(conversation):
        query  = turn.get("query", "")
        answer = turn.get("answer", "")

        if i == 0:
            # First turn: just accumulate
            cumulative_entities |= _extract_identifiers(query + " " + answer)
            continue

        # Fraction of the answer's identifiers that come from prior context
        answer_ids = _extract_identifiers(answer)
        if answer_ids:
            overlap = len(answer_ids & cumulative_entities) / len(answer_ids)
        else:
            overlap = 0.0
        scores.append(overlap)
        cumulative_entities |= _extract_identifiers(query + " " + answer)

    return float(np.mean(scores)) if scores else 1.0


def _extract_identifiers(text: str) -> set:
    """Extract code-like identifiers: snake_case, camelCase, ALL_CAPS."""
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', text)
    return {t.lower() for t in tokens}


# ---------------------------------------------------------------------------
# Coreference Accuracy
# ---------------------------------------------------------------------------

def coreference_accuracy(
    conversation: List[Dict[str, Any]],
    ground_truth_refs: Optional[List[Dict[str, str]]] = None,
) -> float:
    """
    Measures whether the model correctly resolved coreferences.

    If ground_truth_refs is provided (list of {pronoun, referent, answer_contains}),
    checks exact match against gold. Otherwise, uses a heuristic:
    pronouns (it/they/this/that) in queries should be responded to with
    tokens from the immediately preceding turn.

    Returns accuracy in [0, 1].
    """
    if ground_truth_refs is not None:
        # Gold labels provided
        correct = sum(
            1 for r in ground_truth_refs
            if r.get("referent", "").lower() in r.get("answer", "").lower()
        )
        return correct / len(ground_truth_refs) if ground_truth_refs else 1.0

    # Heuristic
    PRONOUNS = re.compile(r'\b(it|they|this|that|these|those|its|their)\b', re.I)
    scores = []

    for i, turn in enumerate(conversation[1:], 1):
        query = turn.get("query", "")
        if not PRONOUNS.search(query):
            continue  # No pronoun — skip

        prev_answer = conversation[i - 1].get("answer", "")
        curr_answer = turn.get("answer", "")

        # Overlap between current answer and previous answer (referent context)
        overlap = _token_overlap(curr_answer, prev_answer)
        scores.append(overlap)

    return float(np.mean(scores)) if scores else 1.0


# ---------------------------------------------------------------------------
# Redundancy Rate
# ---------------------------------------------------------------------------

def redundancy_rate(
    conversation: List[Dict[str, Any]],
) -> float:
    """
    Fraction of retrieved chunks (across all turns) that were already
    retrieved in a prior turn within the same session.

    conversation turns need "retrieved_chunks": [chunk_id, ...]
    Returns rate in [0, 1]. Lower is better.
    """
    seen: set = set()
    total = 0
    redundant = 0

    for turn in conversation:
        chunks = turn.get("retrieved_chunks", [])
        for chunk in chunks:
            total += 1
            if chunk in seen:
                redundant += 1
            seen.add(chunk)

    return redundant / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Turn Consistency Score
# ---------------------------------------------------------------------------

def turn_consistency(
    conversation: List[Dict[str, Any]],
) -> float:
    """
    Detects contradictions between answers across turns.

    Heuristic: for each pair of answers (i, j), if the overlap of their
    content is high (> 0.3) but they contain opposing keywords, flag as
    potential contradiction.

    Returns consistency score in [0, 1]. 1 = no contradictions detected.
    """
    NEGATION_PAIRS = [
        (r'\bnot\b', r'\byes\b'),
        (r'\bfalse\b', r'\btrue\b'),
        (r'\bdoes not\b', r'\bdoes\b'),
        (r'\bcannot\b', r'\bcan\b'),
        (r'\bno\b', r'\byes\b'),
    ]
    answers = [t.get("answer", "") for t in conversation]
    n = len(answers)
    if n <= 1:
        return 1.0

    inconsistent = 0
    pairs_checked = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            a1, a2 = answers[i].lower(), answers[j].lower()
            overlap = _token_overlap(a1, a2)
            if overlap < 0.15:
                continue  # Likely different topics — skip
            pairs_checked += 1
            for neg_pat, pos_pat in NEGATION_PAIRS:
                a1_neg = bool(re.search(neg_pat, a1))
                a2_pos = bool(re.search(pos_pat, a2))
                if a1_neg and a2_pos and overlap > 0.25:
                    inconsistent += 1
                    break

    if pairs_checked == 0:
        return 1.0
    return 1.0 - inconsistent / pairs_checked


# ---------------------------------------------------------------------------
# Session Coherence
# ---------------------------------------------------------------------------

def session_coherence(conversation: List[Dict[str, Any]]) -> float:
    """
    Average token overlap between consecutive (query, answer) pairs.
    High coherence = answers are on-topic relative to the question.
    """
    scores = []
    for turn in conversation:
        q = turn.get("query", "")
        a = turn.get("answer", "")
        if q and a:
            scores.append(_token_overlap(q, a))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Response Relevance
# ---------------------------------------------------------------------------

def response_relevance(query: str, answer: str) -> float:
    """
    Token-overlap relevance between a single query and its answer.
    """
    return _token_overlap(query, answer)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_conversation(
    sessions: List[List[Dict[str, Any]]],
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """
    Evaluate multi-turn conversation quality over a list of sessions.

    Each session is a list of turns:
      {
        "query":            str,
        "answer":           str,
        "retrieved_chunks": [str, ...]   # optional chunk IDs
      }

    Returns per_session metrics + aggregate summary.
    """
    per_session = []

    for sid, session in enumerate(sessions):
        row: Dict[str, Any] = {"session_id": sid, "n_turns": len(session)}

        row["context_carry_over"]  = context_carry_over(session)
        row["coreference_accuracy"] = coreference_accuracy(session)
        row["redundancy_rate"]      = redundancy_rate(session)
        row["turn_consistency"]     = turn_consistency(session)
        row["session_coherence"]    = session_coherence(session)

        # Per-turn response relevance
        relevances = [
            response_relevance(t.get("query", ""), t.get("answer", ""))
            for t in session
        ]
        row["avg_response_relevance"] = float(np.mean(relevances)) if relevances else 0.0

        per_session.append(row)

    # Aggregate
    all_scores: Dict[str, list] = defaultdict(list)
    for row in per_session:
        for k, v in row.items():
            if k not in ("session_id", "n_turns"):
                all_scores[k].append(v)

    summary = {}
    for metric, vals in all_scores.items():
        arr = np.array(vals)
        summary[metric] = bootstrap_ci(arr, n_resamples=n_bootstrap)

    return {
        "per_session": per_session,
        "summary":     summary,
        "n_sessions":  len(sessions),
        "n_turns_total": sum(s["n_turns"] for s in per_session),
    }


def print_conversation_report(results: Dict[str, Any],
                               title: str = "Conversation Results") -> None:
    print(f"\n{'='*60}")
    print(f"  {title}  (sessions={results['n_sessions']}, turns={results['n_turns_total']})")
    print(f"{'='*60}")
    key = ["context_carry_over", "coreference_accuracy", "redundancy_rate",
           "turn_consistency", "session_coherence", "avg_response_relevance"]
    summary = results["summary"]
    for m in key:
        if m in summary:
            s = summary[m]
            arrow = "↓" if "redundancy" in m else "↑"
            print(f"  {m:<28} {s['mean']:.4f}  {arrow}  95% CI [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]")
    print()
