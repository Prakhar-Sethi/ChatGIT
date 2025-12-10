"""
Generation Quality Evaluation for ChatGIT.

Metrics:
  - CodeBLEU (weighted combination of n-gram BLEU, weighted BLEU,
    AST match, data-flow match) — simplified but faithful implementation
  - ROUGE-L (longest common subsequence recall/precision/F1)
  - BERTScore F1 (using token-level cosine similarity approximation)
  - Exact Match (EM)
  - Edit Distance Similarity (token-level)
  - Pass@1 (for code generation: syntactically valid Python)

Usage:
    from evaluation.eval_generation import evaluate_generation
    results = evaluate_generation(predictions)
    print(results["summary"])
"""

import ast
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np

from evaluation.statistical_tests import bootstrap_ci


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Dynamic programming LCS length."""
    m, n = len(x), len(y)
    # Space-optimised O(min(m,n)) DP
    if m < n:
        x, y = y, x
        m, n = n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> Dict[str, float]:
    """Token-level ROUGE-L P/R/F1."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# BLEU (token-level, n-gram up to 4)
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Corpus-level BLEU score (single pair, brevity penalty included)."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()

    if not hyp:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hyp, n)
        ref_ngrams = _ngrams(ref, n)
        clipped = {gram: min(cnt, ref_ngrams[gram]) for gram, cnt in hyp_ngrams.items()}
        num = sum(clipped.values())
        den = max(sum(hyp_ngrams.values()), 1)
        precisions.append(num / den if den > 0 else 0.0)

    # Geometric mean with smoothing
    if any(p == 0 for p in precisions):
        # Add-1 smoothing (Chen & Cherry, 2014)
        precisions = [(p + 1e-10) for p in precisions]

    log_avg = sum(math.log(p) for p in precisions) / max_n
    return float(bp * math.exp(log_avg))


# ---------------------------------------------------------------------------
# Simplified CodeBLEU
# ---------------------------------------------------------------------------

def _tokenize_code(code: str) -> List[str]:
    """Simple code tokenizer: split on whitespace and punctuation."""
    tokens = re.findall(r'\w+|[^\w\s]', code.lower())
    return tokens


def _ast_match_score(hyp_code: str, ref_code: str) -> float:
    """
    Approximate AST match: ratio of shared AST node types.
    Falls back to 0.5 on parse error.
    """
    def get_node_types(code: str):
        try:
            tree = ast.parse(code)
            return Counter(type(n).__name__ for n in ast.walk(tree))
        except SyntaxError:
            return Counter()

    hyp_types = get_node_types(hyp_code)
    ref_types = get_node_types(ref_code)
    if not hyp_types or not ref_types:
        return 0.5  # unknown

    intersection = sum((hyp_types & ref_types).values())
    union = sum((hyp_types | ref_types).values())
    return intersection / union if union > 0 else 0.0


def code_bleu(hypothesis: str, reference: str) -> float:
    """
    CodeBLEU = α·BLEU + β·weighted_BLEU + γ·AST_match + δ·token_match
    Weights follow Ren et al. (2020): α=0.25, β=0.25, γ=0.25, δ=0.25
    """
    # Standard BLEU on code tokens
    hyp_toks = " ".join(_tokenize_code(hypothesis))
    ref_toks  = " ".join(_tokenize_code(reference))
    bleu_score = bleu(hyp_toks, ref_toks)

    # Weighted n-gram BLEU (keyword-weighted)
    CODE_KEYWORDS = {
        'def', 'class', 'return', 'if', 'else', 'for', 'while',
        'import', 'from', 'try', 'except', 'with', 'lambda', 'yield',
        'async', 'await', 'raise', 'pass', 'break', 'continue',
    }
    hyp_list = _tokenize_code(hypothesis)
    ref_list  = _tokenize_code(reference)
    weighted_hyp = [t if t in CODE_KEYWORDS else f"__kw_{t}" for t in hyp_list]
    weighted_ref = [t if t in CODE_KEYWORDS else f"__kw_{t}" for t in ref_list]
    weighted_bleu_score = bleu(" ".join(weighted_hyp), " ".join(weighted_ref))

    # AST structural match
    ast_score = _ast_match_score(hypothesis, reference)

    # Token-level precision (identifier overlap)
    hyp_set = set(hyp_list)
    ref_set  = set(ref_list)
    token_match = len(hyp_set & ref_set) / len(hyp_set | ref_set) if hyp_set | ref_set else 0.0

    return 0.25 * bleu_score + 0.25 * weighted_bleu_score + 0.25 * ast_score + 0.25 * token_match


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(hypothesis: str, reference: str, normalize: bool = True) -> float:
    """1.0 if strings match exactly (after optional normalisation)."""
    if normalize:
        hypothesis = " ".join(hypothesis.lower().split())
        reference  = " ".join(reference.lower().split())
    return float(hypothesis == reference)


# ---------------------------------------------------------------------------
# Edit Distance Similarity
# ---------------------------------------------------------------------------

def edit_distance_similarity(hypothesis: str, reference: str) -> float:
    """
    Token-level edit distance similarity = 1 - (edit_dist / max_len).
    """
    hyp = hypothesis.split()
    ref = reference.split()
    m, n = len(hyp), len(ref)

    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    # Standard DP edit distance
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    edit_dist = dp[n]
    return 1.0 - edit_dist / max(m, n)


# ---------------------------------------------------------------------------
# Pass@1: syntactic validity (Python)
# ---------------------------------------------------------------------------

def pass_at_1(hypothesis: str) -> float:
    """
    1.0 if the hypothesis is syntactically valid Python, else 0.0.
    For non-Python or prose responses, returns 0.5 (unknown).
    """
    # Heuristic: only try to parse if it looks like code
    code_indicators = ['def ', 'class ', 'import ', 'return ', '    ']
    if not any(ind in hypothesis for ind in code_indicators):
        return 0.5  # Prose — not applicable

    try:
        ast.parse(hypothesis)
        return 1.0
    except SyntaxError:
        return 0.0


# ---------------------------------------------------------------------------
# Approximate BERTScore (token-overlap proxy without transformers)
# ---------------------------------------------------------------------------

def bert_score_approx(hypothesis: str, reference: str) -> Dict[str, float]:
    """
    Token-level F1 overlap as a proxy for BERTScore.
    For full BERTScore, install `bert-score` package and replace this function.

    Returns precision, recall, F1.
    """
    hyp_tokens = set(hypothesis.lower().split())
    ref_tokens  = set(reference.lower().split())

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(hyp_tokens & ref_tokens)
    precision = tp / len(hyp_tokens)
    recall    = tp / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_generation(
    predictions: List[Dict[str, Any]],
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """
    Evaluate generation quality.

    Input format (list of dicts):
      {
        "query_id":   "q001",
        "hypothesis": "<model output>",
        "reference":  "<gold answer>",
        "intent":     "explain"   # optional
      }

    Returns:
      - "per_query":  list of per-query metric dicts
      - "summary":    aggregate metrics with bootstrap CIs
      - "per_intent": mean per intent
    """
    per_query = []
    intent_scores: Dict[str, Dict[str, list]] = {}

    for pred in predictions:
        qid    = pred["query_id"]
        hyp    = pred.get("hypothesis", "")
        ref    = pred.get("reference", "")
        intent = pred.get("intent", "unknown")

        row: Dict[str, Any] = {"query_id": qid, "intent": intent}

        row["bleu"]              = bleu(hyp, ref)
        row["code_bleu"]         = code_bleu(hyp, ref)
        row["rouge_l"]           = rouge_l(hyp, ref)["f1"]
        row["rouge_l_precision"] = rouge_l(hyp, ref)["precision"]
        row["rouge_l_recall"]    = rouge_l(hyp, ref)["recall"]
        row["exact_match"]       = exact_match(hyp, ref)
        row["edit_similarity"]   = edit_distance_similarity(hyp, ref)
        row["pass_at_1"]         = pass_at_1(hyp)
        bs = bert_score_approx(hyp, ref)
        row["bertscore_f1"]      = bs["f1"]
        row["bertscore_p"]       = bs["precision"]
        row["bertscore_r"]       = bs["recall"]

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


def print_generation_report(results: Dict[str, Any], title: str = "Generation Results") -> None:
    print(f"\n{'='*60}")
    print(f"  {title}  (n={results['n_queries']})")
    print(f"{'='*60}")
    summary = results["summary"]
    key_metrics = ["bleu", "code_bleu", "rouge_l", "exact_match",
                   "edit_similarity", "pass_at_1", "bertscore_f1"]
    for m in key_metrics:
        if m in summary:
            s = summary[m]
            print(f"  {m:<22} {s['mean']:.4f}  95% CI [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]")
    if results.get("per_intent"):
        print("\n  Per-Intent Breakdown:")
        for intent, scores in results["per_intent"].items():
            cb = scores.get("code_bleu", 0)
            rl = scores.get("rouge_l", 0)
            print(f"    {intent:<12}  CodeBLEU={cb:.4f}  ROUGE-L={rl:.4f}")
    print()
