"""
Retrieval Evaluation — MRR@k, Recall@k, NDCG@k, P@k, MAP
with bootstrap confidence intervals and Wilcoxon significance tests.

Input format (list of dicts):
    [
      {
        "query_id":     "q001",
        "retrieved":    ["file_a.py::fn1", "file_b.py::fn2", ...],  # ranked
        "ground_truth": ["file_a.py::fn1"],                          # relevant set
        "intent":       "locate"   # optional, for per-intent breakdown
      },
      ...
    ]

Usage:
    from evaluation.eval_retrieval import evaluate_retrieval
    results = evaluate_retrieval(predictions, ks=[1,5,10])
    print(results["summary"])
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def reciprocal_rank(retrieved: List[str], ground_truth: set) -> float:
    for i, item in enumerate(retrieved, 1):
        if item in ground_truth:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved: List[str], ground_truth: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in ground_truth)
    return hits / k


def recall_at_k(retrieved: List[str], ground_truth: set, k: int) -> float:
    if not ground_truth:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(retrieved: List[str], ground_truth: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(retrieved[:k], 1):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: all relevant docs ranked first
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: List[str], ground_truth: set) -> float:
    if not ground_truth:
        return 0.0
    hits, precision_sum = 0, 0.0
    for i, item in enumerate(retrieved, 1):
        if item in ground_truth:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(ground_truth)


def success_at_k(retrieved: List[str], ground_truth: set, k: int) -> float:
    """Binary: 1 if any relevant doc in top-k, else 0."""
    return float(any(item in ground_truth for item in retrieved[:k]))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(scores: np.ndarray, n_resamples: int = 10_000,
                 confidence: float = 0.95) -> Dict[str, float]:
    """Return mean and [lo, hi] bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return {"mean": float(scores.mean()), "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    predictions: List[Dict[str, Any]],
    ks: List[int] = [1, 5, 10],
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """
    Compute retrieval metrics for all predictions.

    Returns a dict with:
      - "per_query":  list of per-query metric dicts
      - "summary":    aggregate metrics with bootstrap CIs
      - "per_intent": breakdown by intent (if 'intent' key present)
    """
    per_query = []
    intent_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for pred in predictions:
        qid      = pred["query_id"]
        retrieved = pred.get("retrieved", [])
        gt_raw    = pred.get("ground_truth", [])
        gt        = set(gt_raw)
        intent    = pred.get("intent", "unknown")

        row: Dict[str, Any] = {"query_id": qid, "intent": intent}

        row["mrr"] = reciprocal_rank(retrieved, gt)
        row["map"] = average_precision(retrieved, gt)
        for k in ks:
            row[f"p@{k}"]       = precision_at_k(retrieved, gt, k)
            row[f"recall@{k}"]  = recall_at_k(retrieved, gt, k)
            row[f"ndcg@{k}"]    = ndcg_at_k(retrieved, gt, k)
            row[f"success@{k}"] = success_at_k(retrieved, gt, k)

        per_query.append(row)

        # Aggregate by intent
        for metric, val in row.items():
            if metric not in ("query_id", "intent"):
                intent_scores[intent][metric].append(val)

    # Aggregate summary
    all_scores: Dict[str, List[float]] = defaultdict(list)
    for row in per_query:
        for k, v in row.items():
            if k not in ("query_id", "intent"):
                all_scores[k].append(v)

    summary = {}
    for metric, vals in all_scores.items():
        arr = np.array(vals)
        summary[metric] = bootstrap_ci(arr, n_resamples=n_bootstrap)

    # Per-intent breakdown — mean + bootstrap CI for each metric
    per_intent = {}
    for intent, metrics in intent_scores.items():
        per_intent[intent] = {}
        for m, vals in metrics.items():
            arr = np.array(vals)
            if len(arr) >= 5:
                ci = bootstrap_ci(arr, n_resamples=n_bootstrap)
                per_intent[intent][m] = ci
            else:
                per_intent[intent][m] = {"mean": float(arr.mean()),
                                          "ci_lo": float(arr.min()),
                                          "ci_hi": float(arr.max())}

    return {
        "per_query":  per_query,
        "summary":    summary,
        "per_intent": per_intent,
        "n_queries":  len(predictions),
    }


def print_retrieval_report(results: Dict[str, Any], title: str = "Retrieval Results") -> None:
    print(f"\n{'='*60}")
    print(f"  {title}  (n={results['n_queries']})")
    print(f"{'='*60}")
    summary = results["summary"]

    key_metrics = [k for k in summary if "mrr" in k or "recall" in k or "ndcg" in k or "p@1" in k]
    for m in sorted(key_metrics):
        s = summary[m]
        print(f"  {m:<20} {s['mean']:.4f}  95% CI [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]")

    if results.get("per_intent"):
        print("\n  Per-Intent Breakdown:")
        for intent, scores in results["per_intent"].items():
            mrr_s = scores.get("mrr", {})
            r5_s  = scores.get("recall@5", {})
            mrr   = mrr_s.get("mean", mrr_s) if isinstance(mrr_s, dict) else float(mrr_s)
            r5    = r5_s.get("mean", r5_s)   if isinstance(r5_s,  dict) else float(r5_s)
            mrr_ci = mrr_s.get("ci_hi", mrr) - mrr_s.get("ci_lo", mrr) \
                     if isinstance(mrr_s, dict) else 0.0
            print(f"    {intent:<12} MRR={mrr:.4f}±{mrr_ci/2:.4f}  Recall@5={r5:.4f}")
    print()


# ---------------------------------------------------------------------------
# Compare two systems
# ---------------------------------------------------------------------------

def compare_systems(
    system_preds: List[Dict[str, Any]],
    baseline_preds: List[Dict[str, Any]],
    metric: str = "mrr",
    ks: List[int] = [1, 5, 10],
) -> Dict[str, Any]:
    """
    Compare a system against a baseline on a single metric.
    Returns Wilcoxon test result + effect size.
    """
    from evaluation.statistical_tests import wilcoxon_test, rank_biserial

    sys_res  = evaluate_retrieval(system_preds,   ks=ks, n_bootstrap=1000)
    base_res = evaluate_retrieval(baseline_preds, ks=ks, n_bootstrap=1000)

    qid_to_sys  = {r["query_id"]: r for r in sys_res["per_query"]}
    qid_to_base = {r["query_id"]: r for r in base_res["per_query"]}

    common = sorted(set(qid_to_sys) & set(qid_to_base))
    sys_scores  = np.array([qid_to_sys[q][metric]  for q in common])
    base_scores = np.array([qid_to_base[q][metric] for q in common])

    stat_result = wilcoxon_test(sys_scores, base_scores)
    effect      = rank_biserial(sys_scores, base_scores)

    return {
        "metric":          metric,
        "system_mean":     float(sys_scores.mean()),
        "baseline_mean":   float(base_scores.mean()),
        "delta":           float(sys_scores.mean() - base_scores.mean()),
        "p_value":         stat_result["p_value"],
        "significant":     stat_result["significant"],
        "effect_size_r":   effect,
        "n_compared":      len(common),
    }
