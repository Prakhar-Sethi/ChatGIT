"""
Statistical significance tests for ChatGIT evaluation.

Provides:
  - Wilcoxon signed-rank test (paired, non-parametric)
  - Bootstrap confidence intervals
  - Cohen's d effect size (parametric)
  - Rank-biserial correlation r (non-parametric effect size)
  - Fleiss' κ inter-annotator agreement
  - Bonferroni correction for multiple comparisons
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test  (no scipy dependency — pure numpy)
# ---------------------------------------------------------------------------

def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test for paired samples.

    Args:
        scores_a: metric scores for system A (shape: [n_queries])
        scores_b: metric scores for system B (shape: [n_queries])
        alpha:    significance level

    Returns dict with p_value, statistic, significant, n_pairs.
    Uses normal approximation for n >= 10 (accurate for n > 20).
    """
    assert len(scores_a) == len(scores_b), "Paired arrays must have equal length"
    diffs = scores_a - scores_b
    nonzero = diffs[diffs != 0]
    n = len(nonzero)

    if n == 0:
        return {"p_value": 1.0, "statistic": 0.0, "significant": False,
                "n_pairs": len(scores_a), "n_nonzero": 0}

    ranks = _rank_array(np.abs(nonzero))
    W_plus  = float(ranks[nonzero > 0].sum())
    W_minus = float(ranks[nonzero < 0].sum())
    W = min(W_plus, W_minus)

    # Normal approximation with continuity correction
    mean_W = n * (n + 1) / 4
    std_W  = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_W == 0:
        p_value = 1.0
    else:
        z = (W - mean_W) / std_W
        p_value = 2 * _standard_normal_cdf(z)   # two-tailed

    return {
        "p_value":     float(np.clip(p_value, 0, 1)),
        "statistic":   W,
        "W_plus":      W_plus,
        "W_minus":     W_minus,
        "n_pairs":     len(scores_a),
        "n_nonzero":   n,
        "significant": bool(p_value < alpha),
    }


def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Average-rank ties handling (like scipy.stats.rankdata)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)

    # Handle ties: replace with average
    sorted_arr = arr[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        if j > i + 1:  # tie group
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _standard_normal_cdf(z: float) -> float:
    """Φ(z) via math.erfc for numerical stability."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohen_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Cohen's d = (mean_a - mean_b) / pooled_std.
    Positive → system A better.
    """
    diff = float(scores_a.mean() - scores_b.mean())
    pooled_var = (scores_a.var(ddof=1) + scores_b.var(ddof=1)) / 2
    if pooled_var == 0:
        return 0.0
    return diff / math.sqrt(pooled_var)


def rank_biserial(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Rank-biserial correlation r for paired Wilcoxon test.
    r = (W+ - W-) / (n*(n+1)/2)
    Range: [-1, 1]. Positive → A tends to beat B.
    """
    diffs = scores_a - scores_b
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0
    ranks = _rank_array(np.abs(nonzero))
    W_plus  = float(ranks[nonzero > 0].sum())
    W_minus = float(ranks[nonzero < 0].sum())
    total = n * (n + 1) / 2
    return (W_plus - W_minus) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Bootstrap CI  (also used stand-alone)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Return mean and percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = (1 - confidence) / 2
    return {
        "mean":  float(scores.mean()),
        "ci_lo": float(np.percentile(means, 100 * alpha)),
        "ci_hi": float(np.percentile(means, 100 * (1 - alpha))),
        "std":   float(scores.std(ddof=1)),
    }


def bootstrap_diff_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for the difference (A - B)."""
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs.append(scores_a[idx].mean() - scores_b[idx].mean())
    diffs = np.array(diffs)
    alpha = (1 - confidence) / 2
    return {
        "mean_diff": float(scores_a.mean() - scores_b.mean()),
        "ci_lo":     float(np.percentile(diffs, 100 * alpha)),
        "ci_hi":     float(np.percentile(diffs, 100 * (1 - alpha))),
    }


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------

def bonferroni_correct(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    """
    Bonferroni correction for a list of p-values.
    Returns list of dicts with original p, corrected threshold, significant.
    """
    n = len(p_values)
    corrected_alpha = alpha / n if n > 0 else alpha
    return [
        {
            "p_value":          p,
            "corrected_alpha":  corrected_alpha,
            "significant":      bool(p < corrected_alpha),
        }
        for p in p_values
    ]


def holm_correct(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    """
    Holm-Bonferroni step-down correction (more powerful than Bonferroni).
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * n
    reject = True
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        if not reject or p > threshold:
            reject = False
        results[orig_idx] = {
            "p_value":    p,
            "threshold":  threshold,
            "rank":       rank + 1,
            "significant": reject,
        }
    return results


# ---------------------------------------------------------------------------
# Fleiss' κ (inter-annotator agreement)
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings: List[List[int]], n_categories: int) -> Dict[str, float]:
    """
    Fleiss' κ for multiple raters × multiple items.

    Args:
        ratings:      list of items; each item is a list of category labels
                      (one per rater). E.g. [[1,2,1], [3,3,2], ...]
        n_categories: total number of possible categories (1-indexed OK)

    Returns dict with kappa, p_o (observed agreement), p_e (expected).
    """
    n_items = len(ratings)
    if n_items == 0:
        return {"kappa": 0.0, "p_o": 0.0, "p_e": 0.0}

    n_raters = len(ratings[0])
    N = n_items * n_raters

    # Proportion of assignments to each category
    cat_counts = Counter()
    for item_ratings in ratings:
        for r in item_ratings:
            cat_counts[r] += 1
    p_j = {c: cat_counts[c] / N for c in range(1, n_categories + 1)}

    # P_i for each item
    p_i_sum = 0.0
    for item_ratings in ratings:
        item_cat = Counter(item_ratings)
        p_i = sum(c * (c - 1) for c in item_cat.values())
        p_i /= n_raters * (n_raters - 1) if n_raters > 1 else 1
        p_i_sum += p_i

    p_o = p_i_sum / n_items
    p_e = sum(v ** 2 for v in p_j.values())

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0.0
    return {"kappa": float(kappa), "p_o": float(p_o), "p_e": float(p_e)}


# ---------------------------------------------------------------------------
# Full system comparison report
# ---------------------------------------------------------------------------

def full_comparison_report(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_name: str = "metric",
    system_a_name: str = "System",
    system_b_name: str = "Baseline",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Comprehensive paired comparison: Wilcoxon + Cohen's d + r + bootstrap diff CI.
    """
    wil = wilcoxon_test(scores_a, scores_b, alpha=alpha)
    d   = cohen_d(scores_a, scores_b)
    r   = rank_biserial(scores_a, scores_b)
    ci  = bootstrap_diff_ci(scores_a, scores_b)

    magnitude = _effect_magnitude(abs(d))

    return {
        "metric":         metric_name,
        "system_a":       system_a_name,
        "system_b":       system_b_name,
        "mean_a":         float(scores_a.mean()),
        "mean_b":         float(scores_b.mean()),
        "delta":          float(scores_a.mean() - scores_b.mean()),
        "delta_ci_lo":    ci["ci_lo"],
        "delta_ci_hi":    ci["ci_hi"],
        "wilcoxon_p":     wil["p_value"],
        "significant":    wil["significant"],
        "cohen_d":        d,
        "rank_biserial_r": r,
        "effect_magnitude": magnitude,
        "n_pairs":        len(scores_a),
    }


def _effect_magnitude(abs_d: float) -> str:
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def print_comparison_table(reports: List[Dict[str, Any]]) -> None:
    """Pretty-print a list of full_comparison_report dicts."""
    header = f"{'Metric':<18} {'Δ':>8} {'95% CI':>20} {'p':>8} {'sig':>4} {'d':>7} {'r':>7} {'mag':<10}"
    print("\n" + "=" * 82)
    print(header)
    print("-" * 82)
    for r in reports:
        ci_str = f"[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]"
        sig_str = "*" if r["significant"] else ""
        print(
            f"{r['metric']:<18} {r['delta']:>+8.4f} {ci_str:>20} "
            f"{r['wilcoxon_p']:>8.4f} {sig_str:>4} "
            f"{r['cohen_d']:>7.3f} {r['rank_biserial_r']:>7.3f} {r['effect_magnitude']:<10}"
        )
    print("=" * 82 + "\n")
