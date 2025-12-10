"""
Ablation Study Framework for ChatGIT.

Toggles each novelty independently to measure individual contribution:

  N1 — Git-History Volatility Weighting
  N2 — Hybrid PageRank + QC-Graph Attention
  N3 — Multi-Turn Session-Aware Retrieval Memory
  N4 — Intent-Driven Granularity-Adaptive Retrieval
  N5 — Bidirectional Call-Context Neighbourhood Augmentation

Ablation configurations:
  - Full system (all N1-N5 ON)
  - Remove one novelty at a time (-N1, -N2, ..., -N5)
  - Remove all novelties = VanillaRAG equivalent

Usage:
    from evaluation.ablation import AblationConfig, run_ablation_study
    results = run_ablation_study(predictions_by_config, ground_truth)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from evaluation.eval_retrieval import evaluate_retrieval
from evaluation.eval_generation import evaluate_generation
from evaluation.statistical_tests import full_comparison_report, print_comparison_table


# ---------------------------------------------------------------------------
# Ablation Configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    Flags controlling which novelties are active.
    All True = full ChatGIT system.
    """
    name: str = "Full"

    # N1: Git-history volatility weighting
    use_volatility: bool = True

    # N2: Hybrid PageRank + QC-attention
    use_hybrid_importance: bool = True

    # N3: Session-aware retrieval memory
    use_session_memory: bool = True

    # N4: Intent-adaptive granularity routing
    use_intent_routing: bool = True

    # N5: Call-context neighbourhood augmentation
    use_neighborhood: bool = True

    # Meta
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Predefined ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: List[AblationConfig] = [
    # ── Incremental build-up (bottom-up ablation) ──────────────────────────────
    # Row 1: Vanilla baseline — all novelties off, same as VanillaRAG
    AblationConfig(
        name="Vanilla",
        description="All novelties disabled — VanillaRAG equivalent",
        use_volatility=False, use_hybrid_importance=False,
        use_session_memory=False, use_intent_routing=False,
        use_neighborhood=False,
    ),
    # Row 2: N3 alone — session memory only, fixed retrieval parameters
    AblationConfig(
        name="N3 only",
        description="Session-aware retrieval memory only; no intent routing, no graph signals",
        use_volatility=False, use_hybrid_importance=False,
        use_session_memory=True, use_intent_routing=False,
        use_neighborhood=False,
    ),
    # Row 3: N4 alone — intent routing only, no session memory
    AblationConfig(
        name="N4 only",
        description="Intent-adaptive granularity routing only; no session memory, no graph signals",
        use_volatility=False, use_hybrid_importance=False,
        use_session_memory=False, use_intent_routing=True,
        use_neighborhood=False,
    ),
    # Row 4: N3+N4 together — the confirmed session+intent base
    AblationConfig(
        name="N3+N4",
        description="Session memory + intent routing — the conversational base system",
        use_volatility=False, use_hybrid_importance=False,
        use_session_memory=True, use_intent_routing=True,
        use_neighborhood=False,
    ),
    # Row 5: Add N1 on top of N3+N4
    AblationConfig(
        name="N3+N4+N1",
        description="Add git-history volatility weighting over N3+N4 base",
        use_volatility=True, use_hybrid_importance=False,
        use_session_memory=True, use_intent_routing=True,
        use_neighborhood=False,
    ),
    # Row 6: Add N2 on top of N3+N4
    AblationConfig(
        name="N3+N4+N2",
        description="Add hybrid PageRank+QC-attention over N3+N4 base",
        use_volatility=False, use_hybrid_importance=True,
        use_session_memory=True, use_intent_routing=True,
        use_neighborhood=False,
    ),
    # Row 7: Add N5 on top of N3+N4 — the biggest single gain
    AblationConfig(
        name="N3+N4+N5",
        description="Add bidirectional call-context neighbourhood over N3+N4 base",
        use_volatility=False, use_hybrid_importance=False,
        use_session_memory=True, use_intent_routing=True,
        use_neighborhood=True,
    ),
    # Row 8: Full system — all novelties active
    AblationConfig(
        name="Full",
        description="All novelties active — ChatGIT full system (N1+N2+N3+N4+N5)",
        use_volatility=True, use_hybrid_importance=True,
        use_session_memory=True, use_intent_routing=True,
        use_neighborhood=True,
    ),
]

# Convenience: original subtract-one configs kept for backward compatibility
SUBTRACT_ONE_CONFIGS: List[AblationConfig] = [
    AblationConfig(
        name="-N1",
        description="Full system minus git-history volatility weighting",
        use_volatility=False, use_hybrid_importance=True,
        use_session_memory=True, use_intent_routing=True,
        use_neighborhood=True,
    ),
    AblationConfig(
        name="-N2",
        description="Full system minus hybrid PageRank+QC-attention",
        use_volatility=True, use_hybrid_importance=False,
        use_session_memory=True, use_intent_routing=True,
        use_neighborhood=True,
    ),
    AblationConfig(
        name="-N3",
        description="Full system minus session-aware retrieval memory",
        use_volatility=True, use_hybrid_importance=True,
        use_session_memory=False, use_intent_routing=True,
        use_neighborhood=True,
    ),
    AblationConfig(
        name="-N4",
        description="Full system minus intent-adaptive granularity routing",
        use_volatility=True, use_hybrid_importance=True,
        use_session_memory=True, use_intent_routing=False,
        use_neighborhood=True,
    ),
    AblationConfig(
        name="-N5",
        description="Full system minus bidirectional call-context neighbourhood",
        use_volatility=True, use_hybrid_importance=True,
        use_session_memory=True, use_intent_routing=True,
        use_neighborhood=False,
    ),
]


# ---------------------------------------------------------------------------
# Ablation result storage
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    config: AblationConfig
    retrieval_summary: Dict[str, Any]
    generation_summary: Optional[Dict[str, Any]] = None
    n_queries: int = 0

    def key_metrics(self) -> Dict[str, float]:
        """Extract the most important scalar metrics."""
        ret = self.retrieval_summary
        gen = self.generation_summary or {}
        return {
            "mrr":        ret.get("mrr", {}).get("mean", 0),
            "recall@5":   ret.get("recall@5", {}).get("mean", 0),
            "ndcg@5":     ret.get("ndcg@5", {}).get("mean", 0),
            "code_bleu":  gen.get("code_bleu", {}).get("mean", 0),
            "rouge_l":    gen.get("rouge_l", {}).get("mean", 0),
        }


# ---------------------------------------------------------------------------
# Run ablation
# ---------------------------------------------------------------------------

def run_ablation_study(
    predictions_by_config: Dict[str, List[Dict[str, Any]]],
    gen_predictions_by_config: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ks: List[int] = [1, 5, 10],
    primary_metric: str = "mrr",
) -> List[AblationResult]:
    """
    Run ablation study.

    Args:
        predictions_by_config: dict mapping config name -> list of retrieval predictions
            Each prediction: {"query_id", "retrieved": [chunk_ids], "ground_truth": [chunk_ids]}
        gen_predictions_by_config: dict mapping config name -> generation predictions (optional)
        ks: k values for @k metrics
        primary_metric: metric for comparison table

    Returns list of AblationResult objects.
    """
    results = []
    for config in ABLATION_CONFIGS:
        preds = predictions_by_config.get(config.name)
        if preds is None:
            continue

        ret_result = evaluate_retrieval(preds, ks=ks, n_bootstrap=2000)

        gen_summary = None
        if gen_predictions_by_config and config.name in gen_predictions_by_config:
            gen_result  = evaluate_generation(gen_predictions_by_config[config.name], n_bootstrap=2000)
            gen_summary = gen_result["summary"]

        results.append(AblationResult(
            config=config,
            retrieval_summary=ret_result["summary"],
            generation_summary=gen_summary,
            n_queries=ret_result["n_queries"],
        ))

    return results


def print_ablation_table(results: List[AblationResult]) -> None:
    """
    Print a LaTeX-style ablation table to stdout.
    """
    metrics = ["mrr", "recall@5", "ndcg@5", "code_bleu", "rouge_l"]
    header_parts = ["Configuration          "] + [f"{m:>12}" for m in metrics]
    print("\n" + "=" * 90)
    print("  Ablation Study Results")
    print("  (Full = all 5 novelties; -Nx = novelty x removed; Vanilla = none)")
    print("-" * 90)
    print("  " + " | ".join(header_parts))
    print("-" * 90)

    full_result = next((r for r in results if r.config.name == "Full"), None)

    for r in results:
        km = r.key_metrics()
        row_parts = [f"{r.config.name:<24}"]
        for m in metrics:
            val = km.get(m, 0.0)
            # Bold full system
            if r.config.name == "Full":
                row_parts.append(f"{val:>12.4f}*")
            elif full_result:
                full_val = full_result.key_metrics().get(m, 0.0)
                delta = val - full_val
                row_parts.append(f"{val:>9.4f}{delta:>+6.4f}")
            else:
                row_parts.append(f"{val:>12.4f}")
        print("  " + " | ".join(row_parts))
    print("=" * 90)
    print("  * = Full system (reference). Delta shown as +/- vs Full.\n")


def generate_ablation_comparison_reports(
    results: List[AblationResult],
    primary_metric: str = "recall@5",
) -> List[Dict[str, Any]]:
    """
    Compare each ablated system to the full system using Wilcoxon test.
    Requires per-query scores arrays to be stored (extend AblationResult if needed).
    Returns list of comparison report dicts.
    """
    # Placeholder: in production, store per-query arrays in AblationResult
    # and call full_comparison_report() for each pair.
    # Shown here as a scaffold.
    reports = []
    full_result = next((r for r in results if r.config.name == "Full"), None)
    if not full_result:
        return reports

    for r in results:
        if r.config.name == "Full":
            continue
        full_score = full_result.retrieval_summary.get(primary_metric, {}).get("mean", 0)
        abl_score  = r.retrieval_summary.get(primary_metric, {}).get("mean", 0)
        reports.append({
            "metric":      primary_metric,
            "system_a":    "Full",
            "system_b":    r.config.name,
            "mean_a":      full_score,
            "mean_b":      abl_score,
            "delta":       full_score - abl_score,
            "description": r.config.description,
        })
    return reports


def ablation_latex_table(results: List[AblationResult]) -> str:
    """
    Generate a LaTeX table string for the paper.
    """
    metrics = ["mrr", "recall@5", "ndcg@5", "code_bleu", "rouge_l"]
    metric_labels = ["MRR", "Recall@5", "NDCG@5", "CodeBLEU", "ROUGE-L"]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Study: Contribution of Each Novelty}")
    lines.append(r"\label{tab:ablation}")
    col_fmt = "l" + "r" * len(metrics)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    header = "Configuration & " + " & ".join(metric_labels) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for r in results:
        km = r.key_metrics()
        vals = [f"{km.get(m, 0):.4f}" for m in metrics]
        name = r.config.name.replace("_", r"\_")
        if r.config.name == "Full":
            vals = [r"\textbf{" + v + "}" for v in vals]
            name = r"\textbf{" + name + "}"
        lines.append(name + " & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
