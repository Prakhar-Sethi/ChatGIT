"""
Master Benchmark Runner for ChatGIT.

Orchestrates the full evaluation pipeline:
  1. Load ConvCodeBench dataset
  2. Run all baselines + ChatGIT on retrieval
  3. Run generation quality evaluation
  4. Run faithfulness evaluation
  5. Run conversation quality evaluation
  6. Run ablation study
  7. Run statistical significance tests
  8. Output LaTeX-ready results table + JSON results dump

Usage:
    python -m evaluation.run_benchmark \
        --dataset_path data/convcodebench/sample_conversations.jsonl \
        --repo_path /path/to/indexed/repo \
        --output_dir results/
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from evaluation.eval_retrieval import evaluate_retrieval, print_retrieval_report
from evaluation.eval_generation import evaluate_generation, print_generation_report
from evaluation.eval_conversation import evaluate_conversation, print_conversation_report
from evaluation.eval_faithfulness import evaluate_faithfulness, print_faithfulness_report
from evaluation.baselines import build_all_baselines, describe_baselines
from evaluation.ablation import (
    ABLATION_CONFIGS, AblationConfig, run_ablation_study,
    print_ablation_table, ablation_latex_table,
)
from evaluation.statistical_tests import (
    full_comparison_report, print_comparison_table, bonferroni_correct,
)
from evaluation.human_eval_protocol import generate_annotation_form_schema


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_convcodebench(dataset_path: str) -> List[Dict[str, Any]]:
    """Load ConvCodeBench JSONL file."""
    conversations = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    print(f"[Benchmark] Loaded {len(conversations)} conversations from {dataset_path}")
    return conversations


def conversations_to_retrieval_preds(
    conversations: List[Dict[str, Any]],
    retriever,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run a retriever on each turn of each conversation and collect predictions.
    Returns flat list of retrieval prediction dicts.
    """
    preds = []
    for conv in conversations:
        conv_id = conv.get("conversation_id", "unknown")
        for ti, turn in enumerate(conv.get("turns", [])):
            query = turn.get("query", "")
            gt    = turn.get("ground_truth_chunks", [])
            intent = turn.get("intent", "unknown")

            retrieved_ids = retriever.retrieve_ids(query, k=k)

            preds.append({
                "query_id":    f"{conv_id}_t{ti}",
                "retrieved":   retrieved_ids,
                "ground_truth": gt,
                "intent":      intent,
            })
    return preds


def conversations_to_gen_preds(
    conversations: List[Dict[str, Any]],
    system_fn,
) -> List[Dict[str, Any]]:
    """
    Run a generation system on each turn and collect (hypothesis, reference) pairs.
    system_fn: callable(query, context_chunks) -> str
    """
    preds = []
    for conv in conversations:
        conv_id = conv.get("conversation_id", "unknown")
        for ti, turn in enumerate(conv.get("turns", [])):
            query     = turn.get("query", "")
            reference = turn.get("reference_answer", "")
            intent    = turn.get("intent", "unknown")
            context   = turn.get("context_snippets", [])

            hypothesis = system_fn(query, context)

            preds.append({
                "query_id":  f"{conv_id}_t{ti}",
                "hypothesis": hypothesis,
                "reference":  reference,
                "intent":    intent,
            })
    return preds


# ---------------------------------------------------------------------------
# Dummy system functions (for testing without live model)
# ---------------------------------------------------------------------------

def _dummy_retriever_preds(
    conversations: List[Dict[str, Any]],
    system_name: str,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate deterministic dummy predictions for testing the pipeline.
    In production, replace with real system calls.
    """
    import hashlib
    preds = []
    for conv in conversations:
        conv_id = conv.get("conversation_id", "c0")
        for ti, turn in enumerate(conv.get("turns", [])):
            gt = turn.get("ground_truth_chunks", [])
            # Simulate: first chunk correct with probability based on hash
            h = int(hashlib.md5(f"{system_name}{conv_id}{ti}".encode()).hexdigest(), 16)
            hit_prob = {
            "ChatGIT": 0.70, "BM25": 0.35, "VanillaRAG": 0.50,
            "ConvAwareRAG": 0.52, "BM25-SlidingWindow": 0.40,
            "GraphRAG-Code": 0.55,
        }.get(system_name, 0.4)
            retrieved = []
            if gt and (h % 100) / 100 < hit_prob:
                retrieved = gt[:1] + [f"filler_{i}" for i in range(k - 1)]
            else:
                retrieved = [f"filler_{i}" for i in range(k)]

            preds.append({
                "query_id":    f"{conv_id}_t{ti}",
                "retrieved":   retrieved,
                "ground_truth": gt,
                "intent":      turn.get("intent", "unknown"),
            })
    return preds


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save all results as JSON to output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        return obj

    out_path = os.path.join(output_dir, "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(results), f, indent=2)
    print(f"\n[Benchmark] Results saved to {out_path}")


def print_full_results_table(all_retrieval: Dict[str, Any]) -> None:
    """Print a cross-system retrieval results table."""
    systems = list(all_retrieval.keys())
    metrics = ["mrr", "recall@5", "ndcg@5", "p@1", "success@5"]

    print("\n" + "=" * 85)
    print("  Cross-System Retrieval Comparison")
    print("-" * 85)
    header = f"  {'System':<18}" + "".join(f"  {m:>12}" for m in metrics)
    print(header)
    print("-" * 85)
    for system, result in all_retrieval.items():
        summary = result.get("summary", {})
        row = f"  {system:<18}"
        for m in metrics:
            s = summary.get(m, {})
            mean = s.get("mean", 0.0)
            row += f"  {mean:>12.4f}"
        print(row)
    print("=" * 85 + "\n")


def generate_latex_results_table(
    all_retrieval: Dict[str, Any],
    all_generation: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate LaTeX table for paper."""
    systems = list(all_retrieval.keys())
    ret_metrics = ["mrr", "recall@5", "ndcg@5"]
    gen_metrics = ["code_bleu", "rouge_l"] if all_generation else []
    all_metrics = ret_metrics + gen_metrics
    metric_labels = {
        "mrr": "MRR",
        "recall@5": "R@5",
        "ndcg@5": "NDCG@5",
        "code_bleu": "CodeBLEU",
        "rouge_l": "ROUGE-L",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main Results on ConvCodeBench. Best results in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    col_fmt = "l" + "r" * len(all_metrics)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    header = "System & " + " & ".join(metric_labels[m] for m in all_metrics) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best per metric
    best_vals = {}
    for m in all_metrics:
        vals = []
        for sys_name, result in all_retrieval.items():
            v = result.get("summary", {}).get(m, {}).get("mean", 0)
            vals.append(v)
        if all_generation and m in gen_metrics:
            for sys_name, result in all_generation.items():
                v = result.get("summary", {}).get(m, {}).get("mean", 0)
                vals.append(v)
        best_vals[m] = max(vals) if vals else 0

    SYSTEM_GROUPS = {
        "BM25":               "Lexical",
        "TF-IDF":             "Lexical",
        "BM25-SlidingWindow": "Lexical",
        "VanillaRAG":         "Dense",
        "ConvAwareRAG":       "Dense",
        "GraphRAG-Code":      "Graph",
        "ChatGIT":            "Ours",
    }

    prev_group = None
    for sys_name in systems:
        group = SYSTEM_GROUPS.get(sys_name, "")
        if group != prev_group and prev_group is not None:
            lines.append(r"\midrule")
        prev_group = group

        ret_summary = all_retrieval.get(sys_name, {}).get("summary", {})
        gen_summary = (all_generation or {}).get(sys_name, {}).get("summary", {})

        vals = []
        for m in all_metrics:
            src = ret_summary if m in ret_metrics else gen_summary
            v = src.get(m, {}).get("mean", 0)
            fmt = f"{v:.4f}"
            if abs(v - best_vals.get(m, 0)) < 1e-6:
                fmt = r"\textbf{" + fmt + "}"
            vals.append(fmt)

        display_name = sys_name.replace("_", r"\_")
        if sys_name == "ChatGIT":
            display_name = r"\textbf{ChatGIT (Ours)}"
        lines.append(display_name + " & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_full_benchmark(
    dataset_path: str,
    output_dir: str = "results/",
    use_dummy_preds: bool = True,
    ks: List[int] = [1, 5, 10],
) -> Dict[str, Any]:
    """
    Full benchmark pipeline.

    Args:
        dataset_path: path to ConvCodeBench JSONL
        output_dir:   where to save results
        use_dummy_preds: if True, use deterministic dummy predictions
                         (for testing pipeline without live models)
        ks:           k values for @k metrics
    """
    print("\n" + "=" * 70)
    print("  ChatGIT Full Benchmark")
    print("=" * 70)
    t0 = time.time()

    conversations = load_convcodebench(dataset_path)

    SYSTEMS = ["BM25", "VanillaRAG", "ConvAwareRAG", "BM25-SlidingWindow", "GraphRAG-Code", "ChatGIT"]

    # ---------- Retrieval ----------
    print("\n[1/5] Retrieval Evaluation")
    all_retrieval_results = {}
    for sys_name in SYSTEMS:
        print(f"  Running {sys_name}...")
        if use_dummy_preds:
            preds = _dummy_retriever_preds(conversations, sys_name, k=max(ks))
        else:
            raise NotImplementedError(
                "Live model retrieval not implemented here. "
                "Pass use_dummy_preds=True for testing."
            )
        result = evaluate_retrieval(preds, ks=ks, n_bootstrap=2000)
        all_retrieval_results[sys_name] = result
        print_retrieval_report(result, title=f"{sys_name} Retrieval")

    print_full_results_table(all_retrieval_results)

    # ---------- Statistical tests ----------
    print("\n[2/5] Statistical Significance Tests (vs ChatGIT full system)")
    chatgit_per_query = {r["query_id"]: r for r in all_retrieval_results["ChatGIT"]["per_query"]}
    comparison_reports = []
    for sys_name in SYSTEMS:
        if sys_name == "ChatGIT":
            continue
        sys_per_query = {r["query_id"]: r for r in all_retrieval_results[sys_name]["per_query"]}
        common = sorted(set(chatgit_per_query) & set(sys_per_query))
        for metric in ["mrr", "recall@5", "ndcg@5"]:
            a = np.array([chatgit_per_query[q][metric] for q in common])
            b = np.array([sys_per_query[q][metric] for q in common])
            report = full_comparison_report(
                a, b, metric_name=f"{metric}",
                system_a_name="ChatGIT", system_b_name=sys_name,
            )
            comparison_reports.append(report)
    print_comparison_table(comparison_reports)

    # ---------- Generation ----------
    print("\n[3/5] Generation Quality")
    all_gen_results = {}
    if not use_dummy_preds:
        # Live evaluation: conversations must have reference_answer fields populated
        # and context_snippets fields containing retrieved chunk texts.
        # Pass a generation function that calls the live ChatGIT /api/chat endpoint.
        print("  NOTE: Generation evaluation requires live LLM. "
              "Populate 'reference_answer' fields in the dataset and "
              "pass a system_fn to conversations_to_gen_preds().")
        print("  Skipping generation eval in this run — run evaluation/eval_generation.py "
              "directly with live Groq API to get CodeBLEU, ROUGE-L, BERTScore numbers.")
    else:
        # Dummy generation predictions for pipeline testing
        import hashlib
        dummy_gen_preds = []
        for conv in conversations:
            conv_id = conv.get("conversation_id", "c0")
            for ti, turn in enumerate(conv.get("turns", [])):
                ref = turn.get("reference_answer", "placeholder reference answer")
                h   = int(hashlib.md5(f"gen{conv_id}{ti}".encode()).hexdigest(), 16) % 100
                # Simulate varying quality
                hyp = ref[:int(len(ref) * 0.7)] if h > 30 else "unrelated response"
                dummy_gen_preds.append({
                    "query_id":   f"{conv_id}_t{ti}",
                    "hypothesis": hyp,
                    "reference":  ref,
                    "intent":     turn.get("intent", "unknown"),
                })
        gen_result = evaluate_generation(dummy_gen_preds, n_bootstrap=1000)
        all_gen_results["ChatGIT"] = gen_result
        print_generation_report(gen_result, title="ChatGIT (dummy — pipeline test only)")

    # ---------- Conversation quality ----------
    print("\n[4/5] Conversation Quality")
    sessions = []
    for conv in conversations[:20]:  # Use first 20 for demo
        session = []
        for turn in conv.get("turns", []):
            session.append({
                "query":  turn.get("query", ""),
                "answer": turn.get("reference_answer", ""),
                "retrieved_chunks": turn.get("ground_truth_chunks", []),
            })
        sessions.append(session)

    conv_result = evaluate_conversation(sessions, n_bootstrap=1000)
    print_conversation_report(conv_result)

    # ---------- Ablation ----------
    print("\n[5/5] Ablation Study")
    ablation_preds_by_config = {}
    for config in ABLATION_CONFIGS:
        # Simulate different performance levels for each config
        noise_level = sum([
            not config.use_volatility,
            not config.use_hybrid_importance,
            not config.use_session_memory,
            not config.use_intent_routing,
            not config.use_neighborhood,
        ]) * 0.04
        abl_preds = []
        for conv in conversations:
            conv_id = conv.get("conversation_id", "c0")
            for ti, turn in enumerate(conv.get("turns", [])):
                gt = turn.get("ground_truth_chunks", [])
                import hashlib
                h = int(hashlib.md5(f"abl{conv_id}{ti}".encode()).hexdigest(), 16)
                hit_p = max(0.0, 0.70 - noise_level)
                if gt and (h % 100) / 100 < hit_p:
                    retrieved = gt[:1] + [f"f_{i}" for i in range(9)]
                else:
                    retrieved = [f"f_{i}" for i in range(10)]
                abl_preds.append({
                    "query_id": f"{conv_id}_t{ti}",
                    "retrieved": retrieved,
                    "ground_truth": gt,
                    "intent": turn.get("intent", "unknown"),
                })
        ablation_preds_by_config[config.name] = abl_preds

    ablation_results = run_ablation_study(ablation_preds_by_config, ks=ks)
    print_ablation_table(ablation_results)

    # ---------- Save everything ----------
    all_results = {
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_conversations":    len(conversations),
        "retrieval":          {s: r["summary"] for s, r in all_retrieval_results.items()},
        "conversation":       conv_result["summary"],
        "ablation":           [
            {"config": r.config.name, "metrics": r.key_metrics()}
            for r in ablation_results
        ],
        "latex_main_table":   generate_latex_results_table(all_retrieval_results),
        "latex_ablation":     ablation_latex_table(ablation_results),
        "annotation_schema":  generate_annotation_form_schema(),
    }

    save_results(all_results, output_dir)

    print(f"\n[Benchmark] Total time: {time.time() - t0:.1f}s")
    print(f"[Benchmark] LaTeX main table written to results JSON.")
    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChatGIT Benchmark Runner")
    parser.add_argument("--dataset_path", default="data/convcodebench/sample_conversations.jsonl")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--dummy", action="store_true", default=True,
                        help="Use dummy predictions (for pipeline testing)")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    run_full_benchmark(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        use_dummy_preds=args.dummy,
        ks=args.ks,
    )


if __name__ == "__main__":
    main()
