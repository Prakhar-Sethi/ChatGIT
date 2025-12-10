"""
ConvCodeBench Dataset Quality Validation Script.

Checks:
  1. Schema compliance (all required fields present)
  2. No empty queries or reference answers
  3. Ground truth chunk IDs are non-empty
  4. Coreference annotations are consistent
  5. Intent sequence diversity (not all same intent)
  6. Minimum conversation length (>= 2 turns)
  7. Language distribution balance
  8. Duplicate conversation detection
  9. Difficulty distribution
  10. Intent distribution balance

Usage:
    python data/convcodebench/validate_dataset.py \
        --dataset data/convcodebench/conversations.jsonl \
        --schema  data/convcodebench/schema.json \
        --report  data/convcodebench/validation_report.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re


REQUIRED_CONV_FIELDS = ["conversation_id", "repo_id", "language", "turns"]
REQUIRED_TURN_FIELDS = ["turn_id", "query", "intent", "ground_truth_chunks", "reference_answer"]
VALID_INTENTS = {"locate", "explain", "summarize", "debug"}
VALID_LANGUAGES = {"python", "javascript", "typescript", "java", "go"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
MIN_TURNS = 2
MAX_TURNS = 8
MIN_QUERY_LEN = 10
MIN_ANSWER_LEN = 20


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_schema(conv: Dict[str, Any]) -> List[str]:
    errors = []
    for f in REQUIRED_CONV_FIELDS:
        if f not in conv:
            errors.append(f"Missing required field: {f}")

    turns = conv.get("turns", [])
    if not isinstance(turns, list):
        errors.append("'turns' must be a list")
        return errors

    if len(turns) < MIN_TURNS:
        errors.append(f"Too few turns: {len(turns)} < {MIN_TURNS}")
    if len(turns) > MAX_TURNS:
        errors.append(f"Too many turns: {len(turns)} > {MAX_TURNS}")

    for i, turn in enumerate(turns):
        prefix = f"Turn {i}"
        for f in REQUIRED_TURN_FIELDS:
            if f not in turn:
                errors.append(f"{prefix}: Missing field '{f}'")
        if "intent" in turn and turn["intent"] not in VALID_INTENTS:
            errors.append(f"{prefix}: Invalid intent '{turn['intent']}'")
        if "query" in turn and len(turn["query"]) < MIN_QUERY_LEN:
            errors.append(f"{prefix}: Query too short ({len(turn['query'])} chars)")
        if "reference_answer" in turn:
            ans = turn["reference_answer"]
            if "[TO BE ANNOTATED]" in ans:
                errors.append(f"{prefix}: Reference answer not yet annotated")
            elif len(ans) < MIN_ANSWER_LEN:
                errors.append(f"{prefix}: Reference answer too short ({len(ans)} chars)")
        if "ground_truth_chunks" in turn and not turn["ground_truth_chunks"]:
            errors.append(f"{prefix}: ground_truth_chunks is empty")

    if "language" in conv and conv["language"] not in VALID_LANGUAGES:
        errors.append(f"Invalid language: {conv['language']}")

    return errors


def validate_coreferences(conv: Dict[str, Any]) -> List[str]:
    warnings = []
    turns = conv.get("turns", [])
    for i, turn in enumerate(turns):
        for coref in turn.get("coreferences", []):
            ref_turn = coref.get("referent_turn_id", -1)
            if ref_turn >= i:
                warnings.append(
                    f"Turn {i}: Coreference referent_turn_id={ref_turn} "
                    f"must be < current turn {i}"
                )
            if not coref.get("pronoun"):
                warnings.append(f"Turn {i}: Coreference missing 'pronoun'")
            if not coref.get("referent"):
                warnings.append(f"Turn {i}: Coreference missing 'referent'")
    return warnings


def validate_intent_diversity(conv: Dict[str, Any]) -> List[str]:
    warnings = []
    intents = [t.get("intent", "") for t in conv.get("turns", [])]
    if len(set(intents)) == 1 and len(intents) > 2:
        warnings.append(f"Low intent diversity: all turns are '{intents[0]}'")
    return warnings


def check_duplicate(conv: Dict, seen_queries: set) -> List[str]:
    issues = []
    for turn in conv.get("turns", []):
        q = turn.get("query", "").strip().lower()
        if q in seen_queries:
            issues.append(f"Duplicate query found: '{q[:60]}...'")
        seen_queries.add(q)
    return issues


# ---------------------------------------------------------------------------
# Dataset-level statistics
# ---------------------------------------------------------------------------

def compute_statistics(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    lang_counter    = Counter()
    intent_counter  = Counter()
    diff_counter    = Counter()
    turns_lengths   = []
    coref_count     = 0
    topic_shift_count = 0
    domain_counter  = Counter()
    complexity_counter = Counter()

    for conv in conversations:
        lang_counter[conv.get("language", "unknown")] += 1
        domain_counter[conv.get("domain", "unknown")] += 1
        complexity_counter[conv.get("complexity_tier", "unknown")] += 1
        meta = conv.get("metadata", {})
        if meta.get("has_coreference"):
            coref_count += 1
        if meta.get("topic_shift"):
            topic_shift_count += 1

        turns = conv.get("turns", [])
        turns_lengths.append(len(turns))
        for turn in turns:
            intent_counter[turn.get("intent", "unknown")] += 1
            diff_counter[turn.get("difficulty", "unknown")] += 1

    total_turns = sum(turns_lengths)
    n = len(conversations)

    return {
        "n_conversations":     n,
        "n_turns_total":       total_turns,
        "avg_turns_per_conv":  round(total_turns / n, 2) if n > 0 else 0,
        "min_turns":           min(turns_lengths) if turns_lengths else 0,
        "max_turns":           max(turns_lengths) if turns_lengths else 0,
        "language_dist":       dict(lang_counter),
        "intent_dist":         dict(intent_counter),
        "difficulty_dist":     dict(diff_counter),
        "domain_dist":         dict(domain_counter),
        "complexity_dist":     dict(complexity_counter),
        "n_with_coreference":  coref_count,
        "n_with_topic_shift":  topic_shift_count,
        "pct_with_coreference": round(100 * coref_count / n, 1) if n > 0 else 0,
    }


def check_balance_warnings(stats: Dict[str, Any]) -> List[str]:
    warnings = []

    # Language balance: no language should be > 50%
    n = stats["n_conversations"]
    for lang, cnt in stats["language_dist"].items():
        if cnt / n > 0.5:
            warnings.append(
                f"Language imbalance: '{lang}' is {cnt/n:.1%} of conversations "
                f"(recommended ≤ 50%)"
            )

    # Intent balance: no intent should be > 40%
    total_turns = stats["n_turns_total"]
    for intent, cnt in stats["intent_dist"].items():
        if cnt / total_turns > 0.40:
            warnings.append(
                f"Intent imbalance: '{intent}' is {cnt/total_turns:.1%} of turns "
                f"(recommended ≤ 40%)"
            )

    # Difficulty balance
    for diff, cnt in stats["difficulty_dist"].items():
        if cnt / total_turns > 0.60:
            warnings.append(
                f"Difficulty imbalance: '{diff}' is {cnt/total_turns:.1%} of turns"
            )

    # Coreference coverage
    if stats["pct_with_coreference"] < 20:
        warnings.append(
            f"Low coreference coverage: {stats['pct_with_coreference']}% "
            f"(recommended ≥ 20%)"
        )

    return warnings


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_dataset(
    dataset_path: str,
    schema_path: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    conversations = []
    with open(dataset_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[validate] Line {line_no}: JSON parse error: {e}")

    print(f"[validate] Loaded {len(conversations)} conversations from {dataset_path}")

    all_errors: List[Dict] = []
    all_warnings: List[Dict] = []
    seen_queries: set = set()
    seen_ids: set = set()

    for conv in conversations:
        conv_id = conv.get("conversation_id", "unknown")

        # Duplicate conversation ID check
        if conv_id in seen_ids:
            all_errors.append({"conversation_id": conv_id, "error": "Duplicate conversation_id"})
        seen_ids.add(conv_id)

        errors = validate_schema(conv)
        warnings  = []
        warnings += validate_coreferences(conv)
        warnings += validate_intent_diversity(conv)
        warnings += check_duplicate(conv, seen_queries)

        for err in errors:
            all_errors.append({"conversation_id": conv_id, "error": err})
        for warn in warnings:
            all_warnings.append({"conversation_id": conv_id, "warning": warn})

    stats = compute_statistics(conversations)
    balance_warnings = check_balance_warnings(stats)
    for w in balance_warnings:
        all_warnings.append({"conversation_id": "DATASET", "warning": w})

    report = {
        "dataset_path":   dataset_path,
        "n_conversations": len(conversations),
        "n_errors":        len(all_errors),
        "n_warnings":      len(all_warnings),
        "valid":           len(all_errors) == 0,
        "statistics":      stats,
        "errors":          all_errors[:100],    # Cap at 100 for readability
        "warnings":        all_warnings[:100],
        "balance_warnings": balance_warnings,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Validation Report: {Path(dataset_path).name}")
    print(f"{'='*60}")
    print(f"  Conversations:  {len(conversations)}")
    print(f"  Errors:         {len(all_errors)}   {'✓ PASS' if not all_errors else '✗ FAIL'}")
    print(f"  Warnings:       {len(all_warnings)}")
    print(f"  Languages:      {dict(sorted(stats['language_dist'].items()))}")
    print(f"  Intents:        {dict(sorted(stats['intent_dist'].items()))}")
    print(f"  Avg turns:      {stats['avg_turns_per_conv']}")
    print(f"  Coreferences:   {stats['pct_with_coreference']}% of conversations")

    if all_errors:
        print(f"\n  First 5 errors:")
        for e in all_errors[:5]:
            print(f"    [{e['conversation_id']}] {e['error']}")

    if balance_warnings:
        print(f"\n  Balance warnings:")
        for w in balance_warnings:
            print(f"    {w}")
    print()

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[validate] Report saved to {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate ConvCodeBench dataset")
    parser.add_argument("--dataset", default="data/convcodebench/sample_conversations.jsonl")
    parser.add_argument("--schema",  default="data/convcodebench/schema.json")
    parser.add_argument("--report",  default="data/convcodebench/validation_report.json")
    args = parser.parse_args()

    report = validate_dataset(args.dataset, args.schema, args.report)
    exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
