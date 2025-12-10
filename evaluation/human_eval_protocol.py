"""
Human Evaluation Protocol for ChatGIT.

Dimensions (5-point Likert scale, 1=worst, 5=best):
  - Relevance:    Does the answer address the question?
  - Accuracy:     Is the code/information factually correct?
  - Completeness: Does the answer cover all aspects needed?
  - Clarity:      Is the answer clear and well-explained?
  - Groundedness: Is the answer grounded in the actual repository?

Inter-Annotator Agreement:
  - Fleiss' κ (across all raters)
  - Krippendorff's α (ordinal)
  - Pairwise Cohen's κ

Rater Qualifications:
  - Minimum 2 years professional coding experience
  - Familiarity with Python (primary language in ConvCodeBench)
  - Blind to which system generated each response

Protocol:
  - Each response rated by 3 independent annotators
  - Annotation interface: simple web form or CSV
  - Estimated time: ~3 min per response
  - Total load: 100 samples × 3 raters = ~300 ratings per session
"""

import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from evaluation.statistical_tests import fleiss_kappa, bootstrap_ci


# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------

DIMENSIONS = ["relevance", "accuracy", "completeness", "clarity", "groundedness"]
SCALE_MIN, SCALE_MAX = 1, 5
N_RATERS_REQUIRED = 3


def validate_annotation(annotation: Dict[str, Any]) -> List[str]:
    """
    Validate a single annotation dict.
    Returns list of error strings (empty = valid).
    """
    errors = []
    required_fields = ["sample_id", "rater_id"] + DIMENSIONS
    for field in required_fields:
        if field not in annotation:
            errors.append(f"Missing field: {field}")

    for dim in DIMENSIONS:
        if dim in annotation:
            val = annotation[dim]
            if not isinstance(val, (int, float)):
                errors.append(f"{dim} must be numeric, got {type(val)}")
            elif not (SCALE_MIN <= val <= SCALE_MAX):
                errors.append(f"{dim}={val} out of range [{SCALE_MIN}, {SCALE_MAX}]")

    return errors


def validate_annotation_set(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a full set of annotations.
    Returns summary with valid/invalid counts.
    """
    valid, invalid = [], []
    for ann in annotations:
        errors = validate_annotation(ann)
        if errors:
            invalid.append({"annotation": ann, "errors": errors})
        else:
            valid.append(ann)
    return {
        "n_total": len(annotations),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "invalid_details": invalid,
    }


# ---------------------------------------------------------------------------
# Per-sample aggregate scores
# ---------------------------------------------------------------------------

def aggregate_ratings(
    annotations: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate ratings across raters for each sample.

    Returns dict: sample_id -> {dimension: {mean, std, ratings}}
    """
    by_sample: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for ann in annotations:
        sid = ann["sample_id"]
        for dim in DIMENSIONS:
            if dim in ann:
                by_sample[sid][dim].append(ann[dim])

    result = {}
    for sid, dim_ratings in by_sample.items():
        result[sid] = {}
        for dim, ratings in dim_ratings.items():
            arr = np.array(ratings)
            result[sid][dim] = {
                "mean":    float(arr.mean()),
                "std":     float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "ratings": ratings,
                "n_raters": len(ratings),
            }

    return result


# ---------------------------------------------------------------------------
# Inter-Annotator Agreement (IAA)
# ---------------------------------------------------------------------------

def compute_iaa(
    annotations: List[Dict[str, Any]],
    n_categories: int = 5,
) -> Dict[str, Any]:
    """
    Compute inter-annotator agreement for each dimension.

    Returns Fleiss' κ per dimension + overall average.
    """
    # Group by sample_id
    by_sample: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        sid = ann["sample_id"]
        for dim in DIMENSIONS:
            if dim in ann:
                by_sample[sid][dim].append(int(ann[dim]))

    iaa_results = {}
    for dim in DIMENSIONS:
        # Build ratings matrix: list of items, each item = list of ratings
        ratings_matrix = []
        for sid, dim_ratings in by_sample.items():
            if dim in dim_ratings and len(dim_ratings[dim]) >= 2:
                ratings_matrix.append(dim_ratings[dim])
        if not ratings_matrix:
            iaa_results[dim] = {"kappa": None, "n_items": 0}
            continue
        kappa_result = fleiss_kappa(ratings_matrix, n_categories=n_categories)
        iaa_results[dim] = {
            "kappa":    kappa_result["kappa"],
            "p_o":      kappa_result["p_o"],
            "p_e":      kappa_result["p_e"],
            "n_items":  len(ratings_matrix),
        }

    # Overall average κ
    valid_kappas = [v["kappa"] for v in iaa_results.values() if v.get("kappa") is not None]
    iaa_results["overall"] = {
        "mean_kappa": float(np.mean(valid_kappas)) if valid_kappas else None
    }

    return iaa_results


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) κ interpretation."""
    if kappa < 0:
        return "poor (less than chance)"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost perfect"


# ---------------------------------------------------------------------------
# System-level human evaluation summary
# ---------------------------------------------------------------------------

def human_eval_summary(
    annotations: List[Dict[str, Any]],
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    """
    Full human evaluation summary:
      - Mean ± CI for each dimension
      - Overall score (mean of all dimensions)
      - IAA (Fleiss' κ) per dimension
    """
    by_dim: Dict[str, list] = defaultdict(list)
    for ann in annotations:
        for dim in DIMENSIONS:
            if dim in ann:
                by_dim[dim].append(ann[dim])

    dim_summaries = {}
    for dim, vals in by_dim.items():
        arr = np.array(vals, dtype=float)
        dim_summaries[dim] = bootstrap_ci(arr, n_resamples=n_bootstrap)

    # Overall
    all_vals = [v for vals in by_dim.values() for v in vals]
    overall_arr = np.array(all_vals, dtype=float)
    dim_summaries["overall"] = bootstrap_ci(overall_arr, n_resamples=n_bootstrap)

    iaa = compute_iaa(annotations)

    return {
        "dimensions": dim_summaries,
        "iaa":        iaa,
        "n_annotations": len(annotations),
        "n_samples":  len({a["sample_id"] for a in annotations}),
    }


def print_human_eval_report(
    results: Dict[str, Any],
    system_name: str = "ChatGIT",
) -> None:
    print(f"\n{'='*65}")
    print(f"  Human Evaluation: {system_name}  (n={results['n_samples']} samples, {results['n_annotations']} annotations)")
    print(f"{'='*65}")
    print(f"  {'Dimension':<18} {'Mean':>6}  95% CI               IAA κ    Agreement")
    print(f"  {'-'*62}")
    dims = results["dimensions"]
    iaa  = results["iaa"]
    for dim in DIMENSIONS + ["overall"]:
        if dim not in dims:
            continue
        s = dims[dim]
        kappa_info = iaa.get(dim, {})
        kappa = kappa_info.get("kappa")
        kappa_str = f"{kappa:.3f} ({interpret_kappa(kappa)})" if kappa is not None else "N/A"
        print(f"  {dim:<18} {s['mean']:>6.3f}  [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]   {kappa_str}")
    print()


# ---------------------------------------------------------------------------
# Annotation templates
# ---------------------------------------------------------------------------

def generate_annotation_form_schema() -> Dict[str, Any]:
    """
    Returns JSON-compatible annotation form schema for integration
    with annotation tools (Label Studio, Argilla, etc.).
    """
    return {
        "form_name": "ChatGIT Human Evaluation",
        "instructions": (
            "Rate the following AI-generated response to a code question. "
            "Each dimension uses a 1-5 Likert scale (1=very poor, 5=excellent). "
            "Do NOT consider which AI system generated the response (blind evaluation)."
        ),
        "fields": [
            {
                "name": "relevance",
                "label": "Relevance",
                "description": "Does the response directly address the question asked?",
                "type": "likert",
                "scale": 5,
                "anchors": {1: "Completely off-topic", 5: "Perfectly addresses the question"},
            },
            {
                "name": "accuracy",
                "label": "Accuracy",
                "description": "Is the code or information factually correct?",
                "type": "likert",
                "scale": 5,
                "anchors": {1: "Contains major errors", 5: "Fully correct"},
            },
            {
                "name": "completeness",
                "label": "Completeness",
                "description": "Does the response cover all aspects needed to answer the question?",
                "type": "likert",
                "scale": 5,
                "anchors": {1: "Very incomplete", 5: "Fully complete"},
            },
            {
                "name": "clarity",
                "label": "Clarity",
                "description": "Is the response clear, well-structured, and easy to understand?",
                "type": "likert",
                "scale": 5,
                "anchors": {1: "Confusing or poorly written", 5: "Very clear and well-explained"},
            },
            {
                "name": "groundedness",
                "label": "Groundedness",
                "description": "Is the response grounded in the actual repository code (not hallucinated)?",
                "type": "likert",
                "scale": 5,
                "anchors": {1: "Entirely hallucinated", 5: "Fully grounded in the codebase"},
            },
        ],
        "metadata_fields": ["sample_id", "rater_id", "system_name", "timestamp"],
    }
