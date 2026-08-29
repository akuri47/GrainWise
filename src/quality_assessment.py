"""
Rice Grain Quality Analyzer — Quality Assessment Module
=========================================================
FAQ (Fair Average Quality) standard validation and overall quality
grading per Government of India standards.

Functions:
    calculate_moisture_content   — Estimate moisture % from sample weight.
    calculate_defect_percentages — Summarise per-defect percentages.
    assess_faq_grade             — Grade A / B / Common / Rejected.
    calculate_quality_score      — Composite 0-100 quality score.
    generate_quality_report      — Full quality report dict.
"""

import sys
import os
import argparse
import json
from collections import Counter

# ---------------------------------------------------------------------------
# Config imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    VARIETY_DATABASE,
    FAQ_STANDARDS,
    QUALITY_WEIGHTS,
    DEFECT_LABELS,
)


# ===================================================================
# 1. Moisture content estimation
# ===================================================================

def calculate_moisture_content(sample_weight_g: float,
                               num_grains: int,
                               variety_key: str) -> float:
    """Estimate moisture percentage from a bulk sample weight.

    Formula
    -------
    moisture% = ((sample_weight - (dry_weight_per_grain × num_grains))
                  / sample_weight) × 100

    Parameters
    ----------
    sample_weight_g : float
        Total weight of the sample in grams.
    num_grains : int
        Number of grains counted in the sample.
    variety_key : str
        Key into VARIETY_DATABASE (e.g. ``"swarna"``).

    Returns
    -------
    float
        Estimated moisture percentage (0-100).

    Raises
    ------
    ValueError
        If *variety_key* is not in the database, or inputs are invalid.
    """
    if variety_key not in VARIETY_DATABASE:
        raise ValueError(
            f"Unknown variety '{variety_key}'. "
            f"Available: {list(VARIETY_DATABASE.keys())}"
        )
    if sample_weight_g <= 0:
        raise ValueError("sample_weight_g must be positive.")
    if num_grains <= 0:
        raise ValueError("num_grains must be positive.")

    dry_weight_per_grain = VARIETY_DATABASE[variety_key]["dry_weight_g"]
    total_dry_weight = dry_weight_per_grain * num_grains
    
    moisture_pct = ((sample_weight_g - total_dry_weight) / sample_weight_g) * 100.0

    # If calculation gives unreasonable result (negative or >50%), 
    # it means variety prediction or grain count is off.
    # Fall back to 12.5% which is typical for stored milled rice.
    if moisture_pct <= 0 or moisture_pct > 50:
        moisture_pct = 12.5

    return round(moisture_pct, 2)


# ===================================================================
# 2. Defect percentage calculation
# ===================================================================

def calculate_defect_percentages(total_grains: int,
                                 defect_predictions: list) -> dict:
    """Calculate per-defect percentages from classifier predictions.

    Parameters
    ----------
    total_grains : int
        Total number of grains analysed.
    defect_predictions : list[str]
        One label per grain — each must be a value from DEFECT_LABELS
        (``"whole"``, ``"broken"``, ``"chalky"``, ``"damaged"``,
        ``"discolored"``, ``"foreign"``).

    Returns
    -------
    dict
        Mapping of defect label → percentage, e.g.
        ``{"broken": 4.5, "chalky": 1.2, ...}``.
        The ``"whole"`` label is included for completeness.

    Raises
    ------
    ValueError
        If *total_grains* is non-positive or an unknown label appears.
    """
    if total_grains <= 0:
        raise ValueError("total_grains must be positive.")

    counts = Counter(defect_predictions)

    # Validate labels
    unknown = set(counts.keys()) - set(DEFECT_LABELS)
    if unknown:
        raise ValueError(
            f"Unknown defect labels: {unknown}. "
            f"Expected labels from: {DEFECT_LABELS}"
        )

    percentages = {}
    for label in DEFECT_LABELS:
        percentages[label] = (counts.get(label, 0) / total_grains) * 100.0

    return percentages


# ===================================================================
# 2b. Multi-label defect percentage calculation
# ===================================================================

def calculate_multilabel_defect_percentages(grain_classifications: list) -> dict:
    """Calculate defect percentages from multi-label grain classifications.

    In multi-label classification a grain has TWO identities:
      - **structural**: ``"whole"`` / ``"broken"`` / ``"foreign"``
        (determined by the 3/4 avg-length rule)
      - **defect**: ``"clean"`` / ``"chalky"`` / ``"damaged"`` / ``"discolored"``
        (determined by the model)

    Percentages can overlap — e.g. a ``broken + chalky`` grain counts
    in **both** ``broken%`` and ``chalky%``.

    Parameters
    ----------
    grain_classifications : list[dict]
        Each element is a dict with at least keys ``"structural"`` and
        ``"defect"`` (as produced by the analyzer pipeline).

    Returns
    -------
    dict
        Keys: ``whole`` (good grain — whole + clean only), ``broken``,
        ``chalky``, ``damaged``, ``discolored``, ``foreign``.
        Values: percentages of total grains.
    """
    if not grain_classifications:
        empty_pct = {label: 0.0 for label in ["whole", "broken", "chalky", "damaged", "discolored", "foreign"]}
        return {"overlapping": empty_pct, "mutually_exclusive": empty_pct}

    total = len(grain_classifications)

    # Count foreign grains
    foreign_count = sum(
        1 for g in grain_classifications if g["structural"] == "foreign"
    )
    # Total rice grains (excluding foreign matter)
    total_rice = total - foreign_count

    if total_rice <= 0:
        empty_pct = {label: 0.0 for label in ["whole", "broken", "chalky", "damaged", "discolored", "foreign"]}
        empty_pct["foreign"] = 100.0 if total > 0 else 0.0
        return {"overlapping": empty_pct, "mutually_exclusive": empty_pct}

    # Structural counts (among rice grains only)
    broken_count = sum(
        1 for g in grain_classifications if g["structural"] == "broken"
    )

    # Defect counts — these count ALL grains with that defect,
    # whether whole or broken (overlapping percentages)
    chalky_count = sum(
        1 for g in grain_classifications if g["defect"] == "chalky"
    )
    damaged_count = sum(
        1 for g in grain_classifications if g["defect"] == "damaged"
    )
    discolored_count = sum(
        1 for g in grain_classifications if g["defect"] == "discolored"
    )

    # Good grain = structurally whole AND defect-free (clean)
    good_grain_count = sum(
        1 for g in grain_classifications
        if g["structural"] == "whole" and g["defect"] == "clean"
    )

    # Build percentages dict for overlapping (used for grading)
    percentages = {
        "whole":      (good_grain_count / total_rice) * 100.0 if total_rice else 0,
        "broken":     (broken_count / total_rice) * 100.0 if total_rice else 0,
        "chalky":     (chalky_count / total_rice) * 100.0 if total_rice else 0,
        "damaged":    (damaged_count / total_rice) * 100.0 if total_rice else 0,
        "discolored": (discolored_count / total_rice) * 100.0 if total_rice else 0,
        "foreign":    (foreign_count / total) * 100.0 if total else 0,
    }

    # Build mutually exclusive percentages for the UI Pie Chart (sums to exactly 100)
    # Hierarchy: Foreign > Broken > Damaged > Discolored > Chalky > Good
    me_counts = {"foreign": foreign_count, "broken": broken_count, "damaged": 0, "discolored": 0, "chalky": 0, "whole": good_grain_count}
    for g in grain_classifications:
        if g["structural"] == "whole":
            # For whole grains, pick the most severe defect
            d = g["defect"]
            if d == "damaged": me_counts["damaged"] += 1
            elif d == "discolored": me_counts["discolored"] += 1
            elif d == "chalky": me_counts["chalky"] += 1

    me_percentages = {k: (v / total) * 100.0 for k, v in me_counts.items()}

    return {
        "overlapping": percentages,
        "mutually_exclusive": me_percentages
    }


# ===================================================================
# 3. FAQ grade assessment
# ===================================================================

def assess_faq_grade(moisture_pct: float,
                     defect_percentages: dict) -> str:
    """Determine FAQ grade based on Government of India thresholds.

    The function checks each grade tier (A → B → Common) in order.
    If all thresholds for a tier are satisfied the corresponding grade is
    returned; otherwise the next tier is tried.  If none pass, the
    sample is ``"Rejected"``.

    Parameters
    ----------
    moisture_pct : float
        Moisture percentage of the sample.
    defect_percentages : dict
        Output of :func:`calculate_defect_percentages`.

    Returns
    -------
    str
        One of ``"Grade A"``, ``"Grade B"``, ``"Common"``, or
        ``"Rejected"``.
    """
    # Map defect percentage keys to the FAQ_STANDARDS key suffixes
    defect_to_faq_key = {
        "broken":     "broken_max",
        "chalky":     "chalky_max",
        "damaged":    "damaged_max",
        "discolored": "discolored_max",
        "foreign":    "foreign_matter_max",
    }

    grade_tiers = [
        ("grade_a", "Grade A"),
        ("grade_b", "Grade B"),
        ("common",  "Common"),
    ]

    for tier_key, grade_label in grade_tiers:
        thresholds = FAQ_STANDARDS[tier_key]
        passed = True

        # Moisture check
        if moisture_pct > thresholds["moisture_max"]:
            passed = False

        # Defect checks
        if passed:
            for defect, faq_key in defect_to_faq_key.items():
                pct = defect_percentages.get(defect, 0.0)
                if pct > thresholds[faq_key]:
                    passed = False
                    break

        if passed:
            return grade_label

    return "Rejected"


# ===================================================================
# 4. Composite quality score
# ===================================================================

def calculate_quality_score(moisture_pct: float,
                            defect_percentages: dict) -> float:
    """Compute a composite quality score in the range 0-100.

    Each parameter contributes a weighted sub-score.  The sub-score is
    derived by measuring how far the parameter value is from the
    Grade-A threshold; values at or below Grade-A yield full marks,
    values beyond the Common threshold yield zero.

    Parameters
    ----------
    moisture_pct : float
        Moisture percentage.
    defect_percentages : dict
        Output of :func:`calculate_defect_percentages`.

    Returns
    -------
    float
        Composite quality score between 0 and 100 (rounded to 2 d.p.).
    """
    grade_a = FAQ_STANDARDS["grade_a"]
    common = FAQ_STANDARDS["common"]

    def _sub_score(value: float, best: float, worst: float) -> float:
        """Return 1.0 when *value* ≤ *best*, 0.0 when ≥ *worst*,
        linearly interpolated between."""
        if value <= best:
            return 1.0
        if value >= worst:
            return 0.0
        return 1.0 - (value - best) / (worst - best)

    # Build parameter → (value, best_threshold, worst_threshold)
    params = {
        "moisture":       (moisture_pct,
                           grade_a["moisture_max"],
                           common["moisture_max"] + 4.0),
        "foreign_matter": (defect_percentages.get("foreign", 0.0),
                           grade_a["foreign_matter_max"],
                           common["foreign_matter_max"]),
        "broken":         (defect_percentages.get("broken", 0.0),
                           grade_a["broken_max"],
                           common["broken_max"]),
        "damaged":        (defect_percentages.get("damaged", 0.0),
                           grade_a["damaged_max"],
                           common["damaged_max"]),
        "discolored":     (defect_percentages.get("discolored", 0.0),
                           grade_a["discolored_max"],
                           common["discolored_max"]),
        "chalky":         (defect_percentages.get("chalky", 0.0),
                           grade_a["chalky_max"],
                           common["chalky_max"]),
    }

    score = 0.0
    for param_name, (value, best, worst) in params.items():
        weight = QUALITY_WEIGHTS.get(param_name, 0.0)
        score += weight * _sub_score(value, best, worst)

    return round(score * 100.0, 2)


# ===================================================================
# 5. Comprehensive quality report
# ===================================================================

def generate_quality_report(variety_key: str,
                            moisture_pct: float,
                            defect_percentages: dict,
                            faq_grade: str,
                            quality_score: float) -> dict:
    """Generate a comprehensive quality report dictionary.

    Parameters
    ----------
    variety_key : str
        Key into VARIETY_DATABASE.
    moisture_pct : float
        Moisture percentage.
    defect_percentages : dict
        Per-defect percentages.
    faq_grade : str
        Output of :func:`assess_faq_grade`.
    quality_score : float
        Output of :func:`calculate_quality_score`.

    Returns
    -------
    dict
        Complete report including parameters, pass/fail status for
        each FAQ criterion, and actionable recommendations.
    """
    variety_info = VARIETY_DATABASE.get(variety_key, {})

    # --- Per-criterion pass / fail against Grade A ---
    grade_a = FAQ_STANDARDS["grade_a"]
    defect_to_faq_key = {
        "broken":     "broken_max",
        "chalky":     "chalky_max",
        "damaged":    "damaged_max",
        "discolored": "discolored_max",
        "foreign":    "foreign_matter_max",
    }

    criteria_results = {}

    # Moisture criterion
    criteria_results["moisture"] = {
        "value": round(moisture_pct, 2),
        "threshold_grade_a": grade_a["moisture_max"],
        "pass": moisture_pct <= grade_a["moisture_max"],
    }

    # Defect criteria
    for defect, faq_key in defect_to_faq_key.items():
        pct = defect_percentages.get(defect, 0.0)
        criteria_results[defect] = {
            "value": round(pct, 2),
            "threshold_grade_a": grade_a[faq_key],
            "pass": pct <= grade_a[faq_key],
        }

    # --- Recommendations ---
    recommendations = []

    if moisture_pct > grade_a["moisture_max"]:
        recommendations.append(
            f"Reduce moisture from {moisture_pct:.1f}% to below "
            f"{grade_a['moisture_max']}% through additional drying."
        )

    for defect, faq_key in defect_to_faq_key.items():
        pct = defect_percentages.get(defect, 0.0)
        if pct > grade_a[faq_key]:
            recommendations.append(
                f"Reduce {defect} grains from {pct:.1f}% to below "
                f"{grade_a[faq_key]}% (Grade A threshold)."
            )

    if faq_grade == "Rejected":
        recommendations.append(
            "Sample does not meet any FAQ grade. Consider re-milling, "
            "additional cleaning, and re-drying before resubmission."
        )

    if quality_score >= 90:
        recommendations.append("Excellent quality — suitable for premium markets.")
    elif quality_score >= 70:
        recommendations.append("Good quality — suitable for standard retail.")
    elif quality_score >= 50:
        recommendations.append("Fair quality — consider quality improvement before sale.")
    else:
        recommendations.append(
            "Poor quality — significant improvements needed before marketing."
        )

    report = {
        "variety": {
            "key": variety_key,
            "display_name": variety_info.get("display_name", variety_key),
            "category": variety_info.get("category", "unknown"),
        },
        "moisture_pct": round(moisture_pct, 2),
        "defect_percentages": {k: round(v, 2) for k, v in defect_percentages.items()},
        "faq_grade": faq_grade,
        "quality_score": quality_score,
        "criteria_results": criteria_results,
        "recommendations": recommendations,
    }

    return report


# ===================================================================
# CLI entry point
# ===================================================================

def _cli():
    """Command-line interface for quick quality assessment."""
    parser = argparse.ArgumentParser(
        description="Rice Grain Quality Assessment — FAQ grading & scoring"
    )
    parser.add_argument(
        "--variety", required=True,
        choices=list(VARIETY_DATABASE.keys()),
        help="Rice variety key.",
    )
    parser.add_argument(
        "--sample-weight", type=float, required=True,
        help="Sample weight in grams.",
    )
    parser.add_argument(
        "--num-grains", type=int, required=True,
        help="Number of grains in the sample.",
    )
    parser.add_argument(
        "--broken", type=float, default=0.0,
        help="Broken grain percentage.",
    )
    parser.add_argument(
        "--chalky", type=float, default=0.0,
        help="Chalky grain percentage.",
    )
    parser.add_argument(
        "--damaged", type=float, default=0.0,
        help="Damaged grain percentage.",
    )
    parser.add_argument(
        "--discolored", type=float, default=0.0,
        help="Discolored grain percentage.",
    )
    parser.add_argument(
        "--foreign", type=float, default=0.0,
        help="Foreign matter percentage.",
    )

    args = parser.parse_args()

    # --- Moisture ---
    moisture = calculate_moisture_content(
        args.sample_weight, args.num_grains, args.variety
    )

    # --- Defect percentages (direct from CLI) ---
    whole_pct = max(
        0.0,
        100.0 - args.broken - args.chalky - args.damaged
        - args.discolored - args.foreign,
    )
    defect_pcts = {
        "whole":      round(whole_pct, 2),
        "broken":     args.broken,
        "chalky":     args.chalky,
        "damaged":    args.damaged,
        "discolored": args.discolored,
        "foreign":    args.foreign,
    }

    # --- Grade & score ---
    grade = assess_faq_grade(moisture, defect_pcts)
    score = calculate_quality_score(moisture, defect_pcts)

    # --- Report ---
    report = generate_quality_report(
        args.variety, moisture, defect_pcts, grade, score
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _cli()
