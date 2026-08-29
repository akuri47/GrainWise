"""
Rice Grain Quality Analyzer — Price Recommendation Module
==========================================================
Recommends market prices based on rice variety and FAQ quality
grade, adjusting for defect exceedances using Government of India
FAQ standards and approximate 2024 MSP / mandi rate tables.

Functions
---------
- get_base_price          : Look up base price from PRICE_TABLE.
- calculate_price_adjustments : Compute deductions for FAQ exceedances.
- recommend_price         : Return recommended price, range, justification.
- compare_grade_prices    : Price comparison across all grades for a variety.
- generate_price_report   : Full pricing report with breakdown.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup — allow running from anywhere
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PRICE_TABLE,
    PRICE_DEDUCTION_PER_PCT,
    FAQ_STANDARDS,
    VARIETY_DATABASE,
)


# ============================================================
# 1. Base Price Lookup
# ============================================================

def get_base_price(variety_key: str, faq_grade: str,
                   custom_price: float = None) -> dict:
    """
    Look up the base price for a given rice variety and FAQ grade.

    Priority order:
        1. custom_price (user-entered price override)
        2. Live Agmarknet API price from data.gov.in
        3. Static PRICE_TABLE fallback

    Parameters
    ----------
    variety_key : str
        Variety identifier as used in PRICE_TABLE.
    faq_grade : str
        FAQ quality grade: "grade_a", "grade_b", "common", or "rejected".
    custom_price : float or None
        User-provided base price override (Rs/quintal). If given, skips API.

    Returns
    -------
    dict
        {
            "base_price": float,
            "price_source": str,     # "custom" | "live_api" | "static"
            "used_other": bool,      # True if API used "Other" variety category
            "api_variety": str,      # Agmarknet variety name queried (or "")
        }
    """
    variety_key = variety_key.lower()

    if variety_key not in PRICE_TABLE:
        available = ", ".join(sorted(PRICE_TABLE.keys()))
        raise ValueError(
            f"Unknown variety '{variety_key}'. Available: {available}"
        )

    grade_prices = PRICE_TABLE[variety_key]
    faq_grade = faq_grade.lower()

    if faq_grade not in grade_prices:
        available = ", ".join(sorted(grade_prices.keys()))
        raise ValueError(
            f"Unknown grade '{faq_grade}'. Available: {available}"
        )

    # Determine if this variety uses the "Other" category on Agmarknet
    # (this flag is set regardless of whether the API call succeeds)
    try:
        from src.mandi_api import VARIETY_MAPPING
        mapping = VARIETY_MAPPING.get(variety_key, {})
        is_other_variety = mapping.get("is_other", False)
    except Exception:
        is_other_variety = False

    result = {
        "base_price": float(grade_prices[faq_grade]),
        "price_source": "static",
        "used_other": is_other_variety,
        "api_variety": mapping.get("variety", "") if mapping else "",
        "mandi_details": []
    }

    # --- Priority 1: User-entered custom price ---
    if custom_price is not None and custom_price > 0:
        result["base_price"] = float(custom_price)
        result["price_source"] = "custom"
        result["used_other"] = False  # user overrode, no "Other" warning needed
        print(f"  [Price] Using user-entered custom price: Rs.{custom_price:.2f}/qtl")
        return result

    # --- Priority 2: Live Agmarknet API ---
    try:
        from src.mandi_api import fetch_live_price
        api_result = fetch_live_price(variety_key)
        live_modal_price = api_result.get("price")

        if live_modal_price and live_modal_price > 0:
            # The live modal price IS the real market price for this variety.
            # Use it directly as base price — do NOT scale by static grade ratio,
            # as that caused artificial inflation (e.g. ₹3363 for Sonam).
            # Quality deductions applied afterwards will reduce it for poor grades.
            adjusted_price = live_modal_price

            result["base_price"] = round(adjusted_price, 2)
            result["price_source"] = "live_api"
            result["used_other"] = api_result.get("used_other", is_other_variety)
            result["api_variety"] = api_result.get("api_variety", "")
            result["mandi_details"] = api_result.get("mandi_details", [])
            source_label = "Other category" if result["used_other"] else api_result.get("api_variety", "")
            print(f"  [API] Live Agmarknet price: Rs.{result['base_price']:.2f}/qtl"
                  f" (variety: {source_label})")
            return result
    except Exception as e:
        print(f"  [API Error] {e}")

    # --- Priority 3: Static fallback ---
    other_note = " (variety uses 'Other' category on Agmarknet)" if is_other_variety else ""
    print(f"  [Price] Using static price table: Rs.{result['base_price']:.2f}/qtl{other_note}")
    return result


# ============================================================
# 2. Price Adjustments (deductions for FAQ parameter exceedances)
# ============================================================

def calculate_price_adjustments(
    defect_percentages: dict,
    faq_grade: str,
) -> dict:
    """
    Calculate price deductions for each defect type whose measured
    percentage exceeds the FAQ threshold for *faq_grade*.

    For each defect type present in both *defect_percentages* and
    ``PRICE_DEDUCTION_PER_PCT``, the deduction is::

        deduction = max(0, measured_pct - threshold) × deduction_rate

    Parameters
    ----------
    defect_percentages : dict
        Mapping of defect type to measured percentage, e.g.::

            {"broken": 8.5, "damaged": 1.2, "foreign_matter": 0.3, ...}

    faq_grade : str
        FAQ grade whose thresholds are used (``"grade_a"``,
        ``"grade_b"``, ``"common"``).  If the grade has no
        thresholds (e.g. ``"rejected"``), no deductions are applied.

    Returns
    -------
    dict
        ``{"deductions": {defect: amount, ...},
           "total_deduction": float,
           "details": [human-readable strings]}``
    """
    faq_grade = faq_grade.lower()

    deductions: dict[str, float] = {}
    details: list[str] = []

    # Grades without FAQ thresholds (e.g. "rejected") → no deductions
    if faq_grade not in FAQ_STANDARDS:
        return {
            "deductions": deductions,
            "total_deduction": 0.0,
            "details": [f"No FAQ thresholds defined for grade '{faq_grade}'; "
                        "no deductions applied."],
        }

    thresholds = FAQ_STANDARDS[faq_grade]

    # Map defect names to threshold keys
    _defect_to_threshold_key = {
        "foreign_matter": "foreign_matter_max",
        "broken":         "broken_max",
        "damaged":        "damaged_max",
        "discolored":     "discolored_max",
        "chalky":         "chalky_max",
    }

    for defect, measured_pct in defect_percentages.items():
        defect = defect.lower()
        threshold_key = _defect_to_threshold_key.get(defect)
        deduction_rate = PRICE_DEDUCTION_PER_PCT.get(defect, 0)

        if threshold_key is None or deduction_rate == 0:
            continue

        threshold = thresholds.get(threshold_key, float("inf"))
        excess = max(0.0, measured_pct - threshold)

        if excess > 0:
            amount = round(excess * deduction_rate, 2)
            deductions[defect] = amount
            details.append(
                f"{defect}: {measured_pct:.2f}% (limit {threshold:.1f}%) "
                f"→ excess {excess:.2f}% × ₹{deduction_rate}/% = ₹{amount:.2f}"
            )

    total = round(sum(deductions.values()), 2)

    if not deductions:
        details.append("All defect parameters within FAQ limits — no deductions.")

    return {
        "deductions": deductions,
        "total_deduction": total,
        "details": details,
    }


# ============================================================
# 3. Recommend Price
# ============================================================

def _quality_score_multiplier(quality_score: float) -> float:
    """
    Map quality score (0–100) to a price multiplier.

    Formula:  multiplier = 0.65 + (score / 100) × 0.50

    Anchoring:
      score 100 → 1.15  (Grade A equivalent, ~15% above Common market price)
      score  70 → 1.00  (Matches Common grade price exactly — baseline)
      score  50 → 0.90  (10% below Common — acceptable but below standard)
      score   0 → 0.65  (Floor — rejected/feed grade, ~35% below Common)

    Evidence basis:
      - Grade A vs Common open-market differential: 10–18%
        (Agmarknet mandi data, AP/Telangana, 2023–24)
      - FCI procurement: rejected lots at 60–70% of Common MSP
        (FCI Quality Specifications, 2023)
      - CACP MSP: Grade A officially ₹20 above Common (~0.9% at MSP level),
        but open market shows 10–18% premium
        (CACP Report 2024-25, Table 3.1)
      - Score=70 chosen as the "no adjustment" baseline because the FAQ
        'Common' grade threshold corresponds roughly to 70/100 in our
        weighted scoring (moisture 14%, broken 25%, foreign 1%, damaged 3%)
    """
    score = max(0.0, min(100.0, quality_score))
    return round(0.65 + (score / 100.0) * 0.50, 4)


def recommend_price(
    variety_key: str,
    faq_grade: str,
    defect_percentages: dict,
    quality_score: float = 70.0,
    custom_price: float = None,
) -> dict:
    """
    Calculate a recommended price using quality score multiplier.

    The base price is the live Common-grade market price (or static fallback).
    The quality score (0–100) is mapped to a multiplier via
    ``_quality_score_multiplier()`` and applied to derive the recommended price.

    Parameters
    ----------
    variety_key : str
        Variety key (e.g. ``"swarna"``).
    faq_grade : str
        FAQ grade (informational only, not used for multiplier calculation).
    defect_percentages : dict
        Measured defect percentages (kept for justification text).
    quality_score : float
        Overall quality score 0–100 from ``calculate_quality_score()``.
    custom_price : float or None
        User-provided base price override (Rs/quintal). If given, skips API.

    Returns
    -------
    dict
        ``{"base_price": float,
           "quality_multiplier": float,
           "recommended_price": float,
           "price_range": {"min": float, "max": float},
           "price_source": str,
           "used_other": bool,
           "quality_score": float,
           "justification": str}``
    """
    price_info = get_base_price(variety_key, faq_grade, custom_price=custom_price)
    base_price = price_info["base_price"]
    price_source = price_info["price_source"]
    used_other = price_info["used_other"]
    api_variety = price_info.get("api_variety", "")
    mandi_details = price_info.get("mandi_details", [])

    # Quality score multiplier (Option 2 — score-based continuous adjustment)
    multiplier = _quality_score_multiplier(quality_score)
    recommended = round(base_price * multiplier, 2)

    # Price range: ±5% of recommended (natural mandi negotiation band)
    price_min = round(recommended * 0.95, 2)
    price_max = round(recommended * 1.05, 2)

    # Build justification
    variety_display = VARIETY_DATABASE.get(
        variety_key.lower(), {}
    ).get("display_name", variety_key)

    lines = [
        f"Variety          : {variety_display}",
        f"FAQ Grade        : {faq_grade.upper().replace('_', ' ')}",
        f"Quality Score    : {quality_score:.1f}/100",
        f"Base Price       : Rs.{base_price:.2f}/qtl ({price_source}, Common grade)",
        f"Quality Multiplier: {multiplier:.4f}  "
        f"[formula: 0.65 + ({quality_score:.1f}/100) × 0.50]",
        f"Recommended Price : Rs.{recommended:.2f}/qtl",
        f"Price Range       : Rs.{price_min:.2f} – Rs.{price_max:.2f}/qtl",
    ]
    if used_other:
        lines.append(
            "Note: Price based on 'Other' rice category "
            "(variety not directly tracked on Agmarknet)"
        )

    return {
        "base_price": base_price,
        "quality_multiplier": multiplier,
        "quality_score": quality_score,
        "recommended_price": recommended,
        "price_range": {"min": price_min, "max": price_max},
        "price_source": price_source,
        "used_other": used_other,
        "api_variety": api_variety,
        "mandi_details": mandi_details,
        "justification": "\n".join(lines),
        # Keep adjustments key empty for backward compat with frontend
        "adjustments": {"deductions": {}, "total_deduction": 0.0, "details": []},
    }


# ============================================================
# 4. Compare Grade Prices
# ============================================================

def compare_grade_prices(variety_key: str) -> dict:
    """
    Show the base-price comparison across all FAQ grades for a
    given variety.

    Parameters
    ----------
    variety_key : str
        Variety key.

    Returns
    -------
    dict
        ``{"variety": str, "display_name": str,
           "prices": {grade: price, ...},
           "summary": str}``

    Raises
    ------
    ValueError
        If the variety is not found in PRICE_TABLE.
    """
    variety_key = variety_key.lower()
    if variety_key not in PRICE_TABLE:
        available = ", ".join(sorted(PRICE_TABLE.keys()))
        raise ValueError(
            f"Unknown variety '{variety_key}'. Available: {available}"
        )

    prices = PRICE_TABLE[variety_key]
    display_name = VARIETY_DATABASE.get(variety_key, {}).get(
        "display_name", variety_key
    )

    # Pretty summary
    header = f"Price comparison for {display_name}:"
    rows = []
    for grade in ["grade_a", "grade_b", "common", "rejected"]:
        p = prices.get(grade)
        if p is not None:
            label = grade.upper().replace("_", " ")
            rows.append(f"  {label:<12s} : ₹{p:,.2f}/qtl")

    summary = "\n".join([header] + rows)

    return {
        "variety": variety_key,
        "display_name": display_name,
        "prices": dict(prices),
        "summary": summary,
    }


# ============================================================
# 5. Generate Full Price Report
# ============================================================

def generate_price_report(
    variety_key: str,
    faq_grade: str,
    defect_percentages: dict,
    quality_score: float = None,
) -> str:
    """
    Generate a comprehensive price report including base price,
    deductions breakdown, final recommendation, and comparison
    with other grades.

    Parameters
    ----------
    variety_key : str
        Variety key.
    faq_grade : str
        FAQ grade.
    defect_percentages : dict
        Measured defect percentages.
    quality_score : float, optional
        Composite quality score (0–100).  Included in the report
        if provided.

    Returns
    -------
    str
        Multi-line formatted price report.
    """
    recommendation = recommend_price(variety_key, faq_grade, defect_percentages)
    comparison = compare_grade_prices(variety_key)

    display_name = comparison["display_name"]
    grade_label = faq_grade.upper().replace("_", " ")

    sep = "=" * 60
    thin = "-" * 60

    lines: list[str] = [
        sep,
        "  RICE GRAIN PRICE RECOMMENDATION REPORT",
        sep,
        "",
        f"  Variety         : {display_name}",
        f"  Assigned Grade  : {grade_label}",
    ]
    if quality_score is not None:
        lines.append(f"  Quality Score   : {quality_score:.1f} / 100")
    lines.append("")

    # ---- Base price ----
    lines.append(thin)
    lines.append("  BASE PRICE")
    lines.append(thin)
    lines.append(f"  ₹{recommendation['base_price']:,.2f} per quintal "
                 f"({grade_label})")
    lines.append("")

    # ---- Deductions ----
    adj = recommendation["adjustments"]
    lines.append(thin)
    lines.append("  DEDUCTIONS BREAKDOWN")
    lines.append(thin)

    if adj["deductions"]:
        lines.append(f"  {'Defect':<18s} {'Measured':>9s} {'Limit':>7s} "
                      f"{'Excess':>7s} {'Rate':>8s} {'Deduction':>10s}")
        lines.append(f"  {'-'*18} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")

        _defect_to_key = {
            "foreign_matter": "foreign_matter_max",
            "broken":         "broken_max",
            "damaged":        "damaged_max",
            "discolored":     "discolored_max",
            "chalky":         "chalky_max",
        }

        thresholds = FAQ_STANDARDS.get(faq_grade.lower(), {})

        for defect, amount in adj["deductions"].items():
            measured = defect_percentages.get(defect, 0.0)
            threshold_key = _defect_to_key.get(defect, "")
            limit = thresholds.get(threshold_key, 0.0)
            excess = max(0.0, measured - limit)
            rate = PRICE_DEDUCTION_PER_PCT.get(defect, 0)
            lines.append(
                f"  {defect:<18s} {measured:>8.2f}% {limit:>6.1f}% "
                f"{excess:>6.2f}% ₹{rate:>5d}/% ₹{amount:>8.2f}"
            )
        lines.append(f"  {'':>52s} {'─'*10}")
        lines.append(f"  {'Total deduction':<52s} ₹{adj['total_deduction']:>8.2f}")
    else:
        lines.append("  No deductions — all parameters within FAQ limits.")
    lines.append("")

    # ---- Recommended price ----
    lines.append(thin)
    lines.append("  RECOMMENDED PRICE")
    lines.append(thin)
    rec = recommendation["recommended_price"]
    pr = recommendation["price_range"]
    lines.append(f"  ₹{rec:,.2f} per quintal")
    lines.append(f"  Expected range : ₹{pr['min']:,.2f} – ₹{pr['max']:,.2f}/qtl")
    lines.append("")

    # ---- Grade comparison ----
    lines.append(thin)
    lines.append("  GRADE-WISE PRICE COMPARISON")
    lines.append(thin)
    for grade in ["grade_a", "grade_b", "common", "rejected"]:
        p = comparison["prices"].get(grade)
        if p is not None:
            marker = " ◄" if grade == faq_grade.lower() else ""
            label = grade.upper().replace("_", " ")
            lines.append(f"  {label:<12s} : ₹{p:>8,.2f}/qtl{marker}")
    lines.append("")

    lines.append(sep)
    lines.append("  Report generated by Rice Grain Quality Analyzer")
    lines.append(sep)

    return "\n".join(lines)


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rice Grain Price Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python price_recommendation.py --variety swarna --grade grade_a\n"
            "  python price_recommendation.py --variety ir64 --grade grade_b "
            "--broken 18 --damaged 4\n"
            "  python price_recommendation.py --variety sonam --compare\n"
        ),
    )

    parser.add_argument(
        "--variety", type=str, required=True,
        help="Variety key (e.g. swarna, ir64, sonam)",
    )
    parser.add_argument(
        "--grade", type=str, default="grade_a",
        help="FAQ grade: grade_a, grade_b, common, rejected (default: grade_a)",
    )
    parser.add_argument("--broken", type=float, default=0.0,
                        help="Broken grain percentage")
    parser.add_argument("--damaged", type=float, default=0.0,
                        help="Damaged grain percentage")
    parser.add_argument("--discolored", type=float, default=0.0,
                        help="Discolored grain percentage")
    parser.add_argument("--chalky", type=float, default=0.0,
                        help="Chalky grain percentage")
    parser.add_argument("--foreign_matter", type=float, default=0.0,
                        help="Foreign matter percentage")
    parser.add_argument("--quality_score", type=float, default=None,
                        help="Composite quality score (0–100)")
    parser.add_argument(
        "--compare", action="store_true",
        help="Show price comparison across all grades for the variety",
    )

    args = parser.parse_args()

    # Build defect dict from CLI args
    defect_pcts = {
        "broken":         args.broken,
        "damaged":        args.damaged,
        "discolored":     args.discolored,
        "chalky":         args.chalky,
        "foreign_matter": args.foreign_matter,
    }

    if args.compare:
        result = compare_grade_prices(args.variety)
        print(result["summary"])
    else:
        report = generate_price_report(
            args.variety, args.grade, defect_pcts, args.quality_score
        )
        print(report)
