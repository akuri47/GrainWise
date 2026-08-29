"""
Rice Grain Quality Analyzer -- Shelf-Life Estimation Module
============================================================
Estimates expected rice shelf life using the halving-rule model
based on IRRI Rice Knowledge Bank & NDSU Extension (Hellevang).

Core model:
    shelf_life_days = 365 * 2^(-(T - 21) / 5)
                          * 2^(-(MC - 14) / 2)
                          * defect_factor
    Clamped to [10, 1095] days.

Humidity is NOT a core input -- it serves as a background
moisture-drift flag (RH > 70% sustained = risk of MC increase).

Functions:
    estimate_shelf_life          -- Expected shelf life in days/months.
    get_risk_level               -- MC/T-based risk categorization.
    get_storage_recommendations  -- Actionable storage advice.
    generate_shelf_life_report   -- Full report with factor breakdown.
"""

import sys
import os
import math
import argparse
import json

# ---------------------------------------------------------------------------
# Config imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    VARIETY_DATABASE,
    BASELINE_SHELF_LIFE_DAYS,
    T_REF_C,
    MC_REF_PCT,
    TEMP_HALVING_C,
    MC_HALVING_PCT,
    BREAKAGE_PENALTY_PER_PCT,
    DAMAGE_PENALTY_PER_PCT,
    DEFECT_FACTOR_FLOOR,
    MIN_SHELF_LIFE_DAYS,
    MAX_SHELF_LIFE_DAYS,
    HUMIDITY_DRIFT_THRESHOLD_RH,
    HIGH_TEMP_THRESHOLD_C,
)


# ===================================================================
# 1. Core shelf-life estimation (halving rule)
# ===================================================================

def estimate_shelf_life(temp_c: float,
                        moisture_pct_wb: float,
                        humidity_pct: float = 60.0,
                        broken_pct: float = 0.0,
                        damaged_pct: float = 0.0,
                        variety: str = None) -> dict:
    """Estimate expected shelf life using the halving-rule model.

    Formula
    -------
    shelf_life_days = BASELINE * 2^(-(T - T_ref)/5)
                                * 2^(-(MC - MC_ref)/2)
                                * defect_factor
    Clamped to [10, 1095] days.

    Parameters
    ----------
    temp_c : float
        Grain / storage temperature in degrees C.
    moisture_pct_wb : float
        Grain moisture content in % wet basis.
    humidity_pct : float, optional
        Ambient relative humidity in % (default 60).
        NOT used in the core formula -- only for the drift flag.
    broken_pct : float, optional
        Percentage of broken grains (default 0).
    damaged_pct : float, optional
        Percentage of damaged grains (default 0).
    variety : str or None, optional
        Variety key (stored for future per-variety tuning, not used
        in v1 math).

    Returns
    -------
    dict
        {
            "shelf_life_days": int,
            "shelf_life_months": float,
            "shelf_life_display": str,       # e.g. "8.2 months"
            "risk_level": str,               # Low | Moderate | Elevated | High
            "moisture_drift_flag": bool,     # True if RH > threshold
            "factors": {
                "temp_factor": float,
                "moisture_factor": float,
                "defect_factor": float,
            },
        }
    """
    # --- Temperature factor: halves every +5 C above T_ref ---
    temp_factor = 2.0 ** (-(temp_c - T_REF_C) / TEMP_HALVING_C)

    # --- Moisture factor: halves every +2% MC above MC_ref ---
    moisture_factor = 2.0 ** (-(moisture_pct_wb - MC_REF_PCT) / MC_HALVING_PCT)

    # --- Defect factor: linear penalty, floored ---
    defect_factor = max(
        DEFECT_FACTOR_FLOOR,
        1.0 - BREAKAGE_PENALTY_PER_PCT * broken_pct
            - DAMAGE_PENALTY_PER_PCT * damaged_pct,
    )

    # --- Composite shelf life ---
    shelf_life_days = (
        BASELINE_SHELF_LIFE_DAYS
        * temp_factor
        * moisture_factor
        * defect_factor
    )

    # Clamp to sane range
    shelf_life_days = max(MIN_SHELF_LIFE_DAYS,
                          min(shelf_life_days, MAX_SHELF_LIFE_DAYS))
    shelf_life_days = int(round(shelf_life_days))

    shelf_life_months = round(shelf_life_days / 30.0, 1)

    # Display string
    if shelf_life_days < 60:
        shelf_life_display = f"{shelf_life_days} days"
    else:
        shelf_life_display = f"{shelf_life_months} months"

    # --- Risk level ---
    risk_level = get_risk_level(moisture_pct_wb, temp_c)

    # --- Humidity drift flag ---
    moisture_drift_flag = humidity_pct > HUMIDITY_DRIFT_THRESHOLD_RH

    return {
        "shelf_life_days": shelf_life_days,
        "shelf_life_months": shelf_life_months,
        "shelf_life_display": shelf_life_display,
        "risk_level": risk_level,
        "moisture_drift_flag": moisture_drift_flag,
        "factors": {
            "baseline_days": BASELINE_SHELF_LIFE_DAYS,
            "temp_factor": round(temp_factor, 4),
            "moisture_factor": round(moisture_factor, 4),
            "defect_factor": round(defect_factor, 4),
        },
    }


# ===================================================================
# 2. Risk categorization
# ===================================================================

def get_risk_level(moisture_pct_wb: float, temp_c: float) -> str:
    """Categorize storage risk based on moisture and temperature.

    Risk bands (MC-based, escalated by high temperature):

    | MC (wb)  | Risk (normal T) | Risk (T > 30 C) |
    |----------|-----------------|------------------|
    | <= 12%   | Low             | Moderate         |
    | 12-14%   | Moderate        | Elevated         |
    | 14-16%   | Elevated        | High             |
    | > 16%    | High            | High             |

    Parameters
    ----------
    moisture_pct_wb : float
        Grain moisture content in % wet basis.
    temp_c : float
        Storage temperature in degrees C.

    Returns
    -------
    str
        One of "Low", "Moderate", "Elevated", "High".
    """
    hot = temp_c > HIGH_TEMP_THRESHOLD_C

    if moisture_pct_wb <= 12.0:
        return "Moderate" if hot else "Low"
    elif moisture_pct_wb <= 14.0:
        return "Elevated" if hot else "Moderate"
    elif moisture_pct_wb <= 16.0:
        return "High" if hot else "Elevated"
    else:
        return "High"


# ===================================================================
# 3. Storage recommendations
# ===================================================================

def get_storage_recommendations(shelf_life_months: float,
                                moisture_pct: float,
                                faq_grade: str,
                                risk_level: str = "Moderate",
                                moisture_drift_flag: bool = False) -> list:
    """Return actionable storage advice based on conditions.

    Parameters
    ----------
    shelf_life_months : float
        Estimated shelf life in months.
    moisture_pct : float
        Measured moisture percentage.
    faq_grade : str
        FAQ grade string.
    risk_level : str
        Risk level from get_risk_level().
    moisture_drift_flag : bool
        Whether the humidity drift flag is set.

    Returns
    -------
    list[str]
        List of recommendation strings.
    """
    recommendations = []

    # --- Moisture-based advice ---
    if moisture_pct > 16.0:
        recommendations.append(
            f"CRITICAL: Moisture is {moisture_pct:.1f}% (well above 14% safe limit). "
            "Dry the grain immediately to 12-13% before storage to prevent "
            "rapid mould growth and insect infestation."
        )
    elif moisture_pct > 14.0:
        recommendations.append(
            f"WARNING: Moisture is {moisture_pct:.1f}% (above 14% safe limit). "
            "Dry the grain to 12-13% before long-term storage."
        )
    elif moisture_pct > 12.0:
        recommendations.append(
            f"Moisture ({moisture_pct:.1f}%) is within commercial storage range "
            "(12-14%). For extended storage (>6 months), consider drying to 12%."
        )
    else:
        recommendations.append(
            f"Moisture level ({moisture_pct:.1f}%) is excellent for "
            "long-term storage."
        )

    # --- Humidity drift warning ---
    if moisture_drift_flag:
        recommendations.append(
            "ALERT: Ambient humidity exceeds 70%. In open/breathable storage, "
            "grain moisture may drift upward over time. Increase MC monitoring "
            "frequency or switch to airtight/hermetic storage."
        )

    # --- Risk-level advice ---
    if risk_level == "High":
        recommendations.append(
            "HIGH RISK: Current conditions pose serious spoilage risk. "
            "Prioritise immediate sale, processing, or emergency drying."
        )
    elif risk_level == "Elevated":
        recommendations.append(
            "ELEVATED RISK: Spoilage risk is above normal. Monitor closely "
            "and improve storage conditions (reduce temperature, lower moisture)."
        )

    # --- Shelf-life-based advice ---
    if shelf_life_months < 2:
        recommendations.append(
            "Very short shelf life estimated. Prioritise immediate sale or "
            "processing to avoid quality loss."
        )
    elif shelf_life_months < 6:
        recommendations.append(
            "Moderate shelf life. Sell within 3-6 months or improve storage "
            "conditions (cool, dry environment)."
        )
    else:
        recommendations.append(
            f"Good shelf life of ~{shelf_life_months:.0f} months expected "
            "under current conditions."
        )

    # --- General storage best practices ---
    recommendations.append(
        "Store in clean, dry, well-ventilated warehouses at <=25 C and "
        "<=60% RH."
    )
    recommendations.append(
        "Use hermetic (airtight) storage bags or metal bins where possible "
        "to protect against insects and moisture re-absorption."
    )

    # --- Grade-specific advice ---
    if faq_grade == "Grade A":
        recommendations.append(
            "Grade A quality -- suitable for premium packaging. Maintain "
            "conditions to preserve grade."
        )
    elif faq_grade == "Grade B":
        recommendations.append(
            "Grade B quality -- marketable but monitor for further quality "
            "degradation during storage."
        )
    elif faq_grade == "Common":
        recommendations.append(
            "Common grade -- consider selling sooner to prevent further "
            "downgrading."
        )
    elif faq_grade == "Rejected":
        recommendations.append(
            "Sample is rejected grade. Not recommended for long-term "
            "storage; consider reprocessing or blending."
        )

    return recommendations


# ===================================================================
# 4. Comprehensive shelf-life report
# ===================================================================

def generate_shelf_life_report(variety_key: str,
                               moisture_pct: float,
                               broken_pct: float,
                               damaged_pct: float,
                               storage_temp_c: float = 25.0,
                               storage_humidity_rh: float = 60.0,
                               faq_grade: str = "Common") -> dict:
    """Generate a full shelf-life report.

    Parameters
    ----------
    variety_key : str
        Key into VARIETY_DATABASE.
    moisture_pct : float
        Moisture percentage.
    broken_pct : float
        Broken grain percentage.
    damaged_pct : float
        Damaged grain percentage.
    storage_temp_c : float, optional
        Storage temperature in C (default 25).
    storage_humidity_rh : float, optional
        Storage relative humidity % (default 60).
    faq_grade : str, optional
        FAQ grade string (default "Common").

    Returns
    -------
    dict
        Complete report with variety info, input conditions, shelf-life
        estimate, factor breakdown, risk level, and recommendations.
    """
    variety_info = VARIETY_DATABASE.get(variety_key, {})

    # Shelf-life calculation using new halving-rule model
    sl_result = estimate_shelf_life(
        temp_c=storage_temp_c,
        moisture_pct_wb=moisture_pct,
        humidity_pct=storage_humidity_rh,
        broken_pct=broken_pct,
        damaged_pct=damaged_pct,
        variety=variety_key,
    )

    shelf_life_months = sl_result["shelf_life_months"]

    # Recommendations
    recommendations = get_storage_recommendations(
        shelf_life_months, moisture_pct, faq_grade,
        risk_level=sl_result["risk_level"],
        moisture_drift_flag=sl_result["moisture_drift_flag"],
    )

    report = {
        "variety": {
            "key": variety_key,
            "display_name": variety_info.get("display_name", variety_key),
            "category": variety_info.get("category", "unknown"),
        },
        "input_conditions": {
            "moisture_pct": round(moisture_pct, 2),
            "broken_pct": round(broken_pct, 2),
            "damaged_pct": round(damaged_pct, 2),
            "storage_temp_c": round(storage_temp_c, 1),
            "storage_humidity_rh": round(storage_humidity_rh, 1),
        },
        "shelf_life_days": sl_result["shelf_life_days"],
        "shelf_life_months": shelf_life_months,
        "shelf_life_display": sl_result["shelf_life_display"],
        "risk_level": sl_result["risk_level"],
        "moisture_drift_flag": sl_result["moisture_drift_flag"],
        "factor_breakdown": sl_result["factors"],
        "faq_grade": faq_grade,
        "recommendations": recommendations,
    }

    return report


# ===================================================================
# CLI entry point
# ===================================================================

def _cli():
    """Command-line interface for shelf-life estimation."""
    parser = argparse.ArgumentParser(
        description="Rice Shelf-Life Estimation (Halving Rule Model)"
    )
    parser.add_argument(
        "--variety", required=False, default=None,
        choices=list(VARIETY_DATABASE.keys()),
        help="Rice variety key (optional, stored for future use).",
    )
    parser.add_argument(
        "--moisture", type=float, required=True,
        help="Moisture content (%% wet basis).",
    )
    parser.add_argument(
        "--temp", type=float, default=25.0,
        help="Storage temperature in C (default: 25).",
    )
    parser.add_argument(
        "--humidity", type=float, default=60.0,
        help="Ambient relative humidity %% (default: 60).",
    )
    parser.add_argument(
        "--broken", type=float, default=0.0,
        help="Broken grain percentage.",
    )
    parser.add_argument(
        "--damaged", type=float, default=0.0,
        help="Damaged grain percentage.",
    )
    parser.add_argument(
        "--faq-grade", type=str, default="Common",
        choices=["Grade A", "Grade B", "Common", "Rejected"],
        help="FAQ grade (default: Common).",
    )

    args = parser.parse_args()

    # Use the report function if variety is provided, else direct estimate
    if args.variety:
        report = generate_shelf_life_report(
            variety_key=args.variety,
            moisture_pct=args.moisture,
            broken_pct=args.broken,
            damaged_pct=args.damaged,
            storage_temp_c=args.temp,
            storage_humidity_rh=args.humidity,
            faq_grade=args.faq_grade,
        )
    else:
        report = estimate_shelf_life(
            temp_c=args.temp,
            moisture_pct_wb=args.moisture,
            humidity_pct=args.humidity,
            broken_pct=args.broken,
            damaged_pct=args.damaged,
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _cli()
