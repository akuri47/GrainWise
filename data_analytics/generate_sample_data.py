"""
Generate Sample Grain Feature Data
====================================
Creates realistic synthetic grain feature data for the data analytics module.
Saves to data_analytics/data/grain_features.csv.

Usage:
    python data_analytics/generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Variety-specific morphological parameters (mean, std)
# Based on actual rice grain measurements
VARIETY_PARAMS = {
    "swarna":         {"length": (4.5, 0.35), "width": (2.2, 0.20), "category": "short"},
    "1010":           {"length": (5.8, 0.40), "width": (2.0, 0.18), "category": "medium"},
    "1001":           {"length": (4.3, 0.30), "width": (2.3, 0.22), "category": "short"},
    "ganga_kaveri":   {"length": (4.6, 0.32), "width": (2.4, 0.20), "category": "short"},
    "mansuri":        {"length": (5.5, 0.38), "width": (1.9, 0.16), "category": "medium"},
    "golden_mansuri": {"length": (5.7, 0.42), "width": (2.1, 0.18), "category": "medium"},
    "nati_mansuri":   {"length": (4.8, 0.34), "width": (2.3, 0.20), "category": "short"},
    "sonam":          {"length": (6.8, 0.45), "width": (1.8, 0.15), "category": "long"},
    "ir64":           {"length": (5.6, 0.40), "width": (2.0, 0.17), "category": "medium"},
}

# Color feature baselines (slight variety-specific offsets)
COLOR_PARAMS = {
    "swarna":         {"rg": 1.08, "rb": 1.28, "gb": 1.18, "ns": 0.14},
    "1010":           {"rg": 1.12, "rb": 1.32, "gb": 1.20, "ns": 0.16},
    "1001":           {"rg": 1.10, "rb": 1.30, "gb": 1.19, "ns": 0.15},
    "ganga_kaveri":   {"rg": 1.09, "rb": 1.27, "gb": 1.17, "ns": 0.13},
    "mansuri":        {"rg": 1.11, "rb": 1.31, "gb": 1.21, "ns": 0.15},
    "golden_mansuri": {"rg": 1.15, "rb": 1.35, "gb": 1.22, "ns": 0.17},
    "nati_mansuri":   {"rg": 1.10, "rb": 1.29, "gb": 1.18, "ns": 0.14},
    "sonam":          {"rg": 1.13, "rb": 1.33, "gb": 1.21, "ns": 0.16},
    "ir64":           {"rg": 1.11, "rb": 1.30, "gb": 1.19, "ns": 0.15},
}

DEFECT_PROBS = [0.65, 0.12, 0.08, 0.06, 0.05, 0.04]
DEFECT_TYPES = ["whole", "broken", "chalky", "damaged", "discolored", "foreign"]

GRAINS_PER_VARIETY = 130


def generate_data() -> pd.DataFrame:
    """Generate realistic synthetic grain feature data."""
    np.random.seed(42)
    records = []

    for variety, params in VARIETY_PARAMS.items():
        l_mean, l_std = params["length"]
        w_mean, w_std = params["width"]
        cp = COLOR_PARAMS[variety]

        for _ in range(GRAINS_PER_VARIETY):
            # Morphological features
            length = max(2.0, np.random.normal(l_mean, l_std))
            width = max(1.0, np.random.normal(w_mean, w_std))
            aspect_ratio = length / width
            area = np.pi / 4 * length * width  # ellipse approximation

            circularity = np.clip(np.random.normal(0.70, 0.10), 0.30, 1.0)
            solidity = np.clip(np.random.normal(0.92, 0.05), 0.70, 1.0)
            eccentricity = np.clip(1.0 - 1.0 / (aspect_ratio + 0.01), 0.1, 0.99)

            # Color features (with variety-specific offsets)
            rg_ratio = max(0.5, np.random.normal(cp["rg"], 0.06))
            rb_ratio = max(0.5, np.random.normal(cp["rb"], 0.08))
            gb_ratio = max(0.5, np.random.normal(cp["gb"], 0.06))
            norm_s = max(0.01, np.random.normal(cp["ns"], 0.03))

            # Defect assignment
            defect = np.random.choice(DEFECT_TYPES, p=DEFECT_PROBS)

            # Quality grade based on defect
            if defect == "whole":
                grade = np.random.choice(
                    ["Grade A", "Grade B", "Common"],
                    p=[0.55, 0.30, 0.15]
                )
            elif defect == "broken":
                grade = np.random.choice(
                    ["Grade B", "Common", "Rejected"],
                    p=[0.3, 0.5, 0.2]
                )
            else:
                grade = np.random.choice(
                    ["Common", "Rejected"],
                    p=[0.4, 0.6]
                )

            records.append({
                "variety": variety,
                "defect_type": defect,
                "length_mm": round(length, 3),
                "width_mm": round(width, 3),
                "aspect_ratio": round(aspect_ratio, 3),
                "area_mm2": round(area, 3),
                "circularity": round(circularity, 4),
                "solidity": round(solidity, 4),
                "eccentricity": round(eccentricity, 4),
                "rg_ratio": round(rg_ratio, 4),
                "rb_ratio": round(rb_ratio, 4),
                "gb_ratio": round(gb_ratio, 4),
                "norm_s": round(norm_s, 4),
                "quality_grade": grade,
            })

    return pd.DataFrame(records)


def main():
    """Generate and save sample grain feature data."""
    print("=" * 50)
    print("  GENERATING SAMPLE GRAIN DATA")
    print("=" * 50)

    os.makedirs(DATA_DIR, exist_ok=True)

    df = generate_data()
    output_path = os.path.join(DATA_DIR, "grain_features.csv")
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} grain records")
    print(f"Saved to: {output_path}")
    print(f"\nVariety distribution:")
    for variety, count in df["variety"].value_counts().items():
        print(f"  {variety:<20s} {count:>5d} grains")
    print(f"\nDefect distribution:")
    for defect, count in df["defect_type"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {defect:<15s} {count:>5d} ({pct:.1f}%)")
    print(f"\nGrade distribution:")
    for grade, count in df["quality_grade"].value_counts().items():
        print(f"  {grade:<15s} {count:>5d}")
    print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()
