"""
Rice Grain Quality Analyzer -- Variety Classifier Module
========================================================
Identifies rice variety using **per-grain** morphological and color
features with a Random Forest classifier.

Each individual grain is classified independently, enabling:
- Mixed variety detection
- Per-grain variety distribution reporting
- Confidence scoring per grain

Functions
---------
- build_training_data : Extract per-grain features from whole_grain images.
- train_classifier    : Train a Random Forest with stratified CV.
- predict_variety_per_grain : Classify each grain individually.
- predict_variety     : Convenience wrapper returning dominant variety.
- detect_mixed_variety: Flag mixed-variety lots.
- save_model / load_model : Serialize / deserialize with joblib.
- evaluate_model      : Accuracy, confusion matrix, classification report.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grain_segmentation import (
    process_variety_folder,
    aggregate_image_features,
    detect_aruco_marker,
    segment_grains,
    extract_all_grain_features,
)
from src.config import (
    VARIETY_FEATURES,
    PER_GRAIN_VARIETY_FEATURES,
    VARIETY_N_ESTIMATORS,
    VARIETY_MAX_DEPTH,
    VARIETY_RANDOM_STATE,
    VARIETY_DATABASE,
    WHOLE_GRAIN_DIR,
    MODEL_DIR,
    FOLDER_TO_VARIETY,
    MIXED_VARIETY_THRESHOLD_PCT,
)

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Optional, List
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. Build Per-Grain Training Data
# ============================================================

def build_training_data(pixels_per_mm: float) -> pd.DataFrame:
    """
    Process every variety folder under ``WHOLE_GRAIN_DIR`` and
    construct a per-grain labelled training DataFrame.

    Each grain extracted from ``whole_grain/swarna/`` is labelled
    "swarna", each grain from ``whole_grain/IR64/`` is labelled
    "ir64", etc.  This gives ~1000+ per-grain training samples
    instead of ~22 image-level samples.

    Parameters
    ----------
    pixels_per_mm : float
        ArUCo-derived calibration factor.

    Returns
    -------
    pd.DataFrame
        Per-grain training data with columns from
        ``PER_GRAIN_VARIETY_FEATURES`` plus a ``"variety"`` label.
    """
    import cv2

    all_grains: list[dict] = []

    for variety_folder in sorted(os.listdir(WHOLE_GRAIN_DIR)):
        folder_path = os.path.join(WHOLE_GRAIN_DIR, variety_folder)
        if not os.path.isdir(folder_path):
            continue

        variety_key = FOLDER_TO_VARIETY.get(
            variety_folder.lower(), variety_folder.lower()
        )

        print(f"\n[Train] Processing variety: {variety_key} "
              f"(folder: {variety_folder})")

        # Process each image in the variety folder
        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        for img_file in sorted(images):
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [SKIP] Cannot read: {img_file}")
                continue

            # Segment and extract per-grain features
            contours, masks, binary, _ = segment_grains(image, pixels_per_mm)
            if len(contours) == 0:
                print(f"  [SKIP] No grains in: {img_file}")
                continue

            grain_df = extract_all_grain_features(
                image, contours, masks, pixels_per_mm
            )

            # Add variety label to each grain
            grain_count = len(grain_df)
            for _, row in grain_df.iterrows():
                grain_features = {feat: row.get(feat, 0.0)
                                  for feat in PER_GRAIN_VARIETY_FEATURES}
                grain_features["variety"] = variety_key
                all_grains.append(grain_features)

            print(f"  {img_file}: {grain_count} grains extracted")

    if not all_grains:
        print("[Train] WARNING: No training samples collected.")
        return pd.DataFrame()

    training_df = pd.DataFrame(all_grains)

    # Drop rows with NaN values
    required_cols = PER_GRAIN_VARIETY_FEATURES + ["variety"]
    available = [c for c in required_cols if c in training_df.columns]
    training_df = training_df[available].dropna()

    print(f"\n[Train] Per-grain training data:")
    print(f"  Total grains : {len(training_df)}")
    print(f"  Varieties    : {training_df['variety'].nunique()}")
    print(f"  Features     : {len(PER_GRAIN_VARIETY_FEATURES)}")
    print(f"\n  Per-variety counts:")
    for variety, count in training_df["variety"].value_counts().items():
        print(f"    {variety:<20s} {count:>5d} grains")

    return training_df


# ============================================================
# 2. Train Classifier
# ============================================================

def train_classifier(
    training_df: pd.DataFrame,
    n_folds: int = 5,
) -> Tuple[RandomForestClassifier, LabelEncoder, dict]:
    """
    Train a ``RandomForestClassifier`` on per-grain features.

    Parameters
    ----------
    training_df : pd.DataFrame
        Must contain ``PER_GRAIN_VARIETY_FEATURES`` plus ``"variety"``.
    n_folds : int, optional
        Number of stratified CV folds (default 5).

    Returns
    -------
    model : RandomForestClassifier
    label_encoder : LabelEncoder
    metrics : dict
    """
    feature_cols = [c for c in PER_GRAIN_VARIETY_FEATURES
                    if c in training_df.columns]
    if not feature_cols:
        raise ValueError("No matching feature columns found in training data.")

    X = training_df[feature_cols].values
    y_labels = training_df["variety"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    print(f"[Train] Features : {len(feature_cols)}")
    print(f"[Train] Samples  : {len(X)} (before balancing)")
    print(f"[Train] Classes  : {list(le.classes_)}")

    # ── Removed SMOTE Balancing ──────────────────────────────────────
    # We now rely entirely on Random Forest's class_weight="balanced"
    # to handle class imbalance, preventing SMOTE from corrupting
    # features and creating synthetic unrealistic grains.
    
    print(f"[Train] Samples  : {len(X)} (without SMOTE)")

    # Build model
    model = RandomForestClassifier(
        n_estimators=VARIETY_N_ESTIMATORS,
        max_depth=VARIETY_MAX_DEPTH,
        random_state=VARIETY_RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Stratified cross-validation
    n_folds_actual = min(n_folds, min(np.bincount(y)))
    if n_folds_actual < 2:
        print("[Train] Too few samples per class for CV; "
              "training on full data.")
        model.fit(X, y)
        cv_scores = []
    else:
        skf = StratifiedKFold(
            n_splits=n_folds_actual, shuffle=True,
            random_state=VARIETY_RANDOM_STATE,
        )
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        print(f"[Train] {n_folds_actual}-fold CV accuracy: "
              f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Fit on full data
        model.fit(X, y)

    # Feature importances
    importances = dict(zip(feature_cols, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("[Train] Feature importances:")
    for feat, imp in sorted_imp:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<20s} {imp:.4f}  {bar}")

    metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()) if len(cv_scores) else None,
        "cv_accuracy_std": float(cv_scores.std()) if len(cv_scores) else None,
        "cv_scores": [float(s) for s in cv_scores],
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
        "feature_importances": importances,
    }

    return model, le, metrics


# ============================================================
# 3. Per-Grain Variety Prediction
# ============================================================

def predict_variety_per_grain(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    grain_df: pd.DataFrame,
) -> dict:
    """
    Classify each grain individually and return variety distribution.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained per-grain classifier.
    label_encoder : LabelEncoder
        Fitted label encoder.
    grain_df : pd.DataFrame
        Per-grain features from ``extract_all_grain_features()``.

    Returns
    -------
    dict
        {
            "dominant_variety": str,
            "dominant_confidence": float,
            "is_mixed": bool,
            "variety_distribution": {variety: {"count": int, "percentage": float}, ...},
            "per_grain_predictions": [{"grain_id": int, "variety": str, "confidence": float}, ...],
            "total_grains": int,
        }
    """
    if grain_df.empty:
        return {
            "dominant_variety": "unknown",
            "dominant_confidence": 0.0,
            "is_mixed": False,
            "variety_distribution": {},
            "per_grain_predictions": [],
            "total_grains": 0,
        }

    # Build feature matrix
    feature_cols = [c for c in PER_GRAIN_VARIETY_FEATURES
                    if c in grain_df.columns]

    X = grain_df[feature_cols].fillna(0.0).values

    # Predict each grain
    pred_indices = model.predict(X)
    pred_proba = model.predict_proba(X)

    pred_labels = label_encoder.inverse_transform(pred_indices)

    # Per-grain results
    per_grain = []
    for i, (label, proba) in enumerate(zip(pred_labels, pred_proba)):
        conf = float(proba[pred_indices[i]])
        per_grain.append({
            "grain_id": int(grain_df.iloc[i].get("grain_id", i)),
            "variety": label,
            "confidence": round(conf, 4),
        })

    # Variety distribution
    total = len(pred_labels)
    counts = Counter(pred_labels)
    distribution = {}
    for variety, count in counts.most_common():
        pct = (count / total) * 100
        display = VARIETY_DATABASE.get(variety, {}).get("display_name", variety)
        distribution[variety] = {
            "count": count,
            "percentage": round(pct, 1),
            "display_name": display,
        }

    # Dominant variety
    dominant = counts.most_common(1)[0]
    dominant_variety = dominant[0]
    dominant_pct = (dominant[1] / total) * 100

    # Mixed variety detection
    secondary_varieties = [
        v for v, info in distribution.items()
        if v != dominant_variety and info["percentage"] >= MIXED_VARIETY_THRESHOLD_PCT
    ]
    is_mixed = len(secondary_varieties) > 0

    return {
        "dominant_variety": dominant_variety,
        "dominant_confidence": round(dominant_pct / 100.0, 4),
        "is_mixed": is_mixed,
        "variety_distribution": distribution,
        "per_grain_predictions": per_grain,
        "total_grains": total,
    }


def predict_variety(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    image_features_or_grain_df,
) -> dict:
    """
    Convenience wrapper — accepts either a grain DataFrame (per-grain)
    or an aggregated features dict (legacy).

    Returns a dict compatible with the analyzer's expected format:
        {"predicted_variety", "confidence", "probabilities", "display_name",
         "is_mixed", "variety_distribution", ...}
    """
    if isinstance(image_features_or_grain_df, pd.DataFrame):
        # Per-grain mode
        result = predict_variety_per_grain(
            model, label_encoder, image_features_or_grain_df
        )

        predicted = result["dominant_variety"]
        display = VARIETY_DATABASE.get(predicted, {}).get(
            "display_name", predicted
        )

        # Build probabilities dict from distribution
        probabilities = {
            v: info["percentage"] / 100.0
            for v, info in result["variety_distribution"].items()
        }

        return {
            "predicted_variety": predicted,
            "confidence": result["dominant_confidence"],
            "probabilities": probabilities,
            "display_name": display,
            "is_mixed": result["is_mixed"],
            "variety_distribution": result["variety_distribution"],
            "per_grain_predictions": result["per_grain_predictions"],
            "total_grains": result["total_grains"],
        }

    else:
        # Legacy: aggregated features dict
        features = image_features_or_grain_df
        X = []
        for feat in PER_GRAIN_VARIETY_FEATURES:
            val = features.get(feat, 0.0)
            X.append(float(val) if val is not None else 0.0)
        X = np.array(X).reshape(1, -1)

        pred_idx = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        predicted = label_encoder.inverse_transform([pred_idx])[0]

        probabilities = {
            label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }

        display = VARIETY_DATABASE.get(predicted, {}).get(
            "display_name", predicted
        )

        return {
            "predicted_variety": predicted,
            "confidence": float(proba[pred_idx]),
            "probabilities": probabilities,
            "display_name": display,
            "is_mixed": False,
            "variety_distribution": {},
            "per_grain_predictions": [],
            "total_grains": 1,
        }


# ============================================================
# 4. Save / Load Model
# ============================================================

def save_model(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    path: str = None,
) -> str:
    """Serialize the trained model and label encoder with joblib."""
    if path is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "variety_classifier.joblib")

    bundle = {"model": model, "label_encoder": label_encoder}
    joblib.dump(bundle, path)
    print(f"[Model] Saved to {path}")
    return path


def load_model(path: str = None) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """Deserialize a previously saved model bundle."""
    if path is None:
        path = os.path.join(MODEL_DIR, "variety_classifier.joblib")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    bundle = joblib.load(path)
    print(f"[Model] Loaded from {path}")
    return bundle["model"], bundle["label_encoder"]


# ============================================================
# 5. Evaluate Model
# ============================================================

def evaluate_model(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate the model on test data."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    per_class = {}
    for i, cls_name in enumerate(label_encoder.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            cls_acc = accuracy_score(y_test[mask], y_pred[mask])
            per_class[cls_name] = round(float(cls_acc), 4)

    print(f"[Eval] Overall accuracy: {acc:.4f}")
    print(f"[Eval] Classification report:\n{report}")

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class_accuracy": per_class,
    }


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rice Variety Classifier -- Per-Grain Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python variety_classifier.py --train\n"
            "  python variety_classifier.py --predict --image path/to/image.jpg\n"
        ),
    )

    parser.add_argument("--train", action="store_true",
        help="Train per-grain variety classifier.")
    parser.add_argument("--evaluate", action="store_true",
        help="Evaluate saved model.")
    parser.add_argument("--predict", action="store_true",
        help="Predict varieties in an image (detects mixed varieties).")
    parser.add_argument("--image", type=str, default=None,
        help="Path to grain image (used with --predict).")
    parser.add_argument("--aruco", type=str, default=None,
        help="Path to ArUCo calibration image.")
    parser.add_argument("--model_path", type=str, default=None,
        help="Path to model file.")

    args = parser.parse_args()

    # ---- Calibrate ----
    print("=" * 60)
    print("ArUCo Calibration")
    print("=" * 60)
    px_per_mm = detect_aruco_marker(args.aruco)
    print(f"Calibration: {px_per_mm:.2f} px/mm\n")

    # ---- Train ----
    if args.train:
        print("=" * 60)
        print("Building Per-Grain Training Data")
        print("=" * 60)
        training_df = build_training_data(px_per_mm)

        if training_df.empty:
            print("[ERROR] No training data. Exiting.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Training Per-Grain Classifier")
        print("=" * 60)
        model, le, metrics = train_classifier(training_df)

        model_path = save_model(model, le, args.model_path)
        print(f"\nModel saved to: {model_path}")
        if metrics.get("cv_accuracy_mean"):
            print(f"CV accuracy: {metrics['cv_accuracy_mean']:.4f} "
                  f"+/- {metrics['cv_accuracy_std']:.4f}")

    # ---- Evaluate ----
    elif args.evaluate:
        print("=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        model, le = load_model(args.model_path)

        training_df = build_training_data(px_per_mm)
        if training_df.empty:
            print("[ERROR] No data. Exiting.")
            sys.exit(1)

        feature_cols = [c for c in PER_GRAIN_VARIETY_FEATURES
                        if c in training_df.columns]
        X = training_df[feature_cols].values
        y = le.transform(training_df["variety"].values)

        evaluate_model(model, le, X, y)

    # ---- Predict ----
    elif args.predict:
        if not args.image:
            parser.error("--predict requires --image <path>")

        import cv2
        print("=" * 60)
        print("Per-Grain Variety Prediction")
        print("=" * 60)
        model, le = load_model(args.model_path)

        image = cv2.imread(args.image)
        if image is None:
            print(f"[ERROR] Cannot read: {args.image}")
            sys.exit(1)

        contours, masks, binary, _ = segment_grains(image, px_per_mm)
        grain_df = extract_all_grain_features(
            image, contours, masks, px_per_mm
        )

        result = predict_variety(model, le, grain_df)

        print(f"\nTotal grains: {result['total_grains']}")
        print(f"Dominant variety: {result['display_name']} "
              f"({result['confidence']*100:.1f}%)")

        if result["is_mixed"]:
            print("\n*** MIXED VARIETY DETECTED ***")

        print("\nVariety Distribution:")
        for v, info in sorted(result["variety_distribution"].items(),
                              key=lambda x: -x[1]["percentage"]):
            bar = "#" * int(info["percentage"] / 2)
            print(f"  {info['display_name']:<20s} "
                  f"{info['count']:>4d} grains  "
                  f"({info['percentage']:>5.1f}%)  {bar}")

    else:
        parser.print_help()
