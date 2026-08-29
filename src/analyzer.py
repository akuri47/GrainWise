"""
Rice Grain Quality Analyzer — Main Analysis Pipeline
=====================================================
End-to-end orchestrator that combines all components:
  Calibrate -> Segment -> Extract Features -> Classify Variety ->
  Classify Defects -> Assess Quality -> Estimate Shelf Life ->
  Recommend Price

Outputs a comprehensive JSON report.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ARUCO_IMAGE_PATH, MODEL_DIR, OUTPUT_DIR,
    VARIETY_DATABASE, DEFECT_LABELS, FOLDER_TO_VARIETY,
    BROKEN_LENGTH_THRESHOLD,
)
from src.grain_segmentation import (
    detect_aruco_marker, detect_aruco_from_image,
    segment_grains, extract_all_grain_features,
    aggregate_image_features, draw_segmentation_overlay,
)
from src.quality_assessment import (
    calculate_moisture_content, calculate_defect_percentages,
    calculate_multilabel_defect_percentages,
    assess_faq_grade, calculate_quality_score, generate_quality_report,
)
from src.shelf_life import estimate_shelf_life, generate_shelf_life_report
from src.price_recommendation import recommend_price, compare_grade_prices


class RiceGrainAnalyzer:
    """
    End-to-end rice grain quality analyzer.

    Combines grain segmentation, variety identification, defect classification,
    FAQ quality assessment, shelf-life estimation, and price recommendation
    into a single pipeline.
    """

    def __init__(self, aruco_image_path: str = None):
        """
        Initialize the analyzer.

        Parameters
        ----------
        aruco_image_path : str, optional
            Path to ArUCo calibration image. Defaults to config path.
        """
        self.aruco_path = aruco_image_path or ARUCO_IMAGE_PATH
        self.pixels_per_mm = None
        self.variety_model = None
        self.variety_label_encoder = None
        self.defect_model = None
        self.device = "cpu"

        # Calibrate on initialization
        self._calibrate()

    def _calibrate(self):
        """Perform ArUCo calibration."""
        print("=" * 60)
        print("INITIALIZING RICE GRAIN ANALYZER")
        print("=" * 60)
        self.pixels_per_mm = detect_aruco_marker(self.aruco_path)
        print(f"Calibration: {self.pixels_per_mm:.2f} px/mm\n")

    def load_models(self):
        """Load pretrained variety and defect classification models."""
        # Load variety classifier
        variety_model_path = os.path.join(MODEL_DIR, "variety_classifier.joblib")
        if os.path.exists(variety_model_path):
            try:
                from src.variety_classifier import load_model as load_variety_model
                loaded_model, loaded_encoder = load_variety_model(variety_model_path)
                self.variety_model = loaded_model
                self.variety_label_encoder = loaded_encoder
                print("[Models] Variety classifier loaded.")
            except Exception as e:
                print(f"[Models] Could not load variety classifier: {e}")
        else:
            print(f"[Models] Variety classifier not found at {variety_model_path}")
            print("[Models] Variety will be predicted from morphological features (rule-based).")

        # Load defect classifier
        defect_model_path_onnx = os.path.join(MODEL_DIR, "defect_resnet18_best.onnx")
        if os.path.exists(defect_model_path_onnx):
            try:
                import onnxruntime as ort
                self.defect_model = ort.InferenceSession(defect_model_path_onnx)
                print("[Models] Defect classifier (ONNX) loaded.")
            except Exception as e:
                print(f"[Models] Could not load defect classifier: {e}")
        else:
            print(f"[Models] Defect classifier (ONNX) not found at {defect_model_path_onnx}")
            print("[Models] Defect classification will use rule-based fallback.")

        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"[Models] Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("[Models] Using CPU.")
        except ImportError:
            self.device = "cpu"

    def _predict_variety_rule_based(self, image_features: Dict) -> Tuple[str, Dict]:
        """
        Rule-based variety prediction using morphological features.
        Fallback when no trained model is available.

        Uses aspect ratio to determine category, then matches
        against known variety dimensions.
        """
        mean_ar = image_features.get("mean_aspect_ratio", 0)
        mean_len = image_features.get("mean_length_mm", 0)
        mean_wid = image_features.get("mean_width_mm", 0)

        # Determine category from AR
        if mean_ar < 2.0:
            category = "short"
        elif mean_ar <= 3.0:
            category = "medium"
        else:
            category = "long"

        # Find best matching variety by feature similarity
        best_variety = None
        best_score = float('inf')
        scores = {}

        for vkey, vdata in VARIETY_DATABASE.items():
            if vdata["category"] != category:
                # Penalize category mismatch but don't exclude
                penalty = 5.0
            else:
                penalty = 0.0

            # Simple distance metric (would be replaced by trained model)
            score = penalty
            scores[vkey] = max(0, 1.0 - score / 10.0)

            if score < best_score:
                best_score = score
                best_variety = vkey

        # Normalize scores to probabilities
        total = sum(scores.values()) or 1
        confidences = {k: v / total for k, v in scores.items()}

        if best_variety is None:
            best_variety = "unknown"

        return best_variety, confidences

    def _predict_defects_rule_based(
        self, image: np.ndarray, contours, masks, grain_df: pd.DataFrame
    ) -> list:
        """
        Rule-based defect prediction as fallback.
        Uses morphological and color features to estimate defect type.
        Compares each grain against image-level statistics for better detection.
        """
        predictions = []

        # Compute image-level statistics for relative comparison
        if not grain_df.empty and "mean_intensity" in grain_df.columns:
            overall_mean_int = grain_df["mean_intensity"].mean()
            overall_std_int = grain_df["mean_intensity"].std()
            overall_mean_s = grain_df["mean_s"].mean() if "mean_s" in grain_df.columns else 0
        else:
            overall_mean_int = 128
            overall_std_int = 30
            overall_mean_s = 0

        for i, (_, row) in enumerate(grain_df.iterrows()):
            ar = row.get("aspect_ratio", 0)
            area = row.get("area_mm2", 0)
            circularity = row.get("circularity", 0)
            solidity = row.get("solidity", 0)
            mean_intensity = row.get("mean_intensity", 0)
            chalkiness = row.get("chalkiness_score", 0)
            mean_r = row.get("mean_r", 0)
            mean_g = row.get("mean_g", 0)
            mean_b = row.get("mean_b", 0)
            mean_s = row.get("mean_s", 0)
            std_intensity = row.get("std_intensity", 0)
            mean_h = row.get("mean_h", 0)

            # 1. BROKEN: very small area
            if area < 2.0:
                predictions.append("broken")

            # 2. FOREIGN: unusual shape, very dark, or high saturation + dark
            elif ar < 1.5:
                predictions.append("foreign")
            elif mean_intensity < overall_mean_int - 2.5 * overall_std_int:
                # Much darker than other grains
                predictions.append("foreign")
            elif mean_s > 50 and mean_intensity < 140:
                # Very saturated + dark = non-grain object
                predictions.append("foreign")

            # 3. CHALKY: genuinely bright patches (with 1.0*std, normal ~0.16, chalky ~0.25+)
            elif chalkiness > 0.20:
                predictions.append("chalky")
            elif (mean_intensity > overall_mean_int + 2.0 * overall_std_int
                  and mean_s < 15):
                # Much brighter than average + very low saturation = chalky
                predictions.append("chalky")

            # 4. DISCOLORED: higher saturation than average (relative comparison)
            elif mean_s > overall_mean_s + 15:
                # Significantly more saturated than the average grain
                predictions.append("discolored")
            elif mean_r > mean_g * 1.08 and mean_r > mean_b * 1.08:
                # Clear reddish/brownish tint
                predictions.append("discolored")

            # 5. DAMAGED: irregular shape or high internal variance
            elif solidity < 0.82:
                predictions.append("damaged")
            elif std_intensity > 28:
                predictions.append("damaged")

            # 6. WHOLE (default)
            else:
                predictions.append("whole")

        return predictions

    # ------------------------------------------------------------------
    # Multi-label classification: structural (length) + defect (model)
    # ------------------------------------------------------------------

    def _apply_length_based_classification(
        self,
        grain_df: pd.DataFrame,
        model_predictions: list,
    ) -> list:
        """Assign dual identity to each grain: structural + defect.

        1. Separate foreign grains (model says ``"foreign"``).
        2. From model-predicted ``"whole"`` grains, compute avg length.
        3. Apply length rule: ``< 3/4 * avg_whole_length`` = broken.
        4. Map model prediction to defect type.

        Parameters
        ----------
        grain_df : pd.DataFrame
            Must contain a ``"length_mm"`` column.
        model_predictions : list[str]
            One label per grain from the defect model.

        Returns
        -------
        list[dict]
            Each dict: ``{"structural": ..., "defect": ..., "model_raw": ...}``
        """
        classifications = []

        # --- Step 1: Identify model-predicted whole grains for avg length ---
        whole_lengths = []
        for i, pred in enumerate(model_predictions):
            if pred == "whole" and i < len(grain_df):
                length = grain_df.iloc[i].get("length_mm", 0)
                if length > 0:
                    whole_lengths.append(length)

        # Avg whole grain length (fallback: mean of all non-foreign grains)
        if whole_lengths:
            avg_whole_length = sum(whole_lengths) / len(whole_lengths)
        else:
            # Fallback: use all non-foreign grain lengths
            non_foreign_lengths = [
                grain_df.iloc[i]["length_mm"]
                for i, pred in enumerate(model_predictions)
                if pred != "foreign" and i < len(grain_df)
                and grain_df.iloc[i].get("length_mm", 0) > 0
            ]
            avg_whole_length = (
                sum(non_foreign_lengths) / len(non_foreign_lengths)
                if non_foreign_lengths else 5.0  # safe default mm
            )

        broken_threshold = BROKEN_LENGTH_THRESHOLD * avg_whole_length

        # --- Step 2: Classify each grain ---
        for i, pred in enumerate(model_predictions):
            if i >= len(grain_df):
                break

            length = grain_df.iloc[i].get("length_mm", 0)
            area = grain_df.iloc[i].get("area_mm2", 0)

            # Foreign stays foreign
            if pred == "foreign":
                classifications.append({
                    "structural": "foreign",
                    "defect": "clean",
                    "model_raw": pred,
                    "length_mm": length,
                    "area_mm2": area,
                })
                continue

            # Structural status: length rule always wins
            if length < broken_threshold:
                structural = "broken"
            else:
                structural = "whole"

            # Defect type from model prediction
            if pred in ("chalky", "damaged", "discolored"):
                defect = pred
            else:
                # Model said "whole" or "broken" -> no surface defect detected
                defect = "clean"

            classifications.append({
                "structural": structural,
                "defect": defect,
                "model_raw": pred,
                "length_mm": length,
                "area_mm2": area,
            })

        # Store avg whole length for the report
        self._avg_whole_length = round(avg_whole_length, 2)
        self._broken_threshold = round(broken_threshold, 2)

        return classifications

    def _calculate_equivalent_head_rice(
        self,
        grain_classifications: list,
    ) -> dict:
        """Calculate Equivalent Head Rice from multi-label classifications.

        Equivalent Head Rice = Actual whole grains
                             + SUM(broken_grain_area / avg_whole_grain_area)

        Foreign matter is excluded from the calculation.

        Parameters
        ----------
        grain_classifications : list[dict]
            Output of ``_apply_length_based_classification()``.

        Returns
        -------
        dict
            ``actual_whole_count``, ``broken_count``,
            ``broken_grain_equivalents``, ``equivalent_head_rice``,
            ``equivalent_head_rice_pct``, ``total_rice_grains``,
            ``avg_whole_grain_area_mm2``.
        """
        # Separate whole and broken rice grains (exclude foreign)
        whole_grains = [
            g for g in grain_classifications if g["structural"] == "whole"
        ]
        broken_grains = [
            g for g in grain_classifications if g["structural"] == "broken"
        ]
        total_rice = len(whole_grains) + len(broken_grains)

        if total_rice == 0:
            return {
                "actual_whole_count": 0,
                "broken_count": 0,
                "broken_grain_equivalents": 0.0,
                "equivalent_head_rice": 0.0,
                "equivalent_head_rice_pct": 0.0,
                "total_rice_grains": 0,
                "avg_whole_grain_area_mm2": 0.0,
            }

        # Avg area of whole grains
        whole_areas = [g["area_mm2"] for g in whole_grains if g["area_mm2"] > 0]
        avg_whole_area = (
            sum(whole_areas) / len(whole_areas) if whole_areas else 1.0
        )

        # Broken grain equivalents: each broken grain contributes
        # its area fraction relative to a whole grain
        broken_equivalents = sum(
            g["area_mm2"] / avg_whole_area
            for g in broken_grains if g["area_mm2"] > 0
        )

        equivalent_head_rice = len(whole_grains) + broken_equivalents
        ehr_pct = (equivalent_head_rice / total_rice) * 100.0 if total_rice > 0 else 0.0

        return {
            "actual_whole_count": len(whole_grains),
            "broken_count": len(broken_grains),
            "broken_grain_equivalents": round(broken_equivalents, 2),
            "equivalent_head_rice": round(equivalent_head_rice, 2),
            "equivalent_head_rice_pct": round(ehr_pct, 2),
            "total_rice_grains": total_rice,
            "avg_whole_grain_area_mm2": round(avg_whole_area, 2),
        }

    def analyze(
        self,
        image_path: str,
        sample_weight_g: float = None,
        num_grains_weighed: int = None,
        variety_hint: str = None,
        storage_temp_c: float = 25.0,
        storage_humidity_rh: float = 60.0,
        save_output: bool = True,
        marker_width_mm: float = None,
        marker_height_mm: float = None,
        custom_price_per_qtl: float = None,
    ) -> Dict:
        """
        Run the complete analysis pipeline on a rice grain image.

        Parameters
        ----------
        image_path : str
            Path to the rice grain image.
        sample_weight_g : float, optional
            Weight of the sample in grams (wet weight).
            Required for moisture calculation.
        num_grains_weighed : int, optional
            Number of grains in the weighed sample.
            If not provided, uses the count from segmentation.
        variety_hint : str, optional
            If the user already knows the variety, skip prediction.
        storage_temp_c : float
            Expected storage temperature in °C.
        storage_humidity_rh : float
            Expected storage relative humidity (%).
        save_output : bool
            Whether to save output files.

        Returns
        -------
        dict
            Comprehensive analysis report.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "status": "success",
            "calibration_px_per_mm": self.pixels_per_mm,
        }

        # ============================================================
        # STEP 1: Read image
        # ============================================================
        print("\n" + "=" * 60)
        print(f"ANALYZING: {os.path.basename(image_path)}")
        print("=" * 60)

        image = cv2.imread(image_path)
        if image is None:
            report["status"] = "error"
            report["error"] = f"Could not read image: {image_path}"
            return report

        report["image_size"] = {"width": image.shape[1], "height": image.shape[0]}

        # ============================================================
        # STEP 1b: Re-detect calibration marker FROM the uploaded image
        # This is the primary calibration — uses the white paper/marker
        # that the user placed next to their rice grains.
        # Falls back to the startup calibration if not found.
        # ============================================================
        print("[Calibration] Detecting marker in uploaded image...")
        from src.config import ARUCO_MARKER_WIDTH_MM, ARUCO_MARKER_HEIGHT_MM
        mw = marker_width_mm or ARUCO_MARKER_WIDTH_MM
        mh = marker_height_mm or ARUCO_MARKER_HEIGHT_MM
        aruco_result = detect_aruco_from_image(image, mw, mh)
        if aruco_result is not None:
            live_ppm, aruco_bx, aruco_by, aruco_bw, aruco_bh = aruco_result
            report["calibration_px_per_mm"] = live_ppm
            print(f"[Calibration] Using image marker: {live_ppm:.2f} px/mm "
                  f"(marker {aruco_bw}x{aruco_bh} px = {mw}x{mh} mm)")
            pixels_per_mm_for_analysis = live_ppm
            image_aruco_bbox = (aruco_bx, aruco_by, aruco_bw, aruco_bh)
        else:
            print(f"[Calibration] Marker not found in image. "
                  f"Using startup calibration: {self.pixels_per_mm:.2f} px/mm")
            pixels_per_mm_for_analysis = self.pixels_per_mm
            image_aruco_bbox = None

        # ============================================================
        # STEP 2: Segment grains
        # ============================================================
        print("\n[Step 1/7] Segmenting grains...")
        contours, masks, binary, aruco_info = segment_grains(image, pixels_per_mm_for_analysis)
        # Prefer the freshly detected ArUCo bbox from the uploaded image over
        # any bbox found incidentally during segmentation
        if image_aruco_bbox:
            aruco_info["aruco_bbox"] = image_aruco_bbox
        grain_df = extract_all_grain_features(image, contours, masks, pixels_per_mm_for_analysis)
        image_features = aggregate_image_features(grain_df)

        report["segmentation"] = {
            "total_grains": len(contours),
            "mean_length_mm": round(grain_df["length_mm"].mean(), 2) if not grain_df.empty else 0,
            "mean_width_mm": round(grain_df["width_mm"].mean(), 2) if not grain_df.empty else 0,
            "mean_aspect_ratio": round(grain_df["aspect_ratio"].mean(), 2) if not grain_df.empty else 0,
            "mean_area_mm2": round(grain_df["area_mm2"].mean(), 2) if not grain_df.empty else 0,
        }

        if len(contours) == 0:
            report["status"] = "error"
            report["error"] = "No grains detected in the image."
            return report

        # ============================================================
        # STEP 3: Identify variety (per-grain classification)
        # ============================================================
        print("[Step 2/7] Identifying variety...")

        if variety_hint and variety_hint.lower() in VARIETY_DATABASE:
            predicted_variety = variety_hint.lower()
            variety_confidence = {predicted_variety: 1.0}
            report["variety_source"] = "user_provided"
            report["is_mixed"] = False
            report["variety_distribution"] = {}
        elif self.variety_model is not None:
            try:
                from src.variety_classifier import predict_variety
                # Pass grain_df for per-grain classification
                variety_result = predict_variety(
                    self.variety_model, self.variety_label_encoder, grain_df
                )
                predicted_variety = variety_result["predicted_variety"]
                variety_confidence = variety_result.get("probabilities", {predicted_variety: variety_result.get("confidence", 1.0)})
                report["variety_source"] = "ml_model"
                report["is_mixed"] = variety_result.get("is_mixed", False)
                report["variety_distribution"] = variety_result.get("variety_distribution", {})

                if report["is_mixed"]:
                    print("  ** MIXED VARIETY DETECTED **")
                    for v, info in variety_result.get("variety_distribution", {}).items():
                        print(f"     {info.get('display_name', v)}: "
                              f"{info['count']} grains ({info['percentage']:.1f}%)")
            except Exception as e:
                print(f"  ML prediction failed, using rule-based: {e}")
                predicted_variety, variety_confidence = self._predict_variety_rule_based(image_features)
                report["variety_source"] = "rule_based"
                report["is_mixed"] = False
                report["variety_distribution"] = {}
        else:
            predicted_variety, variety_confidence = self._predict_variety_rule_based(image_features)
            report["variety_source"] = "rule_based"
            report["is_mixed"] = False
            report["variety_distribution"] = {}

        variety_info = VARIETY_DATABASE.get(predicted_variety, {})
        report["variety"] = {
            "predicted": predicted_variety,
            "display_name": variety_info.get("display_name", predicted_variety),
            "category": variety_info.get("category", "unknown"),
            "confidence": {k: round(v, 4) for k, v in sorted(
                variety_confidence.items(), key=lambda x: -x[1]
            )[:5]},
            "is_mixed": report.get("is_mixed", False),
            "variety_distribution": report.get("variety_distribution", {}),
        }
        print(f"  -> Variety: {report['variety']['display_name']} "
              f"(category: {report['variety']['category']})")

        # ============================================================
        # STEP 4: Classify defects
        # ============================================================
        print("[Step 3/7] Classifying grain defects...")

        if self.defect_model is not None:
            try:
                # Extract individual grain images
                grain_images = []
                for cnt, msk in zip(contours, masks):
                    if isinstance(msk, tuple):
                        roi_mask, mx, my, mw, mh = msk
                    else:
                        mx, my, mw, mh = cv2.boundingRect(cnt)
                        roi_mask = msk[my:my+mh, mx:mx+mw]
                    grain_crop = image[my:my+mh, mx:mx+mw].copy()
                    grain_crop[roi_mask == 0] = 0
                    grain_images.append(grain_crop)

                # ---------------------------------------------------------
                # Pure ONNX Inference (No PyTorch/Torchvision required)
                # ---------------------------------------------------------
                input_size = 224
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                tensors = []
                for crop in grain_images:
                    # BGR to RGB
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    
                    # Resize to 256 (maintaining aspect ratio or just 256x256)
                    h, w = rgb.shape[:2]
                    scale = 256 / min(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Center Crop 224x224
                    start_y = (new_h - input_size) // 2
                    start_x = (new_w - input_size) // 2
                    cropped = resized[start_y:start_y+input_size, start_x:start_x+input_size]
                    
                    # Normalize (0-1), subtract mean, divide std
                    normalized = (cropped / 255.0 - mean) / std
                    
                    # Transpose from HWC to CHW format for ONNX/PyTorch
                    transposed = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
                    tensors.append(transposed)
                
                if tensors:
                    batch = np.stack(tensors)
                    ort_inputs = {self.defect_model.get_inputs()[0].name: batch}
                    logits = self.defect_model.run(None, ort_inputs)[0]
                    # Softmax
                    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                else:
                    probs = np.array([])
                
                # Format results just like the old PyTorch output
                results = []
                for i in range(len(grain_images)):
                    confidence = {DEFECT_LABELS[j]: float(probs[i, j]) for j in range(len(DEFECT_LABELS))}
                    max_idx = int(np.argmax(probs[i]))
                    max_prob = float(probs[i, max_idx])
                    max_label = DEFECT_LABELS[max_idx]
                    
                    if max_label != "whole" and max_prob >= 0.3:
                        label = max_label
                    elif max_label == "whole" and max_prob >= 0.5:
                        label = "whole"
                    else:
                        label = "unknown"
                    results.append((label, confidence))
                # ---------------------------------------------------------
                
                # Hybrid ML + rule-based approach:
                # - ML detects a defect → trust ML
                # - ML is "unknown" (low confidence) → use rule-based
                # - Rule says "chalky" → ALWAYS trust rules (ML can't detect chalky at all)
                # - ML says "whole" with < 0.7 confidence + rules detect defect → trust rules
                # - ML says "whole" with HIGH confidence → trust ML
                rule_based_preds = self._predict_defects_rule_based(image, contours, masks, grain_df)
                defect_predictions = []
                for i, r in enumerate(results):
                    ml_pred = r[0]
                    ml_conf = r[1]  # confidence dict per class
                    rule_pred = rule_based_preds[i]
                    ml_max_conf = max(ml_conf.values()) if ml_conf else 0
                    
                    if ml_pred == "unknown":
                        # ML has no confident prediction → use rules
                        defect_predictions.append(rule_pred)
                    elif ml_pred != "whole":
                        # ML detected a specific defect → trust ML
                        defect_predictions.append(ml_pred)
                    elif rule_pred == "chalky":
                        # ML can't detect chalky (always ~0% probability)
                        # Always trust rule-based for this class
                        defect_predictions.append("chalky")
                    elif rule_pred != "whole" and ml_max_conf < 0.7:
                        # ML says whole but not very confident, rules say defect → trust rules
                        defect_predictions.append(rule_pred)
                    else:
                        # ML confidently says whole, or both agree whole
                        defect_predictions.append("whole")
                        
                report["defect_source"] = "ml_model_with_fallback"
            except Exception as e:
                print(f"  ML prediction failed, using rule-based: {e}")
                defect_predictions = self._predict_defects_rule_based(
                    image, contours, masks, grain_df
                )
                report["defect_source"] = "rule_based"
        else:
            defect_predictions = self._predict_defects_rule_based(
                image, contours, masks, grain_df
            )
            report["defect_source"] = "rule_based"

        # ============================================================
        # STEP 4: Multi-label classification (structural + defect)
        # ============================================================
        print("[Step 4/7] Applying length-based classification...")

        grain_classifications = self._apply_length_based_classification(
            grain_df, defect_predictions
        )

        # Calculate multi-label defect percentages
        percentages_dict = calculate_multilabel_defect_percentages(
            grain_classifications
        )
        defect_percentages = percentages_dict["overlapping"]
        me_percentages = percentages_dict["mutually_exclusive"]

        # Calculate Equivalent Head Rice
        ehr_result = self._calculate_equivalent_head_rice(grain_classifications)

        # Multi-label breakdown counts
        structural_counts = {}
        defect_counts = {}
        combined_counts = {}
        for gc in grain_classifications:
            s = gc["structural"]
            d = gc["defect"]
            structural_counts[s] = structural_counts.get(s, 0) + 1
            defect_counts[d] = defect_counts.get(d, 0) + 1
            combo = f"{s}+{d}"
            combined_counts[combo] = combined_counts.get(combo, 0) + 1

        report["grain_classification"] = {
            "method": "multi_label",
            "avg_whole_grain_length_mm": self._avg_whole_length,
            "broken_length_threshold_mm": self._broken_threshold,
            "broken_length_fraction": BROKEN_LENGTH_THRESHOLD,
            "structural_counts": structural_counts,
            "defect_counts": defect_counts,
            "combined_counts": combined_counts,
        }

        report["equivalent_head_rice"] = ehr_result

        report["defects"] = {
            "predictions_summary": {
                label: defect_predictions.count(label)
                for label in set(defect_predictions)
            },
            "multilabel_percentages": {k: round(v, 2) for k, v in defect_percentages.items()},
            "mutually_exclusive_percentages": {k: round(v, 2) for k, v in me_percentages.items()},
            "note": "Percentages can overlap (e.g., a broken+chalky grain counts in both broken% and chalky%). "
                    "'whole' here means good grain (whole + clean, no defects).",
        }
        print(f"  -> Avg whole grain length: {self._avg_whole_length} mm")
        print(f"  -> Broken threshold (3/4): {self._broken_threshold} mm")
        print(f"  -> Good grain: {defect_percentages.get('whole', 0):.1f}%, "
              f"Broken: {defect_percentages.get('broken', 0):.1f}%, "
              f"Chalky: {defect_percentages.get('chalky', 0):.1f}%, "
              f"Damaged: {defect_percentages.get('damaged', 0):.1f}%, "
              f"Discolored: {defect_percentages.get('discolored', 0):.1f}%, "
              f"Foreign: {defect_percentages.get('foreign', 0):.1f}%")
        print(f"  -> Equivalent Head Rice: {ehr_result['equivalent_head_rice_pct']:.1f}%")

        # ============================================================
        # STEP 5: Calculate moisture & assess quality
        # ============================================================
        print("[Step 5/7] Assessing quality (FAQ standards)...")

        moisture_pct = None
        if sample_weight_g is not None and sample_weight_g > 0:
            grain_count_for_moisture = num_grains_weighed or len(contours)
            moisture_pct = calculate_moisture_content(
                sample_weight_g, grain_count_for_moisture, predicted_variety
            )
            
        if moisture_pct is not None:
            report["moisture_source"] = "weight_based"
        else:
            raise ValueError(
                "Invalid or missing sample weight. Please enter the accurate physical "
                "weight (in grams) of the rice grains in the image to calculate moisture."
            )

        # FAQ grading
        faq_grade = assess_faq_grade(moisture_pct, defect_percentages)
        quality_score = calculate_quality_score(moisture_pct, defect_percentages)
        quality_report = generate_quality_report(
            predicted_variety, moisture_pct, defect_percentages,
            faq_grade, quality_score
        )

        report["quality"] = {
            "moisture_pct": round(moisture_pct, 2),
            "faq_grade": faq_grade,
            "quality_score": round(quality_score, 2),
            "details": quality_report,
        }
        print(f"  -> Moisture: {moisture_pct:.1f}%, Grade: {faq_grade}, "
              f"Score: {quality_score:.1f}/100")

        # ============================================================
        # STEP 6: Shelf life estimation
        # ============================================================
        print("[Step 6/7] Estimating shelf life...")

        shelf_raw = generate_shelf_life_report(
            predicted_variety, moisture_pct,
            defect_percentages.get("broken", 0),
            defect_percentages.get("damaged", 0),
            storage_temp_c, storage_humidity_rh,
        )
        # Normalize shelf report keys for frontend compatibility
        from src.config import BASELINE_SHELF_LIFE_DAYS
        report["shelf_life"] = {
            "shelf_life_days": shelf_raw.get("shelf_life_days", 0),
            "shelf_life_months": shelf_raw.get("shelf_life_months", 0),
            "shelf_life_display": shelf_raw.get("shelf_life_display", ""),
            "risk_level": shelf_raw.get("risk_level", "Moderate"),
            "moisture_drift_flag": shelf_raw.get("moisture_drift_flag", False),
            "baseline_days": BASELINE_SHELF_LIFE_DAYS,
            "factors": shelf_raw.get("factor_breakdown", shelf_raw.get("factors", {})),
            "storage_conditions": {
                "temperature": storage_temp_c,
                "humidity": storage_humidity_rh,
            },
            "recommendations": shelf_raw.get("recommendations", []),
        }
        print(f"  -> Estimated shelf life: {report['shelf_life']['shelf_life_display']}"
              f" (Risk: {report['shelf_life']['risk_level']})")

        # ============================================================
        # STEP 7: Price recommendation
        # ============================================================
        print("[Step 7/7] Recommending price...")

        # Convert FAQ grade to price-module format: "Grade A" -> "grade_a"
        faq_grade_key = faq_grade.lower().replace(" ", "_")
        if faq_grade_key not in ("grade_a", "grade_b", "common", "rejected"):
            faq_grade_key = "rejected"

        try:
            price_result = recommend_price(
                predicted_variety, faq_grade_key, defect_percentages,
                quality_score=quality_score,
                custom_price=custom_price_per_qtl,
            )
            grade_comparison = compare_grade_prices(predicted_variety)
            report["pricing"] = {
                "base_price": price_result.get("base_price", 0),
                "recommended_price": price_result.get("recommended_price", 0),
                "quality_multiplier": price_result.get("quality_multiplier", 1.0),
                "quality_score": price_result.get("quality_score", 0),
                "price_range": price_result.get("price_range", {}),
                "price_source": price_result.get("price_source", "static"),
                "used_other": price_result.get("used_other", False),
                "api_variety": price_result.get("api_variety", ""),
                "mandi_details": price_result.get("mandi_details", []),
                "faq_grade": faq_grade,
                "deductions": price_result.get("adjustments", {}),
                "grade_comparison": grade_comparison.get("prices", {}),
                "justification": price_result.get("justification", ""),
            }
        except Exception as e:
            print(f"  ! Price recommendation error: {e}")
            report["pricing"] = {
                "base_price": 0,
                "recommended_price": 0,
                "price_source": "static",
                "used_other": False,
                "faq_grade": faq_grade,
                "error": str(e),
            }
        print(f"  -> Recommended price: Rs.{report['pricing'].get('recommended_price', 'N/A')}/quintal")

        # ============================================================
        # Save outputs
        # ============================================================
        if save_output:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            basename = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save overlay image
            overlay = draw_segmentation_overlay(image, contours, grain_df, self.pixels_per_mm)

            # Color-code grains by multi-label identity
            # Broken grains (any defect) = Red, Foreign = Dark Red
            # Whole grains colored by defect type
            defect_colors = {
                # Whole grains by defect
                ("whole", "clean"):      (0, 255, 0),        # Green - perfect grain
                ("whole", "chalky"):     (255, 255, 0),     # Cyan
                ("whole", "damaged"):    (0, 165, 255),     # Orange
                ("whole", "discolored"): (255, 0, 255),     # Magenta
                # All broken grains = Red
                ("broken", "clean"):     (0, 0, 255),       # Red
                ("broken", "chalky"):    (0, 0, 255),       # Red
                ("broken", "damaged"):   (0, 0, 255),       # Red
                ("broken", "discolored"): (0, 0, 255),      # Red
                # Foreign = Dark red
                ("foreign", "clean"):    (0, 0, 128),       # Dark red
            }
            for i, (cnt, gc) in enumerate(zip(contours, grain_classifications)):
                key = (gc["structural"], gc["defect"])
                color = defect_colors.get(key, (255, 255, 255))
                cv2.drawContours(overlay, [cnt], -1, color, 2)

            # Draw ArUCo marker bounding box in yellow
            aruco_bbox = aruco_info.get("aruco_bbox")
            if aruco_bbox:
                ax, ay, aw, ah = aruco_bbox
                cv2.rectangle(overlay, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 3)  # Cyan/Yellow
                cv2.putText(overlay, "ArUCo", (ax, ay - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            overlay_path = os.path.join(OUTPUT_DIR, f"{basename}_{timestamp}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
            report["output_overlay"] = overlay_path

            # Save JSON report
            report_path = os.path.join(OUTPUT_DIR, f"{basename}_{timestamp}_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            report["output_report"] = report_path

            # Save per-grain features CSV (with multi-label columns)
            grain_df["model_prediction"] = defect_predictions
            grain_df["structural"] = [
                gc["structural"] for gc in grain_classifications
            ][:len(grain_df)]
            grain_df["defect"] = [
                gc["defect"] for gc in grain_classifications
            ][:len(grain_df)]
            csv_path = os.path.join(OUTPUT_DIR, f"{basename}_{timestamp}_grains.csv")
            grain_df.to_csv(csv_path, index=False)
            report["output_csv"] = csv_path

            print(f"\n[Output] Overlay: {overlay_path}")
            print(f"[Output] Report: {report_path}")
            print(f"[Output] CSV: {csv_path}")

        # ============================================================
        # Print summary
        # ============================================================
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"  Variety:          {report['variety']['display_name']} ({report['variety']['category']})")
        print(f"  Total Grains:     {report['segmentation']['total_grains']}")
        print(f"  Avg Whole Length: {self._avg_whole_length} mm")
        print(f"  Good Grain:       {defect_percentages.get('whole', 0):.1f}%")
        print(f"  Broken:           {defect_percentages.get('broken', 0):.1f}%")
        print(f"  Equiv Head Rice:  {ehr_result['equivalent_head_rice_pct']:.1f}%")
        print(f"  Moisture:         {report['quality']['moisture_pct']}%")
        print(f"  FAQ Grade:        {report['quality']['faq_grade']}")
        print(f"  Quality Score:    {report['quality']['quality_score']}/100")
        print(f"  Shelf Life:       {report['shelf_life']['shelf_life_display']} (Risk: {report['shelf_life']['risk_level']})")
        print(f"  Price:            Rs.{report['pricing'].get('recommended_price', 'N/A')}/quintal")
        print("=" * 60)

        return report


def analyze_image(
    image_path: str,
    sample_weight_g: float = None,
    num_grains: int = None,
    variety: str = None,
    storage_temp: float = 25.0,
    storage_humidity: float = 60.0,
    aruco_path: str = None,
) -> Dict:
    """
    Convenience function for single-image analysis.

    Parameters
    ----------
    image_path : str
        Path to the rice grain image.
    sample_weight_g : float, optional
        Sample weight in grams.
    num_grains : int, optional
        Number of grains weighed.
    variety : str, optional
        Known variety name.
    storage_temp : float
        Storage temperature (°C).
    storage_humidity : float
        Storage humidity (%).
    aruco_path : str, optional
        Path to ArUCo calibration image.

    Returns
    -------
    dict
        Analysis report.
    """
    analyzer = RiceGrainAnalyzer(aruco_path)
    analyzer.load_models()
    return analyzer.analyze(
        image_path,
        sample_weight_g=sample_weight_g,
        num_grains_weighed=num_grains,
        variety_hint=variety,
        storage_temp_c=storage_temp,
        storage_humidity_rh=storage_humidity,
    )


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rice Grain Quality Analyzer — End-to-End Pipeline"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to rice grain image")
    parser.add_argument("--aruco", type=str, default=ARUCO_IMAGE_PATH,
                        help="Path to ArUCo calibration image")
    parser.add_argument("--sample-weight", type=float, default=None,
                        help="Sample weight in grams (for moisture calculation)")
    parser.add_argument("--num-grains", type=int, default=None,
                        help="Number of grains in weighed sample")
    parser.add_argument("--variety", type=str, default=None,
                        help="Known variety name (skip prediction)")
    parser.add_argument("--storage-temp", type=float, default=25.0,
                        help="Storage temperature (°C)")
    parser.add_argument("--storage-humidity", type=float, default=60.0,
                        help="Storage relative humidity (%%)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory")

    args = parser.parse_args()

    report = analyze_image(
        args.image,
        sample_weight_g=args.sample_weight,
        num_grains=args.num_grains,
        variety=args.variety,
        storage_temp=args.storage_temp,
        storage_humidity=args.storage_humidity,
        aruco_path=args.aruco,
    )

    # Print final JSON report
    print("\n" + json.dumps(report, indent=2, default=str))
