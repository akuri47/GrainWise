"""
Rice Grain Quality Analyzer — Grain Segmentation & Feature Extraction
======================================================================
Handles:
  1. ArUCo marker detection for pixel-to-mm calibration
  2. Grain segmentation using the user-specified pipeline:
     Grayscale → Gaussian Blur (5×5) → Otsu Threshold →
     Morphological Opening → Contour filtering (100–50,000 px²)
  3. Per-grain morphological and color feature extraction
  4. Batch processing of image folders
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    ARUCO_IMAGE_PATH, ARUCO_MARKER_SIZE_MM,
    ARUCO_MARKER_WIDTH_MM, ARUCO_MARKER_HEIGHT_MM,
    ARUCO_DICT_TYPE,
    GAUSSIAN_BLUR_KERNEL, MORPH_KERNEL_SIZE, MORPH_ITERATIONS,
    CONTOUR_AREA_MIN, CONTOUR_AREA_MAX, WHOLE_GRAIN_DIR, OUTPUT_DIR,
)


# ============================================================
# ArUCo Calibration
# ============================================================

def detect_aruco_marker(image_path: str = None,
                        marker_width_mm: float = None,
                        marker_height_mm: float = None) -> float:
    """
    Detect the calibration marker in an image and compute px/mm ratio.
    Accepts both coded ArUCo markers and plain white rectangles.

    Parameters
    ----------
    image_path : str, optional
        Path to the image. Defaults to config path.
    marker_width_mm : float, optional
        Real-world width of marker in mm.
    marker_height_mm : float, optional
        Real-world height of marker in mm.

    Returns
    -------
    float
        Pixels per millimeter (px/mm) ratio.
    """
    if image_path is None:
        image_path = ARUCO_IMAGE_PATH

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Calibration image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    result = detect_aruco_from_image(image, marker_width_mm, marker_height_mm)
    if result is None:
        raise ValueError(
            "Could not detect calibration marker. "
            "Ensure the white reference marker is clearly visible."
        )
    return result[0]  # Return just pixels_per_mm



def _fallback_aruco_detection(gray: np.ndarray,
                              marker_width_mm: float = None,
                              marker_height_mm: float = None):
    """
    Fallback marker detection using contour analysis.
    Accepts both square and rectangular calibration markers (plain white paper).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
    marker_width_mm : float, optional
        Real-world width of marker in mm. Defaults to ARUCO_MARKER_SIZE_MM.
    marker_height_mm : float, optional
        Real-world height of marker in mm. Defaults to ARUCO_MARKER_SIZE_MM.

    Returns
    -------
    tuple : (pixels_per_mm, x, y, w, h)
        Calibration value and bounding box of the detected marker.
    """
    if marker_width_mm is None:
        marker_width_mm = ARUCO_MARKER_WIDTH_MM
    if marker_height_mm is None:
        marker_height_mm = ARUCO_MARKER_HEIGHT_MM

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try both high-brightness threshold (white paper) and Otsu
    _, binary_bright = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_marker = None
    best_score = float('inf')

    for binary in [binary_bright, binary_otsu]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > 5000000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / h if h > 0 else 0

            # The marker is rectangular (10×6 mm), so the expected pixel
            # aspect ratio is ~1.67 or ~0.6 depending on orientation.
            # Allow 0.4–2.5 to handle perspective distortion.
            if aspect < 0.4 or aspect > 2.5:
                continue

            # High solidity = filled rectangle (not a grain outline)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.85:
                continue

            # Roughly rectangular: 4-8 sides
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) < 4 or len(approx) > 10:
                continue

            # Derive px/mm and sanity check
            px_per_mm_x = w / marker_width_mm
            px_per_mm_y = h / marker_height_mm
            pixels_per_mm = (px_per_mm_x + px_per_mm_y) / 2.0

            # Sanity: px/mm must be reasonable for phone camera images
            # (typical range 10-200 px/mm for 10mm markers)
            if pixels_per_mm < 5 or pixels_per_mm > 200:
                continue

            # Score: prefer contours with aspect ratio closest to expected
            # marker ratio (10/6 ≈ 1.67 or 6/10 = 0.6) and high solidity.
            # Lower score = better.
            expected_ratio = marker_width_mm / marker_height_mm  # 10/6 = 1.667
            # Check which orientation is closer
            aspect_diff = min(abs(aspect - expected_ratio),
                              abs(aspect - 1.0 / expected_ratio))
            aspect_penalty = aspect_diff * 5.0
            solidity_bonus = (1.0 - solidity) * 10.0       # 0 for perfect fill
            size_bonus = -area / 100000.0                  # prefer larger
            score = aspect_penalty + solidity_bonus + size_bonus

            if score < best_score:
                best_score = score
                best_marker = (pixels_per_mm, x, y, w, h)

    if best_marker:
        pixels_per_mm, x, y, w, h = best_marker
        print(f"[ArUCo Fallback] Detected marker: {w}x{h} px "
              f"(real: {marker_width_mm}x{marker_height_mm} mm)")
        print(f"[ArUCo Fallback] Calibration: {pixels_per_mm:.2f} px/mm")
        return best_marker  # (pixels_per_mm, x, y, w, h)

    raise ValueError(
        "Could not detect calibration marker in image. "
        "Please ensure the white reference marker is clearly visible."
    )


def detect_aruco_from_image(image: np.ndarray,
                            marker_width_mm: float = None,
                            marker_height_mm: float = None):
    """
    Detect calibration marker from an already-loaded BGR image.
    Used during analysis of uploaded images so the marker in the
    photo is used for calibration (not a separate calibration file).

    Parameters
    ----------
    image : np.ndarray
        BGR image containing rice grains + calibration marker.
    marker_width_mm : float, optional
        Real-world width of marker in mm.
    marker_height_mm : float, optional
        Real-world height of marker in mm.

    Returns
    -------
    tuple : (pixels_per_mm, x, y, w, h) or None
        Calibration value and bounding box, or None if not found.
    """
    if marker_width_mm is None:
        marker_width_mm = ARUCO_MARKER_WIDTH_MM
    if marker_height_mm is None:
        marker_height_mm = ARUCO_MARKER_HEIGHT_MM

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Try standard ArUCo detection (coded markers)
    aruco_dict_types = [
        cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_ARUCO_ORIGINAL,
    ]
    for dict_type in aruco_dict_types:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) > 0:
            marker_corners = corners[0][0]
            perimeter_px = cv2.arcLength(marker_corners.astype(np.float32), closed=True)
            perimeter_mm = 2.0 * (marker_width_mm + marker_height_mm)
            pixels_per_mm = perimeter_px / perimeter_mm
            xs = marker_corners[:, 0]
            ys = marker_corners[:, 1]
            bx, by = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max() - xs.min()), int(ys.max() - ys.min())
            print(f"[ArUCo] Found coded marker ID={ids[0][0]}, "
                  f"{pixels_per_mm:.2f} px/mm")
            return (pixels_per_mm, bx, by, bw, bh)

    # 2. Fallback: contour-based detection (plain white rectangle)
    print("[ArUCo] No coded marker. Trying contour-based detection...")
    try:
        return _fallback_aruco_detection(gray, marker_width_mm, marker_height_mm)
    except ValueError as e:
        print(f"[ArUCo] Detection failed: {e}")
        return None


# ============================================================
# Grain Segmentation Pipeline
# ============================================================

def _split_touching_grains(binary: np.ndarray, pixels_per_mm: float) -> np.ndarray:
    """
    Apply Watershed algorithm to separate touching grains in a binary mask.
    """
    # 1. Background (sure_bg) is dilation of binary
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    
    # 2. Foreground (sure_fg) is peaks of distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Calculate a sensible distance threshold based on typical grain width (e.g. 2.0 mm)
    # The center of a grain will be roughly 1.0 mm away from the edge
    if pixels_per_mm and pixels_per_mm > 0:
        dist_thresh = 0.35 * pixels_per_mm  # 0.35 mm from edge — more aggressive splitting
    else:
        dist_thresh = 0.5 * dist_transform.max() # fallback
        
    _, sure_fg = cv2.threshold(dist_transform, dist_thresh, 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 3. Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 4. Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 5. Apply watershed (requires 3 channel image)
    img_for_ws = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_for_ws, markers)
    
    # 6. Reconstruct binary mask: valid objects are markers > 1
    split_binary = np.zeros_like(binary)
    split_binary[markers > 1] = 255
    
    # Optionally, we can draw the watershed boundaries (markers == -1) as 0
    split_binary[markers == -1] = 0
    return split_binary

def segment_grains(image: np.ndarray, pixels_per_mm: float = None) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, dict]:
    """
    Segment individual rice grains using a multi-threshold union approach.

    Instead of picking a single 'best' threshold, we run multiple thresholds
    on the V-channel (HSV) and MERGE all valid grain masks together.
    This catches both light/translucent and dark/husked grains.

    Shape filters reject non-grain contours (background noise):
      - Aspect ratio must be >= 1.3 (grains are elongated, blobs are round)
      - Solidity must be >= 0.5 (grains are compact, noise is irregular)
    """
    # Determine area bounds based on calibration
    area_min = CONTOUR_AREA_MIN
    area_max = CONTOUR_AREA_MAX
    if pixels_per_mm and pixels_per_mm > 10:
        grain_area_min_mm2 = 1.5   # smallest broken grain
        grain_area_max_mm2 = 80.0  # largest grain
        area_min = max(100, int(grain_area_min_mm2 * pixels_per_mm * pixels_per_mm))
        area_max = max(50000, int(grain_area_max_mm2 * pixels_per_mm * pixels_per_mm))

    # Morphology kernel
    morph_k_size = (5, 5) if (pixels_per_mm and pixels_per_mm > 20) else MORPH_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_k_size)

    aruco_bbox = None

    def _is_aruco_marker(cnt):
        """Return True if contour looks like the rectangular calibration marker."""
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        # Marker is 10x6 mm -> aspect ~1.67. Allow 1.2-2.2 for perspective.
        if not (1.2 <= aspect <= 2.2):
            return False
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0
        # Marker is very solid (filled rectangle)
        if solidity < 0.85:
            return False
        # Marker area should be in the right ballpark
        if pixels_per_mm and pixels_per_mm > 10:
            expected_area = 10.0 * 6.0 * pixels_per_mm * pixels_per_mm  # ~60 mm²
            actual_area = cv2.contourArea(cnt)
            if actual_area > expected_area * 3 or actual_area < expected_area * 0.3:
                return False
        return True

    def _is_grain_shaped(cnt):
        """Return True if contour has grain-like shape (elongated, compact)."""
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        # Rice grains: aspect ratio 1.15+ (broken pieces can be nearly round)
        if aspect < 1.15 or aspect > 7.0:
            return False
        # Solidity check: grains are compact, background blobs are irregular
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0
        if solidity < 0.5:
            return False
        return True

    # ---- Multi-threshold union approach ----
    # Run multiple thresholds and merge all valid grain regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # Also prepare a grayscale for Otsu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)

    # Union mask: accumulates all grain regions across thresholds
    union_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    thresholds = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]

    for thresh_val in thresholds:
        _, binary = cv2.threshold(v_channel, thresh_val, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours_all, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_all:
            area = cv2.contourArea(cnt)
            if area < area_min or area > area_max:
                continue
            if _is_aruco_marker(cnt):
                aruco_bbox = cv2.boundingRect(cnt)
                continue
            if not _is_grain_shaped(cnt):
                continue
            # Draw this grain onto the union mask
            cv2.drawContours(union_mask, [cnt], -1, 255, -1)

    # Also try Otsu on grayscale
    _, otsu_binary = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_binary = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    otsu_binary = cv2.morphologyEx(otsu_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours_otsu, _ = cv2.findContours(otsu_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_otsu:
        area = cv2.contourArea(cnt)
        if area < area_min or area > area_max:
            continue
        if _is_aruco_marker(cnt):
            aruco_bbox = cv2.boundingRect(cnt)
            continue
        if not _is_grain_shaped(cnt):
            continue
        cv2.drawContours(union_mask, [cnt], -1, 255, -1)

    # Clean up the union mask
    union_mask = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Apply watershed to split touching grains in the union mask
    union_mask = _split_touching_grains(union_mask, pixels_per_mm)

    # Extract final contours from the union mask
    final_contours_all, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contours = []
    best_masks = []
    for cnt in final_contours_all:
        area = cv2.contourArea(cnt)
        if area < area_min or area > area_max:
            continue
        if _is_aruco_marker(cnt):
            aruco_bbox = cv2.boundingRect(cnt)
            continue
        # Final shape check (slightly relaxed after watershed may have altered shapes)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if aspect < 1.1 or aspect > 8.0:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0
        if solidity < 0.45:
            continue

        best_contours.append(cnt)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_cnt = cnt - np.array([x, y])
        cv2.drawContours(roi_mask, [shifted_cnt], -1, 255, -1)
        best_masks.append((roi_mask, x, y, w, h))

    if aruco_bbox:
        ax, ay, aw, ah = aruco_bbox
        print(f"[Segment] ArUCo marker excluded: bbox=({ax},{ay},{aw}x{ah})")
    print(f"[Segment] Multi-threshold union: "
          f"found {len(best_contours)} grains (area {area_min}-{area_max} px²)")

    return best_contours, best_masks, union_mask, {"aruco_bbox": aruco_bbox}


# ============================================================
# Per-Grain Feature Extraction
# ============================================================

def extract_grain_features(
    image: np.ndarray,
    contour: np.ndarray,
    mask,
    pixels_per_mm: float,
    grain_id: int = 0,
) -> Dict:
    """
    Extract morphological and color features from a single grain.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    contour : np.ndarray
        Grain contour.
    mask : tuple or np.ndarray
        ROI-based mask tuple ``(roi_mask, x, y, w, h)`` where roi_mask
        covers only the bounding box, **or** a full-image binary mask
        for backward compatibility.
    pixels_per_mm : float
        Calibration factor for converting pixels to mm.
    grain_id : int
        Identifier for this grain.

    Returns
    -------
    dict
        Dictionary of extracted features.
    """
    features = {"grain_id": grain_id}

    # ---- Unpack mask ----
    if isinstance(mask, tuple):
        roi_mask, mx, my, mw, mh = mask
    else:
        # Backward compatibility: full-image mask
        mx, my, mw, mh = cv2.boundingRect(contour)
        roi_mask = mask[my:my+mh, mx:mx+mw]

    # ---- Morphological Features ----

    # Area
    area_px = cv2.contourArea(contour)
    features["area_px"] = area_px
    features["area_mm2"] = area_px / (pixels_per_mm ** 2)

    # Perimeter
    perimeter_px = cv2.arcLength(contour, closed=True)
    features["perimeter_px"] = perimeter_px
    features["perimeter_mm"] = perimeter_px / pixels_per_mm

    # Minimum area bounding rectangle → Length and Width
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    length_px = max(w, h)
    width_px = min(w, h)

    features["length_px"] = length_px
    features["width_px"] = width_px
    features["length_mm"] = length_px / pixels_per_mm
    features["width_mm"] = width_px / pixels_per_mm
    features["aspect_ratio"] = length_px / width_px if width_px > 0 else 0

    # Circularity: 4π × Area / Perimeter²
    if perimeter_px > 0:
        features["circularity"] = (4 * np.pi * area_px) / (perimeter_px ** 2)
    else:
        features["circularity"] = 0

    # Solidity: Area / Convex Hull Area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features["solidity"] = area_px / hull_area if hull_area > 0 else 0

    # Eccentricity (from fitted ellipse)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (ecx, ecy), (ma, MA), angle_e = ellipse
        a = max(ma, MA) / 2.0
        b = min(ma, MA) / 2.0
        features["eccentricity"] = np.sqrt(1 - (b ** 2 / a ** 2)) if a > 0 else 0
    else:
        features["eccentricity"] = 0

    # Bounding box (in original image coordinates)
    features["bbox_x"] = mx
    features["bbox_y"] = my
    features["bbox_w"] = mw
    features["bbox_h"] = mh

    # ---- Color Features (using ROI crop) ----

    # Extract the ROI from the original image
    roi_bgr = image[my:my+mh, mx:mx+mw]
    grain_pixels_bgr = roi_bgr[roi_mask > 0]

    if len(grain_pixels_bgr) > 0:
        # BGR mean and std
        mean_bgr = np.mean(grain_pixels_bgr, axis=0)
        std_bgr = np.std(grain_pixels_bgr, axis=0)
        features["mean_b"] = mean_bgr[0]
        features["mean_g"] = mean_bgr[1]
        features["mean_r"] = mean_bgr[2]
        features["std_b"] = std_bgr[0]
        features["std_g"] = std_bgr[1]
        features["std_r"] = std_bgr[2]

        # Convert ROI to HSV for color analysis
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        grain_pixels_hsv = roi_hsv[roi_mask > 0]
        mean_hsv = np.mean(grain_pixels_hsv, axis=0)
        features["mean_h"] = mean_hsv[0]
        features["mean_s"] = mean_hsv[1]
        features["mean_v"] = mean_hsv[2]

        # Whiteness index (average intensity in grayscale)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        grain_gray = roi_gray[roi_mask > 0]
        features["mean_intensity"] = np.mean(grain_gray)
        features["std_intensity"] = np.std(grain_gray)

        # Chalkiness score: percentage of very bright pixels within grain
        # 1.0*std: normal grains ~0.16, chalky grains ~0.25-0.40
        brightness_threshold = np.mean(grain_gray) + 1.0 * np.std(grain_gray)
        chalky_pixels = np.sum(grain_gray > brightness_threshold)
        features["chalkiness_score"] = chalky_pixels / len(grain_gray) if len(grain_gray) > 0 else 0
        
        # Normalized color ratios (lighting-invariant)
        mr, mg, mb = features["mean_r"], features["mean_g"], features["mean_b"]
        features["rg_ratio"] = mr / mg if mg > 1 else 1.0
        features["rb_ratio"] = mr / mb if mb > 1 else 1.0
        features["gb_ratio"] = mg / mb if mb > 1 else 1.0
        # Normalized saturation (S / V to remove brightness dependency)
        features["norm_s"] = features["mean_s"] / features["mean_v"] if features["mean_v"] > 1 else 0.0
    else:
        for key in ["mean_b", "mean_g", "mean_r", "std_b", "std_g", "std_r",
                     "mean_h", "mean_s", "mean_v", "mean_intensity", "std_intensity",
                     "chalkiness_score", "rg_ratio", "rb_ratio", "gb_ratio", "norm_s"]:
            features[key] = 0

    return features


def extract_all_grain_features(
    image: np.ndarray,
    contours: List[np.ndarray],
    masks: List[np.ndarray],
    pixels_per_mm: float,
) -> pd.DataFrame:
    """
    Extract features from all segmented grains in an image.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    contours : List[np.ndarray]
        List of grain contours.
    masks : List[np.ndarray]
        List of grain binary masks.
    pixels_per_mm : float
        Calibration factor.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per grain and all extracted features.
    """
    all_features = []
    for i, (cnt, msk) in enumerate(zip(contours, masks)):
        features = extract_grain_features(image, cnt, msk, pixels_per_mm, grain_id=i)
        all_features.append(features)

    if all_features:
        return pd.DataFrame(all_features)
    else:
        return pd.DataFrame()


# ============================================================
# Image-Level Feature Aggregation
# ============================================================

def aggregate_image_features(grain_df: pd.DataFrame) -> Dict:
    """
    Aggregate per-grain features into image-level summary statistics.
    Used for variety classification.

    Parameters
    ----------
    grain_df : pd.DataFrame
        Per-grain feature DataFrame from extract_all_grain_features.

    Returns
    -------
    dict
        Aggregated features for the image.
    """
    if grain_df.empty:
        return {}

    agg = {
        "grain_count": len(grain_df),
        "mean_length_mm": grain_df["length_mm"].mean(),
        "std_length_mm": grain_df["length_mm"].std(),
        "median_length_mm": grain_df["length_mm"].median(),
        "mean_width_mm": grain_df["width_mm"].mean(),
        "std_width_mm": grain_df["width_mm"].std(),
        "median_width_mm": grain_df["width_mm"].median(),
        "mean_aspect_ratio": grain_df["aspect_ratio"].mean(),
        "std_aspect_ratio": grain_df["aspect_ratio"].std(),
        "mean_area_mm2": grain_df["area_mm2"].mean(),
        "std_area_mm2": grain_df["area_mm2"].std(),
        "mean_perimeter_mm": grain_df["perimeter_mm"].mean(),
        "mean_circularity": grain_df["circularity"].mean(),
        "mean_solidity": grain_df["solidity"].mean(),
        "mean_eccentricity": grain_df["eccentricity"].mean(),
        "mean_intensity": grain_df["mean_intensity"].mean(),
        "mean_h": grain_df["mean_h"].mean(),
        "mean_s": grain_df["mean_s"].mean(),
        "mean_chalkiness": grain_df["chalkiness_score"].mean(),
    }

    return agg


# ============================================================
# Visualization
# ============================================================

def draw_segmentation_overlay(
    image: np.ndarray,
    contours: List[np.ndarray],
    grain_df: pd.DataFrame = None,
    pixels_per_mm: float = None,
) -> np.ndarray:
    """
    Draw segmentation contours and annotations on the image.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    contours : List[np.ndarray]
        Grain contours.
    grain_df : pd.DataFrame, optional
        Per-grain features for annotation.
    pixels_per_mm : float, optional
        Calibration factor for scale bar.

    Returns
    -------
    np.ndarray
        Annotated image.
    """
    overlay = image.copy()

    # Draw contours
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Draw grain IDs and dimensions
    if grain_df is not None and not grain_df.empty:
        for _, row in grain_df.iterrows():
            x = int(row.get("bbox_x", 0))
            y = int(row.get("bbox_y", 0))
            gid = int(row.get("grain_id", 0))

            # Draw grain ID
            cv2.putText(overlay, f"#{gid}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw scale bar if calibrated
    if pixels_per_mm is not None:
        bar_length_mm = 10
        bar_length_px = int(bar_length_mm * pixels_per_mm)
        h, w = overlay.shape[:2]
        bar_x = w - bar_length_px - 20
        bar_y = h - 30
        cv2.line(overlay, (bar_x, bar_y), (bar_x + bar_length_px, bar_y), (255, 255, 255), 3)
        cv2.putText(overlay, f"{bar_length_mm} mm", (bar_x, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw grain count
    cv2.putText(overlay, f"Grains: {len(contours)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return overlay


# ============================================================
# Batch Processing
# ============================================================

def process_image(
    image_path: str,
    pixels_per_mm: float,
    save_overlay: bool = False,
    output_dir: str = None,
) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
    """
    Process a single rice grain image: segment, extract features, aggregate.

    Parameters
    ----------
    image_path : str
        Path to the grain image.
    pixels_per_mm : float
        ArUCo calibration factor.
    save_overlay : bool
        If True, save annotated overlay image.
    output_dir : str, optional
        Directory for saving output files.

    Returns
    -------
    grain_df : pd.DataFrame
        Per-grain feature DataFrame.
    image_features : dict
        Aggregated image-level features.
    overlay : np.ndarray
        Annotated image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    print(f"\n[Process] Processing: {image_path}")
    print(f"[Process] Image size: {image.shape[1]}×{image.shape[0]}")

    # Segment grains
    contours, masks, binary = segment_grains(image, pixels_per_mm)

    # Extract per-grain features
    grain_df = extract_all_grain_features(image, contours, masks, pixels_per_mm)

    # Aggregate to image-level
    image_features = aggregate_image_features(grain_df)

    # Create overlay
    overlay = draw_segmentation_overlay(image, contours, grain_df, pixels_per_mm)

    # Save outputs
    if save_overlay and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        overlay_path = os.path.join(output_dir, f"{basename}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)
        print(f"[Process] Overlay saved: {overlay_path}")

        csv_path = os.path.join(output_dir, f"{basename}_features.csv")
        grain_df.to_csv(csv_path, index=False)
        print(f"[Process] Features saved: {csv_path}")

    return grain_df, image_features, overlay


def process_variety_folder(
    folder_path: str,
    variety_name: str,
    pixels_per_mm: float,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Process all images in a variety folder.

    Parameters
    ----------
    folder_path : str
        Path to the variety image folder.
    variety_name : str
        Name/label of the variety.
    pixels_per_mm : float
        Calibration factor.
    save_outputs : bool
        Whether to save overlays and CSV files.

    Returns
    -------
    all_grains_df : pd.DataFrame
        Combined per-grain features from all images.
    image_summaries : List[dict]
        Image-level feature aggregations.
    """
    output_dir = os.path.join(OUTPUT_DIR, variety_name) if save_outputs else None

    all_grains = []
    image_summaries = []

    # Find image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    print(f"\n{'='*60}")
    print(f"Processing variety: {variety_name} ({len(image_files)} images)")
    print(f"{'='*60}")

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            grain_df, img_features, _ = process_image(
                img_path, pixels_per_mm,
                save_overlay=save_outputs,
                output_dir=output_dir,
            )

            # Tag grains with variety and source image
            grain_df["variety"] = variety_name
            grain_df["source_image"] = img_file
            all_grains.append(grain_df)

            img_features["variety"] = variety_name
            img_features["source_image"] = img_file
            image_summaries.append(img_features)

        except Exception as e:
            print(f"[Error] Failed to process {img_path}: {e}")

    if all_grains:
        all_grains_df = pd.concat(all_grains, ignore_index=True)
    else:
        all_grains_df = pd.DataFrame()

    return all_grains_df, image_summaries


def process_all_varieties(pixels_per_mm: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all variety folders under the whole_grain directory.

    Parameters
    ----------
    pixels_per_mm : float
        Calibration factor.

    Returns
    -------
    all_grains_df : pd.DataFrame
        Combined per-grain features from all varieties.
    summary_df : pd.DataFrame
        Image-level feature summaries.
    """
    all_grains = []
    all_summaries = []

    for variety_folder in sorted(os.listdir(WHOLE_GRAIN_DIR)):
        folder_path = os.path.join(WHOLE_GRAIN_DIR, variety_folder)
        if not os.path.isdir(folder_path):
            continue

        grain_df, img_summaries = process_variety_folder(
            folder_path, variety_folder, pixels_per_mm
        )

        if not grain_df.empty:
            all_grains.append(grain_df)
        all_summaries.extend(img_summaries)

    all_grains_df = pd.concat(all_grains, ignore_index=True) if all_grains else pd.DataFrame()
    summary_df = pd.DataFrame(all_summaries)

    # Save combined outputs
    if not all_grains_df.empty:
        all_grains_df.to_csv(os.path.join(OUTPUT_DIR, "all_grain_features.csv"), index=False)
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "image_summaries.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"Total grains extracted: {len(all_grains_df)}")
    print(f"Total images processed: {len(summary_df)}")
    print(f"{'='*60}")

    return all_grains_df, summary_df


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rice Grain Segmentation & Feature Extraction")
    parser.add_argument("--image", type=str, help="Path to a single grain image to process")
    parser.add_argument("--aruco", type=str, default=ARUCO_IMAGE_PATH,
                        help="Path to ArUCo calibration image")
    parser.add_argument("--folder", type=str, help="Path to a variety folder to process")
    parser.add_argument("--variety", type=str, default="unknown",
                        help="Variety name (used when processing a folder)")
    parser.add_argument("--all", action="store_true",
                        help="Process all varieties in whole_grain directory")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory")

    args = parser.parse_args()

    # Step 1: Calibrate
    print("=" * 60)
    print("ArUCo Calibration")
    print("=" * 60)
    px_per_mm = detect_aruco_marker(args.aruco)
    print(f"\nCalibration result: {px_per_mm:.2f} pixels/mm\n")

    # Step 2: Process
    if args.all:
        process_all_varieties(px_per_mm)
    elif args.folder:
        process_variety_folder(args.folder, args.variety, px_per_mm)
    elif args.image:
        grain_df, img_features, overlay = process_image(
            args.image, px_per_mm, save_overlay=True, output_dir=args.output
        )
        print(f"\nExtracted {len(grain_df)} grains")
        if not grain_df.empty:
            print(f"Mean length: {grain_df['length_mm'].mean():.2f} mm")
            print(f"Mean width: {grain_df['width_mm'].mean():.2f} mm")
            print(f"Mean AR: {grain_df['aspect_ratio'].mean():.2f}")
    else:
        parser.print_help()
