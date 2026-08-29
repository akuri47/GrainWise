"""
Rice Grain Quality Analyzer — Flask Web Application
====================================================
Provides a web interface for uploading rice grain images
and viewing analysis results.
"""

import os
import sys
import json
import uuid
import base64
import traceback
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from src.analyzer import RiceGrainAnalyzer
from src.config import ARUCO_IMAGE_PATH, OUTPUT_DIR, VARIETY_DATABASE, DEFECT_LABELS

import cv2
import numpy as np

# ============================================================
# Flask App Setup
# ============================================================

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(WEB_DIR, "templates")
STATIC_DIR = os.path.join(WEB_DIR, "static")

app = Flask(
    "rice_analyzer",
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
    root_path=WEB_DIR,
)
CORS(app)

# Upload configuration
UPLOAD_DIR = os.path.join(WEB_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# Initialize analyzer (singleton)
analyzer = None


def get_analyzer():
    """Get or create the analyzer singleton."""
    global analyzer
    if analyzer is None:
        analyzer = RiceGrainAnalyzer(ARUCO_IMAGE_PATH)
        analyzer.load_models()
    return analyzer


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Main upload page."""
    from flask import make_response
    response = make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/varieties", methods=["GET"])
def get_varieties():
    """Return list of supported varieties."""
    varieties = []
    for key, data in VARIETY_DATABASE.items():
        varieties.append({
            "key": key,
            "display_name": data["display_name"],
            "category": data["category"],
            "dry_weight_g": data["dry_weight_g"],
        })
    return jsonify({"varieties": varieties})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze an uploaded rice grain image.

    Expects multipart form data with:
    - image: The grain image file
    - sample_weight (optional): Sample weight in grams
    - num_grains (optional): Number of grains weighed
    - variety (optional): Known variety name
    - storage_temp (optional): Storage temperature °C
    - storage_humidity (optional): Storage humidity %
    """
    print("==================================================", flush=True)
    print(f"RECEIVED REQUEST TO /api/analyze: {request.method} {request.content_length} bytes", flush=True)
    print("Headers:", request.headers, flush=True)
    try:
        # Validate image upload
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

        # Save uploaded file
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{file_ext}"
        upload_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(upload_path)

        # Parse optional parameters
        sample_weight = request.form.get("sample_weight", type=float)
        # Treat 0 or negative as not provided
        if sample_weight is not None and sample_weight <= 0:
            sample_weight = None
        aruco_length = request.form.get("aruco_length", type=float, default=10.0)
        aruco_width = request.form.get("aruco_width", type=float, default=6.0)
        variety = request.form.get("variety", type=str)
        storage_temp = request.form.get("storage_temp", type=float, default=25.0)
        storage_humidity = request.form.get("storage_humidity", type=float, default=60.0)
        custom_price = request.form.get("custom_price", type=float)
        # Treat 0 or negative as not provided
        if custom_price is not None and custom_price <= 0:
            custom_price = None

        # Clean up variety input
        if variety and variety.strip() == "":
            variety = None

        # Update ArUCo marker size if user provided custom dimensions
        rice_analyzer = get_analyzer()
        
        # Store original calibration to restore later
        original_calibration = rice_analyzer.pixels_per_mm
        
        try:
            # Estimate num_grains from segmentation for moisture calc
            report = rice_analyzer.analyze(
                image_path=upload_path,
                sample_weight_g=sample_weight,
                num_grains_weighed=None,
                variety_hint=variety,
                storage_temp_c=storage_temp,
                storage_humidity_rh=storage_humidity,
                save_output=True,
                marker_width_mm=aruco_width,
                marker_height_mm=aruco_length,
                custom_price_per_qtl=custom_price,
            )
        finally:
            # Restore original calibration so we don't permanently break the singleton
            rice_analyzer.pixels_per_mm = original_calibration

        # Encode overlay image as base64 for frontend display
        overlay_path = report.get("output_overlay")
        if overlay_path and os.path.exists(overlay_path):
            with open(overlay_path, "rb") as img_file:
                overlay_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            report["overlay_base64"] = f"data:image/jpeg;base64,{overlay_b64}"

        # Encode original image as base64
        with open(upload_path, "rb") as img_file:
            original_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        report["original_base64"] = f"data:image/{file_ext};base64,{original_b64}"

        # Convert numpy types to native Python types for JSON serialization
        def _make_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return obj

        report = _make_serializable(report)

        return jsonify(report)

    except Exception as e:
        import sys
        # Safely log the error without crashing on Windows cp1252 consoles
        error_msg = traceback.format_exc()
        try:
            print(error_msg.encode('ascii', 'replace').decode('ascii'), file=sys.stderr)
        except Exception:
            pass

        return jsonify({
            "error": str(e),
            "status": "error",
            "traceback": error_msg,
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint — also triggers model pre-loading."""
    global analyzer
    models_ready = analyzer is not None
    # Pre-load models on health ping so they're ready for /api/analyze
    if not models_ready:
        try:
            get_analyzer()
            models_ready = True
        except Exception:
            models_ready = False
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_ready,
    })


@app.route("/output/<path:filename>")
def serve_output(filename):
    """Serve output files (overlays, reports)."""
    return send_from_directory(OUTPUT_DIR, filename)


# ============================================================
# Eager Model Preload (runs when gunicorn --preload imports this module)
# ============================================================
print("[Startup] Pre-loading ML models...")
try:
    get_analyzer()
    print("[Startup] Models loaded successfully.")
except Exception as e:
    print(f"[Startup] WARNING: Could not preload models: {e}")
    print("[Startup] Models will be loaded on first request.")


# ============================================================
# Main (local dev only)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Rice Grain Quality Analyzer -- Web Application")
    print("=" * 60)
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("\nStarting web server...")
    print("Open http://localhost:8080 in your browser\n")

    app.run(host="0.0.0.0", port=8080, debug=False)
