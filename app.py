#!/usr/bin/env python3
"""
Flask application serving a simple skin analysis API and single‑page UI.

The API exposes a single endpoint, ``/analyze``, that accepts an uploaded
image and user information (e.g., age and skin type). It returns a JSON
response containing classification probabilities, recommended products and a
base64‑encoded annotated image highlighting detected issues.

To start the server, run::

    FLASK_APP=app.py flask run --port 5000 --reload

This will serve the front‑end from the ``static`` directory and handle
analysis requests. You can open http://localhost:5000 in your browser to
interact with the app.
"""
import base64
import io
import os
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from training.predict import highlight_issues, predict as predict_probs

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the trained model when the application starts
MODEL_PATH = Path(__file__).resolve().parent / "model" / "skin_classifier.h5"
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
    CLASS_INDICES = {0: "acne", 1: "dark_circles"}
else:
    model = None
    CLASS_INDICES = {}


def allowed_file(filename: str) -> bool:
    """Check if the uploaded filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg"}


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handle skin analysis requests.

    The request must contain form data with the fields ``age``, ``skinType`` and
    ``photo`` (the image file). Returns a JSON payload with classification
    probabilities, detected issues, recommended products and a base64‑encoded
    annotated image.
    """
    if model is None:
        return jsonify({"error": "Model not found. Please train the model first."}), 500

    # Validate and load the uploaded file
    if "photo" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["photo"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Only JPG and PNG images are accepted."}), 400

    # Read file content
    file_bytes = file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    original_np = np.array(image)

    # Preprocess for prediction (resize to 224x224)
    resized = cv2.resize(original_np, (224, 224))
    probs = predict_probs(model, resized, CLASS_INDICES)

    # Decide which issues are present (threshold = 0.5)
    threshold = 0.5
    detected = [cls for cls, prob in probs.items() if prob >= threshold]

    # Recommend products based on detected issues
    recommendations: List[Dict[str, str]] = []
    if "acne" in detected:
        recommendations.append(
            {
                "issue": "acne",
                "product_name": "2% Salicylic Acid Serum",
                "description": (
                    "Salicylic acid is a beta‑hydroxy acid that helps clear pores "
                    "and reduce acne by exfoliating dead skin cells【324999427622202†L186-L206】. A "
                    "2% formulation can be used once daily to manage mild acne【324999427622202†L221-L227】."
                ),
            }
        )
    if "dark_circles" in detected:
        recommendations.append(
            {
                "issue": "dark_circles",
                "product_name": "10% Vitamin C Eye Serum",
                "description": (
                    "A 2009 clinical trial found that applying a 10% vitamin C product "
                    "around the eyes for six months increased skin thickness and "
                    "reduced the visibility of dark circles【474670558632404†L251-L259】. Topical creams containing "
                    "vitamin C may help lighten the appearance of dark circles【439155641374241†L170-L174】."
                ),
            }
        )

    # Generate annotated image
    annotated_img = highlight_issues(original_np[:, :, ::-1], probs, threshold=threshold)
    # Convert BGR to RGB for PIL
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = {
        "probabilities": probs,
        "detected": detected,
        "recommendations": recommendations,
        "annotatedImage": b64_image,
    }
    return jsonify(response)


@app.route("/static/<path:path>")
def send_static(path):
    """Serve static files (JS, CSS)."""
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    # For development, run with debug mode enabled. In production use a WSGI server.
    app.run(host="0.0.0.0", port=5000, debug=True)