# app.py
# Flask + TensorFlow demo: AI Skin Analyzer (acne, dark circles)
# --------------------------------------------------------------
# This file is self-contained and aligned with the templates you added:
#   templates/base.html, templates/index.html, templates/result.html
#   static/app.js
#
# Model: expects an h5 Keras model at model/skin_classifier.h5
# Trained on 224x224 MobileNetV2 preprocessing with 2 classes:
#   index 0 -> "acne", index 1 -> "dark_circles"

import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for, abort
)

# ML deps
import numpy as np
from PIL import Image

# Make sure these imports succeed in your environment
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array

import cv2


# ---------------------------
# App & paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "skin_classifier.h5"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# ---------------------------
# Load model once at startup
# ---------------------------
if not MODEL_PATH.exists():
    # Let the UI load, but warn in logs
    app.logger.warning("Model file not found at %s", MODEL_PATH)

# TF 2.10 lazily loads; wrap to avoid noisy GPU logs breaking app import
MODEL = None
CLASS_NAMES = ["acne", "dark_circles"]

def load_model_once():
    global MODEL
    if MODEL is None:
        app.logger.info("Loading model from %s", MODEL_PATH)
        MODEL = tf.keras.models.load_model(str(MODEL_PATH))
    return MODEL


# ---------------------------
# Utilities
# ---------------------------
ALLOWED_EXTS = {"jpg", "jpeg", "png"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def save_upload(file_storage) -> str:
    """Save upload to static/uploads and return filename (not path)."""
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    fname = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{ext}"
    out_path = UPLOAD_DIR / fname
    file_storage.save(out_path)
    return fname

def pil_open_rgb(path: Path) -> Image.Image:
    # Robust open; converts to RGB
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def predict_probs(img_pil: Image.Image, img_size: int = 224) -> dict:
    """Return softmax probabilities for classes {acne, dark_circles}."""
    model = load_model_once()

    img = img_pil.resize((img_size, img_size))
    arr = img_to_array(img)  # (H,W,3) float32
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    arr = preprocess_input(arr)  # MobileNetV2 preprocessing

    preds = model.predict(arr, verbose=0)  # shape (1, 2)
    probs = preds[0].astype(float)

    # Defensive: ensure numeric & clipped
    probs = np.clip(probs, 0.0, 1.0)
    return {
        "acne": float(probs[0]),
        "dark_circles": float(probs[1]),
    }

def draw_regions(original_path: Path, out_path: Path, probs: dict):
    """
    Lightweight region markers with OpenCV:
      - face + eyes boxes (Haar cascade)
      - if acne prob high -> draw T-zone/mouth small boxes
    """
    img_bgr = cv2.imread(str(original_path))
    if img_bgr is None:
        # fallback: just copy
        cv2.imwrite(str(out_path), cv2.imread(str(original_path)))
        return

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Haar cascades (bundled with OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, 1.15, 5)
    color = (255, 0, 0)  # blue-ish in BGR
    thickness = 2

    # Draw face & eyes
    for (x, y, w, h) in faces[:1]:  # just the largest/first face
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_bgr[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.15, 5)
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, thickness)
            # Dark-circle helper: box just below each eye
            below_y = min(ey + eh + int(0.15 * eh), h - 1)
            below_h = int(0.3 * eh)
            cv2.rectangle(
                roi_color,
                (ex, below_y),
                (ex + ew, min(below_y + below_h, h - 1)),
                color,
                thickness,
            )

    # Acne helper boxes in the T-zone if acne prob is noticeable
    if probs.get("acne", 0.0) >= 0.35 and len(faces) > 0:
        (x, y, w, h) = faces[0]
        # small grid around nose/mouth/cheeks
        patches = [
            (x + int(0.35 * w), y + int(0.50 * h), int(0.30 * w), int(0.12 * h)),  # upper lip area
            (x + int(0.25 * w), y + int(0.65 * h), int(0.18 * w), int(0.10 * h)),  # left lower cheek
            (x + int(0.60 * w), y + int(0.65 * h), int(0.18 * w), int(0.10 * h)),  # right lower cheek
            (x + int(0.42 * w), y + int(0.40 * h), int(0.16 * w), int(0.10 * h)),  # nose tip
        ]
        for (px, py, pw, ph) in patches:
            cv2.rectangle(img_bgr, (px, py), (px + pw, py + ph), color, thickness)

    # Optional label text with main issue + prob
    main_issue = max(probs, key=probs.get)
    label = f"{main_issue.replace('_', ' ')}: {probs[main_issue]:.2f}"
    cv2.putText(img_bgr, label, (12, img_bgr.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img_bgr)


def build_recommendations(probs: dict, age: int, skin_type: str):
    """
    Simple rule-based cards. You can expand easily later.
    Returns list of {title, note}.
    """
    cards = []
    if probs.get("acne", 0.0) >= 0.35:
        cards.append({
            "title": "2% Salicylic Acid (BHA)",
            "note": "Gently unclogs pores and reduces active acne; start 3x/week at night. Patch test first."
        })
        cards.append({
            "title": "Niacinamide 4–10%",
            "note": "Helps with sebum regulation and post-acne marks (PIH). Layer under moisturizer."
        })
    if probs.get("dark_circles", 0.0) >= 0.35:
        cards.append({
            "title": "10% Vitamin C Eye Serum",
            "note": "Brightens the under-eye area and supports collagen. Use AM with sunscreen."
        })
        cards.append({
            "title": "Peptides + Caffeine Eye Gel",
            "note": "Temporarily reduces puffiness and improves skin firmness around the eyes."
        })

    # Always-on sunscreen message
    cards.append({
        "title": "Broad-Spectrum Sunscreen (SPF 30+)",
        "note": "Daily use prevents worsening of acne marks and dark circles pigmentation."
    })
    # Light personalization touch
    if skin_type.lower() == "dry":
        cards.append({
            "title": "Ceramide Moisturizer",
            "note": "Supports skin barrier; pair with actives to reduce irritation."
        })
    elif skin_type.lower() == "oily":
        cards.append({
            "title": "Oil-free Gel Moisturizer",
            "note": "Hydrates without heaviness; pairs well with niacinamide/BHA."
        })
    return cards


# ---------------------------
# Routes
# ---------------------------
@app.route("/health")
def health():
    return "ok", 200


@app.route("/", methods=["GET"])
def index():
    # Renders the new Tailwind form
    return render_template("index.html", title="A-Eye Skin")


@app.route("/analyze", methods=["POST"])
def analyze():
    # Validate form
    if "photo" not in request.files:
        abort(400, "No file part")
    file = request.files["photo"]
    if not file or file.filename == "":
        abort(400, "No selected file")
    if not allowed_file(file.filename):
        abort(400, "Unsupported file type (use JPG/PNG)")

    age_raw = request.form.get("age", "").strip()
    skin_type = request.form.get("skin_type", "").strip() or "Normal"
    try:
        age = int(age_raw)
    except Exception:
        age = 25

    # Save original
    fname_in = save_upload(file)
    path_in = UPLOAD_DIR / fname_in

    # Load image (RGB) and predict
    img_pil = pil_open_rgb(path_in)
    probs = predict_probs(img_pil)

    # Draw regions & save analyzed image
    fname_out = f"analyzed_{fname_in}"
    path_out = UPLOAD_DIR / fname_out
    draw_regions(path_in, path_out, probs)

    # Build cards for UI
    recos = build_recommendations(probs, age, skin_type)

    # Render results page
    return render_template(
        "result.html",
        title="A-Eye Skin",
        analyzed_filename=fname_out,
        prob_acne=probs.get("acne"),
        prob_dark_circles=probs.get("dark_circles"),
        recommendations=recos,
    )


# --------------- Run ---------------
if __name__ == "__main__":
    # For local runs without FLASK_APP; keeps Metal visible in logs
    app.run(host="127.0.0.1", port=5000, debug=False)
