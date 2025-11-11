#!/usr/bin/env python3
"""
Prediction utility for the skin issue classifier.

This script demonstrates how to load a trained Keras model and apply it to
an input image. It outputs the predicted probabilities for each class and
saves a highlighted version of the image showing the detected issues.

Example usage::

    python predict.py --model ../model/skin_classifier.h5 --image example.jpg --out highlighted.jpg

"""
import argparse
import base64
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
import cv2


def load_image(image_path: str) -> np.ndarray:
    """Loads an image from disk and converts it into a format suitable for prediction.

    Args:
        image_path: Path to the image file.

    Returns:
        A numpy array of shape (224, 224, 3) with pixel values in [0, 255].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} could not be read.")
    img_resized = cv2.resize(img, (224, 224))
    return img_resized


def predict(model: tf.keras.Model, image: np.ndarray, class_names: Dict[int, str]):
    """Predicts the class probabilities for the given image.

    Args:
        model: The loaded Keras model.
        image: A numpy array representing the image (224 x 224 x 3).
        class_names: Mapping from integer class indices to their string names.

    Returns:
        Dictionary mapping class name to probability.
    """
    # Preprocess using the same preprocessing as during training
    x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x.astype(np.float32))
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    return {class_names[i]: float(preds[i]) for i in range(len(preds))}


def highlight_issues(original_img: np.ndarray, probs: Dict[str, float], threshold: float = 0.5) -> np.ndarray:
    """Creates an annotated version of the original image highlighting detected issues.

    The function uses Haar cascades bundled with OpenCV to detect faces and eyes.
    If the probability for ``acne`` or ``dark_circles`` exceeds ``threshold``,
    the corresponding regions of the face are outlined.

    Args:
        original_img: The input image array (BGR).
        probs: Mapping of class name to probability.
        threshold: Probability threshold to trigger highlighting.

    Returns:
        Annotated image array.
    """
    annotated = original_img.copy()
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y : y + h, x : x + w]
        face_roi_color = annotated[y : y + h, x : x + w]

        # Highlight acne by drawing a red rectangle around the face
        if probs.get('acne', 0) >= threshold:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"Acne: {probs['acne']:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Highlight dark circles below the eyes
        if probs.get('dark_circles', 0) >= threshold:
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                # Define a rectangle under the eye
                y_start = int(ey + 0.6 * eh)
                y_end = int(ey + 1.2 * eh)
                # Ensure coordinates stay within the face ROI
                y_start = max(0, y_start)
                y_end = min(h, y_end)
                cv2.rectangle(
                    annotated,
                    (x + ex, y + y_start),
                    (x + ex + ew, y + y_end),
                    (255, 0, 0),
                    2,
                )
            cv2.putText(
                annotated,
                f"Dark circles: {probs['dark_circles']:.2f}",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Predict skin issues from an image")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.h5)")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--out",
        type=str,
        default="annotated.jpg",
        help="Path to save the annotated output image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for highlighting issues",
    )
    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model)

    # Infer class names from the model's output layer if available
    class_indices = {0: 'acne', 1: 'dark_circles'}
    # Load and preprocess the image
    original = cv2.imread(args.image)
    if original is None:
        raise FileNotFoundError(f"Could not read image {args.image}")
    resized = cv2.resize(original, (224, 224))

    # Predict
    probs = predict(model, resized, class_indices)
    print("Predicted probabilities:")
    for cls, prob in probs.items():
        print(f"  {cls}: {prob:.2f}")

    # Generate and save annotated image
    annotated_img = highlight_issues(original, probs, threshold=args.threshold)
    cv2.imwrite(args.out, annotated_img)
    print(f"Annotated image saved to {args.out}")


if __name__ == "__main__":
    main()