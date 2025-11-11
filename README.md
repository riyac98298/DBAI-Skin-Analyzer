# AI Skin Analyzer

This project is a complete prototype for a skin analysis application built for
a **Doing Business with AI** class project. It combines a simple
convolutional neural network to recognize two common skin concerns (acne and
dark circles) with a modern single‑page web interface and a lightweight
Python/Flask backend. When a user uploads a selfie and enters a few details
about themselves, the system highlights areas of concern, reports confidence
scores and recommends generic skincare products.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Collecting your dataset](#collecting-your-dataset)
4. [Training the model](#training-the-model)
5. [Running the application locally](#running-the-application-locally)
6. [Project structure](#project-structure)
7. [Limitations & future work](#limitations--future-work)

## Features

- **Simple transfer‑learning model** – uses MobileNetV2 with a custom head and
  data augmentation. Only two classes (`acne` and `dark_circles`) are
  supported.
- **Minimal dataset requirements** – transfer learning enables reasonable
  performance with as few as ~20–50 images per class for demonstration
  purposes. Although the classic rule of thumb suggests around 1 000 images per
  class【51776491345958†L29-L48】, transfer learning can generalise on much
  smaller datasets【942962495547244†L220-L226】.
- **Modern web UI** – built with TailwindCSS for a clean, professional look.
  The page dynamically updates to show analysis results and product
  recommendations without a page refresh.
- **Automatic highlighting** – faces and eye regions are detected using OpenCV.
  When acne is detected, the face is outlined in red; dark circles are
  highlighted under the eyes in blue. Confidence scores are shown next to
  each issue.
- **Product recommendations** – generic suggestions are provided based on
  recognised issues:
  - **Acne**: A 2 % salicylic acid serum. Salicylic acid, a beta‑hydroxy acid,
    exfoliates the skin and helps keep pores clear【324999427622202†L186-L206】;
    over‑the‑counter gels and lotions typically contain 2 % to 7 % salicylic
    acid【324999427622202†L221-L227】.
  - **Dark circles**: A 10 % vitamin C eye serum. A small clinical trial
    reported that a 10 % vitamin C product applied for six months thickened
    the skin under the eyes and reduced the visibility of dark circles【474670558632404†L251-L259】. Topical
    creams containing vitamin C are often used to lighten the appearance of dark
    circles【439155641374241†L170-L176】.

## Prerequisites

Ensure you have Python 3.9 or later installed. On macOS, you can install
Python via [Homebrew](https://brew.sh/) (``brew install python``) or from
[python.org](https://www.python.org/downloads/). Install the Python dependencies
inside a virtual environment using:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The key dependencies are:

| Package                     | Purpose                             |
|----------------------------|-------------------------------------|
| Flask                      | Lightweight web framework           |
| TensorFlow/Keras           | Model definition and training        |
| opencv‑python‑headless     | Face and eye detection, image I/O    |
| Pillow (PIL)               | Image loading and saving             |
| numpy                      | Array manipulation                   |

## Collecting your dataset

For a classroom demonstration you don’t need thousands of images. Transfer
learning allows the model to be trained on a relatively small collection of
well‑representative samples. Aim for **20–50 images per class** as a starting
point. The images should be:

- **Representative** – they should resemble the selfies users will upload. Use
  natural lighting, clear focus and minimal background distractions. Quality
  matters more than quantity【51776491345958†L56-L67】.
- **Balanced** – collect roughly the same number of acne and dark circle
  examples. For acne, look for faces with visible pimples or blemishes; for
  dark circles, choose images showing pronounced under‑eye shadows.
- **Labelled** – organise your images into two folders named `acne` and
  `dark_circles` within `training/data/train/` and `training/data/val/`. A
  simple way is to manually split 20 % of your images into the `val` set.

> **Tip**: To increase diversity without collecting more images, you can use
> the built‑in augmentation in the training script. Random flips, rotations
> and zooms help the model generalise better【51776491345958†L69-L79】.

## Training the model

After populating the `training/data/` directories with your images, run the
training script from within the project root:

```bash
cd skin_analysis_app
python training/train_model.py --epochs 10 --batch_size 16
```

Adjust the number of epochs or batch size as needed. The script will save
`skin_classifier.h5` inside the `model/` folder. This file is loaded by the
Flask app when it starts. If the model isn’t found, the API will return an
error prompting you to train first.

## Running the application locally

1. **Install dependencies** as described in [Prerequisites](#prerequisites).
2. **Train the model** or copy an existing `skin_classifier.h5` into
   `skin_analysis_app/model`.
3. **Start the Flask server**:

   ```bash
   cd skin_analysis_app
   export FLASK_APP=app.py
   flask run --port 5000
   ```

   The server will be accessible at `http://localhost:5000`. When you open
   this URL you’ll see the web interface. Enter your age, select your skin
   type and upload a selfie. After a short delay you’ll receive an analysis.

If you modify the front‑end or Python code, restart the Flask server to pick
up the changes. During development you may run with ``flask run --reload``
for automatic reloading.

## Project structure

The repository is organised as follows:

```
skin_analysis_app/
├── app.py               # Flask application
├── model/
│   └── skin_classifier.h5   # Saved Keras model (created after training)
├── requirements.txt     # Python dependencies
├── static/
│   ├── js/script.js     # Front‑end logic
│   └── css/             # (Unused) place for custom styles if needed
├── templates/
│   └── index.html       # Single‑page user interface
└── training/
    ├── data/            # Place your training/validation images here
    ├── train_model.py   # Script to train the classifier
    └── predict.py       # Stand‑alone script for testing prediction and
                         # generating annotated images
```

## Limitations & future work

This prototype is intended for educational purposes and has several
limitations:

* **Small dataset** – With very limited training data, the classifier will not
  generalise well to all skin tones, lighting conditions and camera qualities.
  It’s adequate for a classroom demonstration but not for production use.
* **Simplistic detection** – Highlighting uses traditional Haar cascades and
  simple heuristics rather than true object detection. Acne may be drawn over
  the entire face and dark circles are highlighted under detected eyes. More
  precise localisation would require annotated bounding boxes and training
  an object detection model.
* **Fixed product list** – The recommendations are static and not
  personalised beyond the detected issue. A commercial system would include
  a broader catalogue and factors like ingredient sensitivities, pricing or
  brand preferences.
* **Two classes only** – Only acne and dark circles are recognised. Extending
  to additional concerns (e.g. hyperpigmentation, fine lines) would require
  more data and possibly multi‑label classification or segmentation.

Despite these limitations, the project demonstrates how to combine modern
machine learning techniques with a friendly user interface and domain
knowledge to solve a simple problem from end to end.# DBAI-Skin-Analyzer
