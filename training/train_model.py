#!/usr/bin/env python3
"""
Training script for the simple skin issue classifier.

This script builds a convolutional neural network using transfer learning to
classify facial images into one of two categories: ``acne`` or
``dark_circles``. It expects a directory structure such as:

.. code-block:: text

    training/data/train/
        acne/
            img001.jpg
            img002.jpg
            ...
        dark_circles/
            img001.jpg
            img002.jpg
            ...
    training/data/val/
        acne/
            img101.jpg
            ...
        dark_circles/
            ...

At a minimum you should collect roughly 20–50 representative images per
category for your class demonstration. Although the general rule of thumb for
training deep neural networks is to use on the order of one thousand images per
class【51776491345958†L29-L48】, transfer learning can dramatically reduce this
requirement. For example, the PyTorch transfer learning tutorial shows that
reasonable performance can be achieved on a dataset with only about 120
training images per class【942962495547244†L220-L226】. A small dataset can still
perform well when you leverage a pretrained network and apply data
augmentation【51776491345958†L45-L79】.

Running this script will create a model file (``skin_classifier.h5``) in
``skin_analysis_app/model``. You can adjust the number of epochs or batch
size via command‑line flags.
"""

import argparse
import os
from pathlib import Path

import tensorflow as tf


def build_model(num_classes: int) -> tf.keras.Model:
    """Builds a transfer learning model based on MobileNetV2.

    Args:
        num_classes: The number of output classes.

    Returns:
        A compiled Keras model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze the convolutional base

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train skin issue classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data"),
        help="Root directory containing train/ and val/ subdirectories.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "model" / "skin_classifier.h5"),
        help="Path to save the trained model.",
    )
    args = parser.parse_args()

    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Could not find train or val directories under {args.data_dir}. "
            "Please create directories as described in the docstring."
        )

    # Determine class names from subdirectory names in train
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError(
            "At least two classes are required. Ensure train/ contains 'acne' and 'dark_circles' subdirectories."
        )

    # Create image data generators with augmentation for training and just rescaling for validation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="sparse",
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="sparse",
    )

    model = build_model(num_classes)
    print(model.summary())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save the model
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()