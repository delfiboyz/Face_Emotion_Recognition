"""Keras 3 model definitions."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
from keras import layers  # noqa: E402

from face_emotion.labels import EMOTIONS, INPUT_SIZE, PRETRAINED_WEIGHTS  # noqa: E402


def build_pretrained_architecture(num_classes: int = len(EMOTIONS)) -> keras.Model:
    """CNN matching the bundled FER-2013 weights (``facial_expression_model_weights.h5``)."""
    return keras.Sequential(
        [
            keras.Input((INPUT_SIZE, INPUT_SIZE, 1)),
            layers.Conv2D(64, 5, activation="relu"),
            layers.MaxPooling2D(5, strides=2),
            layers.Conv2D(64, 3, activation="relu"),
            layers.Conv2D(64, 3, activation="relu"),
            layers.AveragePooling2D(3, strides=2),
            layers.Conv2D(128, 3, activation="relu"),
            layers.Conv2D(128, 3, activation="relu"),
            layers.AveragePooling2D(3, strides=2),
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="fer2013_pretrained",
    )


def _conv_block(x, filters: int, dropout: float):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout)(x)


def build_training_model(num_classes: int = len(EMOTIONS)) -> keras.Model:
    """Modern CNN for training from scratch: augmentation, BatchNorm, GAP head.

    Expects grayscale 48x48 input scaled to [0, 1]; augmentation layers are
    only active during training, so the saved model is ready for inference.
    """
    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomContrast(0.2)(x)
    for filters, dropout in ((64, 0.2), (128, 0.25), (256, 0.3), (512, 0.35)):
        x = _conv_block(x, filters, dropout)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="fer_cnn")


def load_model(path: str | Path | None = None) -> keras.Model:
    """Load a trained ``.keras`` model, or the bundled pretrained weights when ``path`` is None."""
    if path is None:
        model = build_pretrained_architecture()
        model.load_weights(PRETRAINED_WEIGHTS)
        return model
    path = Path(path)
    if path.suffix == ".keras":
        return keras.models.load_model(path)
    # Legacy weight-only .h5 file for the pretrained architecture.
    model = build_pretrained_architecture()
    model.load_weights(path)
    return model
