"""Emotion labels and bundled asset locations."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

# Output order of the pretrained FER-2013 model. Custom models trained with
# `face-emotion train` use the same order so they are drop-in replacements.
EMOTIONS: tuple[str, ...] = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")

# BGR colour per emotion, used for boxes and labels.
EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "Angry": (40, 40, 220),
    "Disgust": (40, 140, 40),
    "Fear": (160, 60, 160),
    "Happy": (0, 200, 255),
    "Sad": (200, 120, 40),
    "Surprise": (0, 140, 255),
    "Neutral": (190, 190, 190),
}

INPUT_SIZE = 48

_ASSETS = files("face_emotion") / "assets"


def asset_path(*parts: str) -> Path:
    """Return the filesystem path of a bundled asset."""
    return Path(str(_ASSETS.joinpath(*parts)))


PRETRAINED_WEIGHTS = asset_path("models", "facial_expression_model_weights.h5")
YUNET_MODEL = asset_path("models", "face_detection_yunet_2023mar.onnx")
HAAR_CASCADE = asset_path("models", "haarcascade_frontalface_default.xml")
EMOTICON_DIR = asset_path("emoticons")
