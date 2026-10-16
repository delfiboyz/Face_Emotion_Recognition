"""Face detection + emotion classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from face_emotion.labels import EMOTIONS, HAAR_CASCADE, INPUT_SIZE, YUNET_MODEL

Box = tuple[int, int, int, int]  # x, y, w, h
DetectorName = Literal["yunet", "haar"]


@dataclass(frozen=True)
class Detection:
    box: Box
    probabilities: np.ndarray

    @property
    def emotion(self) -> str:
        return EMOTIONS[int(self.probabilities.argmax())]

    @property
    def confidence(self) -> float:
        return float(self.probabilities.max())

    def as_dict(self) -> dict:
        return {
            "box": list(self.box),
            "emotion": self.emotion,
            "confidence": round(self.confidence, 4),
            "scores": {e: round(float(p), 4) for e, p in zip(EMOTIONS, self.probabilities, strict=True)},
        }


class FaceDetector:
    """Finds faces with OpenCV's YuNet DNN (default) or a Haar cascade."""

    def __init__(self, backend: DetectorName = "yunet", score_threshold: float = 0.8):
        self.backend = backend
        if backend == "yunet":
            self._yunet = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), score_threshold, 0.3, 5000)
        elif backend == "haar":
            if not hasattr(cv2, "CascadeClassifier"):
                raise RuntimeError(
                    "The Haar detector needs CascadeClassifier, which OpenCV 5 moved out of the main package. "
                    "Use the default YuNet detector or install opencv-contrib-python."
                )
            self._haar = cv2.CascadeClassifier(str(HAAR_CASCADE))
        else:
            raise ValueError(f"Unknown detector backend: {backend!r}")

    def detect(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        if self.backend == "yunet":
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(frame)
            raw = [] if faces is None else [tuple(int(v) for v in f[:4]) for f in faces]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [b for b in (_clip_box(box, w, h) for box in raw) if b is not None]


def _clip_box(box, frame_w: int, frame_h: int) -> Box | None:
    x, y, w, h = (int(v) for v in box)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame_w), min(y + h, frame_h)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return x0, y0, x1 - x0, y1 - y0


def preprocess_faces(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Crop, grayscale, resize to 48x48 and scale to [0, 1] -> (N, 48, 48, 1)."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crops = [
        cv2.resize(gray[y : y + h, x : x + w], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        for x, y, w, h in boxes
    ]
    batch = np.asarray(crops, dtype=np.float32).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
    return batch / 255.0


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


class EmotionRecognizer:
    """Detects faces in BGR frames and classifies their emotion.

    With ``smoothing > 0`` the probabilities of a face are blended with those of
    the overlapping face in the previous frame, which removes flicker on video.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        detector: DetectorName = "yunet",
        smoothing: float = 0.0,
    ):
        from face_emotion.model import load_model  # deferred: importing Keras is slow

        self.model = load_model(model_path)
        self.face_detector = FaceDetector(detector)
        self.smoothing = smoothing
        self._previous: list[Detection] = []

    def predict_faces(self, frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
        if not boxes:
            return np.empty((0, len(EMOTIONS)), dtype=np.float32)
        batch = preprocess_faces(frame, boxes)
        return np.asarray(self.model(batch, training=False))

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        boxes = self.face_detector.detect(frame)
        probs = self.predict_faces(frame, boxes)
        detections = [Detection(box, p) for box, p in zip(boxes, probs, strict=True)]
        if self.smoothing > 0:
            detections = [self._smooth(d) for d in detections]
            self._previous = detections
        return detections

    def _smooth(self, det: Detection) -> Detection:
        match = max(self._previous, key=lambda p: _iou(p.box, det.box), default=None)
        if match is None or _iou(match.box, det.box) < 0.3:
            return det
        a = self.smoothing
        return Detection(det.box, a * match.probabilities + (1 - a) * det.probabilities)
