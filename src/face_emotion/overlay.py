"""Drawing helpers: boxes, labels, emoticons and probability bars."""

from __future__ import annotations

from functools import cache

import cv2
import numpy as np

from face_emotion.detector import Detection
from face_emotion.labels import EMOTICON_DIR, EMOTION_COLORS, EMOTIONS

FONT = cv2.FONT_HERSHEY_SIMPLEX


@cache
def load_emoticon(emotion: str) -> np.ndarray | None:
    """Load the BGRA emoticon for an emotion (cached), or None if missing."""
    img = cv2.imread(str(EMOTICON_DIR / f"{emotion.lower()}.png"), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def overlay_rgba(frame: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a BGRA ``image`` onto a BGR ``frame`` in place, clipped to the frame."""
    fh, fw = frame.shape[:2]
    ih, iw = image.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + iw, fw), min(y + ih, fh)
    if x0 >= x1 or y0 >= y1:
        return
    src = image[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1]
    roi[:] = (src[:, :, :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    scale, thickness = 0.6, 2
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(y, th + baseline + 4)
    cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 6, y), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, y - baseline - 2), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_probability_bars(frame: np.ndarray, probabilities: np.ndarray, margin: int = 10) -> None:
    """Draw a small horizontal bar chart of all emotion scores in the top-right corner."""
    bar_w, bar_h, gap = 140, 14, 4
    panel_w, panel_h = bar_w + 80, len(EMOTIONS) * (bar_h + gap) + gap
    x, y = max(frame.shape[1] - panel_w - margin, 0), margin
    panel = frame[y : y + panel_h, x : x + panel_w]
    panel[:] = (panel * 0.4).astype(np.uint8)
    for i, (emotion, p) in enumerate(zip(EMOTIONS, probabilities, strict=True)):
        top = y + gap + i * (bar_h + gap)
        cv2.putText(frame, emotion, (x + 4, top + bar_h - 3), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        left = x + 70
        cv2.rectangle(frame, (left, top), (left + int(bar_w * float(p)), top + bar_h), EMOTION_COLORS[emotion], -1)
        cv2.putText(frame, f"{p:.0%}", (left + bar_w - 34, top + bar_h - 3), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    show_emoticons: bool = True,
    show_bars: bool = True,
) -> np.ndarray:
    """Draw detections onto ``frame`` in place and return it."""
    if show_bars and detections:
        largest = max(detections, key=lambda d: d.box[2] * d.box[3])
        draw_probability_bars(frame, largest.probabilities)
    for det in detections:
        x, y, w, h = det.box
        color = EMOTION_COLORS[det.emotion]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        draw_label(frame, f"{det.emotion} {det.confidence:.0%}", x, y, color)
        emoticon = load_emoticon(det.emotion) if show_emoticons else None
        if emoticon is not None:
            size = max(24, w // 3)
            icon = cv2.resize(emoticon, (size, size), interpolation=cv2.INTER_AREA)
            overlay_rgba(frame, icon, x + w + 6, y)
    return frame
