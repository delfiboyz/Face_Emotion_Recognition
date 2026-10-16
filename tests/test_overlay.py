import numpy as np

from face_emotion.detector import Detection
from face_emotion.labels import EMOTIONS
from face_emotion.overlay import annotate, load_emoticon, overlay_rgba


def test_overlay_rgba_alpha_and_clipping():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    icon = np.zeros((10, 10, 4), dtype=np.uint8)
    icon[..., :3] = 200
    icon[:, :5, 3] = 255  # left half opaque, right half transparent
    overlay_rgba(frame, icon, -3, 15)  # partially outside the frame
    assert frame[15:, :2].min() == 200
    assert frame[15:, 2:].max() == 0
    overlay_rgba(frame, icon, 100, 100)  # fully outside: no-op, no error


def test_emoticons_load_with_alpha():
    for emotion in EMOTIONS:
        img = load_emoticon(emotion)
        assert img is not None and img.shape[2] == 4


def test_annotate_draws(blank_frame):
    probs = np.full(len(EMOTIONS), 1 / len(EMOTIONS), dtype=np.float32)
    before = blank_frame.copy()
    out = annotate(blank_frame, [Detection((150, 60, 80, 80), probs)])
    assert out is blank_frame
    assert not np.array_equal(before, out)
