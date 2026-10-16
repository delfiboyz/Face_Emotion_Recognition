import cv2
import numpy as np
import pytest

from face_emotion.detector import Detection, FaceDetector, _clip_box, _iou, preprocess_faces
from face_emotion.labels import EMOTIONS


@pytest.mark.parametrize(
    "backend",
    [
        "yunet",
        pytest.param(
            "haar",
            marks=pytest.mark.skipif(not hasattr(cv2, "CascadeClassifier"), reason="no CascadeClassifier in OpenCV 5"),
        ),
    ],
)
def test_no_faces_on_blank_frame(backend, blank_frame):
    assert FaceDetector(backend).detect(blank_frame) == []


def test_unknown_backend():
    with pytest.raises(ValueError):
        FaceDetector("nope")


def test_clip_box():
    assert _clip_box((-10, -10, 50, 50), 100, 100) == (0, 0, 40, 40)
    assert _clip_box((90, 90, 50, 50), 100, 100) == (90, 90, 10, 10)
    assert _clip_box((98, 0, 50, 50), 100, 100) is None


def test_iou():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_preprocess_faces(blank_frame):
    batch = preprocess_faces(blank_frame, [(0, 0, 100, 80), (50, 50, 60, 60)])
    assert batch.shape == (2, 48, 48, 1)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() <= batch.max() <= 1.0


def test_recognizer_blank_frame(recognizer, blank_frame):
    assert recognizer(blank_frame) == []
    probs = recognizer.predict_faces(blank_frame, [(10, 10, 100, 100)])
    assert probs.shape == (1, len(EMOTIONS))


def test_detection_properties():
    probs = np.zeros(len(EMOTIONS), dtype=np.float32)
    probs[EMOTIONS.index("Happy")] = 0.9
    det = Detection((1, 2, 3, 4), probs)
    assert det.emotion == "Happy"
    assert det.as_dict()["box"] == [1, 2, 3, 4]
    assert det.as_dict()["confidence"] == pytest.approx(0.9)


def test_smoothing_blends_overlapping_faces(recognizer):
    recognizer.smoothing = 0.5
    try:
        a = np.eye(len(EMOTIONS), dtype=np.float32)[0]
        b = np.eye(len(EMOTIONS), dtype=np.float32)[1]
        recognizer._previous = [Detection((0, 0, 50, 50), a)]
        smoothed = recognizer._smooth(Detection((2, 2, 50, 50), b))
        np.testing.assert_allclose(smoothed.probabilities[:2], [0.5, 0.5])
        far = recognizer._smooth(Detection((200, 200, 50, 50), b))
        np.testing.assert_allclose(far.probabilities, b)
    finally:
        recognizer.smoothing = 0.0
        recognizer._previous = []
