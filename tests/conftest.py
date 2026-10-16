import numpy as np
import pytest


@pytest.fixture(scope="session")
def recognizer():
    from face_emotion.detector import EmotionRecognizer

    return EmotionRecognizer()


@pytest.fixture
def blank_frame():
    return np.full((240, 320, 3), 127, dtype=np.uint8)
