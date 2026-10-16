import numpy as np

from face_emotion.labels import EMOTICON_DIR, EMOTIONS, PRETRAINED_WEIGHTS, YUNET_MODEL
from face_emotion.model import build_training_model, load_model


def test_bundled_assets_exist():
    assert PRETRAINED_WEIGHTS.is_file()
    assert YUNET_MODEL.is_file()
    for emotion in EMOTIONS:
        assert (EMOTICON_DIR / f"{emotion.lower()}.png").is_file()


def test_pretrained_model_outputs_probabilities():
    model = load_model()
    out = np.asarray(model(np.random.default_rng(0).random((2, 48, 48, 1), dtype=np.float32), training=False))
    assert out.shape == (2, len(EMOTIONS))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-4)


def test_training_model_shape():
    model = build_training_model()
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, len(EMOTIONS))


def test_load_saved_keras_model(tmp_path):
    path = tmp_path / "m.keras"
    build_training_model().save(path)
    assert load_model(path).output_shape == (None, len(EMOTIONS))
