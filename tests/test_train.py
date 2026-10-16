import json

import cv2
import numpy as np
import pytest

from face_emotion.labels import EMOTIONS

pytest.importorskip("sklearn")


def _make_dataset(root, per_class=4):
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        for emotion in EMOTIONS:
            d = root / split / emotion.lower()
            d.mkdir(parents=True)
            for i in range(per_class):
                cv2.imwrite(str(d / f"{i}.png"), rng.integers(0, 255, (48, 48), dtype=np.uint8))


def test_train_end_to_end(tmp_path):
    from face_emotion.train import TrainConfig, train

    _make_dataset(tmp_path / "data")
    out = tmp_path / "model.keras"
    metrics = train(TrainConfig(data_dir=tmp_path / "data", output=out, epochs=1, batch_size=8, validation_split=0.25))
    assert out.is_file()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    saved = json.loads(out.with_suffix(".metrics.json").read_text())
    assert len(saved["confusion_matrix"]) == len(EMOTIONS)
