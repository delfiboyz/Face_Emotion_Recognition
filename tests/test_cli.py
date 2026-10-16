import json

import cv2
import numpy as np

from face_emotion.cli import main


def test_image_command(tmp_path, capsys):
    img = tmp_path / "blank.png"
    cv2.imwrite(str(img), np.full((120, 160, 3), 90, dtype=np.uint8))
    out = tmp_path / "out" / "annotated.png"
    assert main(["image", str(img), "-o", str(out)]) == 0
    assert out.is_file()
    assert json.loads(capsys.readouterr().out) == {str(img): []}


def test_image_command_missing_file(tmp_path):
    assert main(["image", str(tmp_path / "missing.png")]) == 1
