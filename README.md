# Face Emotion Recognition

Deteksi emosi wajah secara real-time dari webcam, gambar, atau video — dibangun
dengan **OpenCV YuNet** (deteksi wajah berbasis DNN) dan **Keras 3**.

Model mengklasifikasikan 7 emosi (dataset FER-2013):
`Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.

## Fitur

- Deteksi wajah modern dengan **YuNet** (lebih akurat & cepat daripada Haar cascade; Haar tetap tersedia via `--detector haar`, di OpenCV 5 butuh `opencv-contrib-python`)
- Model pretrained FER-2013 sudah dibundel — tidak perlu download apa pun saat runtime (tanpa DeepFace)
- Prediksi batch untuk semua wajah dalam satu frame, plus *temporal smoothing* agar label tidak berkedip
- Overlay emotikon dengan alpha-blending, label berwarna per emosi, grafik probabilitas, dan FPS
- CLI tunggal: `webcam`, `image` (output JSON), `video` (simpan hasil `.mp4`), `train`
- Pipeline training lengkap: augmentasi, BatchNorm, AdamW + cosine decay, class weights,
  early stopping, checkpoint terbaik, laporan klasifikasi & confusion matrix
- Tooling modern: `pyproject.toml`, [uv](https://docs.astral.sh/uv/), Ruff, pytest, pre-commit, GitHub Actions

## Instalasi

Membutuhkan Python 3.10+. Disarankan memakai [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # membuat .venv dan menginstal dependensi
uv run face-emotion --help
```

Atau dengan pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"
```

## Penggunaan

```bash
# Webcam real-time (q/Esc = keluar, s = simpan snapshot)
face-emotion webcam
face-emotion webcam --camera 1 --detector haar --smoothing 0.8

# Gambar: cetak hasil JSON dan simpan gambar beranotasi
face-emotion image foto.jpg -o hasil/
face-emotion image foto1.jpg foto2.jpg --show

# Video: simpan hasil beranotasi
face-emotion video input.mp4 -o output.mp4

# Gunakan model hasil training sendiri
face-emotion webcam --model models/emotion_model.keras
```

Contoh output JSON:

```json
{
  "foto.jpg": [
    {"box": [212, 98, 151, 151], "emotion": "Happy", "confidence": 0.9312,
     "scores": {"Angry": 0.004, "Disgust": 0.0, "Fear": 0.011, "Happy": 0.9312, "Sad": 0.02, "Surprise": 0.008, "Neutral": 0.0258}}
  ]
}
```

### Dari Python

```python
import cv2
from face_emotion.detector import EmotionRecognizer
from face_emotion.overlay import annotate

recognizer = EmotionRecognizer()  # atau EmotionRecognizer("models/emotion_model.keras")
frame = cv2.imread("foto.jpg")
detections = recognizer(frame)
for d in detections:
    print(d.box, d.emotion, d.confidence)
cv2.imwrite("hasil.jpg", annotate(frame, detections))
```

## Training model sendiri

Unduh FER-2013 dalam format folder (misalnya dataset Kaggle `msambare/fer2013`):

```
data/fer2013/
  train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
  test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
```

Lalu jalankan:

```bash
uv run --extra train face-emotion train data/fer2013 --epochs 60 -o models/emotion_model.keras
```

Hasil training:

- `models/emotion_model.keras` — checkpoint terbaik (berdasarkan `val_accuracy`)
- `models/emotion_model.history.csv` — riwayat loss/accuracy per epoch
- `models/emotion_model.metrics.json` — akurasi test, classification report, confusion matrix

Training otomatis memakai GPU bila TensorFlow mendeteksinya.

## Struktur proyek

```
src/face_emotion/
  cli.py          # perintah face-emotion
  detector.py     # FaceDetector (YuNet/Haar) + EmotionRecognizer
  model.py        # arsitektur pretrained & model training Keras 3
  overlay.py      # gambar kotak, label, emotikon, grafik probabilitas
  train.py        # pipeline training
  labels.py       # label emosi, warna, lokasi aset
  assets/
    emoticons/    # emotikon PNG transparan
    models/       # bobot FER-2013 + model YuNet ONNX
tests/            # pytest
```

## Pengembangan

```bash
uv sync --group dev
uv run pytest              # jalankan test
uv run ruff check .        # lint
uv run ruff format .       # format kode
uv run pre-commit install  # pasang git hooks
```

## Kredit

- Bobot model emosi FER-2013: `facial_expression_model_weights.h5` dari proyek
  [serengil/deepface](https://github.com/serengil/deepface) (MIT).
- Detektor wajah YuNet: [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT).
