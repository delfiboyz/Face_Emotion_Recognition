"""Train an emotion classifier on a FER-2013 style image folder.

Expected layout (e.g. the Kaggle "msambare/fer2013" dataset)::

    DATA_DIR/
      train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
      test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_emotion.labels import EMOTIONS, INPUT_SIZE

CLASS_DIRS = [e.lower() for e in EMOTIONS]  # fixes the label order to match EMOTIONS


@dataclass
class TrainConfig:
    data_dir: Path
    output: Path = Path("models/emotion_model.keras")
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    patience: int = 10
    seed: int = 42


def _load_split(directory: Path, cfg: TrainConfig, subset: str | None = None):
    import keras

    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_DIRS,
        color_mode="grayscale",
        image_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=cfg.batch_size,
        shuffle=subset != "test",
        seed=cfg.seed,
        validation_split=cfg.validation_split if subset in ("training", "validation") else None,
        subset=subset if subset in ("training", "validation") else None,
    )


def _class_weights(dataset) -> dict[int, float]:
    labels = np.concatenate([y.numpy() for _, y in dataset])
    counts = np.bincount(labels, minlength=len(EMOTIONS)).astype(np.float64)
    weights = counts.sum() / (len(EMOTIONS) * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def train(cfg: TrainConfig) -> dict:
    import keras
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    from face_emotion.model import build_training_model

    keras.utils.set_random_seed(cfg.seed)
    train_dir = cfg.data_dir / "train"
    test_dir = cfg.data_dir / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"{train_dir} not found; see `face-emotion train --help` for the layout")

    def scale(x, y):
        return x / 255.0, y

    train_ds = _load_split(train_dir, cfg, "training")
    val_ds = _load_split(train_dir, cfg, "validation")
    class_weight = _class_weights(train_ds)
    train_ds = train_ds.map(scale, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(scale, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

    model = build_training_model()
    steps = max(1, int(train_ds.cardinality())) * cfg.epochs
    schedule = keras.optimizers.schedules.CosineDecay(cfg.learning_rate, decay_steps=steps)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(cfg.output, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=cfg.patience, restore_best_weights=True),
        keras.callbacks.CSVLogger(cfg.output.with_suffix(".history.csv")),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, class_weight=class_weight, callbacks=callbacks)
    model = keras.models.load_model(cfg.output)  # best checkpoint

    metrics: dict = {"model": str(cfg.output)}
    if test_dir.is_dir():
        test_ds = _load_split(test_dir, cfg, "test").map(scale)
        y_true = np.concatenate([y.numpy() for _, y in test_ds])
        y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
        report = classification_report(
            y_true, y_pred, labels=range(len(EMOTIONS)), target_names=EMOTIONS, zero_division=0
        )
        print(report)
        metrics["accuracy"] = float((y_true == y_pred).mean())
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=range(len(EMOTIONS))).tolist()
        metrics["report"] = classification_report(
            y_true, y_pred, labels=range(len(EMOTIONS)), target_names=EMOTIONS, zero_division=0, output_dict=True
        )
        cfg.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
