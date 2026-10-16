"""Command line interface: ``face-emotion {webcam,image,video,train}``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WINDOW = "Face Emotion Recognition (q: quit, s: snapshot)"


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", type=Path, help="trained .keras model (default: bundled FER-2013 weights)")
    p.add_argument("--detector", choices=["yunet", "haar"], default="yunet", help="face detector (default: yunet)")
    p.add_argument("--no-emoticons", action="store_true", help="do not draw emoticons")
    p.add_argument("--no-bars", action="store_true", help="do not draw the probability bars")


def _recognizer(args, smoothing: float = 0.0):
    from face_emotion.detector import EmotionRecognizer

    return EmotionRecognizer(args.model, detector=args.detector, smoothing=smoothing)


def _annotate(frame, detections, args):
    from face_emotion.overlay import annotate

    return annotate(frame, detections, show_emoticons=not args.no_emoticons, show_bars=not args.no_bars)


def cmd_image(args) -> int:
    import cv2

    recognizer = _recognizer(args)
    results = {}
    for path in args.images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"error: cannot read image {path}", file=sys.stderr)
            return 1
        detections = recognizer(frame)
        results[str(path)] = [d.as_dict() for d in detections]
        if args.output:
            out = args.output / path.name if args.output.is_dir() or len(args.images) > 1 else args.output
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), _annotate(frame, detections, args))
        if args.show:
            cv2.imshow(str(path), _annotate(frame, detections, args))
            cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()
    print(json.dumps(results, indent=2))
    return 0


def _run_stream(capture, args, writer=None, show: bool = True) -> int:
    import cv2

    recognizer = _recognizer(args, smoothing=args.smoothing)
    fps, last = 0.0, time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detections = recognizer(frame)
        _annotate(frame, detections, args)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        h = frame.shape[0]
        cv2.putText(frame, f"{fps:5.1f} FPS", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(name, frame)
                print(f"saved {name}")
    capture.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    return 0


def cmd_webcam(args) -> int:
    import cv2

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"error: cannot open camera {args.camera}", file=sys.stderr)
        return 1
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return _run_stream(capture, args)


def cmd_video(args) -> int:
    import cv2

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"error: cannot open video {args.video}", file=sys.stderr)
        return 1
    writer = None
    if args.output:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return _run_stream(capture, args, writer=writer, show=args.show)


def cmd_train(args) -> int:
    from face_emotion.train import TrainConfig, train

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
    metrics = train(cfg)
    if "accuracy" in metrics:
        print(f"test accuracy: {metrics['accuracy']:.4f}")
    print(f"model saved to {cfg.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from face_emotion import __version__

    parser = argparse.ArgumentParser(prog="face-emotion", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("webcam", help="real-time recognition from a camera")
    _add_model_args(p)
    p.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--smoothing", type=float, default=0.6, help="temporal smoothing 0..1 (default: 0.6)")
    p.set_defaults(func=cmd_webcam)

    p = sub.add_parser("image", help="recognize emotions in image files (prints JSON)")
    _add_model_args(p)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("-o", "--output", type=Path, help="annotated output file or directory")
    p.add_argument("--show", action="store_true", help="display the annotated images")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("video", help="process a video file")
    _add_model_args(p)
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--output", type=Path, help="annotated output .mp4")
    p.add_argument("--show", action="store_true", help="display while processing")
    p.add_argument("--smoothing", type=float, default=0.6)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("train", help="train a new model on a FER-2013 style folder")
    p.add_argument("data_dir", type=Path, help="folder containing train/ and optional test/")
    p.add_argument("-o", "--output", type=Path, default=Path("models/emotion_model.keras"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
