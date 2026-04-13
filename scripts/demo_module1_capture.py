from __future__ import annotations

import time
import argparse
from pathlib import Path

import cv2
import numpy as np

from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor

try:
    from .run_video_inference import _normalize_frame
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.run_video_inference import _normalize_frame  # type: ignore


def capture_sequence(
    seq_len: int = 30,
    normalize: bool = True,
    camera_index: int = 0,
    video_path: Path | None = None,
) -> np.ndarray:
    extractor = MediaPipeExtractor()
    source_desc = f"video '{video_path}'" if video_path else f"camera index {camera_index}"
    cap = cv2.VideoCapture(str(video_path)) if video_path else cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {source_desc}.")

    frames: list[np.ndarray] = []
    if video_path:
        print(f"Reading frames from {video_path}…")
    else:
        print(f"Starting camera (index {camera_index})… move your hands into frame.")
    try:
        while len(frames) < seq_len:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            try:
                feat = extractor.extract(frame)
            except Exception as exc:  # pragma: no cover - realtime fallback
                print(f"[warn] MediaPipe failed on frame: {exc}")
                continue

            if feat.shape[0] != FEATURE_DIM:
                continue
            if normalize:
                feat = _normalize_frame(feat)

            frames.append(feat)
            print(f"Captured frame {len(frames)}/{seq_len}", end="\r", flush=True)
    finally:
        cap.release()

    print("\nCapture complete.\n")
    return np.vstack(frames[:seq_len])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 1 demo: capture 30 frames of MediaPipe landmarks from webcam or video.",
    )
    parser.add_argument("--seq-len", type=int, default=30, help="Number of frames to capture (default: 30).")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to use (default: 0).")
    parser.add_argument("--video", type=str, help="Optional video file path to use instead of a live camera.")
    args = parser.parse_args()

    seq_len = max(1, args.seq_len)
    print("Module 1 – Landmark Extraction Demo")
    video_path = Path(args.video).expanduser() if args.video else None
    if video_path and not video_path.exists():
        print(f"Video file not found: {video_path}")
        return

    if video_path:
        print(f"Using prerecorded video: {args.video}\n")
    else:
        print("This tool captures frames from your webcam and prints the normalized 411-D vectors.\n")

    array = capture_sequence(seq_len=seq_len, normalize=True, camera_index=args.camera, video_path=video_path)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print(f"Sequence shape: {array.shape} (frames × features)\n")
    for idx, vec in enumerate(array, start=1):
        print(f"Frame {idx:02d}:")
        print(np.array2string(vec, separator=", "))
        print()


if __name__ == "__main__":
    main()
