from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from features.mediapipe_extractor import FEATURE_DIM
from models.lstm import LSTMClassifier


def _infer_num_layers(state: dict[str, torch.Tensor]) -> int:
    indices: list[int] = []
    for key in state:
        if key.startswith("lstm.weight_ih_l"):
            suffix = key.split("lstm.weight_ih_l", 1)[1]
            if suffix.isdigit():
                indices.append(int(suffix))
    return (max(indices) + 1) if indices else 2


def _infer_hidden_dim(state: dict[str, torch.Tensor]) -> int:
    w = state.get("lstm.weight_ih_l0")
    if w is None:
        return 128
    return int(w.shape[0] // 4)


def check_artifacts(repo_root: Path) -> tuple[bool, str]:
    artifacts = repo_root / "artifacts"
    weights = artifacts / "lstm_best.pt"
    labels = artifacts / "label_to_id.json"
    meta = artifacts / "lstm_meta.json"

    missing = [str(p.relative_to(repo_root)) for p in (weights, labels, meta) if not p.exists()]
    if missing:
        return False, f"Missing artifacts: {', '.join(missing)}"

    with labels.open("r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    with meta.open("r", encoding="utf-8") as f:
        meta_data = json.load(f)

    if not isinstance(label_to_id, dict) or not label_to_id:
        return False, "label_to_id.json is empty or invalid."

    state: dict[str, torch.Tensor] = torch.load(weights, map_location="cpu")
    seq_len = int(meta_data.get("sequence_length", 30))
    feature_dim = int(meta_data.get("feature_dim", meta_data.get("input_dim", FEATURE_DIM)))

    if feature_dim != FEATURE_DIM:
        return False, f"Feature dimension mismatch: meta={feature_dim}, runtime={FEATURE_DIM}"

    num_classes = int(meta_data.get("num_classes", len(label_to_id)))
    if num_classes <= 0:
        fc_weight = state.get("fc.weight")
        if fc_weight is None:
            return False, "Cannot infer num_classes from state dict."
        num_classes = int(fc_weight.shape[0])

    hidden_dim = int(meta_data.get("lstm_hidden", meta_data.get("hidden_dim", _infer_hidden_dim(state))))
    num_layers = int(meta_data.get("lstm_layers", meta_data.get("layers", _infer_num_layers(state))))

    model = LSTMClassifier(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        _ = model(torch.zeros((1, seq_len, feature_dim), dtype=torch.float32))

    return True, (
        f"Model OK | labels={len(label_to_id)} classes={num_classes} "
        f"seq_len={seq_len} feature_dim={feature_dim} hidden={hidden_dim} layers={num_layers}"
    )


def check_camera(camera_index: int) -> tuple[bool, str]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False, f"Camera open failed (index {camera_index})."
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False, f"Camera read failed (index {camera_index})."
    h, w = frame.shape[:2]
    return True, f"Camera OK | index={camera_index} resolution={w}x{h}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for real-time sign app runtime.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to test.")
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="Skip camera check (useful if another app currently owns webcam).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    print(f"[INFO] Repo root: {repo_root}")

    ok_artifacts, msg_artifacts = check_artifacts(repo_root)
    print(f"[{'OK' if ok_artifacts else 'FAIL'}] {msg_artifacts}")

    ok_camera = True
    if args.skip_camera:
        print("[SKIP] Camera check skipped by flag.")
    else:
        ok_camera, msg_camera = check_camera(args.camera_index)
        print(f"[{'OK' if ok_camera else 'FAIL'}] {msg_camera}")

    if ok_artifacts and (ok_camera or args.skip_camera):
        print("[READY] Runtime preflight passed.")
    else:
        print("[NOT READY] Fix the failed checks before demo.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

