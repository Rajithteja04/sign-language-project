from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor
from models.lstm import LSTMClassifier


def _load_numeric_names() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "data" / "lsa64_numeric_labels.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


NUMERIC_NAMES = _load_numeric_names()


def _format_label(label: str) -> str:
    pretty = NUMERIC_NAMES.get(label)
    return f"{pretty} ({label})" if pretty else label


def _infer_hidden_and_layers(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    w0 = state.get("lstm.weight_ih_l0")
    hidden = int(w0.shape[0] // 4) if w0 is not None else 128
    layers = 1
    indices = []
    for key in state:
        if key.startswith("lstm.weight_ih_l"):
            suffix = key.split("lstm.weight_ih_l", 1)[1]
            if suffix.isdigit():
                indices.append(int(suffix))
    if indices:
        layers = max(indices) + 1
    return hidden, layers


def sample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == target_len:
        return seq
    if len(seq) < target_len:
        pad = np.repeat(seq[-1][None, :], target_len - len(seq), axis=0)
        return np.concatenate([seq, pad], axis=0)
    indices = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return seq[indices]


def load_artifacts(artifacts_dir: Path) -> tuple[LSTMClassifier, Dict[int, str], int, bool, Path | None]:
    weights = artifacts_dir / "lstm_best.pt"
    labels = artifacts_dir / "label_to_id.json"
    meta = artifacts_dir / "lstm_meta.json"

    if not (weights.exists() and labels.exists() and meta.exists()):
        missing = [str(p) for p in (weights, labels, meta) if not p.exists()]
        raise FileNotFoundError(f"Missing artifacts: {', '.join(missing)}")

    with labels.open("r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}

    with meta.open("r", encoding="utf-8") as f:
        meta_data = json.load(f)

    seq_len = int(meta_data.get("sequence_length", 30))
    feature_dim = int(meta_data.get("feature_dim", FEATURE_DIM))
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"Feature dim mismatch: meta={feature_dim}, expected={FEATURE_DIM}")

    num_classes = int(meta_data.get("num_classes", len(id_to_label)))
    normalize = bool(meta_data.get("normalize", False))

    state: dict[str, torch.Tensor] = torch.load(weights, map_location="cpu")
    inferred_hidden, inferred_layers = _infer_hidden_and_layers(state)
    hidden_dim = int(meta_data.get("lstm_hidden", meta_data.get("hidden_dim", inferred_hidden)))
    num_layers = int(meta_data.get("lstm_layers", meta_data.get("layers", inferred_layers)))
    model = LSTMClassifier(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    cache_dir = meta_data.get("cache_dir")
    cache_path = Path(cache_dir) if cache_dir else None

    model.load_state_dict(state)
    model.eval()
    return model, id_to_label, seq_len, normalize, cache_path


def _normalize_frame(features: np.ndarray) -> np.ndarray:
    pts = features.reshape(-1, 3).copy()
    left_idx = 11
    right_idx = 12
    lx, ly = pts[left_idx, :2]
    rx, ry = pts[right_idx, :2]
    torso_center = np.asarray([(lx + rx) * 0.5, (ly + ry) * 0.5], dtype=np.float32)
    shoulder_dist = float(np.hypot(lx - rx, ly - ry))
    if shoulder_dist <= 1e-6:
        return features
    pts[:, :2] -= torso_center
    pts[:, :2] /= (shoulder_dist + 1e-6)
    return pts.reshape(-1)


def extract_sequence(video_path: Path, extractor: MediaPipeExtractor, normalize: bool) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            try:
                feat = extractor.extract(frame)
                if normalize:
                    feat = _normalize_frame(feat)
            except Exception:
                continue
            if feat.shape[0] == FEATURE_DIM:
                frames.append(feat)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("No valid frames extracted from video.")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module-2 model on a prerecorded video.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Directory containing lstm_best.pt, label_to_id.json, lstm_meta.json.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many top probabilities to print.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    artifacts_dir = Path(args.artifacts)

    model, id_to_label, seq_len, normalize, _ = load_artifacts(artifacts_dir)
    extractor = MediaPipeExtractor()

    frames = extract_sequence(video_path, extractor, normalize)
    seq = np.vstack(frames)
    seq = sample_sequence(seq, seq_len)

    x = torch.from_numpy(seq[None, ...]).float()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(args.topk, probs.shape[0])
    top_vals, top_ids = torch.topk(probs, k=topk)
    pred_id = int(top_ids[0].item())
    pred_label = id_to_label.get(pred_id, "UNKNOWN")
    display_label = _format_label(pred_label)
    confidence = float(top_vals[0].item())

    print(f"Video: {video_path}")
    print(f"Predicted gloss: {display_label} (confidence {confidence:.3f})")
    print("Top probabilities:")
    for score, idx in zip(top_vals.tolist(), top_ids.tolist()):
        raw = id_to_label.get(int(idx), str(idx))
        pretty = _format_label(raw)
        print(f"  {pretty:>20s}: {score:.3f}")


if __name__ == "__main__":
    main()
