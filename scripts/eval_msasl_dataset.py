from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor
from models.lstm import LSTMClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an MS-ASL-style folder of clips with the current LSTM artifacts."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Folder containing one subfolder per gloss (each with video files).",
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Directory with lstm_best.pt, label_to_id.json, lstm_meta.json.",
    )
    parser.add_argument(
        "--csv",
        default="artifacts/msasl_eval.csv",
        help="Where to write per-clip results (CSV).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many probabilities to record per clip.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp4", ".mov"],
        help="Video extensions to evaluate.",
    )
    return parser.parse_args()


def _sanitize_token(token: str) -> str:
    return token.strip().lower()


def load_artifacts(artifacts_dir: Path) -> Tuple[LSTMClassifier, Dict[int, str], int, bool]:
    weights = artifacts_dir / "lstm_best.pt"
    labels = artifacts_dir / "label_to_id.json"
    meta = artifacts_dir / "lstm_meta.json"

    if not (weights.exists() and labels.exists() and meta.exists()):
        missing = [str(p) for p in (weights, labels, meta) if not p.exists()]
        raise FileNotFoundError(f"Missing artifact files: {', '.join(missing)}")

    with labels.open("r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {int(idx): label for label, idx in label_to_id.items()}

    with meta.open("r", encoding="utf-8") as f:
        meta_data = json.load(f)

    seq_len = int(meta_data.get("sequence_length", 30))
    feature_dim = int(meta_data.get("feature_dim", FEATURE_DIM))
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"Feature dimension mismatch: meta={feature_dim}, expected={FEATURE_DIM}")

    num_classes = int(meta_data.get("num_classes", len(id_to_label)))
    state: Dict[str, torch.Tensor] = torch.load(weights, map_location="cpu")

    hidden_dim = int(meta_data.get("lstm_hidden", meta_data.get("hidden_dim", 128)))
    num_layers = int(meta_data.get("lstm_layers", meta_data.get("layers", 2)))
    normalize = bool(meta_data.get("normalize", False))

    model = LSTMClassifier(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    model.load_state_dict(state)
    model.eval()

    return model, id_to_label, seq_len, normalize


def sample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == target_len:
        return seq
    if len(seq) < target_len:
        pad = np.repeat(seq[-1][None, :], target_len - len(seq), axis=0)
        return np.concatenate([seq, pad], axis=0)
    indices = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return seq[indices]


def normalize_frame(features: np.ndarray) -> np.ndarray:
    pts = features.reshape(-1, 3).copy()
    left_idx, right_idx = 11, 12
    lx, ly = pts[left_idx, :2]
    rx, ry = pts[right_idx, :2]
    torso = np.asarray([(lx + rx) * 0.5, (ly + ry) * 0.5], dtype=np.float32)
    shoulder = float(np.hypot(lx - rx, ly - ry))
    if shoulder <= 1e-6:
        return features
    pts[:, :2] -= torso
    pts[:, :2] /= (shoulder + 1e-6)
    return pts.reshape(-1)


def iter_clips(root: Path, extensions: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    lower_exts = {ext.lower() for ext in extensions}
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for video in sorted(label_dir.rglob("*")):
            if video.suffix.lower() in lower_exts:
                yield label_dir.name, video


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    artifacts_dir = Path(args.artifacts)
    csv_path = Path(args.csv)

    model, id_to_label, seq_len, normalize_flag = load_artifacts(artifacts_dir)
    extractor = MediaPipeExtractor()
    label_counts = defaultdict(int)
    correct = 0
    total = 0
    results: List[dict] = []

    for true_label, video_path in iter_clips(dataset_root, args.extensions):
        total += 1
        label_counts[true_label] += 1

        cap = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            try:
                feat = extractor.extract(frame)
            except Exception:
                continue
            if feat.shape[0] != FEATURE_DIM:
                continue
            if normalize_flag:
                feat = normalize_frame(feat)
            frames.append(feat)
        cap.release()

        if not frames:
            pred_label = "NO_FRAMES"
            confidence = 0.0
            top_tokens: List[Tuple[str, float]] = []
        else:
            seq = sample_sequence(np.asarray(frames, dtype=np.float32), seq_len)
            x = torch.from_numpy(seq[None, ...]).float()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
            topk = min(args.topk, probs.shape[0])
            conf_vals, idxs = torch.topk(probs, k=topk)
            top_tokens = [(id_to_label.get(int(idx), "UNKNOWN"), float(val)) for idx, val in zip(idxs.tolist(), conf_vals.tolist())]
            pred_label, confidence = top_tokens[0]

        pred_clean = _sanitize_token(pred_label)
        true_clean = _sanitize_token(true_label)
        if pred_clean == true_clean:
            correct += 1

        row = {
            "video": str(video_path),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": f"{confidence:.4f}",
        }
        for i, (token, score) in enumerate(top_tokens, start=1):
            row[f"top{i}_label"] = token
            row[f"top{i}_confidence"] = f"{score:.4f}"
        results.append(row)

        if total % 20 == 0:
            print(f"[{total}] running accuracy: {correct/total:.3f}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video", "true_label", "pred_label", "confidence"]
    for i in range(1, args.topk + 1):
        fieldnames.extend([f"top{i}_label", f"top{i}_confidence"])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTotal clips: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / max(total,1):.4f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
