from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor
from models.lstm import LSTMClassifier
from scripts.run_video_inference import (
    NUMERIC_NAMES,
    _format_label,
    extract_sequence,
    load_artifacts,
    sample_sequence,
)


def _read_label_names() -> Dict[str, str]:
    return NUMERIC_NAMES


def _load_cached_sequence(video_path: Path, cache_dir: Path | None, seq_len: int, normalize: bool):
    if cache_dir is None:
        return None
    suffix = "_norm" if normalize else ""
    cache_name = f"{video_path.stem}_seq{seq_len}{suffix}.pt"
    cache_path = cache_dir / cache_name
    if not cache_path.exists():
        return None
    tensor = torch.load(cache_path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy().astype(np.float32)
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LSA64 model over every video clip.")
    parser.add_argument("--dataset-root", required=True, help="Path to LSA64 root folder (with class subfolders).")
    parser.add_argument(
        "--artifacts",
        default="artifacts/lsa64/lsa64_top64_seq30_h128_l2_bs16_lr5e-04",
        help="Directory containing lstm_best.pt, label_to_id.json, lstm_meta.json.",
    )
    parser.add_argument("--max-videos", type=int, default=0, help="Optional limit for quick testing.")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV path to log per-video predictions.")
    parser.add_argument("--topk", type=int, default=5, help="Store top-k confidences in CSV.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    model, id_to_label, seq_len, normalize, cache_dir = load_artifacts(Path(args.artifacts))
    extractor = MediaPipeExtractor() if cache_dir is None else None
    label_names = _read_label_names()

    files = sorted(dataset_root.rglob("*.mp4"))
    if args.max_videos > 0:
        files = files[: args.max_videos]
    if not files:
        raise RuntimeError("No .mp4 files found under dataset root.")

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        header = ["video_path", "true_label", "pred_label", "correct", "confidence"]
        for i in range(args.topk):
            header.append(f"top{i+1}_label")
            header.append(f"top{i+1}_confidence")
        csv_writer.writerow(header)

    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    total = 0
    correct = 0

    for idx, video in enumerate(files, 1):
        true_label = video.parent.name
        seq_arr = _load_cached_sequence(video, cache_dir, seq_len, normalize)
        if seq_arr is None:
            if extractor is None:
                extractor = MediaPipeExtractor()
            frames = extract_sequence(video, extractor, normalize)
            seq_arr = np.vstack(frames)

        seq = seq_arr
        seq = sample_sequence(seq, seq_len)

        x = torch.from_numpy(seq[None, ...]).float()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
        topk = min(args.topk, probs.shape[0])
        top_vals, top_ids = torch.topk(probs, k=topk)
        pred_label = id_to_label.get(int(top_ids[0].item()), "UNKNOWN")
        confidence = float(top_vals[0].item())
        is_correct = pred_label == true_label

        stats[true_label]["total"] += 1
        stats[true_label]["correct"] += int(is_correct)
        total += 1
        correct += int(is_correct)

        if csv_writer:
            row = [
                str(video),
                f"{label_names.get(true_label, true_label)} ({true_label})",
                _format_label(pred_label),
                int(is_correct),
                f"{confidence:.4f}",
            ]
            for score, idx_tensor in zip(top_vals.tolist(), top_ids.tolist()):
                raw_label = id_to_label.get(int(idx_tensor), str(idx_tensor))
                row.append(_format_label(raw_label))
                row.append(f"{score:.4f}")
            csv_writer.writerow(row)

        if idx % 50 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] Accuracy so far: {correct/total:.4f}")

    if csv_file:
        csv_file.close()

    print(f"\nTotal clips: {total}")
    print(f"Correct: {correct}")
    print(f"Overall accuracy: {correct/total:.4f}")

    per_label = sorted(stats.items(), key=lambda kv: kv[0])
    print("\nPer-label accuracy:")
    for label, counts in per_label:
        acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
        pretty = label_names.get(label, label)
        print(f"  {pretty:>12s} ({label}): {acc:.4f}  ({counts['correct']}/{counts['total']})")


if __name__ == "__main__":
    main()
