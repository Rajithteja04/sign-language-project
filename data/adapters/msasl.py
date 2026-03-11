from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Counter as CounterType, Iterable, List

import cv2
import numpy as np
import torch
from collections import Counter

from data.formats import DatasetSplit, SequenceExample
from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor


def _normalize_frame(features: np.ndarray) -> np.ndarray:
    if features.shape[0] != FEATURE_DIM:
        return features
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


def _sample_indices(total: int, target: int) -> np.ndarray:
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    if total == 1:
        return np.zeros(target, dtype=np.int64)
    return np.linspace(0, total - 1, num=target, dtype=np.int64)


@dataclass
class MSASLConfig:
    root: Path
    seq_len: int = 30
    top_k: int | None = 20
    cache_features: bool = False
    cache_dir: Path | None = None
    normalize: bool = False


def _iter_class_videos(split_dir: Path) -> Iterable[tuple[str, Path]]:
    if not split_dir.exists():
        return []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = class_dir.name.strip().lower().replace(" ", "_")
        for video in sorted(class_dir.glob("*.mp4")):
            yield label, video


def _select_labels(train_dir: Path, top_k: int | None) -> List[str]:
    counts: CounterType[str] = Counter()
    for label, _ in _iter_class_videos(train_dir):
        counts[label] += 1
    if not counts:
        raise RuntimeError(f"No videos found under {train_dir}")
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [label for label, _ in ordered]
    return labels[:top_k] if top_k else labels


def _extract_sequence(
    path: Path,
    extractor: MediaPipeExtractor,
    seq_len: int,
    normalize: bool,
    cache_features: bool,
    cache_dir: Path | None,
    split: str,
    label: str,
) -> np.ndarray | None:
    cache_path = None
    if cache_features and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_norm" if normalize else ""
        safe_label = label.replace("/", "_")
        cache_path = cache_dir / f"{split}_{safe_label}_{path.stem}_seq{seq_len}{suffix}.pt"
        if cache_path.exists():
            arr = torch.load(cache_path, map_location="cpu")
            if isinstance(arr, torch.Tensor):
                arr = arr.numpy()
            if isinstance(arr, np.ndarray) and arr.shape == (seq_len, FEATURE_DIM):
                return arr.astype(np.float32)

    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        return None

    frames: List[np.ndarray] = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        feat = extractor.extract(frame)
        if normalize:
            feat = _normalize_frame(feat)
        frames.append(feat)

    reader.release()

    if not frames:
        return None

    arr = np.asarray(frames, dtype=np.float32)
    if arr.shape[0] >= seq_len:
        idx = _sample_indices(arr.shape[0], seq_len)
        seq = arr[idx]
    else:
        pad = np.zeros((seq_len - arr.shape[0], FEATURE_DIM), dtype=np.float32)
        seq = np.vstack([arr, pad])

    if cache_path is not None:
        torch.save(torch.from_numpy(seq), cache_path)

    return seq


def load_msasl(
    root: str | Path,
    seq_len: int = 30,
    top_k: int | None = 20,
    cache_features: bool = False,
    cache_dir: str | Path | None = None,
    normalize: bool = False,
) -> DatasetSplit:
    cfg = MSASLConfig(
        root=Path(root),
        seq_len=seq_len,
        top_k=top_k,
        cache_features=cache_features,
        cache_dir=Path(cache_dir) if cache_dir else None,
        normalize=normalize,
    )
    return _load(cfg)


def _load(cfg: MSASLConfig) -> DatasetSplit:
    train_dir = cfg.root / "train"
    val_dir = cfg.root / "val"
    test_dir = cfg.root / "test"

    keep_labels = _select_labels(train_dir, cfg.top_k)
    extractor = MediaPipeExtractor()

    def build_examples(split: str, split_dir: Path) -> List[SequenceExample]:
        bucket: List[SequenceExample] = []
        for label, video_path in _iter_class_videos(split_dir):
            if label not in keep_labels:
                continue
            seq = _extract_sequence(
                video_path,
                extractor,
                cfg.seq_len,
                cfg.normalize,
                cfg.cache_features,
                cfg.cache_dir,
                split,
                label,
            )
            if seq is None:
                continue
            features = [frame for frame in seq]
            bucket.append(SequenceExample(features=features, label=label))
        return bucket

    train_examples = build_examples("train", train_dir)
    val_examples = build_examples("val", val_dir)
    test_examples = build_examples("test", test_dir)

    print(
        f"MS-ASL: labels={len(keep_labels)} train={len(train_examples)} "
        f"val={len(val_examples)} test={len(test_examples)} seq_len={cfg.seq_len}"
    )

    return DatasetSplit(train=train_examples, val=val_examples, test=test_examples)
