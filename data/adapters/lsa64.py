from __future__ import annotations

import random
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List
import re

import cv2
import numpy as np
import torch

from data.formats import DatasetSplit, SequenceExample
from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor


TOKEN_SANITIZE_RE = re.compile(r"[^A-Z0-9]+")


def _sanitize_token(word: str) -> str:
    return TOKEN_SANITIZE_RE.sub("_", word.upper()).strip("_")


def _load_numeric_label_map(root: Path) -> dict[str, str]:
    candidate_paths = [
        root / "lsa64_numeric_labels.json",
        root / "data" / "lsa64_numeric_labels.json",
        Path(__file__).resolve().parents[2] / "data" / "lsa64_numeric_labels.json",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            mapping: dict[str, str] = {}
            for k, v in raw.items():
                key = str(k).zfill(3)
                token = _sanitize_token(str(v))
                if token:
                    mapping[key] = token
            if mapping:
                return mapping
        except Exception:
            continue
    return {}


def _sample_indices(total: int, target: int) -> np.ndarray:
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    if total == 1:
        return np.zeros(target, dtype=np.int64)
    return np.linspace(0, total - 1, num=target, dtype=np.int64)


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


@dataclass
class LSA64Config:
    root: Path
    seq_len: int = 30
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    top_k: int = 5
    cache_features: bool = False
    cache_dir: Path | None = None
    normalize: bool = False
    seed: int = 42
    include_labels: list[str] | None = None


def _gather_files(root: Path) -> Dict[str, List[Path]]:
    video_root = root / "all" if (root / "all").exists() else root
    files = sorted(video_root.rglob("*.mp4"))
    numeric_map = _load_numeric_label_map(root)
    groups: Dict[str, List[Path]] = {}
    for path in files:
        stem = path.stem
        parts = stem.split("_")
        if not parts:
            continue
        numeric = parts[0].zfill(3)
        label = numeric_map.get(numeric, _sanitize_token(numeric))
        groups.setdefault(label, []).append(path)
    return groups


def _extract_sequence(
    path: Path,
    extractor: MediaPipeExtractor,
    seq_len: int,
    normalize: bool,
    cache_features: bool,
    cache_dir: Path | None,
) -> np.ndarray | None:
    cache_path = None
    if cache_features and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_suffix = "_norm" if normalize else ""
        cache_path = cache_dir / f"{path.stem}_seq{seq_len}{cache_suffix}.pt"
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
        features = extractor.extract(frame)
        if normalize:
            features = _normalize_frame(features)
        frames.append(features)

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


def load_lsa64(
    root: str | Path,
    seq_len: int = 30,
    top_k: int = 5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cache_dir: str | Path | None = None,
    cache_features: bool = False,
    normalize: bool = False,
    seed: int = 42,
    include_labels: list[str] | None = None,
) -> DatasetSplit:
    cfg = LSA64Config(
        root=Path(root),
        seq_len=seq_len,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        top_k=top_k,
        cache_features=cache_features,
        cache_dir=Path(cache_dir) if cache_dir else None,
        normalize=normalize,
        seed=seed,
        include_labels=[_sanitize_token(x) for x in include_labels] if include_labels else None,
    )
    return _load(cfg)


def _load(cfg: LSA64Config) -> DatasetSplit:
    groups = _gather_files(cfg.root)
    if not groups:
        raise RuntimeError(f"No mp4 files found in {cfg.root}")

    sorted_labels = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if cfg.include_labels:
        requested = [_sanitize_token(x) for x in cfg.include_labels if _sanitize_token(x)]
        keep = [label for label in requested if label in groups]
        missing = [label for label in requested if label not in groups]
        if missing:
            print(f"LSA64 warning: requested labels not found and skipped: {', '.join(missing)}")
        if not keep:
            raise RuntimeError("None of the requested include_labels exist in dataset.")
    else:
        keep = [label for label, _ in sorted_labels[: cfg.top_k]]

    rng = random.Random(cfg.seed)
    extractor = MediaPipeExtractor()

    train_examples: List[SequenceExample] = []
    val_examples: List[SequenceExample] = []
    test_examples: List[SequenceExample] = []

    for label in keep:
        paths = list(groups[label])
        rng.shuffle(paths)
        n = len(paths)
        val_n = max(1, int(round(n * cfg.val_ratio)))
        test_n = max(1, int(round(n * cfg.test_ratio)))
        train_n = n - val_n - test_n
        if train_n < 1:
            train_n = 1
        if train_n + val_n + test_n > n:
            overflow = train_n + val_n + test_n - n
            if test_n > overflow:
                test_n -= overflow
            elif val_n > overflow:
                val_n -= overflow
            else:
                train_n = max(1, train_n - overflow)

        train_paths = paths[:train_n]
        val_paths = paths[train_n : train_n + val_n]
        test_paths = paths[train_n + val_n : train_n + val_n + test_n]

        for bucket_paths, bucket in (
            (train_paths, train_examples),
            (val_paths, val_examples),
            (test_paths, test_examples),
        ):
            for video_path in bucket_paths:
                seq = _extract_sequence(
                    video_path,
                    extractor,
                    cfg.seq_len,
                    cfg.normalize,
                    cfg.cache_features,
                    cfg.cache_dir,
                )
                if seq is None:
                    continue
                features = [frame for frame in seq]
                bucket.append(SequenceExample(features=features, label=label))

    print(
        f"LSA64: labels={len(keep)} train={len(train_examples)} "
        f"val={len(val_examples)} test={len(test_examples)} seq_len={cfg.seq_len}"
    )
    print(f"LSA64 labels used: {', '.join(keep)}")

    return DatasetSplit(train=train_examples, val=val_examples, test=test_examples)
