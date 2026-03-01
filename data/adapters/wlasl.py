from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

from data.formats import DatasetSplit, SequenceExample
from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor


def _sample_indices(total: int, target: int) -> np.ndarray:
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    if total == 1:
        return np.zeros(target, dtype=np.int64)
    return np.linspace(0, total - 1, num=target, dtype=np.int64)


def _resolve_video_path(videos_dir: Path, video_id: str) -> Path | None:
    for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        p = videos_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    p = videos_dir / video_id
    if p.exists() and p.is_file():
        return p
    return None


def _discover_metadata_file(root: Path) -> Path:
    preferred = root / "WLASL_v0.3.json"
    if preferred.exists():
        return preferred
    json_candidates = sorted(root.glob("WLASL_v*.json"))
    if json_candidates:
        return json_candidates[0]
    csv_candidates = sorted(root.glob("WLASL_v*.csv"))
    if csv_candidates:
        return csv_candidates[0]
    raise FileNotFoundError("No WLASL_v*.json or WLASL_v*.csv found in dataset root")


def _safe_int(value, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _iter_samples_from_json(items: Iterable[dict]) -> Iterable[dict]:
    for entry in items:
        gloss = str(entry.get("gloss", "")).strip().lower()
        if not gloss:
            continue
        instances = entry.get("instances", []) or []
        for ins in instances:
            split = str(ins.get("split", "")).strip().lower()
            if split not in {"train", "test"}:
                continue
            video_id = str(ins.get("video_id", "")).strip()
            if not video_id:
                continue
            start = _safe_int(ins.get("frame_start"))
            end = _safe_int(ins.get("frame_end"))
            fps = _safe_int(ins.get("fps"), default=25) or 25
            if start is None or end is None:
                continue
            start0 = max(0, start - 1)
            end0 = max(0, end - 1)
            if end0 <= start0:
                continue
            yield {
                "gloss": gloss,
                "split": split,
                "video_id": video_id,
                "frame_start": start0,
                "frame_end": end0,
                "fps": fps,
                "bbox": ins.get("bbox"),
            }


def _iter_samples_from_csv(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = str(row.get("gloss", "")).strip().lower()
            split = str(row.get("split", "")).strip().lower()
            video_id = str(row.get("video_id", "") or row.get("id", "")).strip()
            start = _safe_int(row.get("frame_start"))
            end = _safe_int(row.get("frame_end"))
            fps = _safe_int(row.get("fps"), default=25) or 25
            if not gloss or split not in {"train", "test"} or not video_id:
                continue
            if start is None or end is None:
                continue
            start0 = max(0, start - 1)
            end0 = max(0, end - 1)
            if end0 <= start0:
                continue
            yield {
                "gloss": gloss,
                "split": split,
                "video_id": video_id,
                "frame_start": start0,
                "frame_end": end0,
                "fps": fps,
                "bbox": None,
            }


def _apply_bbox(frame: np.ndarray, bbox) -> np.ndarray:
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def _cache_path(cache_dir: Path, sample: dict) -> Path:
    key = (
        f"{sample['video_id']}_{sample['frame_start']}_{sample['frame_end']}_"
        f"{sample['gloss'].replace(' ', '_')}.pt"
    )
    return cache_dir / key


def _extract_segment_sequence(
    sample: dict,
    video_path: Path,
    extractor: MediaPipeExtractor,
    seq_len: int,
    use_bbox_crop: bool,
    cache_features: bool,
    cache_dir: Path | None,
) -> List[np.ndarray]:
    if cache_features and cache_dir is not None:
        cpath = _cache_path(cache_dir, sample)
        if cpath.exists():
            arr = torch.load(cpath, map_location="cpu")
            if isinstance(arr, torch.Tensor):
                arr = arr.numpy()
            if arr.ndim == 2 and arr.shape[1] == FEATURE_DIM:
                return [arr[i].astype(np.float32) for i in range(arr.shape[0])]

    reader = cv2.VideoCapture(str(video_path))
    if not reader.isOpened():
        print(f"WARNING: Missing or corrupted video: {video_path}")
        return []

    start_frame = int(sample["frame_start"])
    end_frame = int(sample["frame_end"])
    reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: List[np.ndarray] = []
    for _ in range(start_frame, end_frame + 1):
        ok, frame = reader.read()
        if not ok:
            break
        if use_bbox_crop:
            frame = _apply_bbox(frame, sample.get("bbox"))
        frames.append(frame)
    reader.release()

    if not frames:
        print(f"WARNING: Empty/incomplete segment for video: {video_path}")
        return []

    idxs = _sample_indices(len(frames), seq_len)
    seq = []
    for i in idxs:
        vec = extractor.extract(frames[int(i)])
        if vec.shape[0] != FEATURE_DIM:
            raise ValueError(f"Feature mismatch in {video_path}: {vec.shape[0]} != {FEATURE_DIM}")
        seq.append(vec.astype(np.float32))

    while len(seq) < seq_len:
        seq.append(np.zeros((FEATURE_DIM,), dtype=np.float32))
    seq = seq[:seq_len]

    if cache_features and cache_dir is not None:
        cpath = _cache_path(cache_dir, sample)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(np.asarray(seq), dtype=torch.float32), cpath)

    return seq


def _holdout_to_val(train_samples: List[dict], val_ratio: float) -> Tuple[List[dict], List[dict]]:
    if val_ratio <= 0:
        return train_samples, []
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for s in train_samples:
        grouped[s["gloss"]].append(s)

    train_out: List[dict] = []
    val_out: List[dict] = []
    for gloss, items in grouped.items():
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
        for idx, item in enumerate(items):
            if idx < n_val:
                val_out.append(item)
            else:
                train_out.append(item)
    return train_out, val_out


def load_wlasl(
    root: str,
    top_k: int = 100,
    max_samples_per_class: int | None = None,
    seq_len: int = 30,
    min_samples_per_class: int = 5,
    val_ratio: float = 0.10,
    cache_features: bool = False,
    cache_dir: str | None = None,
    use_bbox_crop: bool = False,
) -> DatasetSplit:
    dataset_root = Path(root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"WLASL root not found: {dataset_root}")
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"WLASL videos directory not found: {videos_dir}")

    meta_file = _discover_metadata_file(dataset_root)
    print(f"WLASL metadata file: {meta_file.name}")

    if meta_file.suffix.lower() == ".json":
        with meta_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        all_samples = list(_iter_samples_from_json(raw))
    else:
        all_samples = list(_iter_samples_from_csv(meta_file))

    if not all_samples:
        print("WARNING: No valid WLASL samples found after metadata parsing.")
        return DatasetSplit(train=[], val=[], test=[])

    counts = Counter(s["gloss"] for s in all_samples)
    keep_labels = {label for label, n in counts.most_common(top_k) if n >= min_samples_per_class}
    if not keep_labels:
        print("WARNING: No labels satisfy frequency constraints.")
        return DatasetSplit(train=[], val=[], test=[])

    selected = [s for s in all_samples if s["gloss"] in keep_labels]
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for s in selected:
        by_label[s["gloss"]].append(s)
    if max_samples_per_class is not None:
        limited = []
        for label in sorted(by_label.keys()):
            limited.extend(by_label[label][: max(0, int(max_samples_per_class))])
        selected = limited

    train_raw = [s for s in selected if s["split"] == "train"]
    test_raw = [s for s in selected if s["split"] == "test"]
    train_raw, val_raw = _holdout_to_val(train_raw, val_ratio=val_ratio)

    extractor = MediaPipeExtractor()
    cache_root = Path(cache_dir) if cache_dir else Path("artifacts/cache/wlasl_features")
    train: List[SequenceExample] = []
    val: List[SequenceExample] = []
    test: List[SequenceExample] = []

    def build(raw_samples: List[dict], bucket: List[SequenceExample]) -> None:
        for s in raw_samples:
            video_path = _resolve_video_path(videos_dir, s["video_id"])
            if video_path is None:
                print(f"WARNING: Missing or corrupted video: {videos_dir / s['video_id']}")
                continue
            seq = _extract_segment_sequence(
                sample=s,
                video_path=video_path,
                extractor=extractor,
                seq_len=seq_len,
                use_bbox_crop=use_bbox_crop,
                cache_features=cache_features,
                cache_dir=cache_root if cache_features else None,
            )
            if not seq:
                continue
            bucket.append(SequenceExample(features=seq, label=s["gloss"]))

    build(train_raw, train)
    build(val_raw, val)
    build(test_raw, test)

    unique_glosses = sorted({ex.label for ex in train + val + test})
    print(f"Loaded {len(train)} train samples")
    print(f"Loaded {len(val)} val samples")
    print(f"Loaded {len(test)} test samples")
    print(f"Unique gloss count: {len(unique_glosses)}")

    return DatasetSplit(train=train, val=val, test=test)
