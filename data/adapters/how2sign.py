import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from data.formats import DatasetSplit, SequenceExample
from utils.keypoint_format import (
    FACE_POINTS,
    HAND_POINTS,
    POSE_POINTS,
    TOTAL_POINTS,
    flatten_points,
    normalize_openpose_like,
)


@dataclass
class How2SignConfig:
    root: Path
    max_samples_per_split: int | None = None
    min_frames: int = 4
    use_confidence: bool = True


def _reshape_keypoints(flat_values: list[float]) -> np.ndarray:
    if not flat_values:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.asarray(flat_values, dtype=np.float32)
    usable = (arr.size // 3) * 3
    return arr[:usable].reshape(-1, 3)


def _load_frame_vector(json_path: Path, use_confidence: bool) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    people = payload.get("people", [])
    if not people:
        dim = TOTAL_POINTS * (3 if use_confidence else 2)
        return np.zeros(dim, dtype=np.float32)

    p0 = people[0]
    keys = [
        ("pose_keypoints_2d", POSE_POINTS),
        ("face_keypoints_2d", FACE_POINTS),
        ("hand_left_keypoints_2d", HAND_POINTS),
        ("hand_right_keypoints_2d", HAND_POINTS),
    ]
    point_parts: list[np.ndarray] = []

    for name, expected_points in keys:
        arr = _reshape_keypoints(p0.get(name, []))
        if arr.shape[0] < expected_points:
            pad = np.zeros((expected_points - arr.shape[0], 3), dtype=np.float32)
            arr = np.vstack([arr, pad])
        arr = arr[:expected_points]
        point_parts.append(arr)

    points = np.vstack(point_parts)
    points = normalize_openpose_like(points)
    return flatten_points(points, use_confidence=use_confidence)


def _iter_rows(csv_path: Path) -> Iterable[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def _resolve_split_paths(root: Path, split: str) -> tuple[Path, Path]:
    # Layout A (legacy flat):
    #   <root>/how2sign_<split>.csv
    #   <root>/<split>_2D_keypoints/openpose_output/json
    flat_csv = root / f"how2sign_{split}.csv"
    flat_json = root / f"{split}_2D_keypoints" / "openpose_output" / "json"
    if flat_csv.exists() and flat_json.exists():
        return flat_csv, flat_json

    # Layout B (How2Sign downloader sentence-level):
    #   <root>/sentence_level/<split>/text/en/raw_text/how2sign_<split>.csv
    #   <root>/sentence_level/<split>/rgb_front/features/openpose_output/json
    sentence_csv = root / "sentence_level" / split / "text" / "en" / "raw_text" / f"how2sign_{split}.csv"
    sentence_json = root / "sentence_level" / split / "rgb_front" / "features" / "openpose_output" / "json"
    if sentence_csv.exists() and sentence_json.exists():
        return sentence_csv, sentence_json

    raise FileNotFoundError(
        "Could not resolve How2Sign split layout. "
        f"Tried: {flat_csv} + {flat_json} and {sentence_csv} + {sentence_json}"
    )


def _load_split(split: str, cfg: How2SignConfig) -> list[SequenceExample]:
    csv_path, base_json = _resolve_split_paths(cfg.root, split)

    examples: list[SequenceExample] = []
    for row in _iter_rows(csv_path):
        sent_name = row["SENTENCE_NAME"]
        sentence = row["SENTENCE"].strip()
        seq_dir = base_json / sent_name
        if not seq_dir.exists():
            continue

        frame_files = sorted(seq_dir.glob("*_keypoints.json"))
        if len(frame_files) < cfg.min_frames:
            continue

        features = [_load_frame_vector(p, cfg.use_confidence) for p in frame_files]
        examples.append(SequenceExample(features=features, label=sentence))

        if cfg.max_samples_per_split is not None and len(examples) >= cfg.max_samples_per_split:
            break

    return examples


def load_how2sign(root: str, max_samples_per_split: int | None = None) -> DatasetSplit:
    cfg = How2SignConfig(root=Path(root), max_samples_per_split=max_samples_per_split)
    return DatasetSplit(
        train=_load_split("train", cfg),
        val=_load_split("val", cfg),
        test=_load_split("test", cfg),
    )
