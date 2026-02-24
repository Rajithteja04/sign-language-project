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


def _candidate_roots(root: Path) -> list[Path]:
    roots = [root]
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                roots.append(child)
    return roots


def _pick_csv(candidates: list[Path], split: str) -> Path | None:
    if not candidates:
        return None
    exact_name = f"how2sign_{split}.csv"
    for path in candidates:
        if path.name == exact_name:
            return path
    for path in candidates:
        if "how2sign" in path.name.lower() and split in path.name.lower():
            return path
    for path in candidates:
        if split in path.name.lower():
            return path
    return candidates[0]


def _split_tokens(split: str) -> tuple[str, ...]:
    return (split, f"{split}_2d_keypoints", f"{split}_2D_keypoints")


def _score_pair(csv_path: Path, json_path: Path, split: str) -> tuple[int, int]:
    split_parts = {p.lower() for p in csv_path.parts + json_path.parts}
    token_hits = sum(1 for tok in _split_tokens(split) if tok.lower() in split_parts)
    common_prefix = len(csv_path.parents) + len(json_path.parents)
    return token_hits, common_prefix


def _resolve_recursive_layout(root: Path, split: str) -> tuple[Path, Path] | None:
    split_l = split.lower()
    csv_candidates = [p for p in root.glob("**/*.csv") if p.is_file() and split_l in p.name.lower()]
    if not csv_candidates:
        return None

    json_candidates = []
    for p in root.glob("**/openpose_output/json"):
        if not p.is_dir():
            continue
        parts_l = [x.lower() for x in p.parts]
        if any(tok.lower() in parts_l for tok in _split_tokens(split)):
            json_candidates.append(p)
    if not json_candidates:
        json_candidates = [p for p in root.glob("**/openpose_output/json") if p.is_dir()]
    if not json_candidates:
        return None

    picked_csv = _pick_csv(csv_candidates, split)
    if picked_csv is None:
        return None

    best_json = max(json_candidates, key=lambda jp: _score_pair(picked_csv, jp, split))
    return picked_csv, best_json


def _resolve_split_paths(root: Path, split: str) -> tuple[Path, Path]:
    tried: list[tuple[Path, Path]] = []

    for base in _candidate_roots(root):
        # Layout A (legacy flat):
        #   <base>/how2sign_<split>.csv
        #   <base>/<split>_2D_keypoints/openpose_output/json
        flat_csv = base / f"how2sign_{split}.csv"
        flat_json = base / f"{split}_2D_keypoints" / "openpose_output" / "json"
        tried.append((flat_csv, flat_json))
        if flat_csv.exists() and flat_json.exists():
            return flat_csv, flat_json

        # Layout B (sentence-level canonical):
        #   <base>/sentence_level/<split>/text/en/raw_text/how2sign_<split>.csv
        #   <base>/sentence_level/<split>/rgb_front/features/openpose_output/json
        sentence_csv = base / "sentence_level" / split / "text" / "en" / "raw_text" / f"how2sign_{split}.csv"
        sentence_json = base / "sentence_level" / split / "rgb_front" / "features" / "openpose_output" / "json"
        tried.append((sentence_csv, sentence_json))
        if sentence_csv.exists() and sentence_json.exists():
            return sentence_csv, sentence_json

        # Layout C (sentence-level variants):
        split_root = base / "sentence_level" / split
        if split_root.exists():
            csv_candidates = [p for p in split_root.glob("**/*.csv") if p.is_file()]
            json_candidates = [p for p in split_root.glob("**/openpose_output/json") if p.is_dir()]
            picked_csv = _pick_csv(csv_candidates, split)
            if picked_csv is not None and json_candidates:
                return picked_csv, json_candidates[0]

    # Layout D (fully recursive fallback for non-standard extracted trees):
    recursive_match = _resolve_recursive_layout(root, split)
    if recursive_match is not None:
        return recursive_match

    tried_msg = " or ".join(f"{csv} + {json_dir}" for csv, json_dir in tried[:6])
    raise FileNotFoundError(
        "Could not resolve How2Sign split layout. "
        f"Tried: {tried_msg}"
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
