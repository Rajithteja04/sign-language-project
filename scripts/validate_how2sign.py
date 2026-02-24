import argparse
import csv
import json
import os
from pathlib import Path


def candidate_roots(root: Path) -> list[Path]:
    roots = [root]
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                roots.append(child)
    return roots


def pick_csv(candidates: list[Path], split: str) -> Path | None:
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


def split_tokens(split: str) -> tuple[str, ...]:
    return (split, f"{split}_2d_keypoints", f"{split}_2D_keypoints")


def score_pair(csv_path: Path, json_path: Path, split: str) -> tuple[int, int]:
    split_parts = {p.lower() for p in csv_path.parts + json_path.parts}
    token_hits = sum(1 for tok in split_tokens(split) if tok.lower() in split_parts)
    common_prefix = len(csv_path.parents) + len(json_path.parents)
    return token_hits, common_prefix


def resolve_recursive_layout(root: Path, split: str) -> tuple[Path, Path] | None:
    split_l = split.lower()
    csv_candidates = [p for p in root.glob("**/*.csv") if p.is_file() and split_l in p.name.lower()]
    if not csv_candidates:
        return None

    json_candidates = []
    for p in root.glob("**/openpose_output/json"):
        if not p.is_dir():
            continue
        parts_l = [x.lower() for x in p.parts]
        if any(tok.lower() in parts_l for tok in split_tokens(split)):
            json_candidates.append(p)
    if not json_candidates:
        json_candidates = [p for p in root.glob("**/openpose_output/json") if p.is_dir()]
    if not json_candidates:
        return None

    picked_csv = pick_csv(csv_candidates, split)
    if picked_csv is None:
        return None

    best_json = max(json_candidates, key=lambda jp: score_pair(picked_csv, jp, split))
    return picked_csv, best_json


def resolve_split_paths(root: Path, split: str) -> tuple[Path, Path]:
    tried: list[tuple[Path, Path]] = []
    for base in candidate_roots(root):
        flat_csv = base / f"how2sign_{split}.csv"
        flat_json = base / f"{split}_2D_keypoints" / "openpose_output" / "json"
        tried.append((flat_csv, flat_json))
        if flat_csv.exists() and flat_json.exists():
            return flat_csv, flat_json

        sentence_csv = base / "sentence_level" / split / "text" / "en" / "raw_text" / f"how2sign_{split}.csv"
        sentence_json = base / "sentence_level" / split / "rgb_front" / "features" / "openpose_output" / "json"
        tried.append((sentence_csv, sentence_json))
        if sentence_csv.exists() and sentence_json.exists():
            return sentence_csv, sentence_json

        split_root = base / "sentence_level" / split
        if split_root.exists():
            csv_candidates = [p for p in split_root.glob("**/*.csv") if p.is_file()]
            json_candidates = [p for p in split_root.glob("**/openpose_output/json") if p.is_dir()]
            picked_csv = pick_csv(csv_candidates, split)
            if picked_csv is not None and json_candidates:
                return picked_csv, json_candidates[0]

    recursive_match = resolve_recursive_layout(root, split)
    if recursive_match is not None:
        return recursive_match

    tried_msg = " or ".join(f"{csv} + {json_dir}" for csv, json_dir in tried[:6])
    raise FileNotFoundError(
        "Could not resolve How2Sign split layout. "
        f"Tried: {tried_msg}"
    )


def iter_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def validate_split(
    split: str,
    csv_path: Path,
    base_json: Path,
    manifest_path: Path,
    min_frames: int,
    max_rows: int | None,
) -> dict:
    stats = {
        "split": split,
        "csv_path": str(csv_path),
        "keypoint_root": str(base_json),
        "rows_total": 0,
        "rows_missing_sentence_name": 0,
        "rows_missing_sentence_text": 0,
        "rows_missing_seq_dir": 0,
        "rows_insufficient_frames": 0,
        "rows_valid": 0,
        "frames_total_valid": 0,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["split", "row_id", "sentence_name", "sentence", "seq_dir", "num_frames"])

        for row_id, row in iter_rows(csv_path):
            if max_rows is not None and stats["rows_total"] >= max_rows:
                break
            stats["rows_total"] += 1

            sentence_name = (row.get("SENTENCE_NAME") or "").strip()
            sentence_text = (row.get("SENTENCE") or "").strip()

            if not sentence_name:
                stats["rows_missing_sentence_name"] += 1
                continue
            if not sentence_text:
                stats["rows_missing_sentence_text"] += 1
                continue

            seq_dir = base_json / sentence_name
            if not seq_dir.exists():
                stats["rows_missing_seq_dir"] += 1
                continue

            num_frames = sum(1 for _ in seq_dir.glob("*_keypoints.json"))
            if num_frames < min_frames:
                stats["rows_insufficient_frames"] += 1
                continue

            writer.writerow([split, row_id, sentence_name, sentence_text, str(seq_dir), num_frames])
            stats["rows_valid"] += 1
            stats["frames_total_valid"] += num_frames

    stats["avg_frames_per_valid_row"] = round(
        stats["frames_total_valid"] / stats["rows_valid"], 2
    ) if stats["rows_valid"] else 0.0
    stats["manifest_path"] = str(manifest_path)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("HOW2SIGN_ROOT", "D:/DATASETS/How2Sign"),
        help="How2Sign root directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/manifests",
        help="Directory to write split manifests and summary report.",
    )
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"dataset_root": str(root), "min_frames": args.min_frames, "splits": []}

    for split in ("train", "val", "test"):
        csv_path, keypoint_root = resolve_split_paths(root, split)
        manifest_path = out_dir / f"{split}_manifest.tsv"
        split_stats = validate_split(
            split=split,
            csv_path=csv_path,
            base_json=keypoint_root,
            manifest_path=manifest_path,
            min_frames=args.min_frames,
            max_rows=args.max_rows,
        )
        summary["splits"].append(split_stats)
        print(
            f"[{split}] total={split_stats['rows_total']} valid={split_stats['rows_valid']} "
            f"missing_dir={split_stats['rows_missing_seq_dir']} short={split_stats['rows_insufficient_frames']}"
        )

    report_path = out_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

