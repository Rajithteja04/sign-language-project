import argparse
import csv
import json
import os
from pathlib import Path


def resolve_split_paths(root: Path, split: str) -> tuple[Path, Path]:
    flat_csv = root / f"how2sign_{split}.csv"
    flat_json = root / f"{split}_2D_keypoints" / "openpose_output" / "json"
    if flat_csv.exists() and flat_json.exists():
        return flat_csv, flat_json

    sentence_csv = root / "sentence_level" / split / "text" / "en" / "raw_text" / f"how2sign_{split}.csv"
    sentence_json = root / "sentence_level" / split / "rgb_front" / "features" / "openpose_output" / "json"
    if sentence_csv.exists() and sentence_json.exists():
        return sentence_csv, sentence_json

    raise FileNotFoundError(
        "Could not resolve How2Sign split layout. "
        f"Tried: {flat_csv} + {flat_json} and {sentence_csv} + {sentence_json}"
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
        default=os.getenv("HOW2SIGN_ROOT", "datasets"),
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
