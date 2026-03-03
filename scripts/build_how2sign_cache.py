import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from data.adapters.how2sign import load_how2sign
from utils.io import load_config
from utils.keypoint_format import FEATURE_DIM, TOTAL_POINTS


def _serialize_examples(examples, dtype):
    out = []
    for ex in examples:
        out.append(
            {
                "label": ex.label,
                "features": [np.asarray(frame, dtype=dtype) for frame in ex.features],
            }
        )
    return out


def _write_split_parts(split_name, examples, cache_dir: Path, part_size: int, dtype):
    serialized = _serialize_examples(examples, dtype=dtype)
    total = len(serialized)
    part_count = 0
    for start in range(0, total, part_size):
        part_items = serialized[start : start + part_size]
        part_path = cache_dir / f"{split_name}.part{part_count:04d}.pt"
        torch.save(part_items, part_path)
        print(f"Saved {split_name} part {part_count + 1}: {part_path.name} ({len(part_items)} items)")
        part_count += 1
    return total, part_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--local-config", default="config/local.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", default="artifacts/cache/how2sign_cache.pt")
    parser.add_argument("--part-size", type=int, default=250)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    cfg = load_config(args.config, args.local_config)
    root = args.dataset_root or os.getenv("HOW2SIGN_ROOT") or cfg.get("dataset_root", "datasets")
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples_per_split", 3000)

    out_path = Path(args.out)
    cache_dir = out_path.with_suffix("") if out_path.suffix else out_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.float16 if args.dtype == "float16" else np.float32

    print(f"Building How2Sign cache from root: {root}")
    print(f"Max samples per split: {max_samples}")
    print(f"Cache mode: chunked | part_size={args.part_size} | dtype={args.dtype}")
    splits = load_how2sign(root=root, max_samples_per_split=max_samples)
    print(f"Loaded: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

    counts = {}
    parts = {}
    for split_name, split_examples in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        c, p = _write_split_parts(
            split_name,
            split_examples,
            cache_dir=cache_dir,
            part_size=args.part_size,
            dtype=dtype,
        )
        counts[split_name] = c
        parts[split_name] = p

    meta = {
        "format": "how2sign_cache_chunked_v1",
        "dataset_root": str(root),
        "max_samples_per_split": max_samples,
        "feature_dim": FEATURE_DIM,
        "total_points": TOTAL_POINTS,
        "feature_dtype": args.dtype,
        "part_size": args.part_size,
        "counts": counts,
        "parts": parts,
        "cache_dir": str(cache_dir),
    }

    meta_path = cache_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if out_path.suffix:
        torch.save(
            {
                "format": "how2sign_cache_chunked_v1",
                "cache_dir": str(cache_dir),
                "meta_file": str(meta_path),
            },
            out_path,
        )
        print(f"Saved cache index: {out_path}")

    print(f"Saved chunked cache directory: {cache_dir}")


if __name__ == "__main__":
    main()
