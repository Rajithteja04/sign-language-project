from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from features.mediapipe_extractor import MediaPipeExtractor

try:
    from .run_video_inference import (
        _format_label,
        extract_sequence,
        load_artifacts,
        sample_sequence,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.run_video_inference import (  # type: ignore
        _format_label,
        extract_sequence,
        load_artifacts,
        sample_sequence,
    )


def prompt_int(message: str, default: int = 1) -> int:
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Enter a positive integer.")


def prompt_path(message: str) -> Path:
    while True:
        raw = input(f"{message}: ").strip().strip('"')
        if not raw:
            print("Please enter a path.")
            continue
        path = Path(raw)
        if path.exists():
            return path
        print("File does not exist, try again.")


def main() -> None:
    print("Module 2 – Temporal Recognition Demo")
    print("Using artifacts from artifacts/lsa64/lsa64_top64_seq30_h128_l2_bs16_lr5e-04\n")

    artifacts_dir = Path("artifacts/lsa64/lsa64_top64_seq30_h128_l2_bs16_lr5e-04")
    if not artifacts_dir.exists():
        print(f"Artifacts directory not found: {artifacts_dir}")
        return

    try:
        model, id_to_label, seq_len, normalize, _cache = load_artifacts(artifacts_dir)
    except Exception as exc:
        print(f"Failed to load artifacts: {exc}")
        return

    extractor = MediaPipeExtractor()
    video_count = prompt_int("How many videos do you want to test?", default=1)

    for idx in range(1, video_count + 1):
        video_path = prompt_path(f"Video path #{idx}")
        try:
            frames = extract_sequence(video_path, extractor, normalize)
        except Exception as exc:
            print(f"[{video_path.name}] Extraction failed: {exc}")
            continue

        seq = np.vstack(frames)
        seq = sample_sequence(seq, seq_len)

        x = torch.from_numpy(seq[None, ...]).float()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]

        top_vals, top_ids = torch.topk(probs, k=min(3, probs.shape[0]))
        pred_id = int(top_ids[0].item())
        pred_label = id_to_label.get(pred_id, "UNKNOWN")
        pretty = _format_label(pred_label)

        print(f"\nResult for {video_path.name}:")
        print(f"  Word: {pretty}  (confidence {float(top_vals[0]):.3f})")
        print("  Top candidates:")
        for score, idx_tensor in zip(top_vals.tolist(), top_ids.tolist()):
            raw = id_to_label.get(int(idx_tensor), str(idx_tensor))
            label = _format_label(raw)
            print(f"    {label:>20s}: {score:.3f}")
        print()


if __name__ == "__main__":
    main()
