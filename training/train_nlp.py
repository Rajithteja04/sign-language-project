import argparse
import csv
import random
from pathlib import Path

from data.adapters.how2sign import load_how2sign


def _degrade_text(text: str) -> str:
    words = text.strip().split()
    if not words:
        return text
    keep = [w.lower().strip(".,!?;:") for w in words]
    if len(keep) > 4:
        # Simulate rough gloss-like order/noise for correction training pairs.
        mid = keep[1:-1]
        random.shuffle(mid)
        keep = [keep[0], *mid, keep[-1]]
    return " ".join(keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--out", default="artifacts/nlp_pairs.tsv")
    args = parser.parse_args()

    split = load_how2sign(args.dataset_root, max_samples_per_split=args.max_samples).train
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source_text", "target_text"])
        for ex in split:
            src = _degrade_text(ex.label)
            tgt = ex.label.strip()
            if src and tgt:
                writer.writerow([src, tgt])

    print(f"Saved NLP training pairs: {out_path}")
    print("Use this file to fine-tune your preferred Transformer seq2seq model.")


if __name__ == "__main__":
    main()
