from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.adapters.lsa64 import load_lsa64
from data.formats import DatasetSplit, SequenceExample
from features.mediapipe_extractor import FEATURE_DIM
from models.lstm import LSTMClassifier


class SequenceDataset(Dataset):
    def __init__(self, examples: list[SequenceExample], label_to_id: dict[str, int]):
        self.examples = examples
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        x = np.asarray(ex.features, dtype=np.float32)
        y = self.label_to_id[ex.label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def _build_label_map(split: DatasetSplit) -> dict[str, int]:
    labels = {ex.label for ex in split.train}
    labels.update(ex.label for ex in split.val)
    labels.update(ex.label for ex in split.test)
    return {label: idx for idx, label in enumerate(sorted(labels))}


def _run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSA64 LSTM classifier")
    parser.add_argument("--dataset-root", required=True, help="Path to LSA64 root (folder containing 'all').")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--cache-dir", default="artifacts/cache/lsa64")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts/lsa64")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_lsa64(
        root=dataset_root,
        seq_len=args.seq_len,
        top_k=args.top_k,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        cache_dir=cache_dir,
        cache_features=args.cache_features,
        normalize=args.normalize,
        seed=args.seed,
    )

    label_to_id = _build_label_map(splits)

    train_ds = SequenceDataset(splits.train, label_to_id)
    val_ds = SequenceDataset(splits.val, label_to_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_classes=len(label_to_id),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    run_tag = (
        f"lsa64_top{args.top_k}_seq{args.seq_len}_h{args.hidden_dim}_l{args.layers}"
        f"_bs{args.batch_size}_lr{args.learning_rate:.0e}"
    )
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), run_dir / "lstm_best.pt")
            with (run_dir / "label_to_id.json").open("w", encoding="utf-8") as f:
                json.dump(label_to_id, f, indent=2)
            with (run_dir / "lstm_meta.json").open("w", encoding="utf-8") as f:
                json.dump({"input_dim": FEATURE_DIM, "sequence_length": args.seq_len}, f, indent=2)

    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
