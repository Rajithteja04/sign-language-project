import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np

from data.adapters.how2sign import load_how2sign
from models.lstm import LSTMClassifier
from utils.io import load_yaml


class SequenceDataset(Dataset):
    def __init__(self, examples, label_to_id, seq_len: int):
        self.examples = examples
        self.label_to_id = label_to_id
        self.seq_len = seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x = torch.from_numpy(np.asarray(ex.features, dtype=np.float32))
        feat_dim = x.shape[1]
        if x.shape[0] >= self.seq_len:
            x = x[: self.seq_len]
        else:
            pad = torch.zeros((self.seq_len - x.shape[0], feat_dim), dtype=torch.float32)
            x = torch.cat([x, pad], dim=0)
        y = torch.tensor(self.label_to_id[ex.label], dtype=torch.long)
        return x, y


def _build_label_map(train_examples, top_k: int):
    counts = Counter(ex.label for ex in train_examples)
    keep = [label for label, _ in counts.most_common(top_k)]
    label_to_id = {label: i for i, label in enumerate(keep)}
    return label_to_id


def _filter_by_label(examples, label_to_id):
    return [ex for ex in examples if ex.label in label_to_id]


def _run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
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
        total_correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = cfg.get("dataset_root", "datasets")
    seq_len = int(cfg.get("sequence_length", 30))
    top_k = int(cfg.get("top_k_classes", 200))
    max_samples = cfg.get("max_samples_per_split", 3000)
    out_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    if args.dataset_root is not None:
        root = args.dataset_root
    if args.max_samples is not None:
        max_samples = args.max_samples
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_how2sign(root=root, max_samples_per_split=max_samples)
    label_to_id = _build_label_map(splits.train, top_k=top_k)
    if not label_to_id:
        raise RuntimeError("No labels found in training split.")

    train_examples = _filter_by_label(splits.train, label_to_id)
    val_examples = _filter_by_label(splits.val, label_to_id)
    num_classes = len(label_to_id)

    if not train_examples:
        raise RuntimeError("No train examples left after class filtering.")

    train_ds = SequenceDataset(train_examples, label_to_id, seq_len)
    val_ds = SequenceDataset(val_examples, label_to_id, seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    inferred_input_dim = train_ds[0][0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_dim=inferred_input_dim,
        hidden_dim=cfg["lstm_hidden"],
        num_layers=cfg["lstm_layers"],
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = _evaluate(model, val_loader, criterion, device) if len(val_ds) else (0.0, 0.0)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_dir / "lstm_best.pt")
            with open(out_dir / "label_to_id.json", "w", encoding="utf-8") as f:
                json.dump(label_to_id, f, indent=2)
            with open(out_dir / "lstm_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"input_dim": inferred_input_dim, "sequence_length": seq_len},
                    f,
                    indent=2,
                )

    print(f"Saved best model to: {out_dir / 'lstm_best.pt'}")


if __name__ == "__main__":
    main()
