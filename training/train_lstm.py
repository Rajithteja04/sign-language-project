from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.adapters.wlasl import load_wlasl
from features.mediapipe_extractor import FEATURE_DIM
from models.lstm import LSTMClassifier
from utils.io import load_yaml


def normalize_openpose_like(x: np.ndarray) -> np.ndarray:
    """
    Lightweight per-sequence normalization to keep runtime/training behavior stable.
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def _examples_to_tensors(
    examples,
    label_to_id: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for ex in examples:
        seq = np.asarray(ex.features, dtype=np.float32)
        if seq.ndim != 2:
            continue
        if seq.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {seq.shape[1]}")
        seq = normalize_openpose_like(seq)
        features.append(seq)
        labels.append(label_to_id[ex.label])

    if not features:
        return (
            torch.empty((0, 0, FEATURE_DIM), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )

    x = torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
    assert x.shape[2] == FEATURE_DIM
    return x, y


def _filter_by_topk(train, val, top_k: int):
    counts = Counter(ex.label for ex in train)
    keep = {label for label, _ in counts.most_common(top_k)}
    train_f = [ex for ex in train if ex.label in keep]
    val_f = [ex for ex in val if ex.label in keep]
    return train_f, val_f


def _build_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1
    return cm


def _validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    avg_loss = (total_loss / total) if total else float("inf")
    acc = (correct / total) if total else 0.0
    return avg_loss, acc, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train LSTM on word-level sign data")
    parser.add_argument("--dataset", choices=["wlasl", "how2sign"], default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--cache-dir", default="artifacts/cache/wlasl_features")
    parser.add_argument("--use-bbox-crop", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--save-confusion-matrix", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml("config/default.yaml")
    dataset = args.dataset or cfg.get("dataset", "wlasl")
    if dataset == "how2sign":
        raise RuntimeError(
            "How2Sign sentence-level LSTM training is disabled in this refactor. "
            "Use --dataset wlasl for word-level training."
        )

    root = args.dataset_root or cfg.get("dataset_root", "")
    if not root:
        raise ValueError("dataset_root is required (arg or config/default.yaml)")

    top_k = args.top_k if args.top_k is not None else int(cfg.get("top_k_classes", 100))
    seq_len = int(cfg.get("sequence_length", 30))
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 30))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 16))
    lr = args.learning_rate if args.learning_rate is not None else float(cfg.get("learning_rate", 1e-3))
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else int(cfg.get("lstm_hidden", 256))
    num_layers = args.layers if args.layers is not None else int(cfg.get("lstm_layers", 2))

    print(f"Dataset: {dataset}")
    print(f"Loading WLASL from: {root}")
    splits = load_wlasl(
        root=root,
        top_k=top_k,
        max_samples_per_class=args.max_samples_per_class,
        seq_len=seq_len,
        val_ratio=args.val_ratio,
        cache_features=args.cache_features,
        cache_dir=args.cache_dir,
        use_bbox_crop=args.use_bbox_crop,
    )
    print(f"Loaded splits: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

    train_examples, val_examples = _filter_by_topk(splits.train, splits.val, top_k=top_k)
    labels = sorted({ex.label for ex in train_examples})
    if not labels:
        raise RuntimeError("No labels found after filtering. Check dataset/videos.")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    num_classes = len(label_to_id)
    print(f"Classes kept: {num_classes}")

    x_train, y_train = _examples_to_tensors(train_examples, label_to_id)
    x_val, y_val = _examples_to_tensors(val_examples, label_to_id)
    if x_train.numel() == 0 or x_val.numel() == 0:
        raise RuntimeError("Training or validation tensors are empty.")
    print(
        f"Tensor shapes: train={tuple(x_train.shape)} val={tuple(x_val.shape)} feature_dim={x_train.shape[2]}"
    )

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = -1.0
    best_state = None
    best_cm = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * yb.size(0)
            pred = logits.argmax(dim=1)
            running_correct += int((pred == yb).sum().item())
            running_total += int(yb.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc, y_true, y_pred = _validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_cm = _build_confusion_matrix(y_true, y_pred, num_classes)

    if best_state is None:
        raise RuntimeError("Training completed without a best model.")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, artifacts_dir / "lstm_best.pt")
    with (artifacts_dir / "label_to_id.json").open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)
    meta = {
        "dataset": dataset,
        "feature_dim": FEATURE_DIM,
        "sequence_length": int(x_train.shape[1]),
        "num_classes": num_classes,
        "top_k_classes": top_k,
        "lstm_hidden": hidden_dim,
        "lstm_layers": num_layers,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_val_acc": best_acc,
    }
    with (artifacts_dir / "lstm_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.save_confusion_matrix and best_cm is not None:
        with (artifacts_dir / "confusion_matrix.json").open("w", encoding="utf-8") as f:
            json.dump(best_cm.tolist(), f)
        print("Saved confusion matrix to artifacts/confusion_matrix.json")

    print("Saved artifacts:")
    print("- artifacts/lstm_best.pt")
    print("- artifacts/label_to_id.json")
    print("- artifacts/lstm_meta.json")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
