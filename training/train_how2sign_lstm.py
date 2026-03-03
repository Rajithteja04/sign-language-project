import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.adapters.how2sign import load_how2sign
from data.formats import DatasetSplit, SequenceExample
from utils.io import load_config


class How2SignLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float, bidirectional: bool):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        hidden_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(hidden_out_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


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


def _build_overlap_label_map(train_examples, val_examples, top_k: int):
    train_counts = Counter(ex.label for ex in train_examples)
    val_counts = Counter(ex.label for ex in val_examples)

    overlap_labels = [label for label in train_counts if label in val_counts]
    overlap_labels.sort(key=lambda label: (-train_counts[label], -val_counts[label], label))
    keep = overlap_labels[:top_k]
    return {label: i for i, label in enumerate(keep)}


def _filter_by_label(examples, label_to_id):
    return [ex for ex in examples if ex.label in label_to_id]


def _pooled_split_by_label(examples, top_k: int, min_class_count: int, seed: int):
    rng = random.Random(seed)
    by_label = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)

    counts = Counter({label: len(items) for label, items in by_label.items()})
    keep_labels = [label for label, count in counts.most_common() if count >= min_class_count][:top_k]

    train_examples = []
    val_examples = []
    test_examples = []

    for label in keep_labels:
        items = list(by_label[label])
        rng.shuffle(items)
        n = len(items)

        if n == 2:
            train_examples.extend(items[:1])
            val_examples.extend(items[1:2])
            continue

        val_n = max(1, int(round(n * 0.15)))
        test_n = max(1, int(round(n * 0.15)))
        train_n = n - val_n - test_n

        if train_n < 1:
            train_n = 1
            if val_n > test_n and val_n > 1:
                val_n -= 1
            elif test_n > 1:
                test_n -= 1
            elif val_n > 1:
                val_n -= 1
            else:
                test_n = 0

        train_examples.extend(items[:train_n])
        val_examples.extend(items[train_n : train_n + val_n])
        test_examples.extend(items[train_n + val_n : train_n + val_n + test_n])

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    rng.shuffle(test_examples)

    return keep_labels, train_examples, val_examples, test_examples


def _apply_label_map(examples, label_map):
    if not label_map:
        return examples
    remapped = []
    for ex in examples:
        new_label = label_map.get(ex.label, ex.label)
        if new_label is None:
            continue
        if new_label != ex.label:
            remapped.append(SequenceExample(features=ex.features, label=new_label))
        else:
            remapped.append(ex)
    return remapped


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


def _to_examples(items):
    examples = []
    for item in items:
        features = [np.asarray(frame, dtype=np.float32) for frame in item["features"]]
        examples.append(SequenceExample(features=features, label=item["label"]))
    return examples


def _load_chunked_split(cache_dir: Path, split_name: str):
    part_files = sorted(cache_dir.glob(f"{split_name}.part*.pt"))
    legacy_file = cache_dir / f"{split_name}.pt"
    if not part_files and legacy_file.exists():
        part_files = [legacy_file]
    if not part_files:
        raise RuntimeError(f"No cache parts found for split '{split_name}' in: {cache_dir}")

    items = []
    for part_file in part_files:
        part_items = torch.load(part_file, map_location="cpu", weights_only=False)
        items.extend(part_items)
    return _to_examples(items)


def _resolve_chunked_cache_dir(cache_path: Path, payload: dict):
    cache_dir_field = payload.get("cache_dir") if isinstance(payload, dict) else None
    if cache_dir_field:
        raw = Path(cache_dir_field)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(raw)
            candidates.append((cache_path.parent / raw).resolve())
            candidates.append((cache_path.parent / raw.name).resolve())

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

    if cache_path.suffix:
        candidate = cache_path.with_suffix("")
        if candidate.is_dir():
            return candidate
    return None


def _load_splits_from_cache(cache_path: Path) -> DatasetSplit:
    if cache_path.is_dir():
        cache_dir = cache_path
        return DatasetSplit(
            train=_load_chunked_split(cache_dir, "train"),
            val=_load_chunked_split(cache_dir, "val"),
            test=_load_chunked_split(cache_dir, "test"),
        )

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "splits" in payload:
        splits = payload["splits"]
        return DatasetSplit(
            train=_to_examples(splits["train"]),
            val=_to_examples(splits["val"]),
            test=_to_examples(splits["test"]),
        )

    if isinstance(payload, dict) and payload.get("format") == "how2sign_cache_chunked_v1":
        cache_dir = _resolve_chunked_cache_dir(cache_path, payload)
        if cache_dir is None or not cache_dir.exists():
            raise RuntimeError(f"Chunked cache directory not found for index: {cache_path}")
        return DatasetSplit(
            train=_load_chunked_split(cache_dir, "train"),
            val=_load_chunked_split(cache_dir, "val"),
            test=_load_chunked_split(cache_dir, "test"),
        )

    raise RuntimeError(f"Invalid cache file format: {cache_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--local-config", default="config/local.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--split-mode", choices=["official", "pooled"], default="pooled")
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--label-list", default=None, help="Path to newline-separated labels to keep")
    parser.add_argument("--label-map", default=None, help="JSON file mapping original sentences to canonical labels")
    args = parser.parse_args()

    label_whitelist = None
    if args.label_list:
        label_list_path = Path(args.label_list)
        if not label_list_path.exists():
            raise FileNotFoundError(f"Label list not found: {label_list_path}")
        label_whitelist = {line.strip() for line in label_list_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"Label whitelist loaded: {len(label_whitelist)} labels")

    label_map = None
    if args.label_map:
        label_map_path = Path(args.label_map)
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {label_map_path}")
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        print(f"Label map loaded: {len(label_map)} entries")

    cfg = load_config(args.config, args.local_config)
    root = (
        args.dataset_root
        or os.getenv("HOW2SIGN_ROOT")
        or cfg.get("dataset_root", "D:/DATASETS/How2Sign")
    )
    seq_len = int(cfg.get("sequence_length", 30))
    top_k = int(cfg.get("top_k_classes", 200))
    max_samples = cfg.get("max_samples_per_split", 3000)
    out_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    if args.max_samples is not None:
        max_samples = args.max_samples
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.top_k is not None:
        top_k = args.top_k
    if args.seq_len is not None:
        seq_len = args.seq_len
    if args.hidden_dim is not None:
        cfg["lstm_hidden"] = args.hidden_dim
    if args.layers is not None:
        cfg["lstm_layers"] = args.layers
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using dataset root: {root}")
    print(
        "Config: "
        f"hidden_dim={cfg['lstm_hidden']} layers={cfg['lstm_layers']} "
        f"dropout={args.dropout} lr={args.lr} seq_len={seq_len} "
        f"bidirectional={args.bidirectional}"
    )
    if args.cache_path:
        cache_path = Path(args.cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}. "
                "Build cache first with: python -m scripts.build_how2sign_cache"
            )
        print(f"Loading cached splits from: {cache_path}")
        splits = _load_splits_from_cache(cache_path)
        print(
            f"Loaded cached splits: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}"
        )
    else:
        print("Loading How2Sign splits (this can take time on large keypoint sets)...")
        splits = load_how2sign(root=root, max_samples_per_split=max_samples)
        print(
            f"Loaded splits: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} "
            f"(max_samples_per_split={max_samples})"
        )
    if label_map:
        splits = DatasetSplit(
            train=_apply_label_map(splits.train, label_map),
            val=_apply_label_map(splits.val, label_map),
            test=_apply_label_map(splits.test, label_map),
        )
    if args.split_mode == "pooled":
        pooled = list(splits.train) + list(splits.val) + list(splits.test)
        keep_labels, train_examples, val_examples, test_examples = _pooled_split_by_label(
            pooled,
            top_k=top_k,
            min_class_count=args.min_class_count,
            seed=args.seed,
        )
        print("Pooled train label counts:", Counter(ex.label for ex in train_examples))
        print("Pooled val label counts:", Counter(ex.label for ex in val_examples))
        print(
            f"Pooled split mode: pool_size={len(pooled)} kept_labels={len(keep_labels)} "
            f"min_class_count={args.min_class_count} seed={args.seed}"
        )
        print(
            f"Pooled split sizes: train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}"
        )
        if label_whitelist:
            train_examples = [ex for ex in train_examples if ex.label in label_whitelist]
            val_examples = [ex for ex in val_examples if ex.label in label_whitelist]
            test_examples = [ex for ex in test_examples if ex.label in label_whitelist]
            print(
                f"Applied label whitelist: train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}"
            )
    else:
        train_examples = list(splits.train)
        val_examples = list(splits.val)
        if label_whitelist:
            train_examples = [ex for ex in train_examples if ex.label in label_whitelist]
            val_examples = [ex for ex in val_examples if ex.label in label_whitelist]
            print(
                f"Applied label whitelist: train={len(train_examples)} val={len(val_examples)}"
            )

    train_label_count = len({ex.label for ex in train_examples})
    val_label_count = len({ex.label for ex in val_examples})
    overlap_label_count = len({ex.label for ex in train_examples} & {ex.label for ex in val_examples})
    print(f"Label coverage: train_labels={train_label_count} val_labels={val_label_count} overlap={overlap_label_count}")

    label_to_id = _build_overlap_label_map(train_examples, val_examples, top_k=top_k)
    if not label_to_id:
        raise RuntimeError(
            "No train/val overlapping labels found after loading splits. "
            "Increase --max-samples or verify dataset quality/manifests."
        )

    train_examples = _filter_by_label(train_examples, label_to_id)
    val_examples = _filter_by_label(val_examples, label_to_id)
    num_classes = len(label_to_id)
    print(
        f"After class filtering: train={len(train_examples)} val={len(val_examples)} "
        f"num_classes={num_classes} top_k={top_k}"
    )

    if not train_examples:
        raise RuntimeError("No train examples left after class filtering.")
    if not val_examples:
        raise RuntimeError(
            "No validation examples left after class filtering. "
            "Increase --max-samples or lower --top-k to improve overlap."
        )

    train_ds = SequenceDataset(train_examples, label_to_id, seq_len)
    val_ds = SequenceDataset(val_examples, label_to_id, seq_len)
    print(f"Dataset tensors: seq_len={seq_len} train_size={len(train_ds)} val_size={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    inferred_input_dim = train_ds[0][0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = How2SignLSTM(
        input_dim=inferred_input_dim,
        hidden_dim=cfg["lstm_hidden"],
        num_layers=cfg["lstm_layers"],
        num_classes=num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
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
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "lstm_best.pt")
            with open(out_dir / "label_to_id.json", "w", encoding="utf-8") as f:
                json.dump(label_to_id, f, indent=2)
            with open(out_dir / "lstm_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"input_dim": inferred_input_dim, "sequence_length": seq_len},
                    f,
                    indent=2,
                )

    if best_epoch > 0:
        cache_tag = "raw"
        if args.cache_path:
            cache_name = Path(args.cache_path).stem.lower()
            for token in cache_name.split("_"):
                if token.isdigit():
                    cache_tag = token
                    break
        tag = (
            f"how2sign_{cache_tag}_top{top_k}_seq{seq_len}_h{cfg['lstm_hidden']}_l{cfg['lstm_layers']}"
            f"_lr{args.lr}_do{args.dropout}"
            f"{'_bi' if args.bidirectional else ''}"
        )
        snapshot_dir = out_dir / tag
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "lstm_best.pt").write_bytes((out_dir / "lstm_best.pt").read_bytes())
        (snapshot_dir / "label_to_id.json").write_bytes((out_dir / "label_to_id.json").read_bytes())
        (snapshot_dir / "lstm_meta.json").write_bytes((out_dir / "lstm_meta.json").read_bytes())
        print(f"Saved best model to: {out_dir / 'lstm_best.pt'}")
        print(f"Saved snapshot to: {snapshot_dir}")
    else:
        print(f"Saved best model to: {out_dir / 'lstm_best.pt'}")


if __name__ == "__main__":
    main()
