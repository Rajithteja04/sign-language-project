import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mediapipe_extractor import MediaPipeExtractor
from models.lstm import LSTMClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = r"D:\personal\Project\Datasets\ASL\asl_alphabet_train\asl_alphabet_train"
FEATURE_DIM = 411


def _normalize_features(features: np.ndarray) -> np.ndarray:
    coords = features.reshape(-1, 3)
    left_shoulder = coords[11]
    right_shoulder = coords[12]
    torso_center = (left_shoulder + right_shoulder) / 2.0
    coords[:, :2] -= torso_center[:2]
    shoulder_dist = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    shoulder_dist = max(float(shoulder_dist), 1e-6)
    coords[:, :2] /= shoulder_dist
    return coords.flatten()


def extract_features(max_images_per_class: int):
    extractor = MediaPipeExtractor()
    X = []
    y = []

    classes = sorted(os.listdir(DATASET_ROOT))
    for cls in classes:
        cls_path = os.path.join(DATASET_ROOT, cls)
        if not os.path.isdir(cls_path):
            continue

        img_names = [n for n in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, n))]
        random.shuffle(img_names)
        img_names = img_names[:max_images_per_class]

        for img_name in tqdm(img_names, desc=f"Processing {cls}"):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            features = extractor.extract(img)
            if features is None:
                continue

            features = _normalize_features(features)
            X.append(features)
            y.append(cls)

    return np.asarray(X, dtype=np.float32), np.asarray(y)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def main():
    parser = argparse.ArgumentParser(description="Train LSTM on ASL alphabet images")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-images-per-class", type=int, default=1000)
    args = parser.parse_args()

    print("Extracting features...")
    X, y = extract_features(args.max_images_per_class)

    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False
    )

    model = LSTMClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_classes=len(le.classes_),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Training...")
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = yb.size(0)
            train_loss += float(loss.item()) * batch_size
            train_acc += _accuracy(logits, yb) * batch_size
            train_total += batch_size

        train_loss = train_loss / max(train_total, 1)
        train_acc = train_acc / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                batch_size = yb.size(0)
                val_loss += float(loss.item()) * batch_size
                val_acc += _accuracy(logits, yb) * batch_size
                val_total += batch_size

        val_loss = val_loss / max(val_total, 1)
        val_acc = val_acc / max(val_total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifacts_dir / "asl_alphabet_lstm.pt")
    with (artifacts_dir / "asl_alphabet_label_map.json").open("w", encoding="utf-8") as f:
        json.dump(
            {label: int(idx) for idx, label in enumerate(le.classes_)},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        "Run summary: "
        f"total_samples={len(y_encoded)} "
        f"train_samples={len(y_train)} "
        f"val_samples={len(y_val)} "
        f"best_val_acc={best_val_acc:.4f}"
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
