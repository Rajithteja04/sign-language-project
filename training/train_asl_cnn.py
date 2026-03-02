import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


TRAIN_ROOT = r"D:\personal\Project\Datasets\ASL\asl_alphabet_train\asl_alphabet_train"


class ASLCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def main():
    parser = argparse.ArgumentParser(description="Train CNN on ASL alphabet dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    full_dataset = datasets.ImageFolder(TRAIN_ROOT, transform=transform)
    num_classes = len(full_dataset.classes)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ASLCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
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
                xb = xb.to(device)
                yb = yb.to(device)
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
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training completed without a best model.")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, artifacts_dir / "asl_cnn_best.pt")
    with (artifacts_dir / "asl_alphabet_label_map.json").open("w", encoding="utf-8") as f:
        json.dump(
            {label: int(idx) for idx, label in enumerate(full_dataset.classes)},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Best val acc: {best_val_acc:.4f}")
    print(
        "Run summary: "
        f"total_samples={len(full_dataset)} "
        f"train_samples={len(train_dataset)} "
        f"val_samples={len(val_dataset)} "
        f"best_val_acc={best_val_acc:.4f}"
    )


if __name__ == "__main__":
    main()
