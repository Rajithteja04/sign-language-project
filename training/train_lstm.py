import torch
from models.lstm import LSTMClassifier
from utils.io import load_yaml


def main():
    cfg = load_yaml("config/default.yaml")
    model = LSTMClassifier(
        input_dim=cfg["feature_dim"],
        hidden_dim=cfg["lstm_hidden"],
        num_layers=cfg["lstm_layers"],
        num_classes=cfg["num_classes"],
    )
    print(model)
    # TODO: dataset, dataloader, training loop


if __name__ == "__main__":
    main()
