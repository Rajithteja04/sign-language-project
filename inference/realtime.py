from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor
from models.lstm import LSTMClassifier
from models.transformer import correct_text


def normalize_openpose_like(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def _load_artifacts(weights: str, labels: str, meta: str):
    with open(labels, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    with open(meta, "r", encoding="utf-8") as f:
        model_meta = json.load(f)

    id_to_label = {int(v): k for k, v in label_to_id.items()}
    model = LSTMClassifier(
        input_dim=int(model_meta["feature_dim"]),
        hidden_dim=int(model_meta["lstm_hidden"]),
        num_layers=int(model_meta["lstm_layers"]),
        num_classes=int(model_meta["num_classes"]),
    )
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, id_to_label, model_meta


def main():
    parser = argparse.ArgumentParser(description="Realtime word-level sign inference")
    parser.add_argument("--weights", default="artifacts/lstm_best.pt")
    parser.add_argument("--labels", default="artifacts/label_to_id.json")
    parser.add_argument("--meta", default="artifacts/lstm_meta.json")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--cooldown-seconds", type=float, default=1.2)
    args = parser.parse_args()

    weights = Path(args.weights)
    labels = Path(args.labels)
    meta = Path(args.meta)
    if not (weights.exists() and labels.exists() and meta.exists()):
        raise FileNotFoundError("Missing artifacts. Expected weights/labels/meta files.")

    model, id_to_label, model_meta = _load_artifacts(str(weights), str(labels), str(meta))
    seq_len = int(model_meta.get("sequence_length", 30))
    if int(model_meta.get("feature_dim", FEATURE_DIM)) != FEATURE_DIM:
        raise ValueError(
            f"Feature mismatch: meta has {model_meta.get('feature_dim')} but runtime expects {FEATURE_DIM}"
        )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    extractor = MediaPipeExtractor()
    buffer = deque(maxlen=seq_len)
    pred_history = deque(maxlen=args.vote_window)
    committed_words = []
    last_emit_ts = 0.0
    last_word = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        feat = extractor.extract(frame)
        if feat.shape[0] != FEATURE_DIM:
            raise ValueError(f"Runtime feature mismatch: {feat.shape[0]} != {FEATURE_DIM}")
        buffer.append(feat)

        status = "Collecting gesture frames..."
        current_pred = "-"
        current_conf = 0.0

        if len(buffer) == seq_len:
            seq = np.asarray(buffer, dtype=np.float32)
            seq = normalize_openpose_like(seq)
            x = torch.tensor(seq[None, ...], dtype=torch.float32)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred_id = torch.max(probs, dim=1)
                current_conf = float(conf.item())
                current_pred = id_to_label.get(int(pred_id.item()), "unknown")

            if current_conf >= args.confidence_threshold:
                pred_history.append(current_pred)
                voted = Counter(pred_history).most_common(1)[0][0]
                now = time.time()
                if voted != last_word and (now - last_emit_ts) >= args.cooldown_seconds:
                    committed_words.append(voted)
                    last_word = voted
                    last_emit_ts = now
                status = f"Detected: {voted}"
            else:
                status = "Low confidence - hold steady sign."

        sentence = correct_text(" ".join(committed_words)) if committed_words else ""

        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"pred={current_pred} conf={current_conf:.3f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"words={' '.join(committed_words) if committed_words else '-'}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"sentence={sentence if sentence else '-'}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Sign Language Translation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            committed_words.clear()
            pred_history.clear()
            last_word = ""
            last_emit_ts = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
