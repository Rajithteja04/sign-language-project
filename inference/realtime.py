import argparse
import cv2
import json
from collections import deque
from pathlib import Path
import numpy as np
import torch

from features.mediapipe_extractor import MediaPipeExtractor
from models.lstm import LSTMClassifier
from models.transformer import TransformerCorrector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="how2sign")
    parser.add_argument("--weights", default="artifacts/lstm_best.pt")
    parser.add_argument("--labels", default="artifacts/label_to_id.json")
    parser.add_argument("--meta", default="artifacts/lstm_meta.json")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--nlp-model", default=None)
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    extractor = MediaPipeExtractor()
    text_corrector = TransformerCorrector(model_name=args.nlp_model)

    weights_path = Path(args.weights)
    labels_path = Path(args.labels)
    meta_path = Path(args.meta)
    model = None
    id_to_label = {}
    model_input_dim = None

    if weights_path.exists() and labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            label_to_id = json.load(f)
        id_to_label = {v: k for k, v in label_to_id.items()}

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_input_dim = int(meta.get("input_dim", 0))
            args.seq_len = int(meta.get("sequence_length", args.seq_len))
        else:
            model_input_dim = 411

        model = LSTMClassifier(
            input_dim=model_input_dim,
            hidden_dim=args.lstm_hidden,
            num_layers=args.lstm_layers,
            num_classes=len(id_to_label),
        )
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

    seq_buffer = deque(maxlen=args.seq_len)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        feats = extractor.extract(frame)
        if model_input_dim is not None:
            if feats.shape[0] > model_input_dim:
                feats = feats[:model_input_dim]
            elif feats.shape[0] < model_input_dim:
                feats = np.pad(feats, (0, model_input_dim - feats.shape[0]))
        seq_buffer.append(feats)

        predicted_gloss = "MODEL_NOT_LOADED"
        if model is not None and len(seq_buffer) == args.seq_len:
            x = torch.tensor(np.stack(seq_buffer), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                pred_id = int(torch.argmax(logits, dim=1).item())
            predicted_gloss = id_to_label.get(pred_id, "UNKNOWN")

        corrected = text_corrector.correct(predicted_gloss)

        cv2.putText(frame, corrected, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Sign Language Translation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
