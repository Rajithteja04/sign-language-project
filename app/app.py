from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template

from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor
from models.lstm import LSTMClassifier
from models.transformer import gloss_to_sentence


def normalize_openpose_like(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


class WordRealtimeService:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.threshold = 0.60
        self.cooldown_seconds = 1.2
        self.vote_window = 5
        self.vote_buffer: deque[str] = deque(maxlen=self.vote_window)
        self.committed_words: list[str] = []
        self.last_emit_ts = 0.0
        self.last_word = ""

        self.current_word = "-"
        self.current_confidence = 0.0
        self.corrected_sentence = ""
        self.status = "Ready. Click Start."
        self.source_mode = "mock"

        self.model = None
        self.id_to_label: dict[int, str] = {}
        self.seq_len = 30
        self.extractor = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        weights = Path("artifacts/lstm_best.pt")
        labels = Path("artifacts/label_to_id.json")
        meta = Path("artifacts/lstm_meta.json")

        if not (weights.exists() and labels.exists() and meta.exists()):
            self.source_mode = "mock"
            self.status = "Model artifacts missing. Running in mock mode."
            return

        try:
            with labels.open("r", encoding="utf-8") as f:
                label_to_id = json.load(f)
            with meta.open("r", encoding="utf-8") as f:
                m = json.load(f)

            self.id_to_label = {int(v): k for k, v in label_to_id.items()}
            self.seq_len = int(m.get("sequence_length", 30))

            model = LSTMClassifier(
                input_dim=int(m["feature_dim"]),
                hidden_dim=int(m["lstm_hidden"]),
                num_layers=int(m["lstm_layers"]),
                num_classes=int(m["num_classes"]),
            )
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state)
            model.eval()

            if int(m.get("feature_dim", FEATURE_DIM)) != FEATURE_DIM:
                self.source_mode = "mock"
                self.status = "Feature mismatch in model metadata. Running in mock mode."
                return

            self.model = model
            self.extractor = MediaPipeExtractor()
            self.source_mode = "live"
            self.status = "Model loaded. Ready."
        except Exception as exc:
            self.source_mode = "mock"
            self.status = f"Model load failed ({exc}). Running in mock mode."

    def _reset_buffers(self) -> None:
        self.vote_buffer.clear()
        self.committed_words.clear()
        self.last_emit_ts = 0.0
        self.last_word = ""
        self.current_word = "-"
        self.current_confidence = 0.0
        self.corrected_sentence = ""

    def reset(self) -> None:
        with self.lock:
            self._reset_buffers()
            self.status = "Cleared."

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self.stop_event.clear()
            self.status = f"Recognition started ({self.source_mode})."

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.running = False
            self.stop_event.set()
            self.status = "Recognition stopped."

    def _accept_prediction(self, pred_word: str, confidence: float) -> None:
        with self.lock:
            self.current_word = pred_word
            self.current_confidence = confidence

            if confidence >= self.threshold:
                self.vote_buffer.append(pred_word)
                voted = Counter(self.vote_buffer).most_common(1)[0][0]
                now = time.time()
                if voted != self.last_word and (now - self.last_emit_ts) >= self.cooldown_seconds:
                    self.committed_words.append(voted)
                    self.last_word = voted
                    self.last_emit_ts = now

            gloss = " ".join(self.committed_words)
            self.corrected_sentence = gloss_to_sentence(gloss) if gloss else ""

    def _mock_loop(self) -> None:
        words = ["HELLO", "I", "GO", "STORE", "THANK", "YOU", "HELP"]
        while not self.stop_event.is_set():
            word = random.choice(words)
            conf = random.uniform(0.45, 0.95)
            self._accept_prediction(word, conf)
            with self.lock:
                self.status = "Running (mock predictions)."
            time.sleep(1.0)

    def _live_loop(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            with self.lock:
                self.status = "Camera open failed. Switched to mock mode."
                self.source_mode = "mock"
            self._mock_loop()
            return

        buffer: deque[np.ndarray] = deque(maxlen=self.seq_len)
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                with self.lock:
                    self.status = "Camera frame read failed."
                time.sleep(0.2)
                continue

            try:
                feat = self.extractor.extract(frame)
                if feat.shape[0] != FEATURE_DIM:
                    raise ValueError(f"Feature dim mismatch: {feat.shape[0]} != {FEATURE_DIM}")
                buffer.append(feat)
            except Exception as exc:
                with self.lock:
                    self.status = f"Feature extraction failed: {exc}"
                time.sleep(0.1)
                continue

            if len(buffer) < self.seq_len:
                with self.lock:
                    self.status = "Collecting gesture frames..."
                continue

            seq = np.asarray(buffer, dtype=np.float32)
            seq = normalize_openpose_like(seq)
            x = torch.tensor(seq[None, ...], dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred_id = torch.max(probs, dim=1)
                pred_word = self.id_to_label.get(int(pred_id.item()), "UNKNOWN")
                confidence = float(conf.item())
            self._accept_prediction(pred_word, confidence)
            with self.lock:
                self.status = "Running (live)."

        cap.release()

    def _loop(self) -> None:
        if self.source_mode == "live":
            self._live_loop()
        else:
            self._mock_loop()

    def state(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "status": self.status,
                "current_word": self.current_word,
                "confidence": self.current_confidence,
                "committed_words": list(self.committed_words),
                "gloss_sequence": " ".join(self.committed_words),
                "corrected_sentence": self.corrected_sentence,
                "mode": self.source_mode,
            }


app = Flask(__name__)
service = WordRealtimeService()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/state")
def state():
    return jsonify(service.state())


@app.route("/start", methods=["POST"])
def start():
    service.start()
    return jsonify({"ok": True, "state": service.state()})


@app.route("/stop", methods=["POST"])
def stop():
    service.stop()
    return jsonify({"ok": True, "state": service.state()})


@app.route("/reset", methods=["POST"])
def reset():
    service.reset()
    return jsonify({"ok": True, "state": service.state()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
