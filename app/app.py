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
from flask import Flask, Response, jsonify, render_template

from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor
from models.lstm import LSTMClassifier
from models.transformer import gloss_to_sentence


def normalize_openpose_like(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


class WordRealtimeService:
    def __init__(self, camera_manager) -> None:
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
        self.class_labels: list[str] = []
        self.system_info = {
            "dataset": "How2Sign",
            "train_samples": 192,
            "val_samples": 41,
            "model": "BiLSTM (128 hidden units)",
            "sequence_length": 30,
        }
        self.seq_len = 30
        self.extractor = MediaPipeExtractor()
        self.camera = camera_manager
        self.frame_counter = 0
        self.debug_module1 = {
            "frame_count": 0,
            "frame_width": 0,
            "frame_height": 0,
            "pose_landmarks": 0,
            "face_landmarks": 0,
            "left_hand_landmarks": 0,
            "right_hand_landmarks": 0,
            "total_landmarks": 0,
            "feature_dimension": FEATURE_DIM,
            "normalization_applied": True,
            "feature_vector_generated": False,
        }

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
            self.class_labels = [label for label, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]
            self.seq_len = int(m.get("sequence_length", 30))
            self.system_info["sequence_length"] = self.seq_len
            self.system_info["model"] = "BiLSTM (128 hidden units)"

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
            self.source_mode = "live"
            self.status = "Model loaded. Ready."
        except Exception as exc:
            self.source_mode = "mock"
            self.status = f"Model load failed ({exc}). Running in mock mode."
            self.class_labels = []

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

    def _camera_loop(self) -> None:
        buffer: deque[np.ndarray] = deque(maxlen=self.seq_len)
        mock_words = ["HELLO", "I", "GO", "STORE", "THANK", "YOU", "HELP"]
        last_mock_emit = 0.0
        while not self.stop_event.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                with self.lock:
                    self.status = "Camera frame read failed."
                time.sleep(0.2)
                continue

            try:
                feat, module1 = self.extractor.extract_with_debug(frame)
                if feat.shape[0] != FEATURE_DIM:
                    raise ValueError(f"Feature dim mismatch: {feat.shape[0]} != {FEATURE_DIM}")
                buffer.append(feat)
                with self.lock:
                    self.frame_counter += 1
                    self.debug_module1 = {
                        "frame_count": self.frame_counter,
                        "frame_width": int(frame.shape[1]),
                        "frame_height": int(frame.shape[0]),
                        "pose_landmarks": int(module1["pose_landmarks"]),
                        "face_landmarks": int(module1["face_landmarks"]),
                        "left_hand_landmarks": int(module1["left_hand_landmarks"]),
                        "right_hand_landmarks": int(module1["right_hand_landmarks"]),
                        "total_landmarks": int(module1["total_landmarks"]),
                        "feature_dimension": FEATURE_DIM,
                        "normalization_applied": True,
                        "feature_vector_generated": True,
                    }
            except Exception as exc:
                with self.lock:
                    self.status = f"Feature extraction failed: {exc}"
                    self.debug_module1["feature_vector_generated"] = False
                time.sleep(0.1)
                continue

            if len(buffer) < self.seq_len:
                with self.lock:
                    self.status = "Collecting gesture frames..."
                continue

            if self.model is not None:
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
            else:
                now = time.time()
                # In mock mode only classification is simulated; extraction/debug stays real.
                if now - last_mock_emit >= 1.0:
                    word = random.choice(mock_words)
                    conf = random.uniform(0.45, 0.95)
                    self._accept_prediction(word, conf)
                    last_mock_emit = now
                with self.lock:
                    self.status = "Running (mock classification, real extraction)."

    def _loop(self) -> None:
        self._camera_loop()

    def state(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "status": self.status,
                "model_status": "Model loaded successfully." if self.source_mode == "live" else "Running in mock mode (no trained model loaded).",
                "mock": self.source_mode != "live",
                "current_word": self.current_word,
                "confidence": self.current_confidence,
                "committed_words": list(self.committed_words),
                "gloss_sequence": " ".join(self.committed_words),
                "corrected_sentence": self.corrected_sentence,
                "mode": self.source_mode,
                "class_labels": list(self.class_labels),
                "system_info": dict(self.system_info),
                "debug_module1": dict(self.debug_module1),
            }


class CameraManager:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self._cap = None
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            with self._lock:
                self._latest_frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


app = Flask(__name__)
camera_manager = CameraManager(camera_index=0)
camera_manager.start()
service = WordRealtimeService(camera_manager)


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


def _mjpeg_stream():
    while True:
        frame = camera_manager.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        data = jpg.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
