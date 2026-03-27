from __future__ import annotations

import datetime as dt
import json
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template

from features.mediapipe_extractor import FEATURE_DIM, MediaPipeExtractor
from models.lstm import LSTMClassifier
from models.transformer import words_to_sentence


def normalize_openpose_like(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


class CameraManager:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.cap: cv2.VideoCapture | None = None
        self.latest_frame: np.ndarray | None = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        self._frame_times: deque[float] = deque(maxlen=60)
        self.status = "Camera not started."

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.latest_frame = None
            self.status = "Camera stopped."

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        with self.lock:
            self.cap = cap
            self.status = "Camera active." if cap.isOpened() else f"Camera open failed (index {self.camera_index})."
        if not cap.isOpened():
            return

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                with self.lock:
                    self.status = "Camera frame read failed."
                time.sleep(0.05)
                continue

            ts = time.perf_counter()
            self._frame_times.append(ts)
            with self.lock:
                self.latest_frame = frame
                self.frame_height, self.frame_width = frame.shape[:2]
                if len(self._frame_times) >= 2:
                    span = self._frame_times[-1] - self._frame_times[0]
                    self.fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
                self.status = "Camera active."

        cap.release()

    def get_frame(self) -> np.ndarray | None:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "width": self.frame_width,
                "height": self.frame_height,
                "fps": float(self.fps),
                "status": self.status,
                "ready": self.latest_frame is not None,
            }


class WordRealtimeService:
    LEFT_START = 75 + 210
    LEFT_END = LEFT_START + 63
    RIGHT_START = LEFT_END
    RIGHT_END = RIGHT_START + 63

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        # Module-2 stability controls
        self.threshold = 0.82
        self.margin_threshold = 0.20
        self.cooldown_seconds = 1.8
        self.vote_window = 7
        self.min_hand_frames = 26
        self.motion_threshold = 0.014

        self.vote_buffer: deque[str] = deque(maxlen=self.vote_window)
        self.committed_words: list[str] = []
        self.last_emit_ts = 0.0
        self.last_word = ""

        self.current_word = "-"
        self.current_confidence = 0.0
        self.current_margin = 0.0
        self.corrected_sentence = ""
        self.status = "Ready. Click Start."
        self.source_mode = "error"
        self.model_status = "Model artifacts not loaded."
        self.last_update_utc = dt.datetime.now(dt.timezone.utc)

        self.model: LSTMClassifier | None = None
        self.id_to_label: dict[int, str] = {}
        self.seq_len = 30
        self.extractor: MediaPipeExtractor | None = None
        self.processed_frames = 0

        self.camera = CameraManager(camera_index=0)
        self.camera.start()
        self._init_extractor()
        self._load_artifacts()

    def _init_extractor(self) -> None:
        try:
            self.extractor = MediaPipeExtractor()
        except Exception as exc:
            self.source_mode = "error"
            self.model_status = f"MediaPipe init failed: {exc}"
            self.status = "Cannot start recognition until MediaPipe is fixed."

    @staticmethod
    def _infer_num_layers(state: dict[str, torch.Tensor]) -> int:
        indices = []
        for key in state:
            if key.startswith("lstm.weight_ih_l"):
                suffix = key.split("lstm.weight_ih_l", 1)[1]
                if suffix.isdigit():
                    indices.append(int(suffix))
        return (max(indices) + 1) if indices else 2

    @staticmethod
    def _infer_hidden_dim(state: dict[str, torch.Tensor]) -> int:
        w = state.get("lstm.weight_ih_l0")
        if w is None:
            return 128
        return int(w.shape[0] // 4)

    def _load_artifacts(self) -> None:
        weights = Path("artifacts/lstm_best.pt")
        labels = Path("artifacts/label_to_id.json")
        meta = Path("artifacts/lstm_meta.json")

        if not (weights.exists() and labels.exists() and meta.exists()):
            self.source_mode = "error"
            self.model_status = "Model artifacts missing. Place lstm_best.pt, label_to_id.json, lstm_meta.json in artifacts/."
            self.status = "Cannot start recognition until model artifacts are available."
            return

        try:
            with labels.open("r", encoding="utf-8") as f:
                label_to_id = json.load(f)
            with meta.open("r", encoding="utf-8") as f:
                meta_data = json.load(f)

            state: dict[str, torch.Tensor] = torch.load(weights, map_location="cpu")
            self.id_to_label = {int(v): k for k, v in label_to_id.items()}
            self.seq_len = int(meta_data.get("sequence_length", 30))

            feature_dim = int(meta_data.get("feature_dim", meta_data.get("input_dim", FEATURE_DIM)))
            if feature_dim != FEATURE_DIM:
                raise ValueError(f"Feature mismatch: meta={feature_dim}, runtime={FEATURE_DIM}")

            num_classes = int(meta_data.get("num_classes", len(self.id_to_label)))
            if num_classes <= 0:
                fc_weight = state.get("fc.weight")
                if fc_weight is None:
                    raise ValueError("Unable to infer num_classes from metadata/state.")
                num_classes = int(fc_weight.shape[0])

            hidden_dim = int(
                meta_data.get(
                    "lstm_hidden",
                    meta_data.get("hidden_dim", self._infer_hidden_dim(state)),
                )
            )
            num_layers = int(
                meta_data.get(
                    "lstm_layers",
                    meta_data.get("layers", self._infer_num_layers(state)),
                )
            )

            model = LSTMClassifier(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
            )
            model.load_state_dict(state)
            model.eval()
            self.model = model

            self.source_mode = "live"
            self.model_status = "Model loaded successfully (real mode)."
            self.status = "Model ready. Click Start."
        except Exception as exc:
            self.source_mode = "error"
            self.model_status = f"Model load failed: {exc}"
            self.status = "Cannot start recognition. Fix artifact mismatch and reload."

    @staticmethod
    def _has_hand_signal(hand_vec: np.ndarray) -> bool:
        return bool(np.any(np.abs(hand_vec) > 1e-6))

    def _hand_presence_and_motion(self, feat: np.ndarray, prev_hand: np.ndarray | None) -> tuple[bool, float, np.ndarray]:
        left = feat[self.LEFT_START:self.LEFT_END]
        right = feat[self.RIGHT_START:self.RIGHT_END]
        hand = np.concatenate([left, right], axis=0)

        has_hand = self._has_hand_signal(left) or self._has_hand_signal(right)
        if prev_hand is None:
            return has_hand, 0.0, hand

        active = (np.abs(hand) > 1e-6) | (np.abs(prev_hand) > 1e-6)
        if not np.any(active):
            return has_hand, 0.0, hand

        motion = float(np.mean(np.abs(hand[active] - prev_hand[active])))
        return has_hand, motion, hand

    def _set_idle(self, message: str) -> None:
        with self.lock:
            self.current_word = "-"
            self.current_confidence = 0.0
            self.current_margin = 0.0
            self.vote_buffer.clear()
            self.status = message
            self.last_update_utc = dt.datetime.now(dt.timezone.utc)

    def _reset_buffers(self) -> None:
        self.vote_buffer.clear()
        self.committed_words.clear()
        self.last_emit_ts = 0.0
        self.last_word = ""
        self.current_word = "-"
        self.current_confidence = 0.0
        self.current_margin = 0.0
        self.corrected_sentence = ""

    def reset(self) -> None:
        with self.lock:
            self._reset_buffers()
            self.status = "Cleared."

    def reload_model(self) -> None:
        with self.lock:
            was_running = self.running
            self.running = False
            self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self.lock:
            self.stop_event = threading.Event()
            self._reset_buffers()
            self.model = None
            self.id_to_label = {}
            self.source_mode = "error"
            self.model_status = "Reloading model..."
        self._load_artifacts()
        if was_running and self.source_mode == "live":
            self.start()

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            if self.source_mode != "live" or self.model is None or self.extractor is None:
                self.status = "Cannot start: model not loaded."
                return
            self.running = True
            self.stop_event.clear()
            self.status = "Recognition started."

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.running = False
            self.stop_event.set()
            self.status = "Recognition stopped."

    def _accept_prediction(self, pred_word: str, confidence: float, margin: float) -> None:
        with self.lock:
            self.current_margin = margin
            self.last_update_utc = dt.datetime.now(dt.timezone.utc)

            if confidence < self.threshold:
                self.current_word = "-"
                self.current_confidence = confidence
                self.status = f"Low confidence ({confidence:.3f})."
                self.vote_buffer.clear()
                return

            if margin < self.margin_threshold:
                self.current_word = "-"
                self.current_confidence = confidence
                self.status = f"Ambiguous sign (margin {margin:.3f})."
                self.vote_buffer.clear()
                return

            self.current_word = pred_word
            self.current_confidence = confidence
            self.vote_buffer.append(pred_word)
            voted = Counter(self.vote_buffer).most_common(1)[0][0]
            now = time.time()
            if voted != self.last_word and (now - self.last_emit_ts) >= self.cooldown_seconds:
                self.committed_words.append(voted)
                self.last_word = voted
                self.last_emit_ts = now
                self.corrected_sentence = words_to_sentence(self.committed_words)

    def _loop(self) -> None:
        assert self.model is not None
        assert self.extractor is not None

        buffer: deque[np.ndarray] = deque(maxlen=self.seq_len)
        hand_flags: deque[bool] = deque(maxlen=self.seq_len)
        motions: deque[float] = deque(maxlen=self.seq_len)

        prev_hand: np.ndarray | None = None
        no_hand_streak = 0

        while not self.stop_event.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                self._set_idle("Waiting for camera...")
                time.sleep(0.05)
                continue

            try:
                feat = self.extractor.extract(frame)
                if feat.shape[0] != FEATURE_DIM:
                    raise ValueError(f"Feature dim mismatch: {feat.shape[0]} != {FEATURE_DIM}")
                has_hand, motion, prev_hand = self._hand_presence_and_motion(feat, prev_hand)
            except Exception as exc:
                self._set_idle(f"Feature extraction failed: {exc}")
                time.sleep(0.05)
                continue

            with self.lock:
                self.processed_frames += 1

            if not has_hand:
                no_hand_streak += 1
                if no_hand_streak >= 3:
                    buffer.clear()
                    hand_flags.clear()
                    motions.clear()
                self._set_idle("Show a hand sign to start recognition.")
                continue

            no_hand_streak = 0
            buffer.append(feat)
            hand_flags.append(has_hand)
            motions.append(motion)

            if len(buffer) < self.seq_len:
                self._set_idle("Collecting gesture frames...")
                continue

            if sum(1 for x in hand_flags if x) < self.min_hand_frames:
                self._set_idle("Keep hand visible for stable recognition.")
                continue

            avg_motion = float(np.mean(motions)) if motions else 0.0
            if avg_motion < self.motion_threshold:
                self._set_idle("Move the hand gesture clearly.")
                continue

            seq = np.asarray(buffer, dtype=np.float32)
            seq = normalize_openpose_like(seq)
            x = torch.tensor(seq[None, ...], dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred_id = torch.max(probs, dim=1)
                top_vals, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
                margin = float(top_vals[0, 0].item() - top_vals[0, 1].item()) if top_vals.shape[1] == 2 else float(top_vals[0, 0].item())
                pred_word = self.id_to_label.get(int(pred_id.item()), "UNKNOWN")
                confidence = float(conf.item())

            self._accept_prediction(pred_word, confidence, margin)
            with self.lock:
                if self.current_word != "-":
                    self.status = "Running (real mode)."

    def state(self) -> dict[str, Any]:
        cam = self.camera.state()
        with self.lock:
            return {
                "running": self.running,
                "status": self.status,
                "model_status": self.model_status,
                "current_word": self.current_word,
                "confidence": self.current_confidence,
                "margin": self.current_margin,
                "committed_words": list(self.committed_words),
                "gloss_sequence": " ".join(self.committed_words),
                "corrected_sentence": self.corrected_sentence,
                "mode": self.source_mode,
                "fps": cam["fps"],
                "recognized_count": len(self.committed_words),
                "camera_status": cam["status"],
                "preview_ready": cam["ready"],
                "timestamp": self.last_update_utc.isoformat(),
            }


app = Flask(__name__)
service = WordRealtimeService()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/state")
def state():
    return jsonify(service.state())


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = service.camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg = encoded.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


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


@app.route("/reload_model", methods=["POST"])
def reload_model():
    service.reload_model()
    return jsonify({"ok": True, "state": service.state()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
