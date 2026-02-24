from __future__ import annotations

import json
import os
import random
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO

from features.mediapipe_extractor import MediaPipeExtractor
from models.lstm import LSTMClassifier
from models.transformer import TransformerCorrector
from utils.io import load_config

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_runtime = {
    "background_started": False,
    "mode": "mock",
    "model_loaded": False,
    "last_prediction": None,
    "status": "idle",
}
_runtime_lock = threading.Lock()


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_runtime_settings() -> dict:
    cfg = load_config("config/default.yaml", "config/local.yaml")

    def env_or_cfg(env_key: str, cfg_key: str, default: object) -> object:
        return os.getenv(env_key, cfg.get(cfg_key, default))

    return {
        "camera_index": int(env_or_cfg("CAMERA_INDEX", "camera_index", 0)),
        "emit_interval_ms": int(env_or_cfg("EMIT_INTERVAL_MS", "emit_interval_ms", 1000)),
        "use_mock_inference": _is_true(env_or_cfg("USE_MOCK_INFERENCE", "use_mock_inference", True)),
        "weights": str(env_or_cfg("LSTM_WEIGHTS", "lstm_weights_path", "artifacts/lstm_best.pt")),
        "labels": str(env_or_cfg("LSTM_LABELS", "labels_path", "artifacts/label_to_id.json")),
        "meta": str(env_or_cfg("LSTM_META", "meta_path", "artifacts/lstm_meta.json")),
        "lstm_hidden": int(cfg.get("lstm_hidden", 256)),
        "lstm_layers": int(cfg.get("lstm_layers", 2)),
        "nlp_model": str(cfg.get("nlp_model", "")).strip() or None,
    }


def _load_model_bundle(settings: dict) -> dict | None:
    weights_path = Path(settings["weights"])
    labels_path = Path(settings["labels"])
    meta_path = Path(settings["meta"])

    if not (weights_path.exists() and labels_path.exists() and meta_path.exists()):
        return None

    with open(labels_path, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    input_dim = int(meta.get("input_dim", 411))
    seq_len = int(meta.get("sequence_length", 30))

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=settings["lstm_hidden"],
        num_layers=settings["lstm_layers"],
        num_classes=len(id_to_label),
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    return {
        "model": model,
        "id_to_label": id_to_label,
        "input_dim": input_dim,
        "seq_len": seq_len,
    }


def _build_payload(mode: str, text: str, confidence: float, raw_text: str | None = None) -> dict:
    return {
        "mode": mode,
        "text": text,
        "raw_text": raw_text or text,
        "confidence": round(float(confidence), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _set_status(message: str) -> None:
    with _runtime_lock:
        _runtime["status"] = message
    socketio.emit("status", {"message": message})


def _capture_loop() -> None:
    settings = _load_runtime_settings()
    use_mock = settings["use_mock_inference"]

    model_bundle = None
    if not use_mock:
        try:
            model_bundle = _load_model_bundle(settings)
            if model_bundle is None:
                _set_status("Model artifacts missing, switching to mock mode.")
                use_mock = True
        except Exception as ex:  # pragma: no cover - runtime defensive guard
            _set_status(f"Model load failed, switching to mock mode: {ex}")
            use_mock = True

    text_corrector = TransformerCorrector(model_name=settings["nlp_model"])

    with _runtime_lock:
        _runtime["mode"] = "mock" if use_mock else "real"
        _runtime["model_loaded"] = model_bundle is not None

    extractor = None
    if not use_mock and model_bundle is not None:
        extractor = MediaPipeExtractor()

    mock_phrases = [
        "Hello.",
        "Thank you.",
        "Please help me.",
        "Nice to meet you.",
        "How are you?",
    ]

    cap = cv2.VideoCapture(settings["camera_index"])
    if not cap.isOpened():
        _set_status("Camera open failed. Check CAMERA_INDEX.")
        return

    _set_status("Realtime pipeline started.")

    seq_len = int(model_bundle["seq_len"]) if model_bundle else 30
    seq_buffer: deque[np.ndarray] = deque(maxlen=seq_len)
    emit_seconds = max(settings["emit_interval_ms"], 200) / 1000.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                _set_status("Camera frame read failed.")
                socketio.sleep(emit_seconds)
                continue

            if use_mock or model_bundle is None or extractor is None:
                raw_text = random.choice(mock_phrases)
                confidence = random.uniform(0.6, 0.95)
                corrected = text_corrector.correct(raw_text)
                payload = _build_payload(mode="mock", text=corrected, confidence=confidence, raw_text=raw_text)
            else:
                feats = extractor.extract(frame)
                input_dim = int(model_bundle["input_dim"])
                if feats.shape[0] > input_dim:
                    feats = feats[:input_dim]
                elif feats.shape[0] < input_dim:
                    feats = np.pad(feats, (0, input_dim - feats.shape[0]))

                seq_buffer.append(feats)
                if len(seq_buffer) < seq_len:
                    payload = _build_payload(
                        mode="real",
                        text="Collecting gesture frames...",
                        confidence=0.0,
                        raw_text="COLLECTING_FRAMES",
                    )
                else:
                    x = torch.tensor(np.stack(seq_buffer), dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = model_bundle["model"](x)
                        probs = torch.softmax(logits, dim=1)
                        pred_id = int(torch.argmax(probs, dim=1).item())
                        confidence = float(probs[0, pred_id].item())
                    raw_text = model_bundle["id_to_label"].get(pred_id, "UNKNOWN")
                    corrected = text_corrector.correct(raw_text)
                    payload = _build_payload(mode="real", text=corrected, confidence=confidence, raw_text=raw_text)

            with _runtime_lock:
                _runtime["last_prediction"] = payload
            socketio.emit("prediction", payload)
            socketio.sleep(emit_seconds)
    finally:
        cap.release()


def _ensure_background_task() -> None:
    with _runtime_lock:
        if _runtime["background_started"]:
            return
        _runtime["background_started"] = True
    socketio.start_background_task(_capture_loop)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health() -> tuple[dict, int]:
    return {"ok": True}, 200


@app.route("/status")
def status():
    with _runtime_lock:
        snapshot = dict(_runtime)
    return jsonify(snapshot)


@app.route("/predict")
def predict():
    with _runtime_lock:
        payload = _runtime.get("last_prediction")
        mode = _runtime.get("mode", "mock")
    if payload is None:
        payload = _build_payload(mode=mode, text="No prediction yet.", confidence=0.0)
    return jsonify(payload)


@socketio.on("connect")
def on_connect():
    _ensure_background_task()
    with _runtime_lock:
        snapshot = dict(_runtime)
    socketio.emit("status", {"message": f"Connected. mode={snapshot['mode']}"})


if __name__ == "__main__":
    _ensure_background_task()
    socketio.run(app, host="0.0.0.0", port=5000)
