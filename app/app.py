from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

from features.mediapipe_extractor import MediaPipeExtractor
from models.lstm import LSTMClassifier
from models.transformer import create_text_corrector
from utils.io import load_config

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_runtime = {
    "mode": "mock",
    "model_loaded": False,
    "last_prediction": None,
    "status": "idle",
    "session_state": "idle",  # idle | running | stopped
    "gesture_mode": None,  # word | sentence
    "camera_active": False,  # browser stream active while frames are received
    "thread_alive": False,
    "nlp_backend": "transformer",
}
_runtime_lock = threading.Lock()
_session_lock = threading.Lock()
_session_ctx: dict | None = None


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
        "emit_interval_ms": int(env_or_cfg("EMIT_INTERVAL_MS", "emit_interval_ms", 1000)),
        "use_mock_inference": _is_true(env_or_cfg("USE_MOCK_INFERENCE", "use_mock_inference", True)),
        "min_confidence": float(env_or_cfg("MIN_PRED_CONFIDENCE", "min_pred_confidence", 0.30)),
        "stability_frames": int(env_or_cfg("STABILITY_FRAMES", "stability_frames", 3)),
        "weights": str(env_or_cfg("LSTM_WEIGHTS", "lstm_weights_path", "artifacts/lstm_best.pt")),
        "labels": str(env_or_cfg("LSTM_LABELS", "labels_path", "artifacts/label_to_id.json")),
        "meta": str(env_or_cfg("LSTM_META", "meta_path", "artifacts/lstm_meta.json")),
        "word_weights": str(env_or_cfg("WORD_LSTM_WEIGHTS", "word_lstm_weights_path", "artifacts/lstm_best.pt")),
        "word_labels": str(env_or_cfg("WORD_LSTM_LABELS", "word_labels_path", "artifacts/label_to_id.json")),
        "word_meta": str(env_or_cfg("WORD_LSTM_META", "word_meta_path", "artifacts/lstm_meta.json")),
        "sentence_weights": str(
            env_or_cfg("SENTENCE_LSTM_WEIGHTS", "sentence_lstm_weights_path", "artifacts/lstm_best.pt")
        ),
        "sentence_labels": str(
            env_or_cfg("SENTENCE_LSTM_LABELS", "sentence_labels_path", "artifacts/label_to_id.json")
        ),
        "sentence_meta": str(env_or_cfg("SENTENCE_LSTM_META", "sentence_meta_path", "artifacts/lstm_meta.json")),
        "lstm_hidden": int(cfg.get("lstm_hidden", 256)),
        "lstm_layers": int(cfg.get("lstm_layers", 2)),
        "nlp_model": str(cfg.get("nlp_model", "")).strip() or None,
        "nlp_backend": str(env_or_cfg("NLP_BACKEND", "nlp_backend", "transformer")).strip().lower(),
        "gemini_model": str(env_or_cfg("GEMINI_MODEL", "gemini_model", "gemini-1.5-flash")).strip(),
        "gemini_api_key": str(env_or_cfg("GEMINI_API_KEY", "gemini_api_key", "")).strip() or None,
        "word_buffer_size": int(cfg.get("word_buffer_size", 12)),
    }


def _build_payload(
    mode: str,
    text: str,
    confidence: float,
    raw_text: str | None = None,
    gesture_mode: str | None = None,
    words: list[str] | None = None,
    sentence: str | None = None,
) -> dict:
    return {
        "mode": mode,
        "gesture_mode": gesture_mode,
        "text": text,
        "raw_text": raw_text or text,
        "words": words or [],
        "sentence": sentence or "",
        "confidence": round(float(confidence), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _set_runtime(**kwargs) -> None:
    with _runtime_lock:
        _runtime.update(kwargs)


def _set_status(message: str) -> None:
    _set_runtime(status=message)
    socketio.emit("status", {"message": message})


def _select_artifacts(settings: dict, gesture_mode: str) -> tuple[Path, Path, Path]:
    if gesture_mode == "word":
        return Path(settings["word_weights"]), Path(settings["word_labels"]), Path(settings["word_meta"])
    return Path(settings["sentence_weights"]), Path(settings["sentence_labels"]), Path(settings["sentence_meta"])


def _load_model_bundle(settings: dict, gesture_mode: str) -> dict | None:
    weights_path, labels_path, meta_path = _select_artifacts(settings, gesture_mode)
    if not (weights_path.exists() and labels_path.exists() and meta_path.exists()):
        return None

    with open(labels_path, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    input_dim = int(meta.get("input_dim", 411))
    seq_len = int(meta.get("sequence_length", 30))

    state_dict = torch.load(weights_path, map_location="cpu")
    weight_ih_l0 = state_dict.get("lstm.weight_ih_l0")
    hidden_dim = int(weight_ih_l0.shape[0] // 4) if weight_ih_l0 is not None else int(settings["lstm_hidden"])
    layer_keys = [k for k in state_dict.keys() if k.startswith("lstm.weight_ih_l")]
    num_layers = max(int(k.rsplit("l", 1)[-1]) for k in layer_keys) + 1 if layer_keys else int(settings["lstm_layers"])

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(id_to_label),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return {"model": model, "id_to_label": id_to_label, "input_dim": input_dim, "seq_len": seq_len}


def _build_session_context(gesture_mode: str) -> dict:
    settings = _load_runtime_settings()
    use_mock = settings["use_mock_inference"]

    model_bundle = None
    if not use_mock:
        try:
            model_bundle = _load_model_bundle(settings, gesture_mode=gesture_mode)
            if model_bundle is None:
                _set_status("Model artifacts missing, switching to mock mode.")
                use_mock = True
        except Exception as ex:
            _set_status(f"Model load failed, switching to mock mode: {ex}")
            use_mock = True

    text_corrector = create_text_corrector(
        backend=settings["nlp_backend"],
        transformer_model=settings["nlp_model"],
        gemini_model=settings["gemini_model"],
        gemini_api_key=settings["gemini_api_key"],
    )

    seq_len = int(model_bundle["seq_len"]) if model_bundle else 30
    ctx = {
        "settings": settings,
        "gesture_mode": gesture_mode,
        "use_mock": use_mock,
        "model_bundle": model_bundle,
        "extractor": MediaPipeExtractor() if (not use_mock and model_bundle is not None) else None,
        "text_corrector": text_corrector,
        "seq_buffer": deque(maxlen=seq_len),
        "candidate_label": None,
        "candidate_count": 0,
        "stable_label": None,
        "words_buffer": deque(maxlen=max(1, int(settings["word_buffer_size"]))),
        "last_committed_word": None,
        "last_emit_ts": 0.0,
        "mock_phrases": ["Hello.", "Thank you.", "Please help me.", "Nice to meet you.", "How are you?"],
    }
    return ctx


def _decode_data_url_to_bgr(image_data_url: str) -> np.ndarray | None:
    if not image_data_url or "," not in image_data_url:
        return None
    try:
        encoded = image_data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def _predict_from_frame(ctx: dict, frame_bgr: np.ndarray) -> dict:
    gesture_mode = ctx["gesture_mode"]
    settings = ctx["settings"]
    text_corrector = ctx["text_corrector"]

    if ctx["use_mock"] or ctx["model_bundle"] is None or ctx["extractor"] is None:
        raw_text = random.choice(ctx["mock_phrases"])
        confidence = random.uniform(0.6, 0.95)
        corrected = text_corrector.correct(raw_text)
        return _build_payload(
            mode="mock",
            gesture_mode=gesture_mode,
            text=corrected,
            confidence=confidence,
            raw_text=raw_text,
            sentence=corrected,
        )

    model_bundle = ctx["model_bundle"]
    feats = ctx["extractor"].extract(frame_bgr)
    input_dim = int(model_bundle["input_dim"])
    if feats.shape[0] > input_dim:
        feats = feats[:input_dim]
    elif feats.shape[0] < input_dim:
        feats = np.pad(feats, (0, input_dim - feats.shape[0]))

    ctx["seq_buffer"].append(feats)
    seq_len = int(model_bundle["seq_len"])
    if len(ctx["seq_buffer"]) < seq_len:
        return _build_payload(
            mode="real",
            gesture_mode=gesture_mode,
            text="Collecting gesture frames...",
            confidence=0.0,
            raw_text="COLLECTING_FRAMES",
        )

    x = torch.tensor(np.stack(ctx["seq_buffer"]), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model_bundle["model"](x)
        probs = torch.softmax(logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_id].item())

    raw_text = model_bundle["id_to_label"].get(pred_id, "UNKNOWN")
    min_confidence = max(0.0, min(1.0, float(settings.get("min_confidence", 0.30))))
    stability_frames = max(1, int(settings.get("stability_frames", 3)))

    if confidence < min_confidence:
        ctx["candidate_label"] = None
        ctx["candidate_count"] = 0
        return _build_payload(
            mode="real",
            gesture_mode=gesture_mode,
            text="No confident sign",
            confidence=confidence,
            raw_text=raw_text,
        )

    if raw_text == ctx["candidate_label"]:
        ctx["candidate_count"] += 1
    else:
        ctx["candidate_label"] = raw_text
        ctx["candidate_count"] = 1

    if ctx["candidate_count"] >= stability_frames:
        ctx["stable_label"] = raw_text

    if ctx["stable_label"] is None:
        return _build_payload(
            mode="real",
            gesture_mode=gesture_mode,
            text="Stabilizing gesture...",
            confidence=confidence,
            raw_text=raw_text,
        )

    stable_label = ctx["stable_label"]
    if gesture_mode == "word":
        if stable_label != ctx["last_committed_word"]:
            ctx["words_buffer"].append(stable_label)
            ctx["last_committed_word"] = stable_label
        assembled = " ".join(ctx["words_buffer"]).strip()
        corrected = text_corrector.correct(assembled) if assembled else stable_label
        return _build_payload(
            mode="real",
            gesture_mode=gesture_mode,
            text=stable_label,
            confidence=confidence,
            raw_text=stable_label,
            words=list(ctx["words_buffer"]),
            sentence=corrected,
        )

    corrected = text_corrector.correct(stable_label)
    return _build_payload(
        mode="real",
        gesture_mode=gesture_mode,
        text=corrected,
        confidence=confidence,
        raw_text=stable_label,
        sentence=corrected,
    )


def _start_session(gesture_mode: str) -> tuple[dict, int]:
    global _session_ctx
    gesture_mode = (gesture_mode or "").strip().lower()
    if gesture_mode not in {"word", "sentence"}:
        return {"ok": False, "error": "Invalid gesture mode. Use 'word' or 'sentence'."}, 400

    with _runtime_lock:
        if _runtime["session_state"] == "running":
            return {"ok": False, "error": "Session already running."}, 409

    try:
        ctx = _build_session_context(gesture_mode)
    except Exception as ex:
        return {"ok": False, "error": f"Session init failed: {ex}"}, 500

    with _session_lock:
        _session_ctx = ctx

    _set_runtime(
        session_state="running",
        gesture_mode=gesture_mode,
        last_prediction=None,
        mode="mock" if ctx["use_mock"] else "real",
        model_loaded=ctx["model_bundle"] is not None,
        camera_active=False,
        nlp_backend=ctx["settings"]["nlp_backend"],
    )
    _set_status(f"Session running ({gesture_mode}). Waiting for browser frames.")
    return {"ok": True, "session_state": "running", "gesture_mode": gesture_mode}, 200


def _stop_session() -> tuple[dict, int]:
    global _session_ctx
    with _session_lock:
        _session_ctx = None
    _set_runtime(session_state="stopped", camera_active=False)
    _set_status("Recognition stopped. Adjust and click Start.")
    return {"ok": True, "session_state": "stopped"}, 200


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


@app.route("/session/start", methods=["POST"])
def session_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "")
    body, code = _start_session(mode)
    return jsonify(body), code


@app.route("/session/stop", methods=["POST"])
def session_stop():
    body, code = _stop_session()
    return jsonify(body), code


@socketio.on("connect")
def on_connect():
    with _runtime_lock:
        snapshot = dict(_runtime)
    socketio.emit("status", {"message": "Connected. Choose mode and press Start."})
    socketio.emit("runtime", snapshot)


@socketio.on("browser_frame")
def on_browser_frame(payload):
    with _runtime_lock:
        running = _runtime.get("session_state") == "running"
    if not running:
        return

    with _session_lock:
        ctx = _session_ctx
    if ctx is None:
        return

    image_data = (payload or {}).get("image")
    frame = _decode_data_url_to_bgr(image_data)
    if frame is None:
        return

    now = time.time()
    emit_interval = max(0.05, float(ctx["settings"]["emit_interval_ms"]) / 1000.0)
    if now - float(ctx["last_emit_ts"]) < emit_interval:
        return
    ctx["last_emit_ts"] = now

    pred = _predict_from_frame(ctx, frame)
    _set_runtime(last_prediction=pred, camera_active=True)
    socketio.emit("prediction", pred)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
