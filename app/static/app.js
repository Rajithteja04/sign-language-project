const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const modeEl = document.getElementById("mode");
const confidenceEl = document.getElementById("confidence");
const timestampEl = document.getElementById("timestamp");
const selfViewEl = document.getElementById("self-view");
const previewStatusEl = document.getElementById("preview-status");

let previewStream = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setPreviewStatus(msg) {
  previewStatusEl.textContent = msg;
}

function setPrediction(payload) {
  if (!payload) return;
  outputEl.textContent = payload.text || "(no text)";
  modeEl.textContent = `mode: ${payload.mode || "unknown"}`;
  const c = Number(payload.confidence || 0);
  confidenceEl.textContent = `confidence: ${c.toFixed(4)}`;
  timestampEl.textContent = `timestamp: ${payload.timestamp || "-"}`;
}

async function initCameraPreview() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setPreviewStatus("Browser camera API not supported.");
    return;
  }

  try {
    previewStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    selfViewEl.srcObject = previewStream;
    setPreviewStatus("Camera preview active.");
  } catch (err) {
    setPreviewStatus(`Camera preview unavailable: ${err.message || err}`);
  }
}

function stopPreview() {
  if (!previewStream) return;
  for (const track of previewStream.getTracks()) {
    track.stop();
  }
  previewStream = null;
}

setStatus("Connecting to realtime server...");
initCameraPreview();

const socket = io();

socket.on("connect", () => {
  setStatus("Connected. Waiting for stream...");
});

socket.on("disconnect", () => {
  setStatus("Disconnected from server.");
});

socket.on("status", (payload) => {
  if (payload && payload.message) {
    setStatus(payload.message);
  }
});

socket.on("prediction", (payload) => {
  setPrediction(payload);
});

window.addEventListener("beforeunload", stopPreview);
