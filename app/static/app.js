const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const wordsOutputEl = document.getElementById("words-output");
const sentenceOutputEl = document.getElementById("sentence-output");
const modeEl = document.getElementById("mode");
const gestureModeMetaEl = document.getElementById("gesture-mode-meta");
const confidenceEl = document.getElementById("confidence");
const timestampEl = document.getElementById("timestamp");
const selfViewEl = document.getElementById("self-view");
const previewStatusEl = document.getElementById("preview-status");
const modeSelectEl = document.getElementById("gesture-mode");
const startBtnEl = document.getElementById("start-btn");
const stopBtnEl = document.getElementById("stop-btn");

let previewStream = null;
let recognitionRunning = false;
let framePumpTimer = null;
const frameCanvas = document.createElement("canvas");

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setPreviewStatus(msg) {
  previewStatusEl.textContent = msg;
}

function setPrediction(payload) {
  if (!payload) return;
  outputEl.textContent = payload.text || "(no text)";
  const words = Array.isArray(payload.words) ? payload.words.join(" ") : "";
  wordsOutputEl.textContent = `words: ${words || "-"}`;
  sentenceOutputEl.textContent = `sentence: ${payload.sentence || "-"}`;
  modeEl.textContent = `mode: ${payload.mode || "unknown"}`;
  gestureModeMetaEl.textContent = `gesture_mode: ${payload.gesture_mode || "-"}`;
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
  selfViewEl.srcObject = null;
}

function setControls(isRunning) {
  recognitionRunning = isRunning;
  modeSelectEl.disabled = isRunning;
  startBtnEl.disabled = isRunning;
  stopBtnEl.disabled = !isRunning;
}

function startFramePump() {
  if (framePumpTimer) return;
  framePumpTimer = setInterval(() => {
    if (!recognitionRunning) return;
    if (!previewStream || !selfViewEl.videoWidth || !selfViewEl.videoHeight) return;
    if (!socket || !socket.connected) return;

    frameCanvas.width = selfViewEl.videoWidth;
    frameCanvas.height = selfViewEl.videoHeight;
    const ctx = frameCanvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(selfViewEl, 0, 0, frameCanvas.width, frameCanvas.height);
    const jpeg = frameCanvas.toDataURL("image/jpeg", 0.75);
    socket.emit("browser_frame", { image: jpeg });
  }, 100); // ~10 FPS
}

async function startRecognition() {
  const mode = modeSelectEl.value;
  setStatus(`Starting ${mode} recognition...`);

  try {
    const resp = await fetch("/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      setStatus(data.error || "Failed to start session.");
      setControls(false);
      return;
    }
    setControls(true);
    setStatus(`Running ${mode} recognition.`);
  } catch (err) {
    setStatus(`Start failed: ${err.message || err}`);
    setControls(false);
  }
}

async function stopRecognition() {
  try {
    await fetch("/session/stop", { method: "POST" });
  } catch (_err) {
    // no-op; UI will still transition back
  }
  setControls(false);
  setStatus("Recognition stopped. Adjust and click Start.");
}

setStatus("Connecting to realtime server...");
initCameraPreview();
startFramePump();
setControls(false);

const socket = io();

socket.on("connect", () => {
  setStatus("Connected. Waiting for stream...");
});

socket.on("disconnect", () => {
  setStatus("Disconnected from server.");
  setControls(false);
});

socket.on("status", (payload) => {
  if (payload && payload.message) {
    setStatus(payload.message);
  }
});

socket.on("runtime", (payload) => {
  if (!payload) return;
  setControls(payload.session_state === "running");
  if (payload.gesture_mode) {
    modeSelectEl.value = payload.gesture_mode;
  }
});

socket.on("prediction", (payload) => {
  setPrediction(payload);
});

startBtnEl.addEventListener("click", startRecognition);
stopBtnEl.addEventListener("click", stopRecognition);

window.addEventListener("beforeunload", stopPreview);
