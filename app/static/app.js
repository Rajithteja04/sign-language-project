const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const modeEl = document.getElementById("mode");
const confidenceEl = document.getElementById("confidence");
const timestampEl = document.getElementById("timestamp");

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setPrediction(payload) {
  if (!payload) return;
  outputEl.textContent = payload.text || "(no text)";
  modeEl.textContent = `mode: ${payload.mode || "unknown"}`;
  const c = Number(payload.confidence || 0);
  confidenceEl.textContent = `confidence: ${c.toFixed(4)}`;
  timestampEl.textContent = `timestamp: ${payload.timestamp || "-"}`;
}

setStatus("Connecting to realtime server...");

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
