const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const statusText = document.getElementById("statusText");
const modelBanner = document.getElementById("modelBanner");
const currentWord = document.getElementById("currentWord");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const glossChips = document.getElementById("glossChips");
const sentenceText = document.getElementById("sentenceText");
const modeValue = document.getElementById("modeValue");
const fpsValue = document.getElementById("fpsValue");
const recognizedCount = document.getElementById("recognizedCount");
const marginValue = document.getElementById("marginValue");
const timestampValue = document.getElementById("timestampValue");
const cameraStatus = document.getElementById("cameraStatus");
const previewFeed = document.getElementById("previewFeed");

async function postJson(url) {
  const response = await fetch(url, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Request failed: ${url}`);
  }
  return response.json();
}

function setConnectionError(message) {
  statusText.textContent = message;
  modelBanner.textContent = "Backend not responding. Check terminal/logs and refresh.";
  modelBanner.classList.remove("success");
  modelBanner.classList.add("warning");
  startBtn.disabled = true;
  stopBtn.disabled = true;
}

function setConfidence(conf) {
  const c = Number(conf || 0);
  const pct = Math.max(0, Math.min(100, c * 100));
  confidenceValue.textContent = c.toFixed(3);
  confidenceBar.style.width = `${pct}%`;
  confidenceBar.classList.remove("low", "mid", "high");
  if (c >= 0.7) {
    confidenceBar.classList.add("high");
  } else if (c >= 0.5) {
    confidenceBar.classList.add("mid");
  } else {
    confidenceBar.classList.add("low");
  }
}

function renderGlossChips(words) {
  glossChips.innerHTML = "";
  if (!words || words.length === 0) {
    const chip = document.createElement("span");
    chip.className = "chip muted";
    chip.textContent = "-";
    glossChips.appendChild(chip);
    return;
  }
  words.forEach((word) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = word;
    glossChips.appendChild(chip);
  });
}

function applyModelBanner(state) {
  modelBanner.textContent = state.model_status || "Model status unavailable.";
  modelBanner.classList.remove("warning", "success");
  if (state.mode === "live") {
    modelBanner.classList.add("success");
  } else {
    modelBanner.classList.add("warning");
  }
}

function updateFromState(state) {
  statusText.textContent = state.status || "Unknown";
  applyModelBanner(state);

  currentWord.textContent = state.current_word || "-";
  setConfidence(state.confidence || 0);
  renderGlossChips(state.committed_words || []);
  sentenceText.textContent = state.corrected_sentence || "-";

  modeValue.textContent = (state.mode || "-").toUpperCase();
  fpsValue.textContent = Number(state.fps || 0).toFixed(1);
  recognizedCount.textContent = String(state.recognized_count || 0);
  if (marginValue) {
    marginValue.textContent = Number(state.margin || 0).toFixed(3);
  }
  timestampValue.textContent = state.timestamp || "-";
  cameraStatus.textContent = state.camera_status || "Camera status unavailable.";

  startBtn.disabled = Boolean(state.running) || state.mode !== "live";
  stopBtn.disabled = !Boolean(state.running);
}

async function pollState() {
  try {
    const response = await fetch("/state", { cache: "no-store" });
    if (!response.ok) {
      setConnectionError(`State API error: ${response.status}`);
      return;
    }
    const state = await response.json();
    updateFromState(state);
  } catch (_err) {
    setConnectionError("Backend disconnected.");
  }
}

startBtn.addEventListener("click", async () => {
  try {
    const data = await postJson("/start");
    updateFromState(data.state);
  } catch (_err) {
    statusText.textContent = "Failed to start recognition.";
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    const data = await postJson("/stop");
    updateFromState(data.state);
  } catch (_err) {
    statusText.textContent = "Failed to stop recognition.";
  }
});

clearBtn.addEventListener("click", async () => {
  try {
    const data = await postJson("/reset");
    updateFromState(data.state);
  } catch (_err) {
    statusText.textContent = "Failed to clear sentence.";
  }
});

if (previewFeed) {
  previewFeed.addEventListener("error", () => {
    cameraStatus.textContent = "Video feed unavailable.";
  });
}

pollState();
setInterval(pollState, 300);
