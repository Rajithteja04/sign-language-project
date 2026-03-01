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
const mockWarning = document.getElementById("mockWarning");
const preview = document.getElementById("preview");

async function postJson(url) {
  const response = await fetch(url, { method: "POST" });
  if (!response.ok) throw new Error(`Request failed: ${url}`);
  return response.json();
}

function setConfidence(conf) {
  const c = Number(conf || 0);
  const pct = Math.max(0, Math.min(100, c * 100));
  confidenceValue.textContent = c.toFixed(2);
  confidenceBar.style.width = `${pct}%`;
  confidenceBar.classList.remove("low", "mid", "high");
  if (c > 0.7) {
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

function updateFromState(state) {
  statusText.textContent = state.status || "Unknown";
  modelBanner.textContent = state.model_status || "Model status unavailable.";
  modelBanner.classList.toggle("ok", !state.mock);
  modelBanner.classList.toggle("warn", Boolean(state.mock));
  mockWarning.classList.toggle("hidden", !state.mock);
  currentWord.textContent = state.current_word || "-";
  setConfidence(state.confidence || 0);
  renderGlossChips(state.committed_words || []);
  sentenceText.textContent = state.corrected_sentence || "-";
  startBtn.disabled = Boolean(state.running);
  stopBtn.disabled = !Boolean(state.running);
}

async function pollState() {
  try {
    const res = await fetch("/state");
    if (!res.ok) return;
    const state = await res.json();
    updateFromState(state);
  } catch (_err) {
    statusText.textContent = "Backend disconnected.";
  }
}

async function initPreview() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusText.textContent = "Camera API unsupported in this browser.";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    preview.srcObject = stream;
  } catch (_err) {
    statusText.textContent = "Camera preview permission denied or unavailable.";
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
    currentWord.textContent = "-";
    setConfidence(0);
    renderGlossChips([]);
    sentenceText.textContent = "-";
    updateFromState(data.state);
  } catch (_err) {
    statusText.textContent = "Failed to clear sentence.";
  }
});

initPreview();
pollState();
setInterval(pollState, 500);
