const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const statusText = document.getElementById("statusText");
const modelBanner = document.getElementById("modelBanner");
const predictionCard = document.getElementById("predictionCard");
const detectedSentence = document.getElementById("detectedSentence");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const glossChips = document.getElementById("glossChips");
const sessionSummary = document.getElementById("sessionSummary");
const demoGestures = document.getElementById("demoGestures");
const mockWarning = document.getElementById("mockWarning");
const preview = document.getElementById("preview");
const dbgFrameCount = document.getElementById("dbgFrameCount");
const dbgResolution = document.getElementById("dbgResolution");
const dbgPose = document.getElementById("dbgPose");
const dbgFace = document.getElementById("dbgFace");
const dbgLeft = document.getElementById("dbgLeft");
const dbgRight = document.getElementById("dbgRight");
const dbgTotal = document.getElementById("dbgTotal");
const dbgDim = document.getElementById("dbgDim");
const dbgNorm = document.getElementById("dbgNorm");
const infoDataset = document.getElementById("infoDataset");
const infoTrain = document.getElementById("infoTrain");
const infoVal = document.getElementById("infoVal");
const infoModel = document.getElementById("infoModel");
const infoSeq = document.getElementById("infoSeq");

const PREDICTION_WINDOW = 5;
let predictionBuffer = [];
let stableSentence = "";

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

function renderDemoList(labels, activeLabel) {
  if (!demoGestures) return;
  demoGestures.innerHTML = "";
  if (!labels || labels.length === 0) {
    const empty = document.createElement("span");
    empty.className = "chip muted";
    empty.textContent = "Awaiting model";
    demoGestures.appendChild(empty);
    return;
  }
  labels.forEach((label) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    if (label === activeLabel) {
      chip.classList.add("active");
    }
    chip.textContent = label;
    demoGestures.appendChild(chip);
  });
}

function resetPredictionState() {
  predictionBuffer = [];
  stableSentence = "";
  if (detectedSentence) {
    detectedSentence.textContent = "Detecting gesture...";
    detectedSentence.classList.add("muted");
  }
}

function updateDetectedSentence(state) {
  if (!detectedSentence) return;
  const prediction = (state.current_word || "").trim();
  if (state.confidence >= 0.6 && prediction && prediction !== "-") {
    predictionBuffer.push(prediction);
  } else {
    predictionBuffer.push("");
  }
  if (predictionBuffer.length > PREDICTION_WINDOW) {
    predictionBuffer.shift();
  }
  const stable =
    predictionBuffer.length === PREDICTION_WINDOW &&
    predictionBuffer.every((item) => item && item === predictionBuffer[0]);
  if (stable) {
    stableSentence = predictionBuffer[0];
  }

  if (state.confidence < 0.6 || !stableSentence) {
    detectedSentence.textContent = "Detecting gesture...";
    detectedSentence.classList.add("muted");
  } else {
    detectedSentence.textContent = `“${stableSentence}”`;
    detectedSentence.classList.remove("muted");
  }
}

function updateFromState(state) {
  statusText.textContent = state.status || "Unknown";
  modelBanner.textContent = state.model_status || "Model status unavailable.";
  modelBanner.classList.toggle("ok", !state.mock);
  modelBanner.classList.toggle("warn", Boolean(state.mock));
  if (mockWarning) {
    mockWarning.classList.toggle("hidden", !state.mock);
  }

  const hidePredictions = Boolean(state.mock);
  if (predictionCard) {
    predictionCard.classList.toggle("hidden", hidePredictions);
  }

  updateDetectedSentence(state);
  renderDemoList(state.class_labels || [], stableSentence);
  setConfidence(state.confidence || 0);
  renderGlossChips(state.committed_words || []);
  sessionSummary.textContent = state.gloss_sequence || "-";

  const dbg = state.debug_module1 || {};
  dbgFrameCount.textContent = String(dbg.frame_count ?? 0);
  const w = dbg.frame_width ?? 0;
  const h = dbg.frame_height ?? 0;
  dbgResolution.textContent = w > 0 && h > 0 ? `${w} x ${h}` : "-";
  dbgPose.textContent = String(dbg.pose_landmarks ?? 0);
  dbgFace.textContent = String(dbg.face_landmarks ?? 0);
  dbgLeft.textContent = String(dbg.left_hand_landmarks ?? 0);
  dbgRight.textContent = String(dbg.right_hand_landmarks ?? 0);
  dbgTotal.textContent = String(dbg.total_landmarks ?? 0);
  dbgDim.textContent = String(dbg.feature_dimension ?? 411);
  dbgNorm.textContent = String(Boolean(dbg.normalization_applied));

  const info = state.system_info || {};
  if (infoDataset) infoDataset.textContent = info.dataset || "How2Sign";
  if (infoTrain) infoTrain.textContent = info.train_samples != null ? String(info.train_samples) : "-";
  if (infoVal) infoVal.textContent = info.val_samples != null ? String(info.val_samples) : "-";
  if (infoModel) infoModel.textContent = info.model || "-";
  if (infoSeq) infoSeq.textContent = info.sequence_length != null ? `${info.sequence_length} frames` : "-";

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
  preview.src = "/video_feed";
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
    resetPredictionState();
    setConfidence(0);
    renderGlossChips([]);
    sessionSummary.textContent = "-";
    updateFromState(data.state);
  } catch (_err) {
    statusText.textContent = "Failed to clear session.";
  }
});

resetPredictionState();
initPreview();
pollState();
setInterval(pollState, 500);
