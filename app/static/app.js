const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const videoInputBtn = document.getElementById("videoInputBtn");
const statusText = document.getElementById("statusText");
const engineDot = document.getElementById("engineDot");
const currentWord = document.getElementById("currentWord");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const topCandidates = document.getElementById("topCandidates");
const glossChips = document.getElementById("glossChips");
const sentenceText = document.getElementById("sentenceText");
const modeValue = document.getElementById("modeValue");
const fpsValue = document.getElementById("fpsValue");
const recognizedCount = document.getElementById("recognizedCount");
const marginValue = document.getElementById("marginValue");
const timestampValue = document.getElementById("timestampValue");
const cameraStatus = document.getElementById("cameraStatus");
const previewFeed = document.getElementById("previewFeed");
const guidelinesBtn = document.getElementById("guidelinesBtn");
const guidelinesModal = document.getElementById("guidelinesModal");
const guidelinesClose = document.getElementById("guidelinesClose");
const guidelinesBackdrop = document.getElementById("guidelinesBackdrop");
const videoInputPanel = document.getElementById("videoInputPanel");
const closeVideoInputBtn = document.getElementById("closeVideoInputBtn");
const videoWordCount = document.getElementById("videoWordCount");
const videoFileRows = document.getElementById("videoFileRows");
const processVideosBtn = document.getElementById("processVideosBtn");
const videoInputStatus = document.getElementById("videoInputStatus");
const pageRoot = document.querySelector(".page");
const scenarioSelect = document.getElementById("scenarioSelect");
const loadScenarioBtn = document.getElementById("loadScenarioBtn");
const clearScenarioBtn = document.getElementById("clearScenarioBtn");
const scenarioStatus = document.getElementById("scenarioStatus");
const scenarioPanel = document.getElementById("scenarioPanel");
const settingsToggle = document.getElementById("settingsToggle");

let videoInputMode = false;
let scenarioList = [];

async function postJson(url) {
  const response = await fetch(url, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Request failed: ${url}`);
  }
  return response.json();
}

function setConnectionError(message) {
  statusText.textContent = message;
  if (engineDot) {
    engineDot.classList.remove("ready", "mock");
    engineDot.classList.add("offline");
  }
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

function setEngineIndicator(state) {
  if (!engineDot) return;
  const mode = (state.mode || "").toLowerCase();
  engineDot.classList.remove("ready", "offline", "mock");
  if (mode === "live") {
    engineDot.classList.add("ready");
  } else if (mode === "mock") {
    engineDot.classList.add("ready");
  } else {
    engineDot.classList.add("offline");
  }
}

function renderTopCandidates(candidates) {
  if (!topCandidates) return;
  topCandidates.innerHTML = "";
  if (!candidates || candidates.length === 0) {
    const chip = document.createElement("span");
    chip.className = "chip muted";
    chip.textContent = "-";
    topCandidates.appendChild(chip);
    return;
  }

  candidates.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "candidate-chip";

    const word = document.createElement("span");
    word.className = "candidate-word";
    word.textContent = item.word || item.token || "UNKNOWN";

    const score = document.createElement("span");
    score.className = "candidate-score";
    score.textContent = Number(item.confidence || 0).toFixed(3);

    chip.appendChild(word);
    chip.appendChild(score);
    topCandidates.appendChild(chip);
  });
}

function toggleVideoInputPanel(show) {
  if (!videoInputPanel) return;
  videoInputMode = show;
  videoInputPanel.classList.toggle("hidden", !show);
  if (pageRoot) {
    pageRoot.classList.toggle("video-input-mode", show);
  }
  if (show) {
    statusText.textContent = "Video input mode active.";
    renderVideoFileInputs();
  }
}

function getRequestedVideoCount() {
  const value = Number(videoWordCount?.value || 1);
  return Math.max(1, Math.min(20, Number.isFinite(value) ? value : 1));
}

function renderVideoFileInputs() {
  if (!videoFileRows) return;
  const count = getRequestedVideoCount();
  videoFileRows.innerHTML = "";

  for (let i = 1; i <= count; i += 1) {
    const row = document.createElement("div");
    row.className = "video-file-row";

    const label = document.createElement("label");
    label.setAttribute("for", `videoFile${i}`);
    label.textContent = `Video Path ${i}`;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.id = `videoFile${i}`;
    input.dataset.videoFile = "1";

    row.appendChild(label);
    row.appendChild(input);
    videoFileRows.appendChild(row);
  }
}

async function fetchScenarios() {
  if (!scenarioSelect) return;
  try {
    const response = await fetch("/mock_scenarios");
    if (!response.ok) return;
    const data = await response.json();
    scenarioList = data.scenarios || [];
    populateScenarioSelect(scenarioList, data.active || "");
  } catch (_err) {
    // silent
  }
}

function populateScenarioSelect(list, active) {
  if (!scenarioSelect) return;
  scenarioSelect.innerHTML = '<option value="">Live Camera (default)</option>';
  list.forEach((scenario) => {
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = scenario.label;
    scenarioSelect.appendChild(option);
  });
  if (active) {
    scenarioSelect.value = active;
  }
}

function updateFromState(state) {
  statusText.textContent = state.status || "Unknown";
  setEngineIndicator(state);

  currentWord.textContent = state.current_word || "-";
  setConfidence(state.confidence || 0);
  renderTopCandidates(state.top_candidates || []);
  renderGlossChips(state.committed_words || []);
  sentenceText.textContent = state.corrected_sentence || "-";

  const modeLabel = state.mode_label || state.mode || "-";
  modeValue.textContent = (modeLabel || "-").toUpperCase();
  fpsValue.textContent = Number(state.fps || 0).toFixed(1);
  recognizedCount.textContent = String(state.recognized_count || 0);
  if (marginValue) {
    marginValue.textContent = Number(state.margin || 0).toFixed(3);
  }
  timestampValue.textContent = state.timestamp || "-";
  cameraStatus.textContent = state.camera_status || "Camera status unavailable.";

  const inMock = Boolean(state.mock_mode);
  if (inMock) {
    startBtn.disabled = videoInputMode || Boolean(state.running);
    stopBtn.disabled = videoInputMode || !Boolean(state.running);
  } else {
    startBtn.disabled = videoInputMode || Boolean(state.running) || state.mode !== "live";
    stopBtn.disabled = videoInputMode || !Boolean(state.running);
  }
  if (scenarioStatus) {
    scenarioStatus.textContent = inMock
      ? `Scripted sentence: ${state.mock_label || state.mock_scenario}`
      : "Mode: Live camera";
  }
  if (scenarioSelect && state.mock_scenario && scenarioSelect.value !== state.mock_scenario) {
    scenarioSelect.value = state.mock_scenario;
  }
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

if (videoInputBtn) {
  videoInputBtn.addEventListener("click", () => {
    const isHidden = videoInputPanel.classList.contains("hidden");
    toggleVideoInputPanel(isHidden);
  });
}

if (closeVideoInputBtn) {
  closeVideoInputBtn.addEventListener("click", () => {
    toggleVideoInputPanel(false);
  });
}

if (processVideosBtn) {
  processVideosBtn.addEventListener("click", async () => {
    const count = getRequestedVideoCount();
    const inputs = Array.from(document.querySelectorAll('input[data-video-file="1"]'));
    if (inputs.length < count) {
      videoInputStatus.textContent = "Please provide all video files.";
      return;
    }

    const form = new FormData();
    form.append("word_count", String(count));
    for (let i = 0; i < count; i += 1) {
      const fileInput = inputs[i];
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        videoInputStatus.textContent = `Please choose file for Video Path ${i + 1}.`;
        return;
      }
      form.append("video_files", file);
    }

    processVideosBtn.disabled = true;
    videoInputStatus.textContent = "Processing videos...";
    try {
      const response = await fetch("/video_input/process", {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Video processing failed.");
      }
      updateFromState(data.state);
      const tokens = data.tokens || [];
      const confidences = data.confidences || [];
  const summary = tokens
    .map((tok, idx) => `${idx + 1}) ${tok} (${Number(confidences[idx] || 0).toFixed(3)})`)
    .join(" | ");
      videoInputStatus.textContent = `Done. ${summary || "No words recognized."}`;
    } catch (err) {
      videoInputStatus.textContent = err.message || "Video processing failed.";
    } finally {
      processVideosBtn.disabled = false;
    }
  });
}

if (videoWordCount) {
  videoWordCount.addEventListener("change", renderVideoFileInputs);
  videoWordCount.addEventListener("input", renderVideoFileInputs);
}

if (loadScenarioBtn) {
  loadScenarioBtn.addEventListener("click", async () => {
    if (!scenarioSelect || !scenarioSelect.value) {
      alert("Select a scenario first.");
      return;
    }
    try {
      const resp = await fetch("/mock/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_id: scenarioSelect.value }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        alert(data.error || "Failed to load scenario.");
        return;
      }
      updateFromState(data.state);
    } catch (_err) {
      alert("Failed to load scenario.");
    }
  });
}

if (clearScenarioBtn) {
  clearScenarioBtn.addEventListener("click", async () => {
    try {
      const resp = await fetch("/mock/clear", { method: "POST" });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        alert(data.error || "Failed to clear scenario.");
        return;
      }
      if (scenarioSelect) scenarioSelect.value = "";
      updateFromState(data.state);
    } catch (_err) {
      alert("Failed to clear scenario.");
    }
  });
}

if (previewFeed) {
  previewFeed.addEventListener("error", () => {
    cameraStatus.textContent = "Video feed unavailable.";
  });
}

function openGuidelines() {
  if (!guidelinesModal) return;
  guidelinesModal.classList.add("open");
  guidelinesModal.setAttribute("aria-hidden", "false");
}

function closeGuidelines() {
  if (!guidelinesModal) return;
  guidelinesModal.classList.remove("open");
  guidelinesModal.setAttribute("aria-hidden", "true");
}

if (guidelinesBtn) {
  guidelinesBtn.addEventListener("click", openGuidelines);
}
if (guidelinesClose) {
  guidelinesClose.addEventListener("click", closeGuidelines);
}
if (guidelinesBackdrop) {
  guidelinesBackdrop.addEventListener("click", closeGuidelines);
}
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    closeGuidelines();
  }
});

pollState();
renderVideoFileInputs();
setInterval(pollState, 300);
fetchScenarios();
if (settingsToggle) {
  settingsToggle.addEventListener("click", () => {
    if (!scenarioPanel) return;
    scenarioPanel.classList.toggle("hidden");
  });
}
