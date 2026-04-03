## Project Workflow Overview

### 1. End-to-End Architecture
| Module | Purpose | Key Tech |
| --- | --- | --- |
| Module I – MediaPipe Landmark Extraction | Converts live or recorded RGB frames into normalized 411-dim feature vectors (pose + face + both hands). | MediaPipe Holistic, torso-centering, shoulder-distance scaling. |
| Module II – LSTM Sequence Modeling | Learns temporal dynamics of the landmark sequences to predict gloss tokens. | PyTorch LSTMClassifier, sequence length 30–60, class-specific checkpoints. |
| Module III – NLP Sentence Correction | Transforms the committed gloss tokens into fluent English sentences. | T5-small + role-aware templates backed by `data/lsa64_labels.json`. |

The runtime loop is: Camera → Module I (per-frame landmarks) → sliding buffer (seq_len frames) → Module II prediction + smoothing → chips list → Module III sentence.

### 2. Dataset Preparation
1. **Landmark caching**  
   - `training/train_*_lstm.py` scripts can read raw datasets directly, but we usually precompute caches (e.g., `artifacts/cache/how2sign_cache_30000.pt`, `artifacts/cache/msasl_seq30`, `artifacts/cache/lsa64_main_seq30`) to avoid re-running MediaPipe.
   - Cache format is chunked tensors per split (train/val/test) stored under `artifacts/cache/<dataset>_seq<len>/`.
2. **Top‑K filtering**  
   - For isolated datasets (LSA64, MS-ASL, WLASL) we use `--top-k` to keep the most frequent glosses. How2Sign uses label pooling.
3. **Normalization**  
   - When `--normalize` is set, cached sequences are torso-centered and scaled by shoulder distance to match live inference.

### 3. Training & Artifacts
Each training script:
- Builds label maps, loads cached tensors, pads/truncates to `seq_len`.
- Trains the LSTM with Adam + CrossEntropy.
- Saves run artifacts under a dataset folder, e.g. `artifacts/msasl/msasl_top6_seq30_h128_l2_bs16_lr5e-04/`.
- Writes `lstm_best.pt`, `label_to_id.json`, `lstm_meta.json` (meta includes feature dim, seq_len, normalize flag, cache_dir, num_classes, etc.).

For the live demo:
- `artifacts/lstm_best.pt`, `artifacts/label_to_id.json`, `artifacts/lstm_meta.json` mirror the active checkpoint (currently `lsa64_top64_seq30_h128_l2_bs16_lr5e-04`).
- `artifacts/final_best/` hosts the curated copy so it’s easy to restore if experiments overwrite the top-level files.

### 4. Runtime Application
1. **Preflight**: `python -m scripts.preflight_runtime --camera-index 0` verifies artifacts and camera connectivity.
2. **Launch**: `python -m app.app` runs the Flask server at `http://127.0.0.1:5000`.
3. **UI highlights**:
   - Recognition Engine dot (green/amber/red) indicates live/mock/error.
   - Toolbar buttons: Start, Stop, Clear Sentence, User Guidelines (modal lists signer instructions).
   - Camera panel shows live feed + status pill.
   - Output panel displays current word, confidence bar, chips, sentence, and metrics (mode, FPS, count, margin, last update).
4. **Smoothing & emission**:
   - Confidence ≥ 0.82, margin ≥ 0.20, ≥ 7 agreeing frames, cooldown 1.8 s.
   - Chips appear immediately once a gloss is committed; Module III reruns on the full word list.

### 5. Module III Behavior
1. **Sanitization**: tokens normalized to uppercase with `_` separators. Immediate repeats removed.
2. **Routing**:
   - If tokens ⊆ MS-ASL demo vocab → deterministic template for the five demo words.
   - Else if tokens ⊆ LSA64 vocab → role-aware fallback (subject/verb/object/adjective/question).
   - Else → T5-small output; falls back to generic sentence if the model echoes the gloss.
3. **Metadata**: `data/lsa64_labels.json` includes `phrase`, `role`, and optional `progressive` entries for all 64 glosses.

### 6. Evaluation & Offline Tools
- `scripts/run_video_inference.py` runs Module I+II on any mp4 and prints top‑k predictions.
- `scripts/eval_lsa64_dataset.py` loads cached sequences and evaluates all 3,200 clips in ~20 s, writing CSV and accuracy stats (97.2%).
- Similar cached-eval scripts can be added for MS-ASL and How2Sign.
- Plots (accuracy comparison, latency breakdown, per-label accuracy) live under `artifacts/plots/`.

### 7. Demo Checklist (Signer Workflow)
1. Neutral pose for ~1 s before each sign.
2. Perform a single LSA64 sign ~1 s, keep hands centered.
3. Hold final pose ~2 s until the chip appears and sentence updates.
4. Repeat for additional signs; use **Clear Sentence** to restart a phrase.
5. If status shows “Low confidence” or “Ambiguous sign,” re-center and repeat.

### 8. Repository Layout Highlights
```
artifacts/
  final_best/           ← snapshot the app consumes
  lsa64/, msasl/, how2sign/, wlasl/, asl/
  cache/                ← reusable landmark tensors
docs/
  reports/              ← Word reports & presentation drafts
  reference/            ← protocols, comparisons, formulas
  notes/                ← scratch files, Mediapipe helper scripts
scripts/                ← helper tooling (cache build, eval, NLP test)
training/               ← dataset-specific training entrypoints
app/                    ← Flask UI + static assets
```

With this structure, swapping datasets for the demo is just a matter of copying the desired run’s `lstm_best.pt`, `label_to_id.json`, and `lstm_meta.json` into `artifacts/` (or pointing the app to `artifacts/final_best/`) and restarting the server.
