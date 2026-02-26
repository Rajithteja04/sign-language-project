# Project Worklog

## Purpose
Single source of truth for progress tracking.  
Before starting any new action/task, add a new entry here.

## Update Protocol
For every action:
1. Add `Action ID`, date, owner.
2. Record objective.
3. Record implementation details (files/commands).
4. Record result/metrics.
5. Record status (`Done`, `Partial`, `Blocked`).
6. Record next action.

---

## Action History

### A-001 | Initial Project Scaffold
- Date: 2026-02-16
- Objective: Initialize end-to-end project structure for ASL sign-to-text pipeline.
- Implementation:
  - Created modules: `app/`, `data/`, `features/`, `models/`, `training/`, `inference/`, `config/`, `utils/`, `scripts/`.
  - Added starter files for LSTM training, realtime inference, Flask UI.
- Result:
  - Baseline code structure in place for all major modules.
- Status: Done
- Next: Dataset selection and integration.

### A-002 | Dataset Strategy Decision
- Date: 2026-02-16
- Objective: Pick a dataset path compatible with project goals and laptop storage.
- Implementation:
  - Chosen: How2Sign.
  - Due to storage constraints, targeted keypoint-driven workflow first.
- Result:
  - Dataset strategy finalized around How2Sign with future portability.
- Status: Done
- Next: Download and validate dataset.

### A-003 | How2Sign Download Troubleshooting
- Date: 2026-02-16 to 2026-02-19
- Objective: Download How2Sign keypoints + translations.
- Implementation:
  - Resolved script issues (`wget`, Google Drive confirmation, argument mismatch).
  - Added/used downloader support and extracted sentence-level structure.
- Result:
  - Dataset became available at `C:\Users\rajit\Datasets\How2Sign`.
- Status: Done
- Next: Align code with actual dataset layout.

### A-004 | Repo Sync and Technical Audit
- Date: 2026-02-21
- Objective: Audit teammate changes and identify current technical state.
- Implementation:
  - Reviewed git history, repo layout, artifacts, adapter/training/inference files.
  - Verified actual dataset presence and split-level volume.
- Result:
  - Found critical mismatch between loader expected paths and actual dataset structure.
- Status: Done
- Next: Make dataset path handling portable and robust.

### A-005 | Cross-Laptop Path Safety
- Date: 2026-02-21
- Objective: Prevent hardcoded dataset paths from breaking teammate workflow.
- Implementation:
  - Added config overlay support: `utils/io.py` (`load_config`).
  - Updated training scripts to resolve dataset root by precedence:
    - `--dataset-root` CLI
    - `HOW2SIGN_ROOT` env var
    - `config/local.yaml`
    - `config/default.yaml`
  - Added `config/local.example.yaml`.
  - Fixed `.gitignore` and ignored `config/local.yaml`.
- Result:
  - Per-laptop dataset configuration now works without committing local paths.
- Status: Done
- Next: Confirm adapter compatibility with downloaded layout.

### A-006 | How2Sign Adapter Layout Compatibility
- Date: 2026-02-21
- Objective: Support both legacy and sentence-level How2Sign layouts.
- Implementation:
  - Updated `data/adapters/how2sign.py` to resolve both:
    - Flat layout
    - `sentence_level/{split}/...` layout
- Result:
  - Loader successfully reads from `C:\Users\rajit\Datasets\How2Sign`.
- Status: Done
- Next: Run smoke training and realtime checks.

### A-007 | Training Smoke Check
- Date: 2026-02-21
- Objective: Verify end-to-end training executes on actual dataset.
- Implementation:
  - Ran tiny LSTM training in virtual environment.
  - Command: `python -m training.train_lstm --max-samples 5 --epochs 1`.
- Result:
  - Training executed successfully.
  - Artifacts were produced/updated in `artifacts/`.
- Status: Done
- Next: Validate realtime pipeline.

### A-008 | Realtime Pipeline Compatibility Fix
- Date: 2026-02-21
- Objective: Fix MediaPipe runtime failure in realtime inference.
- Problem:
  - `AttributeError: module 'mediapipe' has no attribute 'solutions'` with newer env stack.
- Implementation:
  - Created Python 3.11 environment (`.venv311`).
  - Installed MediaPipe stack compatible with `mp.solutions`.
  - Resolved dependency conflict by removing `opencv-python` and standardizing OpenCV stack.
- Result:
  - Camera window launched successfully via `inference/realtime.py`.
- Status: Done
- Next: Lock dependency versions in repo.

### A-009 | Dependency Stabilization
- Date: 2026-02-21
- Objective: Freeze reproducible requirements for working runtime.
- Implementation:
  - Updated `requirements.txt` with compatible versions:
    - `mediapipe==0.10.11`
    - `opencv-contrib-python==4.11.0.86`
    - `numpy<2`
    - `protobuf<4`
    - pinned torch/torchvision
  - Verified with `pip check`.
- Result:
  - Environment reports no broken requirements.
- Status: Done
- Next: Dataset quality validation + manifests.

### A-010 | Dataset Validation + Manifest Generation
- Date: 2026-02-21
- Objective: Add data-quality gate before baseline training.
- Implementation:
  - Added script: `scripts/validate_how2sign.py`.
  - Added docs: `scripts/README.md`, root `README.md`.
  - Generated:
    - `artifacts/manifests/train_manifest.tsv`
    - `artifacts/manifests/val_manifest.tsv`
    - `artifacts/manifests/test_manifest.tsv`
    - `artifacts/manifests/validation_report.json`
- Result (full run):
  - Train: `31165 total`, `31021 valid` (`99.54%`), missing_dir `118`, short `26`
  - Val: `1741 total`, `1737 valid` (`99.77%`), missing_dir `2`, short `2`
  - Test: `2357 total`, `2341 valid` (`99.32%`), missing_dir `14`, short `2`
- Status: Done
- Next: Action A-011 (baseline training + evaluation report).

---

## Current Stage
Implementation and stabilization complete for setup/runtime.  
Now entering model-quality phase (baseline training, evaluation, iterative improvement).

## Next Planned Action
### A-011 | Baseline Training + Evaluation
- Objective: Train reproducible baseline on validated manifests and produce first quality report.
- Planned outputs:
  - Baseline checkpoint.
  - Metrics summary (train/val/test).
  - Error sample review for top failure classes.
- Status: In Progress
- Run setup (2026-02-21):
  - Environment: `.venv311`
  - Dataset root: `C:\Users\rajit\Datasets\How2Sign`
  - Baseline command:
    - `python -m training.train_lstm --max-samples 1000 --epochs 3`
  - Rationale:
    - First stable baseline at moderate scale before full-scale training.
 - Execution note:
   - Run was manually stopped after long silent period.
   - Cause: current training script prints only after dataset loading and epoch start; loading large keypoint splits can take >20 minutes.
 - Adjustment:
   - Add explicit loading progress logs in training flow.
   - Re-run baseline with smaller first pass (`max-samples 300`, `epochs 2`) and then scale up.
 - Sub-run R1 (2026-02-22):
   - Command:
     - `python -m training.train_lstm --max-samples 100 --epochs 1`
   - Output:
     - `Loaded splits: train=100 val=100 test=100`
     - `After class filtering: train=100 val=0 num_classes=99 top_k=200`
     - `Epoch 001 | train_loss=4.6174 train_acc=0.0000 | val_loss=0.0000 val_acc=0.0000`
   - Artifacts updated:
     - `artifacts/lstm_best.pt`
     - `artifacts/label_to_id.json`
     - `artifacts/lstm_meta.json`
   - Interpretation:
     - Pipeline execution is healthy.
     - Validation set collapsed to zero examples after class filtering, so this run is only a technical smoke/baseline sanity run, not quality evaluation.
 - Sub-run R2 setup (2026-02-22):
   - Goal:
     - Force train/val class overlap to obtain meaningful validation metrics.
   - Code adjustment:
     - Added CLI override `--top-k` in `training/train_lstm.py`.
   - Planned command:
     - `python -m training.train_lstm --max-samples 200 --epochs 2 --top-k 20`
 - Sub-run R2 result (2026-02-22):
   - Output:
     - `Loaded splits: train=200 val=200 test=200`
     - `After class filtering: train=22 val=0 num_classes=20 top_k=20`
     - `Epoch 001 | train_loss=3.0104 train_acc=0.0000 | val_loss=0.0000 val_acc=0.0000`
     - `Epoch 002 | train_loss=2.9825 train_acc=0.0909 | val_loss=0.0000 val_acc=0.0000`
   - Interpretation:
     - Even with lower `top_k`, val remained zero due poor label overlap in this small sample window.
     - Current split-loading strategy is not suitable for quick low-sample evaluation at sentence-label granularity.
   - Next adjustment:
     - Implement evaluation-friendly sampling strategy:
       - build train/val from a shared candidate pool with guaranteed class overlap, or
       - add minimum val coverage check and auto-widen sample pool.
 - Sub-run R3 code update (2026-02-23):
   - Owner: Codex
   - Objective:
     - Enforce evaluation-safe class selection so validation metrics cannot silently collapse to zero.
   - Implementation:
     - Updated `training/train_lstm.py`:
       - Added overlap label map builder using intersection of train/val labels.
       - Added label coverage logging (`train_labels`, `val_labels`, `overlap`).
       - Added hard fail when no overlapping labels are found.
       - Added hard fail when filtered validation split becomes empty.
   - Result:
     - Training now fails fast with actionable errors instead of reporting zeroed validation metrics.
     - Baseline runs must satisfy train/val label overlap before epoch training proceeds.
   - Status: Done
   - Next:
     - Re-run baseline with wider sample window to ensure overlap:
       - `python -m training.train_lstm --max-samples 1000 --epochs 3 --top-k 50`
 - Sub-run R4 code update (2026-02-23):
   - Owner: Codex
   - Objective:
     - Replace brittle official-split sampling for baseline classification with overlap-safe pooled splitting.
   - Implementation:
     - Updated `training/train_lstm.py` with pooled split support:
       - Added CLI args:
         - `--split-mode {official,pooled}` (default: `pooled`)
         - `--min-class-count` (default: `2`)
         - `--seed` (default: `42`)
       - Added label-wise pooled splitter that:
         - merges train/val/test into one pool,
         - keeps labels with frequency >= `min_class_count`,
         - performs per-label train/val/test allocation with guaranteed train+val presence.
       - Kept overlap filtering and fail-fast checks after split generation.
   - Result:
     - Baseline training now uses a reproducible class-overlapping split by default, avoiding near-zero effective train/val sets caused by sentence-level split disjointness.
   - Status: Done
   - Next:
     - Run new pooled baseline:
       - `python -m training.train_lstm --max-samples 1000 --epochs 3 --top-k 50 --split-mode pooled --min-class-count 2`
 - Sub-run R5 dataset-path compatibility fix (2026-02-23):
   - Owner: Codex
   - Objective:
     - Resolve `FileNotFoundError` when dataset root uses a different folder nesting/layout than strict expected paths.
   - Problem:
     - Baseline run failed while resolving split files under:
       - `C:\Users\rajit\Datasets\How2Sign`
     - Existing resolver only supported two exact layouts.
   - Implementation:
     - Updated `data/adapters/how2sign.py`:
       - Added candidate-root detection (`root` + one-level child directories).
       - Added flexible sentence-level variant discovery under `sentence_level/<split>/...`.
       - Added CSV candidate picker with preference order:
         - exact `how2sign_<split>.csv`
         - name containing `how2sign` + split
         - name containing split.
     - Updated `scripts/validate_how2sign.py` with same resolver logic to keep validation/training path handling consistent.
   - Result:
     - Loader and validator now handle more real-world How2Sign directory variants without manual code edits.
   - Status: Done
   - Next:
     - Re-run pooled baseline command on target machine and capture output metrics.
 - Sub-run R6 recursive path resolver hardening (2026-02-24):
   - Owner: Codex
   - Objective:
     - Handle additional non-standard How2Sign extracted trees where split files are nested deeper than one level.
   - Implementation:
     - Updated `data/adapters/how2sign.py`:
       - Added fully recursive fallback search for:
         - split CSV files (`**/*.csv`, split-aware),
         - keypoint roots (`**/openpose_output/json`).
       - Added split-token scoring to choose the best CSV+JSON pair for each split.
     - Updated `scripts/validate_how2sign.py` with the same recursive fallback + scoring logic.
   - Result:
     - Resolver now supports strict layouts, one-level nested roots, sentence-level variants, and deep recursive layouts under dataset root.
   - Status: Done
   - Next:
     - Re-run baseline command with explicit dataset root and capture first resolver logs/output.
 - Sub-run R7 web realtime integration (2026-02-24):
   - Owner: Codex
   - Objective:
     - Implement web-app realtime pipeline that works before final model training completes.
   - Implementation:
     - Updated `app/app.py`:
       - Added SocketIO background capture/emission loop.
       - Added runtime modes: mock inference (default) and real inference.
       - Added automatic fallback to mock mode when artifacts are missing or model load fails.
       - Added API routes: `/health`, `/status`, `/predict`.
       - Added runtime config/env support (`camera_index`, `emit_interval_ms`, model artifact paths, `use_mock_inference`).
     - Updated `app/templates/index.html`:
       - Added SocketIO client script and live metadata fields.
     - Updated `app/static/app.js`:
       - Added socket connection handling and live `status` + `prediction` event rendering.
     - Updated `app/static/styles.css`:
       - Added styling for live status and prediction metadata.
     - Updated `config/default.yaml`:
       - Added realtime settings defaults (`use_mock_inference: true`).
   - Result:
     - Web UI can now show live streamed predictions in mock mode immediately.
     - Realtime plumbing is ready to switch to actual model inference after stable multi-class artifacts are produced.
   - Status: Done
   - Next:
     - Complete baseline training run (`max-samples 1000`) and then set `use_mock_inference: false` to validate real inference path.
 - Sub-run R8 pooled baseline success on relocated dataset root (2026-02-24):
   - Owner: Rajit + Codex
   - Objective:
     - Validate stable LSTM baseline training after moving dataset to `D:\DATASETS\How2Sign`.
   - Command:
     - `python -m training.train_lstm --dataset-root "D:\DATASETS\How2Sign" --max-samples 300 --epochs 1 --top-k 50 --split-mode pooled --min-class-count 2`
   - Output:
     - `Loaded splits: train=300 val=300 test=300 (max_samples_per_split=300)`
     - `Pooled split mode: pool_size=900 kept_labels=32 min_class_count=2 seed=42`
     - `Pooled split sizes: train=32 val=32 test=1`
     - `Label coverage: train_labels=32 val_labels=32 overlap=32`
     - `After class filtering: train=32 val=32 num_classes=32 top_k=50`
     - `Dataset tensors: seq_len=30 train_size=32 val_size=32`
     - `Epoch 001 | train_loss=3.4775 train_acc=0.0000 | val_loss=3.4622 val_acc=0.0312`
     - `Saved best model to: artifacts\lstm_best.pt`
   - Result:
     - End-to-end training run completed successfully with non-empty overlapping train/val classes.
     - Artifacts updated (`lstm_best.pt`, `label_to_id.json`, `lstm_meta.json`).
   - Status: Done
   - Next:
     - Run scaled baseline:
       - `python -m training.train_lstm --dataset-root "D:\DATASETS\How2Sign" --max-samples 1000 --epochs 3 --top-k 50 --split-mode pooled --min-class-count 2`
 - Sub-run R9 scaled pooled baseline on local dataset root (2026-02-24):
   - Owner: Rajit
   - Objective:
     - Run first meaningful scaled baseline with non-empty overlapping train/val classes.
   - Command:
     - `python -m training.train_lstm --max-samples 1000 --epochs 3 --top-k 50 --split-mode pooled --min-class-count 2`
     - `HOW2SIGN_ROOT=C:\Users\rajit\Datasets\How2Sign`
   - Output:
     - `Loaded splits: train=1000 val=1000 test=1000 (max_samples_per_split=1000)`
     - `Pooled split mode: pool_size=3000 kept_labels=50 min_class_count=2 seed=42`
     - `Pooled split sizes: train=69 val=51 test=13`
     - `Label coverage: train_labels=50 val_labels=50 overlap=50`
     - `After class filtering: train=69 val=51 num_classes=50 top_k=50`
     - `Dataset tensors: seq_len=30 train_size=69 val_size=51`
     - `Epoch 001 | train_loss=3.9085 train_acc=0.0870 | val_loss=3.8935 val_acc=0.0392`
     - `Epoch 002 | train_loss=3.8453 train_acc=0.1594 | val_loss=3.8714 val_acc=0.0588`
     - `Epoch 003 | train_loss=3.7739 train_acc=0.1884 | val_loss=3.8531 val_acc=0.0588`
     - `Saved best model to: artifacts/lstm_best.pt`
   - Result:
     - End-to-end scaled baseline completed successfully.
     - Previous `val=0` issue is resolved for pooled training mode.
     - Baseline validation is above random chance for 50 classes, but still low and needs further improvement.
   - Status: Done
   - Next:
     - Increase data and epochs for stronger baseline:
       - `python -m training.train_lstm --max-samples 2000 --epochs 5 --top-k 50 --split-mode pooled --min-class-count 2`
 - Sub-run R9 real-mode web demo validation (2026-02-24):
   - Owner: Rajit + Codex
   - Objective:
     - Validate end-to-end Flask + SocketIO demo in real inference mode using the latest trained baseline artifacts.
   - Implementation:
     - Updated `config/default.yaml`:
       - Set `use_mock_inference: false`.
       - Confirmed artifact paths:
         - `artifacts/lstm_best.pt`
         - `artifacts/label_to_id.json`
         - `artifacts/lstm_meta.json`
     - Fixed web client Socket.IO integration:
       - Updated `app/templates/index.html` to load Socket.IO client from CDN (`https://cdn.socket.io/4.7.5/socket.io.min.js`).
     - Installed runtime dependencies in Python 3.11 environment and launched app:
       - `python -m app.app`
   - Validation:
     - Localhost demo:
       - `http://127.0.0.1:5000/` showed:
         - `Connected. mode=real`
         - live prediction text updates
         - live confidence + timestamp updates
     - LAN demo:
       - `http://192.168.0.107:5000/` showed:
         - active Socket.IO connection
         - same real-mode prediction stream
     - Server logs confirmed successful Socket.IO handshake:
       - `GET/POST /socket.io ... 200`
   - Result:
     - Realtime web demo is operational in real inference mode on both localhost and LAN.
     - Integration milestone achieved without waiting for larger (`max-samples 1000`) training runs.
   - Status: Done
   - Next:
     - Improve model quality (confidence/accuracy) via larger staged training runs (`500 -> 700 -> 1000`) and evaluation reporting.
 - Sub-run R10 cache pipeline implementation (2026-02-25):
   - Owner: Codex
   - Objective:
     - Reduce repeated training runtime by avoiding JSON parsing on every run.
   - Implementation:
     - Added cache builder:
       - `scripts/build_how2sign_cache.py`
       - Loads How2Sign once and saves serialized split data to `.pt`.
     - Updated training to support cache input:
       - `training/train_lstm.py`
       - Added `--cache-path` argument.
       - When provided, trainer loads cached splits directly and skips dataset JSON parsing.
     - Updated docs:
       - `README.md` with cache build/train commands.
       - `scripts/README.md` with cache build command.
   - Expected result:
     - Same model input/features as current pipeline, significantly faster repeated runs.
   - Status: Done
   - Next:
     - Build cache and run first cached training benchmark:
       - `python -m scripts.build_how2sign_cache --dataset-root "C:\Users\rajit\Datasets\How2Sign" --max-samples 1000 --out artifacts/cache/how2sign_cache_1000.pt`
       - `python -m training.train_lstm --cache-path artifacts/cache/how2sign_cache_1000.pt --epochs 3 --top-k 50 --split-mode pooled --min-class-count 2`
 - Sub-run R10 cache benchmark validation (2026-02-25):
   - Owner: Codex
   - Objective:
     - Verify cache build and cache-based training path works end-to-end.
   - Commands:
     - `python -m scripts.build_how2sign_cache --max-samples 200 --out artifacts/cache/how2sign_cache_200.pt`
     - `python -m training.train_lstm --cache-path artifacts/cache/how2sign_cache_200.pt --epochs 1 --top-k 20 --split-mode pooled --min-class-count 2`
   - Output:
     - `Loaded cached splits: train=200 val=200 test=200`
     - `Pooled split sizes: train=20 val=20 test=1`
     - `After class filtering: train=20 val=20 num_classes=20 top_k=20`
     - `Epoch 001 | train_loss=3.0054 train_acc=0.0000 | val_loss=2.9898 val_acc=0.1000`
   - Result:
     - Cache pipeline is operational.
     - Cached training starts immediately and avoids expensive JSON parse step for repeat runs.
   - Status: Done
   - Next:
     - Build full cache for target run size (`1000` or `2000`) and continue baseline scaling with `--cache-path`.


### A-012 | Chunked Cache Pipeline for 1000+ Samples
- Date: 2026-02-25
- Objective: Prevent `MemoryError` during cache generation at higher sample counts.
- Implementation:
  - Updated `scripts/build_how2sign_cache.py` to save chunked cache parts per split instead of one huge payload.
  - Added cache options:
    - `--part-size` (default: `250`)
    - `--dtype` (`float16` default, supports `float32`)
  - Added backward-compatible cache index `.pt` file that points to chunk directory.
  - Updated `training/train_lstm.py` cache loader to support:
    - legacy single-file cache,
    - new chunked cache index file,
    - direct chunked cache directory.
- Result:
  - Cache generation is now memory-safe for larger sample sizes on low/medium RAM systems.
- Status: Done
- Next:
  - Rebuild cache and train from cache:
    - `python -m scripts.build_how2sign_cache --max-samples 1000 --out artifacts/cache/how2sign_cache_1000.pt`
    - `python -m training.train_lstm --cache-path artifacts/cache/how2sign_cache_1000.pt --epochs 5 --top-k 50 --split-mode pooled --min-class-count 2`


 - Sub-run R11 cached baseline comparison (2026-02-25):
   - Owner: Rajit
   - Objective:
     - Compare class-budget settings on cached splits and select strongest baseline.
   - Shared setup:
     - `python -m training.train_lstm --cache-path artifacts/cache/how2sign_cache_1000.pt --split-mode pooled --min-class-count 2`
   - Run A (`--epochs 5 --top-k 50`):
     - `Loaded cached splits: train=1000 val=1000 test=1000`
     - `Pooled split sizes: train=69 val=51 test=13`
     - `After class filtering: train=69 val=51 num_classes=50`
     - Best val accuracy: `0.0588`
   - Run B (`--epochs 12 --top-k 30`):
     - `Pooled split sizes: train=49 val=31 test=13`
     - `After class filtering: train=49 val=31 num_classes=30`
     - Best val accuracy: `0.1290`
   - Run C (`--epochs 12 --top-k 20`):
     - `Pooled split sizes: train=39 val=21 test=13`
     - `After class filtering: train=39 val=21 num_classes=20`
     - Best val accuracy: `0.1905`
     - Final epoch: `train_acc=0.4359`, `val_loss=2.5611`, `val_acc=0.1905`
   - Result:
     - Best cached baseline selected: `top_k=20`, `epochs=12` (val_acc `0.1905`).
     - Frozen baseline artifacts created:
       - `artifacts/lstm_best_topk20_e12.pt`
       - `artifacts/label_to_id_topk20_e12.json`
       - `artifacts/lstm_meta_topk20_e12.json`
   - Status: Done
   - Next:
     - Run realtime app validation with frozen baseline artifacts and capture demo observations.


 - Sub-run R12 realtime app smoke validation with frozen baseline (2026-02-26):
   - Owner: Codex
   - Objective:
     - Verify realtime app boots in real mode with frozen top-k20 artifacts.
   - Runtime overrides used:
     - `USE_MOCK_INFERENCE=false`
     - `LSTM_WEIGHTS=artifacts/lstm_best_topk20_e12.pt`
     - `LSTM_LABELS=artifacts/label_to_id_topk20_e12.json`
     - `LSTM_META=artifacts/lstm_meta_topk20_e12.json`
   - Smoke checks:
     - `GET /health` -> `{"ok": true}`
     - `GET /status` -> `mode=real`, `model_loaded=true`, `status=Realtime pipeline started.`
     - `GET /predict` -> `COLLECTING_FRAMES` payload observed (camera stream active, sequence warm-up pending).
   - Result:
     - Realtime server path is healthy with frozen baseline artifacts.
     - Model load and prediction API path validated.
   - Status: Done
   - Next:
     - Manual webcam gesture validation (10 attempts) and log correctness + confidence spread.


 - Sub-run R13 web camera self-preview integration (2026-02-26):
   - Owner: Codex
   - Objective:
     - Let users see themselves in web app mode while predictions stream.
   - Implementation:
     - Updated `app/templates/index.html` with a `video#self-view` preview panel.
     - Updated `app/static/app.js` to initialize browser camera preview using `getUserMedia()`.
     - Added preview status messages and cleanup on page unload.
     - Updated `app/static/styles.css` with responsive two-panel layout.
   - Result:
     - Browser now shows live self-view alongside prediction output.
   - Status: Done
   - Next:
     - Run manual realtime validation and log 10-attempt behavior with confidence trends.

 - Sub-run R14 realtime output stabilization and confidence gating (2026-02-26):
   - Owner: Codex
   - Objective:
     - Reduce noisy repeated predictions when user is idle or confidence is low.
   - Implementation:
     - Updated `app/app.py`:
       - Added runtime settings:
         - `min_confidence` (`MIN_PRED_CONFIDENCE` / `min_pred_confidence`, default `0.30`)
         - `stability_frames` (`STABILITY_FRAMES` / `stability_frames`, default `3`)
       - Added low-confidence gate:
         - shows `No confident sign` when confidence is below threshold.
       - Added temporal stabilization:
         - requires repeated same label across N frames before showing stable prediction.
         - shows `Stabilizing gesture...` while warming label stability.
     - Updated `config/default.yaml` with:
       - `min_pred_confidence: 0.30`
       - `stability_frames: 3`
   - Result:
     - Realtime output is less sticky/noisy and avoids always forcing a label when confidence is weak.
   - Status: Done
   - Next:
     - Tune thresholds during manual demo validation (e.g., confidence `0.30-0.45`, stability `3-5`).


 - Sub-run R15 realtime threshold tuning + sign detection procedure (2026-02-26):
   - Owner: Rajit + Codex
   - Objective:
     - Understand why realtime output stays `No confident sign` and define operating procedure.
   - Observation:
     - Confidence increases slightly when hands are visible (`~0.17-0.18`) and drops when moving away.
     - Waving/`hi` still remains below gate, so output stays `No confident sign`.
   - Interpretation:
     - Pipeline is functioning (landmark/motion sensitivity is present), but current model confidence is low for reliable class commitment at default threshold.
   - Runtime tuning guidance:
     - Start app with lower gate for demo validation:
       - `MIN_PRED_CONFIDENCE=0.15`
       - `STABILITY_FRAMES=2`
     - If still strict, reduce confidence gate gradually (down to ~`0.12`).
     - If output becomes noisy, raise gate toward `0.18-0.22`.
   - Sign detection procedure (current baseline):
     - Use real mode with frozen artifacts (`topk20_e12`).
     - Keep upper body + both hands in frame.
     - Neutral pose ~1 sec before sign.
     - Perform one clear sign for 2-3 sec.
     - Pause ~1 sec between signs.
     - Ensure good front lighting and simple background.
   - Status: Done
   - Next:
     - Manual 10-attempt validation with tuned thresholds and log expected vs predicted labels + confidence.


$env:USE_MOCK_INFERENCE="false"
$env:LSTM_WEIGHTS="artifacts/lstm_best_topk20_e12.pt"
$env:LSTM_LABELS="artifacts/label_to_id_topk20_e12.json"
$env:LSTM_META="artifacts/lstm_meta_topk20_e12.json"
$env:MIN_PRED_CONFIDENCE="0.15"
$env:STABILITY_FRAMES="2"
python -m app.app


 - Sub-run R16 cached baseline (2000 samples, top-k=20) (2026-02-26):
   - Owner: Rajit
   - Objective:
     - Improve baseline accuracy using 2000-sample cache.
   - Command:
     - `python -m training.train_lstm --cache-path artifacts/cache/how2sign_cache_2000.pt --epochs 20 --top-k 20 --split-mode pooled --min-class-count 2`
   - Output:
     - `Loaded cached splits: train=2000 val=1737 test=2000`
     - `Pooled split sizes: train=57 val=22 test=22`
     - `After class filtering: train=57 val=22 num_classes=20`
     - Best val accuracy: `0.2727` (epochs 13-19)
     - Final epoch: `train_acc=0.4386`, `val_loss=2.8142`, `val_acc=0.1818`
   - Result:
     - New best baseline achieved: `val_acc=0.2727` with `top_k=20`, `epochs=20`.
     - Artifacts updated: `artifacts/lstm_best.pt`, `artifacts/label_to_id.json`, `artifacts/lstm_meta.json`.
   - Status: Done
   - Next:
     - Freeze new best artifacts with versioned filenames.
     - Re-run realtime validation with updated model and tuned thresholds.
