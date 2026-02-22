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
