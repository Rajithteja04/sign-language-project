# Worklog

## 2026-03-01 - WLASL Word-Level Refactor

- Added `data/adapters/wlasl.py`:
  - Reads WLASL metadata (`WLASL_v0.3.json` / `WLASL_v0.3.csv`)
  - Uses official `frame_start`/`frame_end` segment boundaries (not full raw video)
  - Honors official `train`/`test` splits and creates `val` from train holdout
  - Skips missing/corrupted videos with warnings (no crash)
  - Prints sanity stats: train/val/test counts + unique gloss count
  - Keeps top-k frequent gloss classes with minimum frequency >= 5
  - Loads videos via OpenCV and extracts MediaPipe features frame-wise
  - Builds fixed-length sequences (30 frames) with pad/truncate
  - Added optional feature cache for faster reruns (`--cache-features`)
  - Returns `DatasetSplit(train/val/test)` of `SequenceExample`
- Reworked `features/mediapipe_extractor.py`:
  - Standardized feature vector to `FEATURE_DIM=411`
  - Added strict dimension check
- Replaced `training/train_lstm.py`:
  - Added `--dataset wlasl|how2sign` (How2Sign training disabled by design)
  - Uses WLASL adapter + top-k word classes
  - Uses `CrossEntropyLoss`
  - Saves only required artifacts:
    - `artifacts/lstm_best.pt`
    - `artifacts/label_to_id.json`
    - `artifacts/lstm_meta.json`
  - Optional confusion matrix export via `--save-confusion-matrix`
- Replaced `inference/realtime.py`:
  - Loads trained artifacts
  - Uses confidence threshold (default `0.6`)
  - Uses majority voting over last 5 predictions
  - Uses cooldown timer to reduce repeated word spam
- Updated defaults in `config/default.yaml` for word-level setup
- Updated `.gitignore`:
  - `artifacts/cache/`
  - `*.part*.pt`
- Updated `README.md` with new training/inference commands
