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

## 2026-03-02 - WLASL Training Prep, Runs, and Ablations (Start Kit)

- Context: Training on WLASL start_kit with `.venv311` and dataset root `D:\personal\Project\Datasets\WLASL\WLASL-master\start_kit`.
- Change: Fixed WLASL adapter handling of `frame_end = -1` and `val` split; reason: previously filtered out most samples; outcome: normal dataset sizes loaded.
- Change: Filter metadata by physically available video IDs and suppress per-file missing warnings; reason: reduce log spam and speed up load; outcome: no missing-file spam and faster iteration.
- Change: Added `--seq-len` CLI flag in `training/train_lstm.py`; reason: enable sequence length ablations without config edits; outcome: seq_len can be overridden per run.
- Change: Added optional per-frame normalization in WLASL preprocessing (`--normalize`); reason: reduce scale variance across samples; outcome: normalization applied before caching when enabled.
- Change: Added LSTM dropout (between layers + pre-FC) and default `num_layers=1`; reason: reduce overfitting and keep PyTorch dropout rules satisfied; outcome: training can use `--layers` to override.
- Config note: `config/default.yaml` now shows `batch_size: 8` (non-default for earlier runs); verify before next baseline if `16` is desired.

- Run 1: `python -m training.train_lstm --dataset wlasl --dataset-root "D:\personal\Project\Datasets\WLASL\WLASL-master\start_kit" --cache-features --cache-dir "artifacts\cache\wlasl_features" --top-k 100 --epochs 30 --val-ratio 0.10` -> train=843, val=286, test=119, classes=100, best val acc=0.1154.
- Run 2: `python -m training.train_lstm --dataset wlasl --dataset-root "D:\personal\Project\Datasets\WLASL\WLASL-master\start_kit" --cache-features --cache-dir "artifacts\cache\wlasl_features" --top-k 20 --epochs 40 --val-ratio 0.10` -> train=211, val=64, test=32, classes=20, best val acc=0.2188.
- Run 3: `python -m training.train_lstm --dataset wlasl --dataset-root "D:\personal\Project\Datasets\WLASL\WLASL-master\start_kit" --cache-features --cache-dir "artifacts\cache\wlasl_features" --top-k 20 --epochs 25 --val-ratio 0.10 --hidden-dim 128` -> train=211, val=64, test=32, classes=20, best val acc=0.2812.
- Run 4: `python -m training.train_lstm --dataset wlasl --dataset-root "D:\personal\Project\Datasets\WLASL\WLASL-master\start_kit" --cache-features --cache-dir "artifacts\cache\wlasl_features" --top-k 20 --epochs 25 --val-ratio 0.10 --hidden-dim 128 --layers 1` -> train=211, val=64, test=32, classes=20, best val acc=0.2344.
- Run 5: `python -m training.train_lstm --dataset wlasl --dataset-root "D:\personal\Project\Datasets\WLASL\WLASL-master\start_kit" --cache-features --cache-dir "artifacts\cache\wlasl_features_seq45" --top-k 20 --epochs 25 --val-ratio 0.10 --seq-len 45 --layers 2 --hidden-dim 128` -> train=211, val=64, test=32, classes=20, best val acc=0.2656.

- Comparison summary: Best val acc across runs is 0.2812 (Run 3, seq_len=30, top_k=20, hidden_dim=128, layers=2).
- Comparison summary: Seq-len impact (30 -> 45, layers=2, hidden_dim=128, top_k=20) decreased best val acc from 0.2812 to 0.2656.
- Comparison summary: Layers impact (2 -> 1, seq_len=30, hidden_dim=128, top_k=20) decreased best val acc from 0.2812 to 0.2344.
