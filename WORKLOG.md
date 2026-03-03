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

## 2026-03-02 - ASL Alphabet Training Script (Standalone)

- Change: Rewrote `training/train_asl_alphabet.py` to be standalone for ASL alphabet images using `MediaPipeExtractor` and `LSTMClassifier` without touching WLASL code; reason: prepare ASL-only pipeline; outcome: supports max 500 images per class, stratified 80/20 split, seq_len=1, DataLoader batching, GPU/CPU selection, and saves model + label map.
- Change: Added repo-root `mediapipe_extractor.py` shim importing from `features.mediapipe_extractor`; reason: satisfy `from mediapipe_extractor import MediaPipeExtractor` without altering WLASL code; outcome: ASL training script can import successfully.
- Run 1: `python -m training.train_asl_alphabet` (dataset root `D:\personal\Project\Datasets\ASL\asl_alphabet_train\asl_alphabet_train`, max 500 images/class, batch_size=64, epochs=20) -> best val acc=0.3666.
- Change: Upgraded `training/train_asl_alphabet.py` with CLI args, per-class sampling cap, shoulder-based normalization, and mini-batch training; reason: improve scalability and control for ASL-only training; outcome: configurable runs with explicit summary metrics.
- Run 2: `python -m training.train_asl_alphabet` (defaults: max_images_per_class=1000, batch_size=128, epochs=50, hidden_dim=256, layers=2, lr=5e-4) -> total=29000, train=23200, val=5800, best val acc=0.2974.
- Change: Added standalone `training/train_asl_cnn.py` for ASL alphabet CNN training with ImageFolder, 4-block Conv+BN+Pool, ImageNet normalization, CLI-configurable batch/epochs/lr/img-size, and test-set evaluation; reason: provide CNN baseline for ASL alphabet without touching WLASL or MediaPipe code; outcome: script saves `artifacts/asl_cnn_best.pt` and label map plus prints per-epoch metrics and test accuracy.
- Change: Simplified `training/train_asl_cnn.py` to train/validate only on the train folder with an 80/20 random split; reason: test folder is flat (no class subfolders) and incompatible with ImageFolder; outcome: best model saved from validation and run summary printed without test evaluation.

## 2026-03-03 - How2Sign Training Prep (Imported From Legacy Repo)

- Change: Added How2Sign adapter `data/adapters/how2sign.py` from legacy project; reason: enable dataset loading for sentence-level keypoints; outcome: How2Sign splits can be resolved across multiple layout conventions.
- Change: Added `training/train_how2sign_lstm.py` and `training/train_how2sign_nlp.py` from legacy project; reason: enable LSTM training and NLP pair generation without touching WLASL/ASL scripts; outcome: How2Sign training and NLP data prep are available in this repo.
- Change: Added `scripts/build_how2sign_cache.py`, `scripts/validate_how2sign.py`, and `scripts/README.md`; reason: support dataset validation/manifests and caching; outcome: can validate dataset and build cache locally for faster training.
- Run 1: `python -m training.train_how2sign_lstm --cache-path "artifacts\cache\how2sign_cache_6000.pt"` -> cached splits train=6000, val=1737, test=2341; pooled split sizes train=176, val=105, test=68; best val acc=0.0762.
- Run 2: `python -m scripts.build_how2sign_cache --dataset-root "D:\personal\Project\Datasets\How2Sign" --max-samples 9000 --out artifacts\cache\how2sign_cache_9000.pt` -> cached splits train=9000, val=1737, test=2341; cache dir `artifacts\cache\how2sign_cache_9000`.
- Run 4: `python -m scripts.build_how2sign_cache --dataset-root "D:\personal\Project\Datasets\How2Sign" --max-samples 20000 --out artifacts\cache\how2sign_cache_20000.pt` -> cached splits train=20000, val=1737, test=2341; cache dir `artifacts\cache\how2sign_cache_20000`.
- Run 5: `python -m training.train_how2sign_lstm --cache-path "artifacts\cache\how2sign_cache_20000.pt" --top-k 5 --split-mode pooled --seq-len 60 --hidden-dim 128 --layers 1 --epochs 60 --lr 5e-4 --dropout 0.1 --bidirectional` -> pooled split sizes train=124, val=26; best val acc=0.6154; snapshot saved to `artifacts\how2sign_20000_top5_seq60_h128_l1_lr0.0005_do0.1_bi`.
- Change: Added `--seq-len`, `--hidden-dim`, and `--layers` overrides to `training/train_how2sign_lstm.py`; reason: allow explicit sequence length and LSTM size overrides for How2Sign experiments; outcome: CLI can override config defaults without touching WLASL/ASL code.
- Change: Updated `training/train_how2sign_lstm.py` to save a snapshot of the best model into a config-tagged folder under `artifacts/` while still writing `artifacts/lstm_best.pt`; reason: preserve best checkpoints for each run; outcome: each run saves a unique folder based on top_k/seq_len/hidden_dim/layers/lr/dropout/bidirectional.
- Change: Snapshot folder now also includes cache size token (from cache filename) in the tag (e.g., `how2sign_20000_...`); reason: distinguish runs by dataset size; outcome: artifacts are uniquely named per cache size.
- Change: Updated `config/default.yaml` dataset_root to `D:/personal/Project/Datasets/How2Sign`; reason: ensure How2Sign training shows correct root by default; outcome: dataset_root now points to How2Sign unless overridden.
- Run 3: `python -m training.train_how2sign_lstm --cache-path "artifacts\cache\how2sign_cache_9000.pt" --top-k 5 --seq-len 60 --hidden-dim 128 --layers 1 --split-mode pooled --epochs 60 --lr 5e-4 --dropout 0.1 --bidirectional` -> pooled split sizes train=62, val=14; best val acc=0.6429; saved snapshot to `artifacts\how2sign_top5_seq60_bi`.
