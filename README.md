# Sign Language Recognition Suite

This repo now supports multiple landmark-based LSTM pipelines:

- **WLASL** word-level glosses (baseline demo).
- **How2Sign** sentence-level phrases (top-5 for live demos).
- **LSA64** isolated Argentine Sign Language words (64 classes × 50 clips).
- **MS-ASL** trimmed YouTube clips with configurable top-k subsets.

All datasets share the same MediaPipe Holistic extractor (411-dim vectors), sequence padding/truncation, and LSTMClassifier architecture. Each training script saves checkpoints in dataset-specific folders under `artifacts/` so runs never overwrite the live-demo weights.

## Setup

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

## WLASL (Word-Level)

```powershell
python -m training.train_lstm ^
    --dataset wlasl ^
    --dataset-root "D:\personal\Project\Datasets\WLASL\WLASL-master\start_kit" ^
    --top-k 20 ^
    --seq-len 30 ^
    --cache-features ^
    --cache-dir "artifacts\cache\wlasl_features" ^
    --normalize ^
    --epochs 40
```

Artifacts land in `artifacts/wlasl/<run_tag>/` (e.g., `top6_seq30_h128_l2_norm`). Copy the snapshot you want into `artifacts/lstm_best.pt`, `label_to_id.json`, `lstm_meta.json` when you need the app to load it.

## How2Sign (Sentence-Level)

Build or reuse caches via `scripts/build_how2sign_cache.py`, then train:

```powershell
python -m training.train_how2sign_lstm ^
    --cache-path "artifacts\cache\how2sign_cache_30000.pt" ^
    --top-k 5 ^
    --seq-len 45 ^
    --hidden-dim 128 ^
    --layers 1 ^
    --epochs 80 ^
    --lr 5e-4 ^
    --dropout 0.1 ^
    --bidirectional
```

Each run snapshots to `artifacts/how2sign_<cache>_top<k>_seq.../` and also refreshes the top-level artifacts if you opt in.

## LSA64 (Argentine Sign Language)

Cache once (optional but recommended):

```powershell
python -m training.train_lsa64_lstm ^
    --dataset-root "D:\personal\Project\Datasets\LSA64" ^
    --seq-len 30 ^
    --top-k 64 ^
    --cache-features ^
    --cache-dir "artifacts\cache\lsa64_seq30" ^
    --epochs 0
```

Then train (example top-5 run):

```powershell
python -m training.train_lsa64_lstm ^
    --dataset-root "D:\personal\Project\Datasets\LSA64" ^
    --seq-len 30 ^
    --top-k 5 ^
    --cache-features ^
    --cache-dir "artifacts\cache\lsa64_seq30" ^
    --normalize ^
    --epochs 40 ^
    --batch-size 16 ^
    --learning-rate 5e-4 ^
    --hidden-dim 128 ^
    --layers 2
```

## MS-ASL (YouTube Clips)

Download/train/val/test folders sit under `D:\personal\Project\Datasets\MSASL`. Run:

```powershell
python -m training.train_msasl_lstm ^
    --dataset-root "D:\personal\Project\Datasets\MSASL" ^
    --seq-len 30 ^
    --top-k 10 ^
    --cache-features ^
    --cache-dir "artifacts\cache\msasl_seq30" ^
    --normalize ^
    --epochs 40 ^
    --batch-size 16 ^
    --learning-rate 5e-4 ^
    --hidden-dim 128 ^
    --layers 2
```

Snapshots appear in `artifacts/msasl/msasl_top<k>_seq30_h128_l2_bs16_lr5e-04/`. Repeat with different `--top-k` to compare demo subsets.

## Realtime Demo (Module I / II)

```powershell
python -m inference.realtime ^
    --weights artifacts/lstm_best.pt ^
    --labels artifacts/label_to_id.json ^
    --meta artifacts/lstm_meta.json
```

- `q` quits, `c` clears committed words.
- Mock mode hides predictions and only shows Module I telemetry; flip `WordRealtimeService.source_mode` to `live` after copying the right artifacts.

## Notes

- All datasets share the same MediaPipe Holistic extractor (`FEATURE_DIM=411`).
- Caches live in `artifacts/cache/<dataset>_seq<len>/`. Delete them if you need to rebuild from scratch.
- Final demo artifacts live in `artifacts/final_best/` so the UI can stay locked to a known-good model while experiments continue elsewhere.
