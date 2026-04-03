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

## Realtime Web App (Final Demo)

The web app now ships with the **LSA64 top-64** checkpoint by default. The files it reads are:

- `artifacts/lstm_best.pt` (copied from `artifacts/lsa64/lsa64_top64_seq30_h128_l2_bs16_lr5e-04/`)
- `artifacts/label_to_id.json` (tokenized gloss map, 64 entries)
- `artifacts/lstm_meta.json` (sequence length, hidden dim, dataset tag, etc.)

If you need to demo a different dataset, copy that run’s trio into either `artifacts/` or `artifacts/final_best/` before launching the app.

Run preflight:

```powershell
python -m scripts.preflight_runtime --camera-index 0
```

Start app:

```powershell
python -m app.app
```

Then open `http://127.0.0.1:5000` and do a hard refresh (`Ctrl+F5`) if browser cache is stale.

### Demo Checklist

1. Run `python -m scripts.preflight_runtime --camera-index 0` to verify MediaPipe, artifacts, and camera.
2. Launch `python -m app.app`, wait for the **Recognition Engine** dot to turn green, then open the UI.
3. Walk the signer through the **User Guidelines** modal (toolbar button) before recording.
4. Signing rhythm: neutral pose ≈1 s → perform sign ≈1 s → hold ≈2 s until the new chip appears.
5. Use **Clear Sentence** between takes; use **Stop** at the end to freeze the stream.
6. If the engine dot goes amber/red, restart the backend or reload the artifacts before continuing.

### Module III Sample Sentences

| Dataset | Gloss tokens | Output sentence |
| --- | --- | --- |
| MS-ASL | `COUSIN EAT` | My cousin is eating. |
| MS-ASL | `COUSIN EAT FINISH` | My cousin finished eating. |
| MS-ASL | `TEACHER NICE` | The teacher is nice. |
| MS-ASL | `FINISH TEACHER` | The teacher finished. |
| MS-ASL | `EAT NICE` | I want to eat, and it is nice. |
| LSA64 | `ARGENTINA HELP` | Argentina is helping. |
| LSA64 | `SON RUN` | My son is running. |
| LSA64 | `WHERE MAP` | Where is the map? |
| LSA64 | `CANDY GREEN` | Candy is green. |
| LSA64 | `BRIGHT` | It is bright. |

T5-small generates sentences when confident; otherwise, the new role-aware fallback ensures every gloss combination still yields a natural sentence.

## Notes

- All datasets share the same MediaPipe Holistic extractor (`FEATURE_DIM=411`).
- Caches live in `artifacts/cache/<dataset>_seq<len>/`. Delete them if you need to rebuild from scratch.
- Final demo artifacts live in `artifacts/final_best/` so the UI can stay locked to a known-good model while experiments continue elsewhere.

## Datasets & Artifacts

- `artifacts/wlasl/top6_seq30_h128_l2_norm/` holds the 6-word WLASL demo model (validation ~62%).
- `artifacts/how2sign_20000_top5_seq45_h128_l1_lr0.0005_do0.1_bi/` stores the 5-phrase How2Sign BiLSTM run (~56% val acc).
- `artifacts/msasl/msasl_top6_seq30_h128_l2_bs16_lr5e-04/` contains the MS-ASL top-6 checkpoint (~74% val acc).
- `artifacts/lsa64/lsa64_top64_seq30_h128_l2_bs16_lr5e-04/` now includes the 64-class LSA64 model (97% val acc) plus cached evaluation logs.
- `artifacts/plots/` holds comparison charts (accuracy vs. prior work, latency breakdown, per-label accuracy).

## Offline Evaluation

Use the new helpers to validate checkpoints without rerunning MediaPipe:

```powershell
# Single video sanity check
python -m scripts.run_video_inference `
    --video "D:\personal\Project\Datasets\LSA64\MAIN\LSA64\001\001_001_001.mp4" `
    --artifacts artifacts\lsa64\lsa64_top64_seq30_h128_l2_bs16_lr5e-04 `
    --topk 5

# Full-dataset sweep with cached sequences
python -m scripts.eval_lsa64_dataset `
    --dataset-root "D:\personal\Project\Datasets\LSA64\MAIN\LSA64" `
    --artifacts artifacts\lsa64\lsa64_top64_seq30_h128_l2_bs16_lr5e-04 `
    --csv artifacts\lsa64\lsa64_eval_cached.csv `
    --topk 5
```

`lstm_meta.json` now records `normalize: true` and `cache_dir`, so both scripts load the cached landmark tensors and finish in ~20 seconds.
