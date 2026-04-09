# LSA64 Subset-10 Training Guide

This document explains how to train a **fixed 10-word LSA64 model** on another system and make it the runtime model for the app.

## Goal
- Reduce confusion from 64-class model in live mode.
- Train only these 10 labels:
  - `BUY, GIVE, HELP, THANKS, WATER, FOOD, RICE, MILK, WHERE, NAME`
- Export artifacts for runtime:
  - `artifacts/lstm_best.pt`
  - `artifacts/label_to_id.json`
  - `artifacts/lstm_meta.json`

## 1) Open Project + Activate Env
```powershell
cd "C:\Users\rajit\FinalYearProject\SignLanguageTranslation\sign-language-project"
.\.venv311\Scripts\Activate.ps1
```

## 2) Train Subset-10 Model
```powershell
python -m training.train_lsa64_lstm `
  --dataset-root "C:\Users\rajit\Datasets\LSA64" `
  --seq-len 30 `
  --include-labels "BUY,GIVE,HELP,THANKS,WATER,FOOD,RICE,MILK,WHERE,NAME" `
  --normalize `
  --cache-features `
  --cache-dir "artifacts/cache/lsa64_subset10_seq30" `
  --epochs 40 `
  --batch-size 16 `
  --learning-rate 5e-4 `
  --hidden-dim 128 `
  --layers 2 `
  --out-dir "artifacts/lsa64"
```

Expected output run folder:
`artifacts/lsa64/lsa64_subset10_seq30_h128_l2_bs16_lr5e-04`

## 3) Promote Best Run to Runtime Artifacts
```powershell
$run="artifacts/lsa64/lsa64_subset10_seq30_h128_l2_bs16_lr5e-04"
Copy-Item "$run\lstm_best.pt" artifacts\lstm_best.pt -Force
Copy-Item "$run\label_to_id.json" artifacts\label_to_id.json -Force
Copy-Item "$run\lstm_meta.json" artifacts\lstm_meta.json -Force
```

## 4) Quick Runtime Check
```powershell
python -m scripts.preflight_runtime
python -m app.app
```

## Notes
- `--include-labels` ensures training uses only the specified 10 words.
- `--normalize` must stay enabled for consistency with runtime preprocessing.
- If dataset path differs on the other system, only update `--dataset-root`.
