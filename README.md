# Sign Language Translation (Full Pipeline)

This project scaffolds a full pipeline for real-time ASL sign-to-text translation:

- Landmark extraction with MediaPipe Holistic
- Temporal modeling with LSTM (sequence classification)
- Text refinement with a Transformer NLP module
- Real-time web UI using Flask + SocketIO

## Quick Start

1. Create a Python venv and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure dataset path per laptop (do not commit absolute paths):

Option A: environment variable

```powershell
$env:HOW2SIGN_ROOT="C:\Users\rajit\Datasets\How2Sign"
```

Option B: local config file

```powershell
Copy-Item config\local.example.yaml config\local.yaml
# edit config\local.yaml with your machine path
```

3. Train LSTM on extracted How2Sign keypoints:

```powershell
python -m training.train_lstm --max-samples 3000 --epochs 10
```

Optional: build and use cache for faster repeated training runs:

```powershell
python -m scripts.build_how2sign_cache --dataset-root "C:\Users\rajit\Datasets\How2Sign" --max-samples 3000 --out artifacts/cache/how2sign_cache.pt
python -m training.train_lstm --cache-path artifacts/cache/how2sign_cache.pt --epochs 10 --top-k 50 --split-mode pooled --min-class-count 2
```

Optional dataset validation (recommended before training):

```powershell
python -m scripts.validate_how2sign --dataset-root "C:\Users\rajit\Datasets\How2Sign"
```

This saves:
- `artifacts/lstm_best.pt`
- `artifacts/label_to_id.json`
- `artifacts/lstm_meta.json`

4. Build NLP training pairs (for Transformer fine-tuning):

```powershell
python -m training.train_nlp --max-samples 5000 --out artifacts/nlp_pairs.tsv
```

5. Run realtime inference (camera + MediaPipe + LSTM + Transformer correction):

```powershell
python -m inference.realtime --weights artifacts/lstm_best.pt --labels artifacts/label_to_id.json --meta artifacts/lstm_meta.json
```

Optional Transformer model:

```powershell
python -m inference.realtime --nlp-model grammarly/coedit-large
```

6. Run the web app:

```powershell
python -m app.app
```

## Dataset Strategy

We start with **How2Sign** and keep a dataset adapter interface so we can switch to **ASL STEM Wiki** later.

- Adapters: `data/adapters/how2sign.py`, `data/adapters/asl_stem_wiki.py`
- Shared format: `data/formats.py`

## Folder Structure

- `app/` Flask + SocketIO web UI
- `data/` dataset adapters + formats
- `features/` MediaPipe landmark extraction
- `models/` LSTM + Transformer modules (starter code)
- `training/` training scripts
- `inference/` realtime inference
- `config/` YAML configs
- `utils/` logging, IO utilities

## Notes

- Path precedence for training scripts:
  - `--dataset-root` CLI argument (highest)
  - `HOW2SIGN_ROOT` environment variable
  - `config/local.yaml`
  - `config/default.yaml` `dataset_root` (lowest)
- The How2Sign adapter supports both layouts:
  - Flat:
    - `<dataset_root>/how2sign_{train,val,test}.csv`
    - `<dataset_root>/{train,val,test}_2D_keypoints/openpose_output/json/...`
  - Sentence-level:
    - `<dataset_root>/sentence_level/{train,val,test}/text/en/raw_text/how2sign_{train,val,test}.csv`
    - `<dataset_root>/sentence_level/{train,val,test}/rgb_front/features/openpose_output/json/...`
- Runtime extraction stays on MediaPipe Holistic.
- Training and runtime keypoints are now unified to one canonical format:
  - topology/order: `pose25 + face70 + left hand21 + right hand21`
  - vector shape: `411 = (25+70+21+21)*3`
  - normalization: center by neck, scale by shoulder distance (bbox fallback)
