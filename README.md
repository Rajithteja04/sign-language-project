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

Recommended recent baseline run (How2Sign):

```powershell
python -m training.train_lstm --cache-path artifacts/cache/how2sign_cache_3000.pt --epochs 25 --top-k 12 --split-mode pooled --min-class-count 3
```

Frozen artifacts from this run:
- `artifacts/lstm_best_topk12_e25_cache3000.pt`
- `artifacts/label_to_id_topk12_e25_cache3000.json`
- `artifacts/lstm_meta_topk12_e25_cache3000.json`

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

## Web App Flow (Word/Sentence Mode)

The UI now supports:
- gesture category selector (`word` or `sentence`)
- self camera preview
- `Start` / `Stop` recognition buttons
- live word stream + assembled sentence output

At runtime:
- `word` mode: stable words are accumulated and sent to NLP for sentence formation.
- `sentence` mode: sentence prediction is directly passed through NLP correction.
- Preview remains visible while recognition runs (browser-camera frame streaming to backend).

Mode-specific artifact config keys (in `config/default.yaml`):
- `word_lstm_weights_path`, `word_labels_path`, `word_meta_path`
- `sentence_lstm_weights_path`, `sentence_labels_path`, `sentence_meta_path`

Example override to run with frozen How2Sign artifacts:

```powershell
$env:USE_MOCK_INFERENCE="false"
$env:SENTENCE_LSTM_WEIGHTS="artifacts/lstm_best_topk12_e25_cache3000.pt"
$env:SENTENCE_LSTM_LABELS="artifacts/label_to_id_topk12_e25_cache3000.json"
$env:SENTENCE_LSTM_META="artifacts/lstm_meta_topk12_e25_cache3000.json"
python -m app.app
```

## NLP Backend Switch (Transformer / Gemini)

Set NLP backend before launching app.

Local Transformer NLP:

```powershell
$env:NLP_BACKEND="transformer"
$env:NLP_MODEL="google/flan-t5-small"
python -m app.app
```

Gemini API NLP:

```powershell
$env:NLP_BACKEND="gemini"
$env:GEMINI_MODEL="gemini-1.5-flash"
$env:GEMINI_API_KEY="<your_api_key>"
python -m app.app
```

Local config keys:
- `nlp_backend`
- `nlp_model`
- `gemini_model`
- `gemini_api_key`

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
