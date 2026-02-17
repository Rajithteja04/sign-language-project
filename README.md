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

2. Run the web app (placeholder pipeline):

```powershell
python -m app.app
```

3. Run realtime inference (camera + landmarks + dummy decoder):

```powershell
python -m inference.realtime --dataset how2sign
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

This is a scaffold. The scripts include TODOs to connect real datasets, trained weights, and evaluation.
