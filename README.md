# Sign Language Translation (Word-Level WLASL Pipeline)

This refactor focuses on a stable final-year demo flow:

- Word-level sign recognition with **WLASL**
- Landmark extraction with **MediaPipe Holistic** (411-dim feature vector)
- LSTM classifier for word classes
- Runtime stabilization: confidence threshold + majority vote + cooldown
- NLP correction with Transformer (`models/transformer.py`)

## Quick Start

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train (WLASL)

```powershell
python -m training.train_lstm --dataset wlasl --dataset-root "C:\Users\rajit\Datasets\WSL\WLASL-master\start_kit"
```

Optional speed-up (feature cache):

```powershell
python -m training.train_lstm --dataset wlasl --dataset-root "C:\Users\rajit\Datasets\WSL\WLASL-master\start_kit" --cache-features
```

Saved artifacts:

- `artifacts/lstm_best.pt`
- `artifacts/label_to_id.json`
- `artifacts/lstm_meta.json`

## Realtime Inference

```powershell
python -m inference.realtime --weights artifacts/lstm_best.pt --labels artifacts/label_to_id.json --meta artifacts/lstm_meta.json
```

Controls:

- `q`: quit
- `c`: clear accumulated words

## Notes

- How2Sign sentence-level LSTM training is intentionally disabled in this refactor.
- Training and runtime both use `MediaPipeExtractor` with `FEATURE_DIM=411`.
