# Scripts

## Validate How2Sign and build manifests

```powershell
python -m scripts.validate_how2sign --dataset-root "C:\Users\rajit\Datasets\How2Sign"
```

Outputs:
- `artifacts/manifests/train_manifest.tsv`
- `artifacts/manifests/val_manifest.tsv`
- `artifacts/manifests/test_manifest.tsv`
- `artifacts/manifests/validation_report.json`

## Build training cache

```powershell
python -m scripts.build_how2sign_cache --dataset-root "C:\Users\rajit\Datasets\How2Sign" --max-samples 3000 --out artifacts/cache/how2sign_cache.pt
```
