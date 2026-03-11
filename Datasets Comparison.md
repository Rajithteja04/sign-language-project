# Datasets Comparison

| Dataset | Top‑k Config | Train / Val / Test Clips | Best Val Acc | Artifact Snapshot | Highest Model |
| --- | --- | --- | --- | --- | --- |
| WLASL | 20 (seq=30, h=128, l=2) | 211 / 64 / 32 | 0.2812 | *(baseline run, artifacts not archived)* |  |
| WLASL | **6** (seq=30, h=128, l=2, norm) | 72 / 21 / 11 | **0.6190** | `artifacts/wlasl/top6_seq30_h128_l2_norm/` | ✓ |
| How2Sign (cache=20000) | 5 (seq=60, h=128, l=1, bi, do=0.1) | 124 / 26 / 26 | 0.6154 | `artifacts/how2sign_20000_top5_seq60_h128_l1_lr0.0005_do0.1_bi/` |  |
| How2Sign (cache=9000) | **5** (seq=60, h=128, l=1, bi, do=0.1) | 62 / 14 / 14 | **0.6429** | `artifacts/how2sign_top5_seq60_bi_run1/` | ✓ |
| MS-ASL | 20 (seq=30, h=128, l=2) | 568 / 114 / 165 | 0.5877 | `artifacts/msasl/msasl_top20_seq30_h128_l2_bs16_lr5e-04/` |  |
| MS-ASL | 10 (seq=30, h=128, l=2) | 300 / 59 / 127 | 0.7797 | `artifacts/msasl/msasl_top10_seq30_h128_l2_bs16_lr5e-04/` |  |
| MS-ASL | 6 (seq=30, h=128, l=2) | 186 / 39 / 59 | 0.7436 | `artifacts/msasl/msasl_top6_seq30_h128_l2_bs16_lr5e-04/` |  |
| MS-ASL | **5** (seq=30, h=128, l=2) | 157 / 32 / 56 | **0.8125** | `artifacts/msasl/msasl_top5_seq30_h128_l2_bs16_lr5e-04/` | ✓ |
| LSA64 | 64 (cache build only) | 3,200 clips cached | — | `artifacts/lsa64/lsa64_top64_seq30_h128_l2_bs1_lr1e-03/` | *(training pending)* |

**Notes & Observations**

- **WLASL:** Reducing the label set from 20 → 6 nearly doubled validation accuracy (0.28 → 0.62). Even though the dataset is tiny (72 train clips), focusing on the most frequent glosses stabilizes the model for demo purposes.
- **How2Sign:** Smaller cache (9k) actually performed slightly better than the 20k cache for the same top‑5 configuration, suggesting diminishing returns (or more noise) beyond the most frequent phrases.
- **MS-ASL:** Accuracy increases as top‑k decreases (0.59 at top‑20 → 0.81 at top‑5). More samples per class help, but model capacity and class imbalance limit gains when too many classes are included.
- **LSA64:** Extraction cache is ready (3,200 clips) but no training run has been completed yet; expect similar trends once top‑k experiments start.

- All models share the MediaPipe Holistic extractor (`FEATURE_DIM=411`) and the same LSTM architecture, so differences are driven by dataset composition rather than code changes.
- “Highest Model” marks the best-performing configuration per dataset to help choose demo-ready checkpoints.
