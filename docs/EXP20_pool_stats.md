# EXP20 / Speechcraft — pool stats (for the report)

Source: `results/EXP20/summary/manifest_*.json` (identical across all 6 models).

## Totals

- **Total clips pooled: 3,980** (not 20k — wording needs to be revised).
- 4 language groups, 4 common emotion labels (`happy, sad, fear, disgust`).

## Labels — NOT balanced

| Label   |     N | Share |
|---------|------:|------:|
| sad     | 1,768 | 44.4% |
| happy   |   831 | 20.9% |
| fear    |   701 | 17.6% |
| disgust |   680 | 17.1% |

- `sad` is ~2.5× the smallest class.
- Majority-class baseline (always-sad) = **44%** — report numbers should be read against this floor.

## Languages — NOT balanced

| Language | Datasets                  |     N | Share |
|----------|---------------------------|------:|------:|
| English  | RAVDESS + IEMOCAP + SAVEE | 2,546 | 64.0% |
| Spanish  | MESD                      |   576 | 14.5% |
| Greek    | AESDD                     |   483 | 12.1% |
| German   | EmoDB                     |   375 |  9.4% |

- English ≈ 64% of the pool → the "pooled" probe is mostly an English probe with a multilingual tail.

## Wording fixes

- "Large-scale 20k-clip benchmark" → **"~4k-clip multilingual benchmark across 6 SSL/speech models"**.
- "Balanced pool" → **"Pooled (imbalanced): sad ~44%, English ~64%"**, or rebalance and rerun.

## Options for next step

1. **Disclose imbalance** as-is and report macro-F1 + per-class F1 alongside accuracy.
2. **Downsample to balanced** (min-class × min-language ≈ EmoDB's 375 cap → ~1.5k clips) and rerun.
3. **Add datasets** to actually reach ~20k (would need new sources per language).
