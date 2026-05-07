# Task 4: Leave-One-Speaker-Out (LOSO) Results

This directory contains the per-(model, dataset) LOSO and 5-fold CV results for the speaker-leakage analysis from Task 4.

## Layout

```
results/LOSO/
├── README.md                       # this file
├── csv/
│   ├── loso_{Model}_{Dataset}.csv  # 42 per-(model, dataset) results, AUTHORITATIVE
│   └── loso_all_combined.csv       # ⚠ STALE: MESD rows have LOSO_Accuracy = 0
└── per_speaker/
    └── per_speaker_{Model}_{Dataset}.json  # per-fold per-speaker accuracies
```

**Models:** HuBERT, wav2vec2, Whisper, WavLM, MERT, w2v-BERT
**Datasets:** RAVDESS, EmoDB, IEMOCAP, SAVEE, AESDD, MESD, MSP-Podcast (= 7 × 6 = 42 runs)

## ⚠ Important: use the per-model CSVs, not the combined one

`csv/loso_all_combined.csv` is stale, every MESD row's `LOSO_Accuracy` is 0.0, which is wrong. The per-model files in the same directory have the correct values. Both summary scripts below already read the per-model files for this reason.

If you want to refresh the combined CSV, re-run `scripts/task4_loso.py --all`. But for any analysis or table generation, prefer the per-model files.

## CSV column reference

Each `loso_{Model}_{Dataset}.csv` has 25 rows (one per layer, layer 0 = CNN front-end) with columns:

| Column | Meaning |
|---|---|
| `Layer` / `Layer_Num` | Human-readable / numeric layer index |
| `FiveFold_Accuracy` / `FiveFold_F1` | Standard 5-fold stratified CV (layers' "official" accuracy) |
| `LOSO_Accuracy` / `LOSO_F1` | Leave-One-Speaker-Out CV (the "honest" accuracy with no speaker leakage) |
| `LOSO_Std` | Std of LOSO accuracy across speaker folds |
| `Accuracy_Drop` | `FiveFold_Accuracy − LOSO_Accuracy` for that layer |
| `LOSO_Min_Speaker_Acc` / `LOSO_Max_Speaker_Acc` | Worst- and best-case held-out speaker |
| `Num_Speakers` | Number of speakers in the dataset |

## Helper scripts

All four live in `scripts/` and read from `results/LOSO/csv/`. Run from repo root.

| Script | Output | Use |
|---|---|---|
| `task4_loso_table.py` | stdout / `docs/task4_table.md` | Headline + MSP markdown tables |
| `task4_loso_plots.py` | `results/LOSO/plots/*.png` (local-only; not tracked) | 42 individual plots + 2 grid plots, bulky, regenerate as needed |
| `task4_loso_heatmaps.py` | `results/LOSO/plots/loso_heatmap_*.png` (tracked) + `results/LOSO/avg_drop_per_layer.csv` | 2 presentation-grade heatmaps (drop + layer shift) and avg-per-layer Q&A backup |
| `task4_class_balance.py` | `results/LOSO/class_balance.{csv,md}` | Per-dataset imbalance audit; documents why MSP-Podcast was excluded from the headline |

### `scripts/task4_loso_table.py`

Generates the headline LOSO summary table in Markdown.

```bash
# Print to stdout
python scripts/task4_loso_table.py

# Save to file
python scripts/task4_loso_table.py --out docs/task4_table.md

# Add LOSO_Std + per-speaker min/max columns
python scripts/task4_loso_table.py --extended
```

Output structure:
1. Headline matrix, 6 models × 6 balanced datasets, sorted by `Drop` descending. Columns: `5F Layer`, `LOSO Layer`, `Δ Layer`, `5F Acc`, `LOSO Acc`, `Drop`, `Chance`, `LOSO Lift`, `# Spk`.
2. Quick-look summary, best honest (highest LOSO acc + small drop), worst leakage (largest drop), smallest drop, average drop per dataset.
3. Top-5 largest layer shifts under LOSO (`|Δ Layer|`): for the "Finding 3" paragraph in the report.
4. Separate MSP-Podcast section (excluded from headline due to severe class imbalance).

**`Drop` semantics:** `(best 5F acc) − (best LOSO acc)`, both computed by `idxmax` on the layer-wise CSV. The two "best layers" can be different layers; the script reports both indices in `5F Layer` and `LOSO Layer` and the difference in `Δ Layer`.

**Chance baselines** are hardcoded in `DATASET_CHANCE` at the top of the script, derived from the midterm-report Appendix A class counts:

| Dataset | Largest class / Total | Chance |
|---|---|---|
| RAVDESS | 384/2880 | 0.133 |
| EmoDB | 127/535 | 0.237 |
| IEMOCAP (4-class) | 1708/5531 | 0.309 |
| SAVEE | 120/480 | 0.250 |
| AESDD | 121/604 | 0.200 |
| MESD | 144/862 | 0.167 |
| MSP-Podcast (4-class) | 5469/10779 | 0.507 |

Edit `DATASET_CHANCE` if you update class counts.

### `scripts/task4_loso_plots.py`

Renders 42 per-(model, dataset) two-panel plots (accuracy curves on top, per-layer drop on bottom) plus a 6×6 grid of small accuracy plots and a separate MSP-Podcast grid.

```bash
# Default: write all plots to results/LOSO/plots/
python scripts/task4_loso_plots.py

# Skip individual plots, only render grids
python scripts/task4_loso_plots.py --no-individual

# Skip grids, only render individual plots
python scripts/task4_loso_plots.py --no-grid

# Custom output directory
python scripts/task4_loso_plots.py --out_dir /some/where
```

The script imports `DATASET_CHANCE` from `task4_loso_table.py` so chance lines stay consistent across both outputs.

**Plot conventions:**
- Blue = 5-fold accuracy
- Red = LOSO accuracy (with shaded ±1σ band on individual plots)
- Gray dotted = majority-class chance baseline
- Vertical dashed lines on individual plots mark the best 5F layer (blue) and best LOSO layer (red)
- Bottom panel shows `5F − LOSO` per layer (positive = leakage)

### `scripts/task4_loso_heatmaps.py`

Renders two presentation-grade summary heatmaps (6 models × 6 balanced datasets) plus a Q&A-backup CSV. MSP-Podcast is excluded; it has its own grid in `task4_loso_plots.py`.

```bash
python scripts/task4_loso_heatmaps.py
```

Outputs:
- `results/LOSO/plots/loso_heatmap_drop.png`: accuracy drop heatmap (Reds colormap). Cell = `(best 5F acc) − (best LOSO acc)`. Columns sorted left-to-right by avg drop (worst leakage → cleanest).
- `results/LOSO/plots/loso_heatmap_layer_shift.png`: best-layer shift heatmap (RdBu_r diverging colormap). Cell = `(LOSO best layer) − (5F best layer)`, labeled `5F→LOSO (Δ±N)`. Same column ordering as the drop heatmap so they pair side-by-side.
- `results/LOSO/avg_drop_per_layer.csv`: for each (model, dataset) cell, both the best-vs-best drop and the avg-per-layer drop (averaged across all 25 layers' `Accuracy_Drop` column). Use this to defend the best-vs-best framing in Q&A.

### `scripts/task4_class_balance.py`

Per-dataset class-imbalance audit. Documents why MSP-Podcast is excluded from the LOSO headline. Class counts hardcoded from the midterm report Appendix A Table 4 (single source of truth).

```bash
python scripts/task4_class_balance.py
```

Outputs:
- `results/LOSO/class_balance.csv`: one row per dataset with `N_Classes`, `N_Samples`, majority/minority class + percentages, majority-class baseline, imbalance ratio, normalized Shannon entropy, and a `Balance_Label` (Balanced / Mostly balanced / Imbalanced / Severely imbalanced).
- `results/LOSO/class_balance.md`: same data as a human-readable markdown table with interpretation paragraph.

The headline finding: **MSP-Podcast is the only `Severely imbalanced` dataset** (50.7% neutral, imbalance ratio 10.2×, normalized entropy 0.76). Every balanced dataset has normalized entropy ≥ 0.94.

## Plot outputs

The 44 generated PNGs live under `results/LOSO/plots/` after running the plot script. **Plots are not committed to git** (per project policy on derived artifacts). For the team-shared rendered versions, see the Drive folder:

https://drive.google.com/drive/folders/1SCyluV-gj-yH8DDRi4z2itIMDlcC6G2d

The Drive folder also has `csv/` and `per_speaker/` mirrors.

## Headline numbers (snapshot, 2026-04-28)

For quick reference without re-running the script:

- **Best honest accuracy:** HuBERT/EmoDB, LOSO 0.9682 at L11 (drop 0.021, 10 speakers)
- **Worst leakage:** MERT/MESD, drop 0.324 (5F 0.869 → LOSO 0.545, 6 speakers)
- **Smallest drop:** wav2vec2/SAVEE, drop 0.006 (5F 0.858 → LOSO 0.852)

**Average drop per dataset (across 6 models):**

| Dataset | Avg Drop |
|---|---:|
| MESD | 0.301 |
| RAVDESS | 0.180 |
| SAVEE | 0.108 |
| AESDD | 0.107 |
| IEMOCAP | 0.031 |
| EmoDB | 0.030 |

**Top 5 largest layer shifts under LOSO:**

- Whisper/SAVEE: 5F=L9 → LOSO=L24 (Δ=+15, deeper); 5F 0.898 → LOSO 0.748 (drop 0.150)
- WavLM/SAVEE: 5F=L6 → LOSO=L18 (Δ=+12, deeper); 5F 0.887 → LOSO 0.844 (drop 0.044)
- HuBERT/SAVEE: 5F=L7 → LOSO=L17 (Δ=+10, deeper); 5F 0.890 → LOSO 0.838 (drop 0.052)
- Whisper/RAVDESS: 5F=L18 → LOSO=L11 (Δ=−7, shallower); 5F 0.975 → LOSO 0.802 (drop 0.173)
- MERT/RAVDESS: 5F=L14 → LOSO=L7 (Δ=−7, shallower); 5F 0.944 → LOSO 0.695 (drop 0.249)

Re-run the table script for the live numbers.

## MSP-Podcast caveat

All 6 MSP-Podcast LOSO runs scored only +0.04 to +0.06 above the 0.507 majority-class baseline (`LOSO Lift` column in the table output). The probe is barely above chance, so the small drops on MSP-Podcast aren't evidence of clean generalization, they're evidence that the probe was barely learning anything to lose. MSP-Podcast is reported separately in §10 of the report, with macro-F1 / class-balanced retraining as future work.

## Source script for the underlying experiments

`scripts/task4_loso.py` (already in the repo): extracts hidden states for each layer, runs both 5-fold and LOSO logistic-regression probes, writes the per-(model, dataset) CSVs in `csv/`. Used as `python scripts/task4_loso.py --all --data_dir /scratch1/minooahm/ser_data`.
