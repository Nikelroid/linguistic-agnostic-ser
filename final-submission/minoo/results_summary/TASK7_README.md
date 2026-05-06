# Task 7 — Probe-trustworthiness Diagnostics

This directory contains the per-(model, ...) outputs for Task 7's two probing experiments and the summary plots used in the presentation.

## Two parts, one theme

Both parts ask: **is this probe's accuracy trustworthy?** They probe different failure modes:

- **Part A — EMIS text-bias.** Train a probe on EMIS_congruent (text emotion = audio emotion) and test on EMIS_incongruent (text ≠ audio). If the probe predicts the *text* emotion, it learned a linguistic shortcut from the frozen encoder. If it predicts the *audio* emotion, it's grounded in acoustics.
- **Part B — IEMOCAP annotator disagreement.** Split IEMOCAP into clips where every annotator agreed vs. clips where ≥1 annotator disagreed. Train probes on each subset and compare. The gap is the annotator-disagreement ceiling — accuracy lost not to model error but to label noise.

Coverage:
- **Part A:** 4 models (HuBERT, wav2vec2, WavLM, Whisper-medium) × 3 categories (all / explicit / implicit) = 12 CSVs. MERT and w2v-BERT excluded (EMIS hidden states not extracted for them).
- **Part B:** 6 models × IEMOCAP only = 6 CSVs. All 6 SSL encoders.

## Layout

```
results/TASK7/
├── README.md                                     # this file
├── probing_{Model}_{all,explicit,implicit}.csv   # Part A: 4 × 3 = 12 files
├── probing_all_combined.csv                      # Part A aggregated (Q&A reference)
├── agreement_probing_{Model}_IEMOCAP.csv         # Part B: 6 files
├── best_worst_summary.csv                        # cross-experiment summary
├── emis_metadata.csv                             # EMIS clip metadata (text/audio emo, category, etc.)
├── plots_EMIS/                                   # Part A per-category line plots (existing)
└── plots_summary/                                # presentation-grade summary plots (new)
    ├── task7B_disagreement_6models.png           # Plot 1
    ├── task7_cross_diagnostic.png                # Plot 2
    └── task7A_bias_swing.png                     # Plot 3
```

## Sign convention (Part A)

```python
text_bias = proxy_acc - target_acc
```
- `target_acc` = probe correctly predicts the **audio** emotion
- `proxy_acc`  = probe predicts the **text** emotion (the would-be shortcut)
- `text_bias > 0` → probe took the text shortcut
- `text_bias < 0` → probe is grounded in audio

Source: [scripts/task7_partA_EMIS.py:285](../../scripts/task7_partA_EMIS.py).

## CSV column references

### Part A — `probing_{Model}_{Category}.csv`
25 rows per file (one per layer 0–24).

| Column | Meaning |
|---|---|
| `layer` | Layer number (0 = CNN front-end, 1–24 = transformer) |
| `target_acc_mean` / `_std` | Audio-emotion accuracy across 5 sentence-stratified folds |
| `proxy_acc_mean` / `_std` | Text-emotion accuracy across same folds |
| `text_bias_mean` / `_std` | `proxy − target` across folds |
| `target_f1_mean` | F1 (weighted) on audio labels |
| `n_train` / `n_test` | Avg fold sizes (probes trained on EMIS_congruent, tested on incongruent) |
| `model` / `category` / `chance` | Constant per file |

### Part B — `agreement_probing_{Model}_IEMOCAP.csv`
25 rows per file.

| Column | Meaning |
|---|---|
| `Layer` / `Layer_Num` | Human-readable / numeric layer index |
| `HighAgree_Acc` / `HighAgree_F1` | Probe accuracy / F1 on the 1308 unanimously-labeled clips |
| `LowAgree_Acc` / `LowAgree_F1` | Same on the 4152 clips with ≥1 annotator disagreement |
| `All_Acc` / `All_F1` | Same on full set (1308 + 4152 = 5460) |
| `HighAgree_N` / `LowAgree_N` | Subset sizes (constant: 1308, 4152) |

## Helper scripts

All run from repo root.

| Script | Output | Use |
|---|---|---|
| `scripts/task7_partA_EMIS.py` | The 12 Part A CSVs above + `plots_EMIS/*.png` | Source experiment runner; ran on Discovery |
| `scripts/task7_disagreement.py` | The 6 Part B CSVs above | Source experiment runner; ran on Discovery |
| `scripts/task7_summary_plots.py` | 3 plots in `plots_summary/` | Presentation-grade summary visuals; reads CSVs only |

## The 3 summary plots (in `plots_summary/`)

### `task7B_disagreement_6models.png` (Plot 1)
Six panels, one per model. Each shows three accuracy curves over 25 layers: high-agree (green), low-agree (red), full set (gray dashed). Each panel title gives the peak-layer numbers: `L{best}: high={x}, low={y}, gap={x−y}`. Shows the agreement gap holds across all 6 architectures.

### `task7_cross_diagnostic.png` (Plot 2)
Two panels side by side. Left: peak text_bias per (model, category) for Part A's 4 models. Right: peak-layer agreement gap per model for Part B's 6 models. Both diagnostics on one figure — the headline image for the Task 7 slide.

### `task7A_bias_swing.png` (Plot 3)
Bar chart, 4 models × 3 test slices (all/explicit/implicit). Same model, same encoder, but bias swings sign as the test slice changes. Strongest internal-falsification visual: if probes used acoustics not text, all three bars per model would match.

## Headline numbers

### Part A (peak text_bias per model, explicit slice — the strongest text-shortcut signal)

| Model | Best Layer | text_bias | target_acc | proxy_acc |
|---|---:|---:|---:|---:|
| HuBERT | L20 | **+0.717** | 0.107 | 0.823 |
| wav2vec2 | L15 | **+0.665** | 0.123 | 0.789 |
| WavLM | L20 | **+0.613** | 0.146 | 0.759 |
| Whisper | L24 | **−0.191** | 0.524 | 0.333 |

### Part B (peak-layer agreement gap, all 6 models)

| Model | Best Layer | High-agree | Low-agree | Gap |
|---|---:|---:|---:|---:|
| Whisper | L22 | 0.838 | 0.672 | +0.166 |
| WavLM | L21 | 0.823 | 0.650 | +0.174 |
| HuBERT | L11 | 0.807 | 0.636 | +0.172 |
| wav2vec2 | L8 | 0.781 | 0.591 | **+0.191** (largest) |
| MERT | L10 | 0.768 | 0.599 | +0.169 |
| w2v-BERT | L16 | 0.771 | 0.625 | **+0.145** (smallest) |

Average peak-layer gap across 6 models: **+0.169** (range 0.145–0.191).

## W&B references

- Part A: group `task7_incongruent_EMIS` (12 runs)
- Part B: group `task7_incongruent` (6 runs, all 6 models)
- Project: https://wandb.ai/AGSER/linguistic-agnostic-ser

## Drive

- Top-level: https://drive.google.com/drive/folders/1uo1l8YeKKlVOAucGRJraBQbLPvWUtGSK
- Subfolders: `task7A_emis_probing/`, `task7B_iemocap_disagreement/`

## Cross-references

- Full study guide: [SER Task 7](../../obsidian/SER/SER Task 7.md) (in Minoo's vault, not in repo)
- Slide script: [SER Task 7 Slide](../../obsidian/SER/SER Task 7 Slide.md) (same)
- Project narrative: [SER Project.md](../../obsidian/SER/SER Project.md) (same)
