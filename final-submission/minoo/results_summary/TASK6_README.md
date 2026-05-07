# Task 6: Adversarial / Subspace Debiasing of Text Shortcut

This directory contains the per-(model, layer, lambda, regime) outputs for Task 6's two debiasing attempts and the summary plots used in the presentation.

## Two attacks, one shortcut

Task 7A established that frozen SSL probes can take a text shortcut on EMIS incongruent clips (HuBERT L20: text_bias = +0.717 on explicit slice). Task 6 asks: **can we remove the shortcut without retraining the encoder?**

| Version | Method | Key parameter | Outcome |
|---|---|---|---|
| **v2: Adversarial GRL** | Train logreg-on-MLP probe with a Gradient Reversal Layer pushing back against a sentence-ID adversary at the shared MLP output. | GRL strength λ ∈ {0, 0.01, 0.1, 1, 10} | Best safe Δ-bias = −0.082 (HuBERT L20, λ=1.0). At λ=10, val_emo collapses to 0.67, bias drop is meaningless. |
| **v3: Linear CCA projection** | Find top-k speech-side directions correlated with BERT text embeddings (no training), then project EMIS hidden states orthogonal to those directions. | k ∈ {0, 1, 2, 5, 10, 16}, two CCA training corpora (RAV+SAV, IEMOCAP) | Best Δ-bias = −0.018 (HuBERT L20, IEMOCAP CCA, k=16). |

Both attacks fail to push the probe off the linguistic subspace at any safe operating point. Combined message: the shortcut is robust to standard linear and adversarial debiasing.

The interesting *positive* finding is upstream of the failed attacks, the **two-regime baseline comparison**:

| Model | Layer | RAV+SAV-trained bias | EMIS-trained bias | Flips? |
|---|---:|---:|---:|---|
| HuBERT | L20 | −0.246 | **+0.928** | YES |
| wav2vec2 | L14 | −0.086 | **+0.926** | YES |
| WavLM | L20 | −0.268 | **+0.850** | YES |
| Whisper | L24 | −0.251 | **+0.003** | **NO** |

All four models show negative bias under RAV+SAV-trained (no text-emotion correlation in training). Three flip strongly positive when EMIS_congruent training offers a text shortcut. **Whisper alone resists the flip**, that's the disentanglement signature.

## Sign convention

```python
text_bias = proxy_acc - target_acc
```
- `target_acc` = probe correctly predicts the **audio** emotion
- `proxy_acc`  = probe predicts the **text** emotion (the would-be shortcut)
- `text_bias > 0` → probe took the text shortcut
- `text_bias < 0` → probe is grounded in audio

Source: [scripts/task6_adversarial.py:534](../../scripts/task6_adversarial.py).

## Layout

```
results/TASK6/
├── README.md                                          # this file
├── csv/
│   ├── adversarial_{Model}.csv                        # v2 RAV+SAV-trained, 4 files
│   └── adversarial_{Model}_emistrain.csv              # v2 EMIS-trained, 4 files
├── projection/
│   ├── cca_proj_{Model}_L{N}.csv                      # v3 RAV+SAV CCA, 4 files
│   └── cca_proj_{Model}_L{N}_iemocap.csv              # v3 IEMOCAP CCA, 4 files
├── plots/                                             # original per-(model, regime) plots, busy, not slide-grade
└── plots_summary/                                     # presentation-grade summary plots
    ├── task6_two_regime_swing.png                     # Figure 1
    ├── task6_lambda_sweep.png                         # Figure 2
    └── task6_two_attacks.png                          # Figure 3
```

## CSV column references

### v2: `csv/adversarial_{Model}{_emistrain}.csv`
Rows: per (layer, lambda, n_seeds=5). Columns relevant to the figures:

| Column | Meaning |
|---|---|
| `model` / `layer` / `layer_name` | identifier |
| `lambda` | GRL strength (0 = no adversary) |
| `n_train` / `n_sentences` | training pool size |
| `best_val_emo_acc_mean` / `_std` | 4-class emotion accuracy on held-out val (signal-quality control) |
| `final_train_sent_acc_mean` | adversary's final train accuracy on sentence-ID |
| `adv_suppression_mean` | how much the adversary's sentence-ID accuracy fell below random (0=random, 1=perfect suppression) |
| `emis_congruent_acc_mean` | sanity: probe accuracy on EMIS_congruent (always near 1.0 by construction in the EMIS-trained regime) |
| `emis_incong_target_acc_mean` / `_std` | audio-emotion accuracy on EMIS incongruent |
| `emis_incong_proxy_acc_mean` / `_std` | text-emotion accuracy on same |
| `emis_incong_text_bias_mean` / `_std` | `proxy − target`, the headline metric |

### v3: `projection/cca_proj_{Model}_L{N}{_iemocap}.csv`
Rows: per k value. Columns:

| Column | Meaning |
|---|---|
| `k` | number of CCA directions projected out (k=0 = no projection, baseline) |
| `val_emo_acc_mean` | sanity: emotion accuracy after projection (should not drop) |
| `target_acc_{all,explicit,implicit}` / `_std` | audio-emotion accuracy per slice |
| `proxy_acc_{all,explicit,implicit}` / `_std` | text-emotion accuracy per slice |
| `text_bias_{all,explicit,implicit}` | `proxy − target` per slice |
| `n_{all,explicit,implicit}` | clip counts per slice |
| `model` / `layer` / `cca_corr_topk_mean` | metadata + canonical correlation strength |

## Helper scripts

| Script | Output | Use |
|---|---|---|
| `scripts/task6_adversarial.py` | The 8 v2 CSVs above + `plots/*.png` | v2 GRL experiment runner; ran on Discovery |
| `scripts/task6_projection.py` | The 8 v3 projection CSVs above | v3 CCA projection runner |
| `scripts/extract_bert_embeddings.py` | BERT embeddings (RAV+SAV) | dependency for v3 |
| `scripts/extract_bert_iemocap.py` | BERT embeddings (IEMOCAP) | dependency for v3 |
| `scripts/task6_summary_plots.py` | 3 plots in `plots_summary/` | presentation-grade summary plots; reads CSVs only |

## The 3 summary plots (in `plots_summary/`)

### `task6_two_regime_swing.png` (Figure 1)
4-panel paired bar chart: RAV+SAV-trained (blue) vs EMIS-trained (red) baseline bias for each (model, best-bias-layer). Demonstrates the bias-flip pattern: 3/4 models swing from negative to strongly positive when training offers a text shortcut; Whisper L24 alone resists.

### `task6_lambda_sweep.png` (Figure 2)
4-panel grid (one per model), bias bars vs λ for the EMIS-trained regime, with `val_emo` overlay (green line) and a "emotion collapse" red shade where val_emo < 0.85. Shows the GRL only "reduces bias" at λ=10 by destroying the emotion signal itself.

### `task6_two_attacks.png` (Figure 3)
Bar chart, 4 models × 2 attacks (GRL purple, CCA green) showing baseline-relative Δ-bias. Visualizes the joint failure of v2 and v3 attacks in one image. **Caveat**: Whisper's baseline is already +0.003, so its "Δ −0.071 from GRL" is not a meaningful debiasing, there was nothing to remove. The HuBERT/wav2vec2/WavLM bars are the real claim.

## Headline numbers (verified against CSVs 2026-04-28)

### v2 baseline bias (λ=0)

| Model | Layer | RAV+SAV | EMIS-trained | val_emo (RAV+SAV) | val_emo (EMIS) |
|---|---:|---:|---:|---:|---:|
| HuBERT | L20 | −0.246 | **+0.928** | 0.95 | 0.99 |
| wav2vec2 | L14 | −0.086 | **+0.926** | 0.95 | 1.00 |
| WavLM | L20 | −0.268 | **+0.850** | 0.96 | 0.99 |
| Whisper | L24 | −0.251 | **+0.003** | 0.95 | 0.95 |

### v2 best safe Δ-bias (EMIS-trained, λ ≤ 1)

| Model | Best safe λ | Δ-bias |
|---|---:|---:|
| HuBERT L20 | 1.0 | −0.082 |
| wav2vec2 L14 | 1.0 | −0.101 |
| WavLM L20 | 1.0 | −0.113 |
| Whisper L24 | 1.0 | −0.071 (baseline already neutral) |

### v3 best Δ-bias (CCA projection, explicit slice)

| Model | Layer | Best CCA corpus | Best k | Δ-bias |
|---|---:|---|---:|---:|
| HuBERT | L20 | iemocap | 16 | −0.018 |
| wav2vec2 | L15 | iemocap | 5 | −0.006 |
| WavLM | L20 | ravsav | 10 | −0.015 |
| Whisper | L24 | iemocap | 10 | −0.005 |

## W&B references

- v2 group: `task6_adversarial_v2` (12 runs, 4 models × 3 setups)
- v3 not logged to W&B (offline run)
- Project: https://wandb.ai/AGSER/linguistic-agnostic-ser

## Drive

- Top-level: https://drive.google.com/drive/folders/1uo1l8YeKKlVOAucGRJraBQbLPvWUtGSK
- Subfolder: `task6_adversarial/` (includes `projection/` for v3)

## Cross-references

- Full study guide: [SER Task 6](../../obsidian/SER/SER Task 6.md) (in Minoo's vault, not in repo)
- Slide script: [SER Task 6 Slide](../../obsidian/SER/SER Task 6 Slide.md) (same)
- Project narrative: [SER Project.md](../../obsidian/SER/SER Project.md) (same)
- Sibling READMEs: [LOSO](../LOSO/README.md), [TASK7](../TASK7/README.md)
