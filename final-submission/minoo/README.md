# Minoo Ahmadi

## Contributions

- **Leave-one-speaker-out evaluation** across 42 (encoder × dataset) pairs to rule out speaker memorization (§5.6). Best layer shifts from middle to deep under LOSO.
- **Adversarial debiasing — text, not speaker**: gradient-reversal against a sentence-ID head + linear BERT-CCA projection on top of the frozen encoder, to test whether the lexical shortcut can be removed at inference time (§5.7c).
- **Congruent vs. incongruent (EMIS)**: gained dataset access and built a sentence-stratified probe trained on synthetic congruent clips, tested on incongruent (§5.7a). Also built the IEMOCAP annotator-disagreement diagnostic that confirms the IEMOCAP ceiling reflects label noise (§5.7b).
- **Midterm notebook** `18exp.ipynb`: 18-experiment layer-wise probing sweep (3 base encoders × 6 datasets) for the midterm report.
- Co-authored the proposals.

## Layout

```
minoo/
├── README.md
├── code/
│   ├── 18exp.ipynb                        midterm probing sweep
│   ├── Task7_PartA_EMIS_clean.ipynb       cleaned EMIS notebook
│   ├── scripts/
│   │   ├── task4_loso*.py                 LOSO runner, heatmaps, plots, table
│   │   ├── task6_adversarial.py           GRL
│   │   ├── task6_projection.py            BERT-CCA projection
│   │   ├── task7_partA_EMIS.py            EMIS text-bias probe
│   │   ├── task7_disagreement.py          IEMOCAP agreement split
│   │   ├── extract_bert_*.py              BERT embeddings used by Task 6 CCA
│   │   └── upload_loso_to_wandb.py
│   └── slurm/                             sbatch jobs for the runs above
└── results_summary/
    ├── LOSO_README.md / TASK6_README.md / TASK7_README.md / class_balance.md
    ├── csv/   loso_all_combined.csv, avg_drop_per_layer.csv, best_worst_summary.csv
    └── plots/
        ├── LOSO/                          drop heatmap, layer-shift heatmap
        ├── TASK6_GRL_CCA/                 lambda sweep, two-attacks, two-regime swing
        └── TASK7_EMIS/                    text bias (all/explicit/implicit), trajectory,
                                           cross-diagnostic, agreement gap across 6 models
```

## Headline numbers

- LOSO: speaker-rich datasets retain accuracy (HuBERT/EmoDB drop 0.021); speaker-poor collapse (MESD ~0.30 drop).
- GRL at λ=1: HuBERT/WavLM keep ~94% emotion accuracy while sentence accuracy drops to ~0.39. CCA projection at most −0.018 Δbias — the SSL shortcut is robust to standard linear/adversarial debiasing.
- EMIS text bias (explicit slice): HuBERT +0.717, wav2vec 2.0 +0.665, WavLM +0.613, **Whisper −0.191** — only Whisper stays audio-grounded.
- IEMOCAP agreement gap: +0.145 to +0.191 across encoders (mean +0.169); the 65–70% IEMOCAP ceiling is label noise, not a probe limitation.
