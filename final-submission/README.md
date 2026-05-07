# Linguistic-Agnostic Speech Emotion Recognition: Final Submission

CSCI-535 final project. Authors: Nima Kelidari, Minoo Ahmadi, Chaitanya Parwatkar, Xiangxu (Henry) Lin.

## Task ownership

| Task | What | Who |
|---|---|---|
| 1 (frozen) | Add WavLM, w2v-BERT, MERT to the 6-encoder grid | Nima |
| 1 (LoRA) | Rank-8 LoRA at best layer, 30 combos | Chaitanya |
| 2 | ESC-50 noise augmentation, 4 SNR levels | Nima |
| 3 | BERT alignment (CKA / Procrustes / centroid transfer) | Chaitanya |
| 4 | Leave-one-speaker-out | Minoo |
| 5 | Learned layer mixer | Chaitanya |
| 6 | GRL + BERT-CCA debiasing | Minoo |
| 7 | EMIS congruent vs. incongruent + IEMOCAP disagreement | Minoo |
| 8 | MSP-Podcast spontaneous-speech evaluation | Nima |
| 9 | SpeechCraft cross-lingual + EN↔ZH | Henry |
| 10 | V/A/D dimensional regression (clean) | Nima |
| 11 | V/A/D under noise | Nima |
| 12 | V/A/D as features for emotion classification | Nima |
| 13 | CREMA-D fixed-lexicon falsification (7,442 clips, 6 emotions) | Chaitanya |
| 14 | BERT alignment on IEMOCAP transcripts (secondary check) | Chaitanya |
| 15 | Original layer-wise probing pipeline (foundation for all probing variants) | Chaitanya |
| 16 | Midterm 18-experiment sweep (3 base encoders × 6 datasets, `18exp.ipynb`) | Minoo |
| Slurm pipeline + `src/` | Shared infrastructure | Nima |

## Layout

```
final-submission/
├── CSCI_535_Final_Report.pdf
├── README.md
└── notebooks/
    ├── nima/        Slurm pipeline, frozen 6-encoder probing, ESC-50 noise,
    │                MSP-Podcast, V/A/D regression
    ├── minoo/       LOSO, GRL + BERT-CCA debiasing, EMIS + IEMOCAP disagreement
    ├── chaitanya/   original probing pipeline, BERT alignment on hidden states,
    │                BERT alignment on IEMOCAP transcripts, layer mixer, LoRA
    │                (CARC Slurm), CREMA-D probe, per-actor/per-sentence
    │                CREMA-D breakdowns
    └── xiangxu/     SpeechCraft cross-lingual benchmark, EN↔ZH transfer
```

Each folder has a `README.md`, a `code/` directory, and a `results_summary/` directory with the key plots and CSVs that back the paper.

## Sources

- GitHub: <https://github.com/Nikelroid/linguistic-agnostic-ser> (branches: `main`, `minoo`, `chaitanya`, `speechcraft`)
- Drive: <https://drive.google.com/drive/folders/1-cqV-IKm7l6tMrgy81dsIN4jYOz8lxli>
- W&B: `AGSER/linguistic-agnostic-ser`

## Reproduction

The shared encoder pipeline lives under `nima/code/src/`; Henry's SpeechCraft scripts and most notebooks import from it. Set up the environment with `nima/code/setup_env.sh` (or `conda env create -f nima/code/environment.yml`), point `nima/code/config/config.yaml` at your dataset roots, then submit the Slurm jobs from each person's `code/slurm/`.

## Headline findings

- Emotion lives in middle-to-late transformer layers, including under CREMA-D's fixed lexicon.
- The signal generalises within Europe but collapses on EN↔ZH transfer; LoRA hurts on 29 of 30 frozen-vs-tuned combinations.
- SSL encoders take a lexical shortcut (EMIS bias +0.61 to +0.72); Whisper resists (−0.19) and is also the most noise-robust (~11% drop at 0 dB vs. 17–30% for SSL).
