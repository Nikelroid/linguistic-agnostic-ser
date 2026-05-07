# Nima Kelidari

## Contributions

- **Slurm-based experiment pipeline** (`src/`, `run_pipeline.py`, `slurm/`): shared infrastructure that all six encoders run through.
- **Frozen probing extension to six encoders**: added WavLM Large, w2v-BERT 2.0, and MERT v1-330M (all 24-layer) on top of the midterm's wav2vec 2.0 / HuBERT / Whisper.
- **ESC-50 noise augmentation** at four SNR levels {20, 10, 5, 0 dB} across 6 encoders × 7 datasets (§5.8).
- **MSP-Podcast** spontaneous-speech evaluation, 4-class probing on all 6 encoders.
- **Dimensional V/A/D regression on MSP-Podcast** end-to-end: clean (§5.9), noisy, and V/A/D-as-features for 4-class classification.
- Co-authored the proposal, led the literature review, coordinated report formatting.

## Layout

```
nima/
├── README.md
├── code/
│   ├── src/                               shared pipeline (data, models, preprocessing, training)
│   ├── config/config.yaml
│   ├── run_pipeline.py                    main entrypoint
│   ├── aggregate_datasets.py              dataset manifest builder
│   ├── probe_metadata.py                  V/A/D-as-features probe
│   ├── slurm/                             one sbatch per experiment family
│   ├── environment.yml / requirements.txt / setup_env.sh
└── results_summary/
    ├── plots/                             noise (Fig. 7), V/A/D (Figs. 8 & 12), MSP 4-vs-8
    ├── noise_analysis_report.txt          best-layer accuracy at each SNR
    ├── vad_statistical_analysis.txt       per-target RMSE and best layer per encoder
    └── vad_as_features_analysis.txt       MI / classifier comparison on 67,929 clips
```

## Headline numbers

- Noise drop at 0 dB: Whisper **10.9%**, w2v-BERT 16.9%, HuBERT 22.8%, WavLM 27.7%, MERT 28.7%, wav2vec 2.0 29.6%.
- V/A/D cross-target mean RMSE: Whisper **0.754**, WavLM 0.761, HuBERT 0.763, MERT 0.775, w2v-BERT 0.779, wav2vec 2.0 0.788.
- Valence is the hardest dimension (mean RMSE 0.881 vs. 0.730 arousal vs. 0.699 dominance).
- V/A/D + gender as features for 4-class emotion: valence MI = 0.341 (1.7× arousal/dominance, 20× gender); RF reaches macro-F1 0.65, but the minority *sad* class collapses (recall 0.06 in LR, 0.25 in RF).

## Drive locations

- Tasks 1/2/8: <https://drive.google.com/drive/folders/1TRX3M80VQhclfX6Jg1D2bnijF9lAVm1x>
- Tasks 10/11/12: <https://drive.google.com/drive/folders/1CU7_0g6e4n_TO0hvAengR6IETASp13bq>
- Embeddings (clean + noisy `.npy` caches): <https://drive.google.com/drive/folders/1weegGLtMUvl5TiX7fiRr72YZtl1ZvoNT>
