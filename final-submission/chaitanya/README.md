# Chaitanya Parwatkar

## Contributions

- **Original layer-wise probing pipeline** (hidden-state extraction, mean pooling, StandardScaler + 5-fold logistic regression): the baseline every variant in the paper sits on top of.
- **BERT representational alignment** between each speech-encoder layer and BERT-base embeddings using linear CKA, Procrustes residual, and centroid transfer; also rerun on IEMOCAP transcripts (§5.5).
- **Learned layer mixer** with softmax attention over all 25 hidden states (§5.3).
- **LoRA fine-tuning** (rank 8, q/v projections) at the per-combination best layer across 30 (encoder × dataset) combinations, compared against the frozen baseline (§5.4).
- **CREMA-D probing** as the cleanest fixed-lexicon falsification: 91 actors saying the same 12 sentences across 6 emotions, 7,442 clips (§5.2).
- Wrote the original Method, Results, and Cross-Model Correlation sections of the midterm.

## Layout

```
chaitanya/
├── README.md
├── code/
│   ├── original_exploratory_probing_analysis.ipynb     original probing pipeline
│   ├── bert_alignment.ipynb                      §5.5 CKA + Procrustes + centroid transfer
│   ├── bert_alignment_transcripts.ipynb          same on IEMOCAP transcripts
│   ├── cremad.ipynb / cremad_finish.py           §5.2 fixed-lexicon CREMA-D probe
│   ├── layer_mixer.ipynb                         §5.3 softmax attention over 25 layers
│   ├── lora_finetune.ipynb / lora_run_one.py     §5.4 LoRA training & re-probe
│   ├── lora_run_all.sh                           submits all 30 combinations
│   ├── scripts/extract_bert_*.py                 BERT-base embeddings used by alignment
│   └── requirements.txt
└── results_summary/
    ├── plots/
    │   ├── bert_alignment/                       Figs. 5 (CKA) + 6 (Procrustes), centroid transfer
    │   ├── cremad/                               Fig. 2 + per-encoder layer curves
    │   ├── layer_mixer/                          Fig. 3 mixer attention heatmap
    │   └── lora/                                 Fig. 4 LoRA delta heatmap, frozen vs tuned
    └── csv/                                      alignment, cremad (summary/per-actor/per-sentence),
                                                  frozen_vs_tuned_comparison, lora_summary
```

## Headline numbers

- CREMA-D best-layer accuracy (6-class, chance 16.7%): WavLM **0.761** (L14), HuBERT 0.750 (L13), Whisper 0.748 (L12), w2v-BERT 0.730 (L17), wav2vec 2.0 0.693 (L10), MERT 0.640 (L9). Every encoder clears chance with the lexicon held constant.
- BERT alignment: Whisper's Procrustes alignment grows monotonically toward L24 on every dataset (e.g. MSP-Podcast 0.861, MESD 0.855, IEMOCAP 0.849); SSL encoders peak in the middle band.
- Layer mixer: Whisper concentrates **64–95%** of attention on L24; SSL encoders spread across L4–L13. Mixer beats best single layer on IEMOCAP (5/6 encoders) and MSP-Podcast (6/6).
- LoRA at best layer: mean delta −0.062, deepest-layer band −0.173. **Frozen wins 29/30** combinations; only WavLM/RAVDESS is positive (+0.003).
