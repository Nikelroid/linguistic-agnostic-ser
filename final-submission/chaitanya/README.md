# Chaitanya Parwatkar

## Contributions

- **Original layer-wise probing pipeline** (hidden-state extraction, mean pooling, StandardScaler + 5-fold logistic regression): the baseline every variant in the paper sits on top of.
- **BERT representational alignment** between each speech-encoder layer and BERT-base embeddings using linear CKA, Procrustes residual, and centroid transfer; also rerun on IEMOCAP transcripts (§5.5).
- **Learned layer mixer** with softmax attention over all 25 hidden states (§5.3).
- **LoRA fine-tuning** (rank 8, q/v projections) at the per-combination best layer across 30 (encoder × dataset) combinations, deployed to USC CARC via Slurm, compared against the frozen baseline (§5.4).
- **CREMA-D probing** as the cleanest fixed-lexicon falsification: 91 actors saying the same 12 sentences across 6 emotions, 7,442 clips (§5.2).
- **Original mass-probing notebook** (`original_exploratory_probing_analysis.ipynb`) that powers the 42 (encoder, dataset) per-layer experiments cited throughout the paper, used as the foundation by every downstream variant on the team.
- **Writing.** Drafted the original Method, Results, and Cross-Model Correlation sections of the midterm report. For the final report, wrote the abstract, problem definition (§1), method (§4), and the core results subsections, layer-wise probing (§5.1), CREMA-D (§5.2), layer mixer (§5.3), Frozen vs. LoRA (§5.4), and BERT representational alignment (§5.5), and contributed substantively to the conclusions and limitations. Co-authored the final presentation deck.

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
│   ├── lora_run_all.sh                           runs all 30 combinations locally
│   ├── slurm/submit_lora.sbatch                  CARC Slurm submission for LoRA
│   ├── scripts/extract_bert_*.py                 BERT-base embeddings used by alignment
│   └── requirements.txt
└── results_summary/
    ├── plots/
    │   ├── bert_alignment/                       Figs. 5 (CKA) + 6 (Procrustes), centroid transfer
    │   ├── cremad/                               Fig. 2 + per-encoder layer curves
    │   ├── layer_mixer/                          Fig. 3 mixer attention heatmap
    │   └── lora/                                 Fig. 4 LoRA delta heatmap, frozen vs tuned
    └── csv/
        ├── bert_alignment/                       alignment_summary, alignment_per_layer_wide
        ├── cremad/                               summary, per-actor, per-sentence, per-encoder layer curves
        ├── layer_mixer/                          summary, attention_weights_all, mixer_vs_probing
        └── lora/                                 lora_summary, frozen_vs_tuned_comparison
```

## Headline numbers

- **CREMA-D fixed-lexicon (Task 13)** best-layer accuracy (6-class, chance 16.7%): WavLM **0.761** (L14), HuBERT 0.750 (L13), Whisper 0.748 (L12), w2v-BERT 0.730 (L17), wav2vec 2.0 0.693 (L10), MERT 0.640 (L9). Every encoder clears chance with the spoken text held constant by construction, the cleanest acoustic-only evidence in the paper.
- **BERT alignment, hidden states (Task 3)**: Whisper's Procrustes alignment grows monotonically toward L24 on every dataset (MSP-Podcast 0.861, MESD 0.855, IEMOCAP 0.849); the SSL encoders peak in the middle band, the deep-layer text fingerprint is specific to supervised pretraining.
- **BERT alignment, IEMOCAP transcripts (Task 14)**: Whisper's CKA reaches **0.392** at L24, the highest any encoder attains on the transcript variant; the same encoder ranking holds as on hidden states, confirming the lexical fingerprint is in the representation itself rather than an artifact of acoustic input.
- **Layer mixer (Task 5)**: Whisper concentrates **64–95%** of softmax attention on L24 across every dataset, while SSL encoders spread attention over L4–L13. The mixer beats the best single layer on IEMOCAP (5/6 encoders, gain +0.02 to +0.04) and MSP-Podcast (6/6 encoders, gain +0.02 to +0.06), the spontaneous corpora where single-layer probing does not localise cleanly.
- **Frozen vs. LoRA (Task 1L)**: mean best-layer delta **−0.062** across 30 (encoder, dataset) combinations; mean deepest-layer-band delta **−0.173**. **Frozen wins 29 of 30** combinations: only WavLM/RAVDESS lifts (+0.003), and MERT/SAVEE collapses by **−0.144** (worst case). The features the probe needs are already in the frozen encoder, and tuning at a single layer overwrites them more often than it helps.
- **Original probing pipeline (Task 15)**: hidden-state extraction + mean pooling + StandardScaler + 5-fold stratified logistic regression, the baseline reused by every probing variant in the report and powering all 42 (encoder, dataset) per-layer experiments.

## Drive locations

- Task 3 (BERT alignment, hidden states): <https://drive.google.com/drive/folders/1mPFfxfOqnUXnJAqwS02ihbFIdcRqU9S6>
- Task 14 (BERT alignment, IEMOCAP transcripts): <https://drive.google.com/drive/folders/1lZvGN4yYkd21Eopw36pdmiHrKmKyVSDP>
- Task 5 (Layer mixer): <https://drive.google.com/drive/folders/1NmksAPrQU2gjhDZ69-yX0BM1kDMotaBI>
- Task 1L (LoRA, full per-(encoder, dataset) tuned probing cache): <https://drive.google.com/drive/folders/13AgZBSJnCqkPg2gOsR6hsJoO8F1Uzjd->
- Task 13 (CREMA-D): <https://drive.google.com/drive/folders/1Tanj8xIpW8Af7OW-xDkMS2VshVKRWeSm>
- Parent folder (all of the above): <https://drive.google.com/drive/folders/1QM6g2K4-5xi9JURScMGX5t-a3EmQO2vo>
