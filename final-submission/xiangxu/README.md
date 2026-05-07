# Xiangxu (Henry) Lin

## Contributions

- **SpeechCraft cross-lingual benchmark** across 6 encoders on the balanced 4-class manifest (English / German / Greek / Spanish; n = 3,980; classes {happy, sad, fear, disgust}). Two protocols: pooled random split and leave-one-language-out (§5.10).
- **Full-scale SpeechCraft + EN↔ZH transfer**: full-scale pooled benchmark on four encoders, plus 5-language pooled (adding Mandarin) and bidirectional EN→ZH / ZH→EN single-language transfer on HuBERT, WavLM, and Whisper (Table 7, Figure 10).
- Initiated the midterm report and wrote the original Data Description section.

## Layout

```
xiangxu/
├── README.md
├── code/
│   ├── data.py        builds the SpeechCraft manifest from the 6 raw datasets, canonicalises labels
│   │                  to {happy, sad, fear, disgust}, and extracts hidden states into a numpy cache.
│   │                  CLI modes: build_manifest / extract_cache.
│   │                  Imports from src.data_ingestion.loader and src.models.feature_extractors
│   │                  (in nima/code/src/).
│   └── analyze.py     runs the layer-wise probe on the cache: pooled_random /
│                      leave_one_language_out / leave_one_dataset_out protocols.
└── results_summary/
    ├── EXP20_pooled_LOLO/
    │   ├── csv/       per-encoder pooled + LOLO layer-wise results
    │   └── summary/   per-encoder cache manifests + best-layer summary
    └── plots/
        ├── speechcraft_pooled_grid.png         Fig. 9 pooled vs LOLO grid
        ├── speechcraft_lolo_heatmap.png        Fig. 11 LOLO heatmap
        ├── task9_protocols.png                 Fig. 10 full-scale 4-condition summary
        └── probing_<encoder>_<protocol>.png    per-encoder layer curves (pooled & LOLO)
```

## Headline numbers

- Balanced 4-language pooled (5-fold CV) saturates near 0.94 for every encoder: Whisper L10 **0.949**, HuBERT L8 0.947, WavLM L6 0.945, w2v-BERT L17 0.936, wav2vec 2.0 L8 0.922, MERT L6 0.908.
- LOLO is asymmetric: German held out generalises (HuBERT 0.952), English held out drops (HuBERT 0.691), Spanish held out is hardest (HuBERT 0.375).
- Full-scale 5-language pooled (with Mandarin) drops to 0.485–0.523. Whisper's peak shifts to L22, the only encoder whose deep layer wins.
- EN↔ZH single-language transfer collapses to 0.28–0.32, only 0.03–0.07 above the 4-class chance baseline (0.25).

## Drive location

- Task 9: <https://drive.google.com/drive/folders/18wbWal1N_Z0taQ3rfdmdxIt-3x7VSPGq>
