# Step 6 — cross-lingual arm (Q ii.2) — BLOCKED, runnable stub

**Status:** blocked on this cluster — **SpeechCraft has no audio/embeddings here.**
`/scratch1/kelidari/ser-experiments/EXP20` (the SpeechCraft results) holds only
CSVs (`results_*_LOLO.csv`, `results_*_pooled.csv`); there is no SpeechCraft pool
under `/scratch1/kelidari/ser_data`. The published EN↔ZH collapse (weighted F1
0.29–0.32) and the European pool (~0.94) come from those CSVs.

**What ran instead:** the cross-**speaker** arm (`run_step6.py`) on CREMA-D's 91
actors — confound-controlled emotion features show ~0 leave-speaker-out drop and
~0.95 actor coverage, isolating the speaker-general emotion signal and
attributing the MESD-LOSO ~0.30 collapse to speaker-entangled features.

**To run the cross-lingual variant once SpeechCraft audio is available:**

1. Place the SpeechCraft pool under `/scratch1/kelidari/ser_data/SpeechCraft/`
   with language tags (EN / ZH / European) and add a `load_speechcraft` loader.
2. Extract per-frame HuBERT activations per language subset
   (`slurm/sae_extract.sbatch`, add SpeechCraft to `src/sae/extraction._LOADERS`).
3. Train an SAE per language (or one shared SAE on the pool) and select
   confound-controlled emotion features (`confounds={"speaker":..., "language":...}`).
4. Cross-lingual feature transfer: because utterances differ across languages,
   match features by their **per-emotion firing profile** (a feature's mean
   firing rate per emotion class) rather than per-utterance correlation, then ask
   whether EN emotion features have ZH counterparts. Expected under the published
   EN↔ZH collapse: **disjoint feature sets across the Indo-European/Sino-Tibetan
   boundary**, but shared features within the European pool.

**Cheaper substitute already on disk** (if SpeechCraft stays unavailable): run the
per-emotion firing-profile matching across the multilingual *acted* datasets
EmoDB (de), MESD (es), AESDD (el), RAVDESS (en) — extract each, train SAEs, and
compare whether emotion features transfer within vs across language families.
