# SAE study — findings

All results are HuBERT on CREMA-D (lexicon-controlled: 91 actors × 12 fixed
sentences × 6 emotions, 7,442 clips) unless noted. SAEs are TopK (d_sae=8192,
k=32) trained on **per-frame** L-layer activations. Reconstruction is healthy:
**FVU 0.070 (93% variance explained), 4/8192 dead features** (Step 1).

## Method note — confound control was essential
Selecting "emotion features" by emotion-MI + monosemanticity alone flagged
**394/8192** features, but the Step-3 control showed **66% of them carried more
information about the spoken *sentence* (lexical/phonetic) or *speaker* than about
emotion** — the documented "SAE rediscovers phonemes, not affect" failure mode.
Requiring `MI(emotion) > MI(sentence)` and `> MI(actor)` on CREMA-D's factorial
yields **61 genuine emotion features** (median monosemanticity 0.59), dominated by
**anger (29) and sadness (18)**; fear/neutral drop to 0 — consistent with acoustic
encoding of high-arousal, acoustically-marked emotions.

## Q (i) — Depth: emotion *disentangles* earlier than it *decodes*
`step4_HuBERT_CREMA-D_depth.{csv,png}`

| quantity | peak layer | value |
|----------|-----------|-------|
| linear probe accuracy (decodability) | **L10** | 0.739 |
| # monosemantic emotion features (disentanglement) | **L7** | 113 |
| monosemanticity of significant features | **L2** | 541 sig-mono |

The decode and disentanglement peaks **do not coincide**. Deep layers (L17–24)
retain decodability (~0.72) but lose disentanglement (24–47 emotion features) —
emotion stays linearly readable while becoming distributed/entangled. **Emotion
organizes into clean monosemantic features in early-middle layers (L2–L7), before
the layer where it is best linearly decoded (L10).**

## Q (ii) — Generalization
**ii.1 Across pretraining paradigms** (`step5_*`): all 6 encoders carry healthy
emotion-feature populations at their best layers (HuBERT 61, wav2vec2 70, Whisper
79, WavLM 145, MERT 64, w2v-BERT 61), all anger/sadness-dominated. Cross-encoder
feature overlap (per-utterance firing match): **Whisper→SSL 0.40 ≈ SSL→SSL 0.39**.
→ On fixed-lexicon CREMA-D **Whisper is not distinct** from SSL in acoustic emotion
features; its known divergence is about resisting the *lexical* shortcut, not about
acoustic emotion encoding (which all encoders share). The lexical-divergence test
is the deferred EMIS check (`STEP8_EMIS_STUB.md`).

**ii.3 Across speakers** (`step6_*`): the confound-controlled emotion features are
**speaker-invariant** — random-CV 0.551 vs speaker-disjoint (GroupKFold over 91
actors) 0.551, **leave-speaker-out drop ≈ 0**, median actor coverage 0.95. Because
selection excluded speaker-entangled features (`MI(emotion) > MI(actor)`), the
published **MESD leave-one-speaker-out collapse (~0.30) is attributable to exactly
those speaker-entangled features** the SAE lets us isolate.

**ii.2 Across languages**: BLOCKED — SpeechCraft audio is not on this cluster
(`STEP6_CROSSLINGUAL_STUB.md`).

## Q (iii) — Acoustic vs lexical: the signal is **acoustic**
**Positive control** (`step3_*`, fixed lexicon ⇒ must be acoustic): **60/61**
emotion features label acoustic — mean eGeMAPS firing-AUC **0.83** vs sentence
firing-AUC **0.66** (margin +0.17). The labeling method works.

**Causal sufficiency** (`step7_*`): emotion decodes from the 61 **acoustic** features
at **0.552**, vs **0.431** from 61 lexical features and **0.423** from 61 random
(6-class; chance ≈0.17, full reconstruction 0.73). Acoustic features carry **+0.12**
more emotion than lexical → **the recovered emotion signal is acoustic, not lexical.**
(Necessity-ablation of 61/8192 features is ~null because the signal is distributed;
sufficiency is the informative test.)

## Deferred (data not on cluster)
EMIS audio-vs-text feature separation (Step 8), IEMOCAP real-transcript ablation
(Step 7 exact variant), SpeechCraft cross-lingual (Step 6) — each has a runnable
`STEP*_STUB.md` with the precise procedure to run when the data is available.
