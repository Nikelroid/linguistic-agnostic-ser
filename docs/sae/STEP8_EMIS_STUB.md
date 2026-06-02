# Step 8 — EMIS two-regime confirmation (DEFERRED, runnable stub)

**Status:** EMIS is NOT available now. This is the planned confirmation that ties
the SAE feature populations to the project's headline C3 finding (Whisper resists
the lexical shortcut; SSL encoders take it). Run it once EMIS access is granted.

## Why EMIS
EMIS dissociates *audio* emotion from *text* emotion by construction (congruent
vs incongruent items), so it is the one corpus where a feature can be labeled
**audio-emotion** vs **text-emotion** directly, rather than via the eGeMAPS/lexical
proxy used on CREMA-D (Step 3). The published two-regime text-bias swing:

| Encoder  | EMIS-trained | RAV+SAV-trained | swing |
|----------|--------------|-----------------|-------|
| HuBERT   | +0.928       | −0.246          | 1.174 |
| wav2vec2 | +0.926       | −0.086          | 1.012 |
| WavLM    | +0.850       | −0.268          | 1.118 |
| Whisper  | +0.003       | −0.251          | 0.254 |

Hypothesis: Whisper has **separable** audio-emotion vs text-emotion SAE feature
populations; SSL encoders **collapse** them onto shared directions.

## To run once EMIS is available
1. Place EMIS audio + congruent/incongruent labels under
   `/scratch1/kelidari/ser_data/EMIS/`; add `load_emis` to `src/sae/extraction._LOADERS`.
2. Extract per-frame activations at L24 for each of the 6 encoders
   (`slurm/sae_extract.sbatch`, `LAYERS_STR=24`).
3. Train one SAE per encoder (`slurm/sae_train.sbatch`).
4. For each feature, compute:
   - `firing_auc(firing, audio_emotion_label)`  → audio-emotion alignment
   - `firing_auc(firing, text_emotion_label)`   → text-emotion alignment
   using the congruent/incongruent split so the two labels diverge.
5. Per encoder, report the count of audio-aligned vs text-aligned features and
   their overlap. **Predicted result:** Whisper shows two well-separated
   populations (low audio↔text feature overlap); SSL encoders show high overlap
   (the same features serve both) — the mechanistic explanation of the C3 swing.
6. Cross-check against Step 5: on fixed-lexicon CREMA-D all encoders converged
   (Whisper↔SSL overlap ≈ SSL↔SSL ≈ 0.40) precisely because no lexical shortcut
   is available there; EMIS is where the divergence should appear.
