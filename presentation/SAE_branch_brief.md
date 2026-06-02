# SAE branch — complete brief (technical + tactical)

Everything you need to know about the `SAE` branch: what it is, what's done, how it
runs, what it found, what's left, and whether/where it's publishable.

---

## 1. What this branch is
A mechanistic-interpretability extension of `linguistic-agnostic-ser` (the "N-1"
direction in `SAE_CONTEXT.md`). It trains **Sparse Autoencoders (SAEs)** on frozen
speech-encoder activations to move the project from **decodability** (what a linear
probe reads out) to **mechanism** (which monosemantic features the encoder uses for
emotion). Branched off `main`; all work lives here; pushed to `origin/SAE`.

## 2. Repository map (on the SAE branch)
```
src/sae/                      # the library
  extraction.py               # per-frame (+ --pooled) frozen-encoder activation extraction
  sae_model.py                # TopK SAE (8x, k=32) + AuxK dead-feature revival + train/encode/decode
  analysis.py                 # vectorized MI, confound-controlled select_emotion_features,
                              #   firing_auc, monosemanticity, streaming encode/pool, plots
  acoustic_lexical.py         # eGeMAPS (openSMILE) + BERT helpers
  steps/                      # pipeline steps as importable modules + CLI dispatcher
    __main__.py               #   python -m src.sae.steps <cmd> [args]
    inventory, step1..step8, step8b, build_summary
slurm/                        # sae_extract.sbatch, sae_train.sbatch, sae_envtest.sbatch (V100 arrays)
docs/sae/                     # README, FINDINGS, EXP_INVENTORY, requirements, deferred-step stubs
results/SAE/                  # committed artifacts: csv/ plots/ json/ embeddings/ summary/
  summary/paper/              # the ACM write-up (LaTeX + PDF)
presentation/                 # this brief, the 4-page report, slides, speaker notes (EN/FA)
```
Large binaries (per-frame activations, SAE `.pt` checkpoints, utterance features,
eGeMAPS cache) live on scratch, **not** git:
`/scratch1/kelidari/ser-experiments/{SAE_frames,SAE_ckpts,SAE_feats,SAE_aux}`.

## 3. Environment (important)
Use the spack module, **not** the repo's `ser_env` conda env (it lacks sae-lens):
```bash
module load gcc/13.3.0 python/3.11.9
pip install --user -r docs/sae/requirements.txt
```
Verified on a Tesla V100-32GB, torch 2.6.0+cu124, sae-lens 6.44. SLURM jobs
`module load` the same python (account `msoleyma_1026`, `--gres=gpu:v100:1`).
Run analysis steps with `OMP_NUM_THREADS=4` to avoid BLAS oversubscription.

## 4. Pipeline (Steps 0–8) — what each does
| Step | Command | Output |
|------|---------|--------|
| 0 | `inventory` | EXP-folder layout → `docs/sae/EXP_INVENTORY.md` |
| 0.5 | `extract` | per-frame activations to scratch |
| 1 | `step1` | train a TopK SAE; FVU 0.07, 4/8192 dead |
| 2 | `step2` | confound-controlled emotion features (394→**61**) |
| 3 | `step3` | acoustic-vs-lexical positive control (**60/61** acoustic) |
| 4 | `step4` | depth sweep: decode L10 vs disentangle L7 |
| 5 | `step5` | cross-encoder feature overlap |
| 6 | `step6` | cross-speaker (leave-speaker-out ≈ 0 drop) |
| 7 | `step7` | causal sufficiency (acoustic 0.55 > lexical 0.43) |
| 8 / 8b | `step8`/`step8b` | EMIS text-bias; layer/TTS sweep |
| — | `summary` | package `results/SAE/` + stats + composite figure |

## 5. Data
- **CREMA-D** (primary, lexicon-controlled): 91 actors × 12 fixed sentences × 6
  emotions, 7,442 clips. Pulled from HuggingFace `confit/cremad` (GitHub LFS was
  exhausted) → `/scratch1/kelidari/ser_data/CREMA-D`.
- **EMIS** (incongruent synthetic speech): 1,248 clips, text vs voice emotion set
  independently. Pulled from the **open Zenodo mirror** (record 19207001; IEEE
  DataPort was paywalled) → `/scratch1/kelidari/ser_data/EMIS`.
- Cached pooled embeddings (`EXP4` etc.) used only for inventory/probe references.

## 6. Findings (headline numbers)
- **SAE quality:** FVU 0.070 (93% variance explained), 4/8192 dead features.
- **Q i — Depth:** probe accuracy peaks **L10**; monosemantic emotion-feature count
  peaks **L7**, monosemanticity **L2**. Deep layers keep decodability (~0.72) but
  lose disentanglement (24–47 features). *Decoding ≠ disentangling.*
- **Q ii — Generalization:** all 6 encoders carry emotion features (61–145); cross-
  encoder overlap Whisper→SSL 0.40 ≈ SSL→SSL 0.39 (Whisper not special on fixed
  lexicon); leave-speaker-out drop ≈ 0, ~95% actor coverage → MESD LOSO collapse is
  due to speaker-entangled features the control removes.
- **Q iii — Acoustic vs lexical:** 60/61 acoustic (eGeMAPS AUC 0.83 vs sentence 0.66);
  sufficiency: emotion decodes 0.55 (acoustic) vs 0.43 (lexical) vs 0.42 (random).
- **Q iv — EMIS lexical shortcut:** a **deep-layer** phenomenon in ASR-SSL encoders
  (HuBERT L20 +0.79, WavLM +0.73, wav2vec2 +0.87); **Whisper resists at every layer**
  (L24 −0.11); MERT/w2v-BERT never take it. Reproduces published text-bias.

## 7. Method gotchas (so they're not rediscovered)
- **Confound control is essential.** MI + monosemanticity alone flags 394 features,
  ~66% phonetic/speaker. Require MI(emotion) > MI(sentence) AND > MI(actor).
- **MI must use the firing indicator** (sparse-appropriate, vectorized) — the kNN
  estimator was ~100× slower (22 min → 14 s).
- **Acoustic/lexical labeling uses ROC-AUC of firing**, not R² (features ~96% zero).
- **Necessity-ablation is near-null** (signal distributed across 8192 features) →
  use the **sufficiency** test.
- **EMIS probe must use RAW activations, not the SAE reconstruction** — the
  reconstruction spuriously inverts Whisper (+0.88 vs the correct −0.11).
- **Per-frame, not pooled:** the cached EXP embeddings are mean-pooled (≈7k/layer),
  too few for an 8× over-complete SAE; we re-extract un-pooled frames (~940k).

## 8. Limitations
- Most results are HuBERT/CREMA-D; cross-encoder & EMIS use one layer per encoder,
  not full 25-layer SAE grids for all six.
- Sufficiency (not necessity) is the causal lever, due to distribution.
- Acoustic axis = eGeMAPS; richer prosody descriptors could refine labeling.
- SAE hyperparameters (k=32, 8×) not exhaustively swept (reconstruction is stable).

## 9. Future work (stubs ready)
- **IEMOCAP real-transcript causal ablation** — `docs/sae/STEP7_IEMOCAP_STUB.md`
  (needs transcripts synced; only wav + meta on cluster).
- **SpeechCraft cross-lingual** feature overlap to explain the EN↔ZH collapse as
  disjoint feature sets — `docs/sae/STEP6_CROSSLINGUAL_STUB.md` (needs audio).
- Full per-encoder 25-layer SAE grids; **feature steering** (causally activate an
  emotion feature); dimensional V/A/D targets.

## 10. Publishability & positioning
**Closest prior work (from a verified literature sweep):**
- **AudioSAE** (EACL 2026, arXiv:2602.05027) — *closest*. Same SAE recipe (BatchTop-K,
  8×, per-frame, all layers) on **2** encoders (Whisper-small, HuBERT-base), with
  cross-model feature universality. BUT emotion is one of four generic probe tasks:
  **no** emotion-feature interpretation, **no** confound control, **no** factorial
  corpus, **no** acoustic-vs-lexical split. Concurrent method work — cite, don't fear.
- arXiv:2605.12225 (Pluth et al., ASR-SAE, final Whisper layer only, no affect, no depth).
- arXiv:2509.24793 (Mariotte et al., SAE for audio FMs; case study = singing technique).
- arXiv:2509.08454 (Ma et al., mechanistic Whisper-SER via probing/CKA — **no SAEs**).
- EMIS (arXiv:2510.25054); lexical-SER (arXiv:2509.05634); text-dependency (arXiv:2403.07767).

**Verdict:** our core combination — confound-controlled SAE **emotion**-feature
identification on a **lexicon-controlled factorial** corpus across **six** encoders,
plus **decodability≠disentanglement-by-depth** and **EMIS-shortcut-by-depth** — is,
to our knowledge, **un-scooped**. AudioSAE is close on method + timing, so position
explicitly: same tool, different (affect-specific, confound-controlled) question.

**Realistic targets:** an affective-computing or interpretability **workshop / short
paper** (ICASSP/Interspeech workshop, an *ACL interpretability track, or an ICML/NeurIPS
interpretability workshop). To reach a full paper, add: (a) full 25-layer grids for
all six encoders, (b) the IEMOCAP causal ablation with real transcripts, (c) a
steering experiment to make a causal (not just correlational) claim.

**Lab fit:** lands in the mechanistic-interpretability line (AVERE; the Ma et al.
LoRA-Whisper work). Frame as a fast, defensible mechanistic result that complements
the lab's multimodal-foundations vector.

## 11. Reproduce in one block
```bash
module load gcc/13.3.0 python/3.11.9
python -m src.sae.steps inventory --markdown docs/sae/EXP_INVENTORY.md
python -m src.sae.steps extract --encoder HuBERT --dataset CREMA-D \
    --data-dir /scratch1/kelidari/ser_data/CREMA-D --layers 13
python -m src.sae.steps step1 --encoder HuBERT --dataset CREMA-D --layer 13
python -m src.sae.steps step2 --encoder HuBERT --dataset CREMA-D --layer 13
python -m src.sae.steps step3 --encoder HuBERT --dataset CREMA-D --layer 13
# depth sweep + cross-encoder fan out on SLURM (see docs/sae/README.md), then:
python -m src.sae.steps step4 --encoder HuBERT --dataset CREMA-D
python -m src.sae.steps step5
python -m src.sae.steps step6 --encoder HuBERT --dataset CREMA-D --layer 13
python -m src.sae.steps step7 --encoder HuBERT --dataset CREMA-D --layer 13
python -m src.sae.steps step8 && python -m src.sae.steps step8b
python -m src.sae.steps summary
```

## 12. Tactical notes
- Commit identity must be `Nima Kelidari <68930046+Nikelroid@users.noreply.github.com>`
  (verify before every commit; global config is set). Past hostname-authored commits
  were already rewritten + force-pushed; backup at `~/agser-prerewrite-backup.bundle`.
- SAE branch is pushed; to get GitHub contribution credit for the bulk of this work,
  **merge SAE → main** via a PR when ready.
- Do **not** modify `final-report-ser/` (Overleaf, read-only) or commit the private
  context files (`PROJECT_CONTEXT.txt`, `SAE_CONTEXT.md`, the feasibility PDF).
- Deliverables for sharing: `presentation/SAE_report.pdf` (4 pp), `SAE_slides.pdf`
  (10 slides), speaker notes EN/FA, and `results/SAE/summary/paper/main.pdf` (3-pp paper).
