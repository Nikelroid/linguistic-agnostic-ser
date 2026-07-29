# Related-work / prior-art search — results

**What this is.** The result of a deep, multi-source literature search (fan-out web
search → fetch sources → adversarial 2-of-3 verification → synthesis) asking: *what
existing work is similar to our SAE-for-SER approach, and could anything scoop us?*
Run mid-2026 against the question framed from `SAE_CONTEXT.md`, the draft report, and
the code. The raw workflow output was ephemeral (temp dir); this file is the
preserved synthesis. **All arXiv IDs below were surfaced by web search — double-check
each one manually before citing in a submission.**

---

## Bottom line
Our specific combination is **NOT scooped** by any single paper, but several 2025–2026
works independently cover *individual facets*. The closest is **AudioSAE** (same SAE
method, but not for emotion and without our controls). Our genuinely novel core —
**confound-controlled SAE emotion-feature identification on a lexicon-controlled
factorial corpus across six encoders**, plus **decodability ≠ disentanglement by
depth** and **EMIS lexical-shortcut by depth** — remains un-scooped. Because AudioSAE
is close on method *and* timing, we must position against it explicitly (done in the
report's Related Work section).

## Closest prior work (verified, high-confidence)

### 1. AudioSAE — the single closest (arXiv:2602.05027, EACL 2026)
- **Overlap:** same core method family — BatchTop-K SAEs, 8× expansion, per-frame
  (frame-level + max-pooled) activations from *every* encoder layer of frozen
  **Whisper-small** and **HuBERT-base**. Even includes **cross-model feature
  universality** (our facet F): same-layer/different-seed coverage > 50%, HuBERT and
  Whisper features do **not** align, inter-layer coverage high only in later layers.
- **How we differ:** emotion is only **one of four generic probe tasks** (gender,
  noise, accent, emotion). No monosemantic emotion-feature interpretation, no MI
  selection / permutation null, **no confound control**, no lexicon-controlled
  factorial corpus, no acoustic-vs-lexical split. Only **2 encoders** (we use 6),
  k=50 (we use 32), general audio understanding (not SER).
- **Takeaway:** concurrent *method* neighbor; cite as related, frame our affect-
  specific, confound-controlled analysis as the new contribution.

### 2. ASR-SAE on Whisper (arXiv:2605.12225, Pluth et al., May 2026)
- TopK SAE (K=45, ~31× expansion) but **only at Whisper's final encoder layer** — so
  **no depth analysis** (explicitly their future work). Discovers linguistic/acoustic
  features (phonetic, semantic, morphological, positional, lexical, noise). **No
  emotion / SER target.**

### 3. SAEs for audio foundation models (arXiv:2509.24793, Mariotte et al., ICASSP 2026 submission)
- Same *framing* as ours ("SAEs go beyond linear probing") on AST / HuBERT / WavLM /
  MERT, **but** the case study is **singing-technique classification** (vocal
  attributes), not SER / affect / paralinguistics.

### 4. Mechanistic Whisper-for-SER (arXiv:2509.08454, Ma et al., "Behind the Scenes", Sept 2025)
- Closest *SER-mechanistic* paper: layer-contribution probing, logit-lens, SVD, CKA on
  LoRA-adapted Whisper (4-class IEMOCAP) — but **explicitly no sparse autoencoders.**
  Confirms no prior Whisper-SER mechanistic work used SAEs.

## Supporting context (the acoustic-vs-lexical question)
- **EMIS** benchmark (arXiv:2510.25054, Corrêa et al.): the incongruent-speech dataset
  we use for Q iv.
- Text-dependency in paralinguistic datasets (arXiv:2403.07767) and the contribution of
  lexical features to SER (arXiv:2509.05634): motivate the acoustic-vs-lexical framing.

## Facet-by-facet verdict
| Facet | Closest work | Scooped? |
|---|---|---|
| A. SAEs on speech models | AudioSAE; Pluth; Mariotte | method exists, **not for emotion** |
| B. SAEs for emotion/affect | — | **no direct precedent** |
| C. decodability vs disentanglement by depth | (general SAE/probing) | **our depth result is new for SER** |
| D. acoustic-vs-lexical / lexical shortcut | EMIS; Ma et al.; 2403.07767; 2509.05634 | studied, **not via SAE features by depth** |
| E. lexicon-controlled factorial corpus + confound control | — | **our recipe is new** |
| F. cross-model SAE feature universality | AudioSAE | done for 2 models, **not emotion** |
| G. where emotion lives / Whisper-vs-SSL | Ma et al. | probing only, **no SAEs** |

## Where these findings are used
- `presentation/latex/report.tex` → §"Related work and how we differ" + §Positioning.
- `presentation/SAE_branch_brief.md` → §10 Publishability & positioning.
- `presentation/latex/slides.tex` → slide 8; `presentation/SAE_briefing.html` → talk guide.
- Citations added to `presentation/latex/references.bib`.

## Provenance / caveats
- Method: parallel web-search agents → source fetch → 2-of-3 adversarial verification →
  synthesis. The listed claims passed high-confidence votes (3-0 / 2-0).
- Some verifier sub-agents failed to return structured output during the run; the search
  + synthesis still completed, but **manually confirm each arXiv ID, venue, and author
  list before relying on them in a paper** (web-surfaced IDs can be imperfect).
