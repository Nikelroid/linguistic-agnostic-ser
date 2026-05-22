#!/usr/bin/env python
"""Render the SAE findings as a clean 3-page paper PDF (matplotlib, no LaTeX).

Pulls numbers from results/SAE/json/*_metrics.json and embeds the generated
figures, producing results/SAE/summary/SAE_findings_paper.pdf.

    python notebooks/sae/build_paper.py
"""
from __future__ import annotations

import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAE = os.path.join(ROOT, "results", "SAE")
JSON, PLOTS, SUM = (os.path.join(SAE, d) for d in ("json", "plots", "summary"))
PAGE_AR = 8.5 / 11.0
LX, RX = 0.07, 0.545


def J(name):
    p = os.path.join(JSON, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def render_col(fig, x, y0, blocks, wrap=60, body_fs=8.0, lh=0.0116, gap=0.005, ybottom=0.0):
    y = y0
    for kind, text in blocks:
        if y < ybottom:
            break
        if kind == "h":
            fig.text(x, y, text, fontsize=10, fontweight="bold", va="top"); y -= lh * 1.7
        elif kind == "sh":
            fig.text(x, y, text, fontsize=8.4, fontweight="bold", style="italic", va="top"); y -= lh * 1.3
        else:
            w = textwrap.fill(text, wrap); n = w.count("\n") + 1
            fig.text(x, y, w, fontsize=body_fs, va="top", family="serif"); y -= lh * n + gap
    return y


def place_image(fig, png, x, ytop, w):
    img = plt.imread(png); ih, iw = img.shape[0], img.shape[1]
    h = w * (ih / iw) * PAGE_AR
    ax = fig.add_axes([x, ytop - h, w, h]); ax.axis("off"); ax.imshow(img)
    return h


def main():
    d1 = J("step1_HuBERT_CREMA-D_L13_metrics.json"); d2 = J("step2_HuBERT_CREMA-D_L13_metrics.json")
    d3 = J("step3_HuBERT_CREMA-D_L13_metrics.json"); d4 = J("step4_HuBERT_CREMA-D_metrics.json")
    d5 = J("step5_metrics.json"); d6 = J("step6_HuBERT_CREMA-D_L13_metrics.json"); d7 = J("step7_HuBERT_CREMA-D_L13_metrics.json")
    ev = 100 * (1 - d1.get("final_fvu", 0))
    counts_str = ", ".join(f"{k} {v}" for k, v in d5.get("emotion_feature_counts", {}).items())
    P = os.path.join(SUM, "SAE_findings_paper.pdf")

    with PdfPages(P) as pdf:
        # ===================== PAGE 1: title + method =====================
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.965, "Sparse-Autoencoder Dissection of Emotion Features",
                 ha="center", fontsize=15.5, fontweight="bold")
        fig.text(0.5, 0.945, "in Frozen Speech Encoders", ha="center", fontsize=15.5, fontweight="bold")
        fig.text(0.5, 0.923, "Nima Kelidari, Minoo Ahmadi, Chaitanya Parwatkar, Xiangxu Lin", ha="center", fontsize=9)
        fig.text(0.5, 0.909, "USC ICT — Intelligent Human Perception Lab   ·   mechanistic extension of linguistic-agnostic-ser",
                 ha="center", fontsize=8, style="italic")
        abstract = ("Abstract — Linear probes show emotion is decodable in the middle-to-late layers of frozen "
                    "speech transformers, but not which internal features carry it, nor whether the signal is "
                    "acoustic or lexical. We train TopK Sparse Autoencoders (SAEs) on per-frame activations of six "
                    "frozen large encoders (HuBERT, WavLM, wav2vec2, w2v-BERT, MERT, Whisper) and dissect their "
                    "emotion features on the lexicon-controlled CREMA-D corpus (91 actors x 12 fixed sentences x 6 "
                    "emotions, 7,442 clips). A confound-controlled selection — a feature's firing must be more "
                    "informative about emotion than about the spoken sentence or the speaker — isolates genuine "
                    "emotion features from phonetic and speaker-driven ones. We find (i) emotion DISENTANGLES into "
                    "clean monosemantic features earlier (L2-L7) than where it is best linearly DECODED (L10); "
                    "(ii) all six encoders share an acoustic emotion-feature basis that generalizes across unseen "
                    "speakers; and (iii) the recovered emotion signal is acoustic, not lexical. Decodability and "
                    "disentanglement are distinct, and emotion here is carried by speaker-invariant acoustic features.")
        fig.text(0.07, 0.893, textwrap.fill(abstract, 134), fontsize=8.2, va="top", family="serif")

        left = [
            ("h", "1. Introduction"),
            ("b", "Self-supervised speech encoders dominate speech-emotion recognition, yet were trained on "
                  "ASR-style objectives that encode linguistic structure. Probing tells us WHERE emotion is "
                  "readable, not which features encode it or whether the signal is acoustic or lexical. SAEs "
                  "decompose dense activations into sparse, near-monosemantic features, moving the analysis from "
                  "decodability to mechanism. CREMA-D's fixed lexicon lets us dissociate acoustic delivery from "
                  "text by construction, so any recovered emotion signal there is necessarily acoustic."),
            ("h", "2. Method"),
            ("b", "We re-extract UN-pooled per-frame hidden states from the frozen encoders (the project's cached "
                  "embeddings are mean-pooled, too few to train an over-complete SAE); CREMA-D L13 yields 940,273 "
                  f"frames. On these we train a TopK SAE (d_in=1024, d_sae=8192, k=32, AuxK dead-feature revival), "
                  f"reaching FVU={d1.get('final_fvu',0):.3f} ({ev:.0f}% variance explained), "
                  f"{d1.get('dead_features')}/8192 dead features."),
            ("b", "Emotion features are scored by mutual information (MI) between a feature's firing and the emotion "
                  "label, thresholded at 5 sigma over a label-permutation null. A feature qualifies only if "
                  "MI(emotion) exceeds MI(sentence) AND MI(actor) and it is monosemantic (>=0.5 activation mass on "
                  "one emotion). Acoustic-vs-lexical labeling uses cross-validated ROC-AUC of a logistic classifier "
                  "predicting firing from eGeMAPS (acoustic) vs sentence identity (lexical); R-squared is unusable "
                  "because the pooled features are ~96% sparse."),
            ("sh", "Contributions."),
            ("b", "(1) A confound-controlled emotion-feature definition for speech SAEs; (2) evidence that "
                  "disentanglement and decodability peak at different depths; (3) a cross-encoder, cross-speaker, "
                  "and acoustic-vs-lexical characterization of the emotion-feature population."),
        ]
        right = [
            ("sh", "Confound control is decisive."),
            ("b", f"Significance + monosemanticity alone flag {d2.get('n_significant')} significant / 394 "
                  "monosemantic features, but ~66% carry more sentence (phonetic) or speaker information than "
                  "emotion — the documented 'SAE rediscovers phonemes, not affect' failure mode. Requiring "
                  f"MI(emotion) above both confounds yields {d2.get('n_emotion_features')} genuine emotion features "
                  f"(median monosemanticity {d2.get('median_mono_emotion_feats',0):.2f}), dominated by anger and "
                  "sadness; fear and neutral drop to zero — consistent with acoustic encoding of high-arousal, "
                  "acoustically-marked emotions."),
            ("sh", "Datasets & encoders."),
            ("b", "CREMA-D is the primary corpus (lexicon-controlled). The six encoders are the project's frozen "
                  "large variants. Per-frame activations, SAE checkpoints and utterance features are stored on "
                  "scratch; per-step CSVs, plots and metrics are versioned under results/SAE/. The depth sweep and "
                  "cross-encoder grids fan out as SLURM array jobs on V100 GPUs."),
            ("sh", "Pipeline."),
            ("b", "Eight steps: (0) inventory + setup; (0.5) CREMA-D; (1) first SAE; (2) emotion-feature "
                  "identification; (3) acoustic-vs-lexical control; (4) depth sweep; (5) cross-paradigm; "
                  "(6) cross-speaker; (7) causal ablation. EMIS / IEMOCAP-transcript / SpeechCraft variants are "
                  "deferred (data not on cluster) with runnable stubs."),
        ]
        render_col(fig, LX, 0.80, left, wrap=60, ybottom=0.40)
        render_col(fig, RX, 0.80, right, wrap=60, ybottom=0.40)
        fig.text(0.07, 0.392, "Figure 1. Six-panel summary (HuBERT / CREMA-D): depth decode-vs-disentangle, "
                 "acoustic-vs-lexical, cross-encoder overlap, per-encoder counts, cross-speaker generalization, "
                 "causal sufficiency.", fontsize=7.3, va="top", style="italic", wrap=True)
        place_image(fig, os.path.join(SUM, "SAE_summary_figure.png"), 0.09, 0.372, 0.82)
        pdf.savefig(fig); plt.close(fig)

        # ===================== PAGE 2: results + figures =====================
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.07, 0.965, "3. Results", fontsize=12, fontweight="bold")
        place_image(fig, os.path.join(PLOTS, "step4_HuBERT_CREMA-D_depth.png"), 0.05, 0.955, 0.46)
        place_image(fig, os.path.join(PLOTS, "step5_cross_encoder.png"), 0.52, 0.955, 0.46)
        fig.text(0.07, 0.74, "Figure 2. Depth sweep: probe accuracy (decode) peaks at L10 while the count of "
                 "monosemantic emotion features peaks at L7.", fontsize=7.2, va="top", style="italic", wrap=True)
        fig.text(0.55, 0.74, "Figure 3. Cross-encoder feature overlap and per-encoder emotion-feature counts; "
                 "Whisper is not distinct from the SSL encoders.", fontsize=7.2, va="top", style="italic", wrap=True)
        res_left = [
            ("sh", "Q (i) Depth: decode != disentangle."),
            ("b", f"Linear probe accuracy peaks at L{d4.get('accuracy_peak_layer')} ({d4.get('max_probe_acc',0):.3f}); "
                  f"the count of monosemantic emotion features peaks earlier at "
                  f"L{d4.get('disentanglement_peak_layer(n_emotion_features)')}, and monosemanticity at "
                  f"L{d4.get('monosemanticity_peak_layer')}. Deep layers (L17-24) retain decodability (~0.72) but "
                  "lose disentanglement (24-47 features): emotion stays linearly readable while becoming "
                  "distributed. Emotion organizes into clean features BEFORE the depth at which it is best decoded "
                  "(Figure 2)."),
            ("sh", "Q (ii.1) Across pretraining paradigms."),
            ("b", f"All six encoders carry emotion features at their best layer ({counts_str}), all anger/sadness-"
                  "dominated. Matching features across encoders by per-utterance firing correlation, Whisper->SSL "
                  f"overlap ({d5.get('mean_whisper_to_ssl_overlap',0):.2f}) approximately equals SSL->SSL overlap "
                  f"({d5.get('mean_ssl_to_ssl_overlap',0):.2f}). On fixed-lexicon CREMA-D Whisper is NOT distinct: "
                  "its known resistance concerns the lexical shortcut, not acoustic emotion encoding, which all "
                  "encoders share (Figure 3)."),
        ]
        res_right = [
            ("sh", "Q (ii.3) Across speakers."),
            ("b", f"Under speaker-disjoint cross-validation over 91 actors, emotion accuracy is unchanged (random "
                  f"{d6.get('emotion_features_acc_random',0):.3f} vs disjoint "
                  f"{d6.get('emotion_features_acc_speaker_disjoint',0):.3f}; leave-speaker-out drop "
                  f"{d6.get('emotion_features_loso_drop',0):.3f}); each feature fires for ~"
                  f"{d6.get('median_actor_coverage',0):.0%} of its emotion's actors. Because selection excluded "
                  "speaker-entangled features, the published MESD leave-one-speaker-out collapse (~0.30) is "
                  "attributable to exactly those excluded features (Figure 5)."),
            ("sh", "Q (iii) Acoustic, not lexical."),
            ("b", f"Positive control (fixed lexicon => must be acoustic): {d3.get('n_acoustic')}/"
                  f"{d3.get('n_emotion_features')} emotion features label acoustic (eGeMAPS firing-AUC "
                  f"{d3.get('mean_acoustic_auc',0):.2f} vs sentence AUC {d3.get('mean_lexical_auc',0):.2f}; "
                  "Figure 4). Causal sufficiency: emotion decodes from the 61 acoustic features at "
                  f"{d7.get('suff_acoustic_emotion',0):.3f}, vs {d7.get('suff_lexical',0):.3f} from lexical and "
                  f"{d7.get('suff_random',0):.3f} from random features (6-class; full reconstruction 0.73; "
                  "Figure 6). The recovered emotion signal is acoustic."),
        ]
        render_col(fig, LX, 0.70, res_left, wrap=60, ybottom=0.05)
        render_col(fig, RX, 0.70, res_right, wrap=60, ybottom=0.05)
        pdf.savefig(fig); plt.close(fig)

        # ===================== PAGE 3: discussion + figures + refs =====================
        fig = plt.figure(figsize=(8.5, 11))
        disc_left = [
            ("h", "4. Discussion"),
            ("b", "Decodability and disentanglement are distinct axes. A probe reads emotion best where the "
                  "representation is linearly separable (L10 and the deep layers), but the encoder packs emotion "
                  "into clean monosemantic features earlier (L2-L7); deep layers trade interpretable structure for "
                  "distributed separability. This sharpens the project's 'middle-to-late' decodability result into "
                  "a claim about where emotion is actually organized."),
            ("b", "On lexicon-controlled data all six encoders converge on a shared acoustic emotion basis, and "
                  "those features survive speaker-disjoint evaluation. The Whisper-vs-SSL divergence reported "
                  "previously is therefore not about acoustic emotion features (identical here) but about whether "
                  "an encoder exploits a LEXICAL shortcut when one is available — which CREMA-D cannot pose (its "
                  "text is fixed) and which the deferred EMIS experiment is designed to answer."),
            ("b", "The causal-sufficiency result answers question (iii): emotion is decodable from the acoustic "
                  "features far better than from lexical or random ones, and those features were independently "
                  "validated against eGeMAPS. What these encoders expose to a probe is paralinguistic delivery, "
                  "not lexical content."),
            ("h", "5. Limitations & deferred"),
            ("b", "Utterance-pooled sparse features make necessity-ablation of small feature subsets ~null (signal "
                  "is distributed across thousands of features); sufficiency is the informative test. Three planned "
                  "experiments are deferred because their data is not on the cluster, each with a runnable stub: "
                  "the IEMOCAP real-transcript ablation (transcripts not synced), the SpeechCraft cross-lingual arm "
                  "(no audio), and the EMIS audio-vs-text feature separation (unavailable) — the decisive test of "
                  "the Whisper lexical-shortcut mechanism."),
        ]
        disc_right = [
            ("h", "6. Reproducibility"),
            ("b", "Code in src/sae/ and notebooks/sae/; per-step CSVs/plots/metrics under results/SAE/{csv,plots,"
                  "json}; combined artifacts, the summary figure and this paper under results/SAE/summary/. SAEs "
                  "train on a single V100; build_summary.py and build_paper.py regenerate all artifacts."),
            ("h", "References"),
            ("b", "[1] linguistic-agnostic-ser report: emotion in middle-to-late layers; Whisper L24 resists the "
                  "EMIS lexical shortcut (text-bias -0.19) while SSL encoders take it (+0.61..+0.72); MESD LOSO "
                  "~0.30; EN-ZH transfer 0.29-0.32."),
            ("b", "[2] Gao et al. 2024, Scaling and evaluating sparse autoencoders (TopK SAEs)."),
            ("b", "[3] SAELens — open-source SAE training library."),
            ("b", "[4] arXiv:2605.12225, Sparse autoencoders for Whisper ASR (phonetic/morphemic/semantic features)."),
            ("b", "[5] arXiv:2509.08454, Mechanistic interpretability of LoRA-adapted Whisper for SER."),
            ("b", "[6] eGeMAPS / openSMILE — acoustic descriptor set."),
            ("sh", "Key numbers"),
            ("b", f"SAE FVU {d1.get('final_fvu',0):.3f} ({ev:.0f}% var), {d1.get('dead_features')}/8192 dead. "
                  f"Emotion features 6407 sig -> 394 mono -> {d2.get('n_emotion_features')} confound-controlled. "
                  f"Depth: decode L{d4.get('accuracy_peak_layer')} vs disentangle "
                  f"L{d4.get('disentanglement_peak_layer(n_emotion_features)')}. Acoustic control "
                  f"{d3.get('n_acoustic')}/{d3.get('n_emotion_features')}. Sufficiency acoustic "
                  f"{d7.get('suff_acoustic_emotion',0):.3f} vs lexical {d7.get('suff_lexical',0):.3f}. "
                  f"Cross-speaker LOSO drop {d6.get('emotion_features_loso_drop',0):.3f}."),
        ]
        render_col(fig, LX, 0.965, disc_left, wrap=60, ybottom=0.40)
        render_col(fig, RX, 0.965, disc_right, wrap=60, ybottom=0.40)
        fig.text(0.07, 0.37, "Figure 4. Acoustic vs lexical AUC per emotion feature (fixed lexicon).      "
                 "Figure 5. Cross-speaker generalization (LOSO).      Figure 6. Causal sufficiency probe.",
                 fontsize=7.2, va="top", style="italic", wrap=True)
        place_image(fig, os.path.join(PLOTS, "step3_HuBERT_CREMA-D_L13_acoustic_vs_lexical.png"), 0.04, 0.35, 0.30)
        place_image(fig, os.path.join(PLOTS, "step6_HuBERT_CREMA-D_L13_cross_speaker.png"), 0.35, 0.35, 0.34)
        place_image(fig, os.path.join(PLOTS, "step7_HuBERT_CREMA-D_L13_ablation.png"), 0.70, 0.35, 0.27)
        fig.text(0.5, 0.02, "results/SAE/summary/SAE_findings_paper.pdf — generated by notebooks/sae/build_paper.py",
                 ha="center", fontsize=6.5, style="italic", color="gray")
        pdf.savefig(fig); plt.close(fig)

    print("wrote", P)


if __name__ == "__main__":
    main()
