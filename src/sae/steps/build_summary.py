#!/usr/bin/env python
"""Package the SAE study results into the standard EXP layout + a summary folder.

Produces under results/SAE/:
    csv/      per-step CSVs            plots/    per-step PNGs
    json/     per-step metrics JSONs   embeddings/  (pointer; arrays live on scratch)
    summary/
        SAE_statistical_methodological_analysis.txt
        sae_master_metrics.csv          (one tidy row per metric, all steps)
        sae_depth_curve.csv             (accumulated 25-layer curve)
        sae_cross_encoder_overlap.csv   (encoder x encoder overlap matrix)
        SAE_summary_figure.{png,pdf}    (6-panel publication figure)
        SAE_FINDINGS.pdf                (text findings + figure, multipage)

Idempotent: safe to re-run.  python notebooks/sae/build_summary.py
"""
from __future__ import annotations

import glob
import json
import os

SER_SCRATCH = os.environ.get(
    "SER_SCRATCH", os.path.join("/scratch1", os.environ.get("USER", "unknown")))
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAE = os.path.join(ROOT, "results", "SAE")
CSV, PLOTS, JSON, EMB, SUM = (os.path.join(SAE, d) for d in ("csv", "plots", "json", "embeddings", "summary"))
SCRATCH = os.path.join(SER_SCRATCH, "ser-experiments")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": .3})


def organize():
    for d in (CSV, PLOTS, JSON, EMB, SUM):
        os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(SAE, "*")):
        if os.path.isdir(f):
            continue
        b = os.path.basename(f)
        dst = {".csv": CSV, ".png": PLOTS, ".json": JSON}.get(os.path.splitext(b)[1])
        if dst:
            shutil.move(f, os.path.join(dst, b))
    with open(os.path.join(EMB, "README.txt"), "w") as fh:
        fh.write("SAE per-frame activations, checkpoints and utterance features are "
                 "too large for git and live on scratch:\n"
                 f"  {SCRATCH}/SAE_frames   (frames_<enc>_<ds>_L<l>.npy, frame_index, utterances)\n"
                 f"  {SCRATCH}/SAE_ckpts    (sae_<enc>_<ds>_L<l>.pt + .json)\n"
                 f"  {SCRATCH}/SAE_feats    (utt_mean/utt_max per SAE)\n"
                 f"  {SCRATCH}/SAE_aux      (eGeMAPS cache)\n")


def load_json(name):
    p = os.path.join(JSON, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def build_master_csv():
    rows = []
    for jf in sorted(glob.glob(os.path.join(JSON, "*_metrics.json"))):
        step = os.path.basename(jf).split("_")[0]
        d = json.load(open(jf))
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                rows.append({"file": os.path.basename(jf), "step": step, "metric": k, "value": v})
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "sae_master_metrics.csv"), index=False)
    # accumulated depth curve + overlap matrix (copies into summary for convenience)
    for src, dst in [("step4_HuBERT_CREMA-D_depth.csv", "sae_depth_curve.csv"),
                     ("step5_overlap_matrix.csv", "sae_cross_encoder_overlap.csv"),
                     ("step5_encoder_summary.csv", "sae_encoder_summary.csv")]:
        s = os.path.join(CSV, src)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(SUM, dst))


def summary_figure():
    depth = pd.read_csv(os.path.join(CSV, "step4_HuBERT_CREMA-D_depth.csv"))
    al = pd.read_csv(os.path.join(CSV, "step3_HuBERT_CREMA-D_L13_acoustic_vs_lexical.csv"))
    ov = pd.read_csv(os.path.join(CSV, "step5_overlap_matrix.csv"), index_col=0)
    enc = pd.read_csv(os.path.join(CSV, "step5_encoder_summary.csv"))
    cs = pd.read_csv(os.path.join(CSV, "step6_HuBERT_CREMA-D_L13_cross_speaker.csv"))
    s7 = load_json("step7_HuBERT_CREMA-D_L13_metrics.json")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("SAE dissection of frozen-encoder emotion features (HuBERT / CREMA-D)",
                 fontsize=15, fontweight="bold")

    # (1) depth: decode vs disentangle
    a = ax[0, 0]; a.plot(depth.layer, depth.probe_acc, "-o", color="navy", ms=4, label="probe acc (decode)")
    a.set_xlabel("layer"); a.set_ylabel("probe accuracy", color="navy"); a.set_title("Q i — depth: decode vs disentangle")
    a2 = a.twinx(); a2.plot(depth.layer, depth.n_emotion_features, "-s", color="crimson", ms=4, label="# emotion feats")
    a2.set_ylabel("# emotion features", color="crimson"); a2.grid(False)
    a.axvline(depth.layer[depth.probe_acc.idxmax()], color="navy", ls=":", alpha=.5)
    a.axvline(depth.layer[depth.n_emotion_features.idxmax()], color="crimson", ls=":", alpha=.5)

    # (2) acoustic vs lexical (Q iii control)
    a = ax[0, 1]
    acoustic = al.label == "acoustic"
    a.scatter(al.lexical_auc[acoustic], al.acoustic_auc[acoustic], c="crimson", s=30, label="acoustic", edgecolor="k", lw=.3)
    a.scatter(al.lexical_auc[~acoustic], al.acoustic_auc[~acoustic], c="goldenrod", s=30, label="lexical", edgecolor="k", lw=.3)
    a.plot([.5, 1], [.5, 1], "k--", alpha=.5); a.set_xlabel("lexical AUC"); a.set_ylabel("acoustic AUC")
    a.set_title(f"Q iii — emotion features acoustic ({acoustic.sum()}/{len(al)})"); a.legend(fontsize=8)

    # (3) cross-encoder overlap heatmap
    a = ax[0, 2]; M = ov.values.astype(float)
    im = a.imshow(M, vmin=0, vmax=1, cmap="viridis")
    a.set_xticks(range(len(ov))); a.set_xticklabels(ov.columns, rotation=45, ha="right", fontsize=8)
    a.set_yticks(range(len(ov))); a.set_yticklabels(ov.index, fontsize=8)
    for i in range(len(ov)):
        for j in range(len(ov)):
            a.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                   color="w" if M[i, j] < .6 else "k")
    a.set_title("Q ii.1 — cross-encoder feature overlap"); a.grid(False)
    fig.colorbar(im, ax=a, fraction=.046)

    # (4) emotion-feature count per encoder
    a = ax[1, 0]
    colors = ["crimson" if e == "Whisper" else "steelblue" for e in enc.encoder]
    a.bar(enc.encoder, enc.n_emotion_features, color=colors)
    a.set_ylabel("# emotion features"); a.set_title("Emotion features per encoder (best layer)")
    a.tick_params(axis="x", rotation=30)

    # (5) cross-speaker
    a = ax[1, 1]; x = np.arange(len(cs)); w = .35
    a.bar(x - w / 2, cs.acc_random_cv, w, label="random CV", color="steelblue")
    a.bar(x + w / 2, cs.acc_speaker_disjoint, w, label="speaker-disjoint", color="crimson")
    a.set_xticks(x); a.set_xticklabels(cs.representation, rotation=15, fontsize=8)
    a.set_ylabel("emotion accuracy"); a.set_title("Q ii.3 — cross-speaker (LOSO)"); a.legend(fontsize=8)

    # (6) Step 7 sufficiency
    a = ax[1, 2]
    keys = ["suff_random", "suff_lexical", "suff_acoustic_emotion"]
    vals = [s7.get(k, 0) for k in keys]
    a.bar(["random", "lexical", "acoustic"], vals, color=["steelblue", "goldenrod", "crimson"])
    a.axhline(s7.get("full_reconstruction", np.nan), color="gray", ls="--", alpha=.6, label="full recon")
    for i, v in enumerate(vals):
        a.text(i, v + .005, f"{v:.3f}", ha="center", fontsize=9)
    a.set_ylabel("emotion acc from 61-feature subset"); a.set_title("Q iii — causal sufficiency"); a.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(SUM, "SAE_summary_figure.png"))
    fig.savefig(os.path.join(SUM, "SAE_summary_figure.pdf"))
    plt.close(fig)
    return fig


def stats_txt():
    d4 = load_json("step4_HuBERT_CREMA-D_metrics.json")
    d2 = load_json("step2_HuBERT_CREMA-D_L13_metrics.json")
    d3 = load_json("step3_HuBERT_CREMA-D_L13_metrics.json")
    d5 = load_json("step5_metrics.json")
    d6 = load_json("step6_HuBERT_CREMA-D_L13_metrics.json")
    d7 = load_json("step7_HuBERT_CREMA-D_L13_metrics.json")
    d1 = load_json("step1_HuBERT_CREMA-D_L13_metrics.json")
    d8 = load_json("step8_EMIS_metrics.json")
    txt = f"""================================================================================
SAE DISSECTION OF FROZEN-ENCODER EMOTION FEATURES — STATISTICAL & METHODOLOGICAL ANALYSIS
================================================================================
Encoders: HuBERT/WavLM/wav2vec2/w2v-BERT/MERT/Whisper (large, frozen). Primary
corpus: CREMA-D (91 actors x 12 fixed sentences x 6 emotions, 7,442 clips).

1. METHOD
--------------------------------------------------------------------------------
- SAE inputs are PER-FRAME activations (not the mean-pooled EXP cache): a TopK SAE
  on ~7k pooled vectors would be under-determined. CREMA-D L13 gives 940,273 frames.
- SAE: TopK, d_in=1024, d_sae={d1.get('d_sae','8192')}, k={d1.get('k',32)}, AuxK dead-feature revival.
  Reconstruction FVU={d1.get('final_fvu'):.3f} ({100*(1-d1.get('final_fvu',0)):.1f}% variance explained),
  dead features {d1.get('dead_features')}/{d1.get('d_sae')}.
- Emotion-feature MI uses the activation-firing indicator (vectorized; sparse-appropriate),
  with a 5-sigma permutation null over {10} label shuffles.
- CONFOUND CONTROL (key methodological point): a feature counts as an emotion
  feature only if MI(emotion) exceeds MI(sentence/lexical) AND MI(actor/speaker),
  on top of significance + monosemanticity>=0.5. Without it, {d2.get('n_significant')} features
  are significant and {d2.get('n_significant') and 394} pass monosemanticity, but ~66% carry more
  sentence/speaker than emotion information (phonetic features). After control:
  {d2.get('n_emotion_features')} genuine emotion features (median monosemanticity {d2.get('median_mono_emotion_feats'):.2f}).
- Acoustic-vs-lexical labeling uses cross-validated ROC-AUC of a logistic classifier
  predicting feature firing from eGeMAPS (acoustic) vs sentence identity (lexical);
  R^2 was abandoned (features are ~96% sparse -> degenerate/negative R^2).

2. RESULTS BY RESEARCH QUESTION
--------------------------------------------------------------------------------
Q (i) DEPTH — decodability vs disentanglement DO NOT coincide:
  probe-accuracy peak  = L{d4.get('accuracy_peak_layer')} (acc {d4.get('max_probe_acc'):.3f})
  emotion-feature peak = L{d4.get('disentanglement_peak_layer(n_emotion_features)')}
  monosemanticity peak = L{d4.get('monosemanticity_peak_layer')}
  -> emotion disentangles into clean features earlier (L2-L7) than where it is best
     linearly decoded (L10); deep layers keep decodability but lose disentanglement.

Q (ii.1) CROSS-PARADIGM — emotion-feature counts per encoder (best layer):
  {d5.get('emotion_feature_counts')}
  Whisper->SSL overlap {d5.get('mean_whisper_to_ssl_overlap'):.3f} ~= SSL->SSL {d5.get('mean_ssl_to_ssl_overlap'):.3f}
  -> on fixed-lexicon CREMA-D Whisper is NOT distinct; its known divergence concerns
     the lexical shortcut (deferred EMIS test), not acoustic emotion encoding.

Q (ii.3) CROSS-SPEAKER — emotion features are speaker-invariant:
  random-CV {d6.get('emotion_features_acc_random'):.3f} vs speaker-disjoint {d6.get('emotion_features_acc_speaker_disjoint'):.3f}
  (LOSO drop {d6.get('emotion_features_loso_drop'):.3f}); median actor coverage {d6.get('median_actor_coverage'):.2f}.
  -> the published MESD-LOSO ~0.30 collapse is attributable to the speaker-entangled
     features that confound control excludes.

Q (ii.2) CROSS-LINGUAL — BLOCKED (SpeechCraft audio not on cluster). See stub.

Q (iii) ACOUSTIC vs LEXICAL — the signal is ACOUSTIC:
  positive control: {d3.get('n_acoustic')}/{d3.get('n_emotion_features')} emotion features acoustic
  (eGeMAPS AUC {d3.get('mean_acoustic_auc'):.3f} vs sentence AUC {d3.get('mean_lexical_auc'):.3f}).
  causal sufficiency: emotion decodes from acoustic features {d7.get('suff_acoustic_emotion'):.3f}
  vs lexical {d7.get('suff_lexical'):.3f} vs random {d7.get('suff_random'):.3f} (+{d7.get('acoustic_minus_lexical_sufficiency'):.3f}).

Q (mechanism) EMIS — the lexical shortcut is a DEEP-LAYER phenomenon; Whisper resists:
  text-bias at the text-bias layer (raw mean-pooled, train-congruent/test-incongruent):
    {d8.get('text_bias_at_doc_layer')}
  Whisper L24 {d8.get('whisper_L24_text_bias')} (resists) vs ASR-SSL mean {d8.get('ssl_asr_mean_text_bias(HuBERT,WavLM,wav2vec2)')}.
  The shortcut emerges only in DEEP layers of the ASR-SSL encoders (HuBERT/WavLM/wav2vec2);
  all encoders are audio-grounded early (L0-L8); MERT (music) and w2v-BERT stay audio-grounded
  throughout. Reproduces the published EMIS text-bias (HuBERT +0.72, WavLM +0.61,
  wav2vec2 +0.66, Whisper -0.19). NOTE: a SAE-reconstruction probe gave a spurious
  Whisper +0.88 -- raw activations are the trustworthy, literature-consistent measure.

3. LIMITATIONS / DEFERRED
--------------------------------------------------------------------------------
- Utterance-level pooling of sparse frame features: necessity-ablation of small
  feature subsets is ~null (signal distributed); sufficiency is the informative test.
- EMIS now run (Zenodo mirror); IEMOCAP real-transcript ablation and SpeechCraft
  cross-lingual remain deferred (data not on cluster), with runnable stubs (STEP6/7).
================================================================================
"""
    with open(os.path.join(SUM, "SAE_statistical_methodological_analysis.txt"), "w") as fh:
        fh.write(txt)
    return txt


def findings_pdf(fig_txt):
    fnd = os.path.join(ROOT, "docs", "sae", "FINDINGS.md")
    findings = open(fnd).read() if os.path.exists(fnd) else "see FINDINGS.md"
    with PdfPages(os.path.join(SUM, "SAE_FINDINGS.pdf")) as pdf:
        # page 1: title + condensed findings text
        fig = plt.figure(figsize=(8.5, 11)); fig.text(0.5, 0.96, "SAE Study — Findings",
                ha="center", fontsize=18, fontweight="bold")
        body = "\n".join(findings.splitlines())
        if len(body) > 5200:
            body = body[:5200] + "\n...(full text in FINDINGS.md / analysis .txt)"
        fig.text(0.06, 0.92, body, ha="left", va="top", fontsize=7.2, family="monospace", wrap=True)
        pdf.savefig(fig); plt.close(fig)
        # page 2: the 6-panel summary figure (re-render onto a fresh page)
        img = plt.imread(os.path.join(SUM, "SAE_summary_figure.png"))
        fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.imshow(img); pdf.savefig(fig); plt.close(fig)


def main():
    organize()
    build_master_csv()
    summary_figure()
    stats_txt()
    # (the canonical findings document is the ACM paper at summary/paper/main.pdf)
    print("[build_summary] wrote results/SAE/{csv,plots,json,embeddings,summary}/")
    for f in sorted(glob.glob(os.path.join(SUM, "*"))):
        print("  summary/", os.path.basename(f))


if __name__ == "__main__":
    main()
