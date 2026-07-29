#!/usr/bin/env python
"""Step 5 — cross-paradigm comparison (Q ii.1): do the 6 encoders encode emotion
with the SAME features?

For each encoder we take its best CREMA-D layer, select confound-controlled
emotion features (Step 2 definition), and represent each feature by its
per-utterance activation. Utterances are aligned across encoders by filename, so
two features from different encoders can be "matched" by the correlation of
their firing patterns. We report a cross-encoder overlap matrix and contrast
Whisper (supervised ASR) against the SSL encoders.

    python notebooks/sae/run_step5.py
"""
from __future__ import annotations

import json
import os

SER_SCRATCH = os.environ.get(
    "SER_SCRATCH", os.path.join("/scratch1", os.environ.get("USER", "unknown")))
import sys

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.sae import analysis as A  # noqa: E402

FRAMES_DIR = os.path.join(SER_SCRATCH, "ser-experiments/SAE_frames")
FEAT_DIR = os.path.join(SER_SCRATCH, "ser-experiments/SAE_feats")
ART_DIR = os.path.join(_REPO, "results", "SAE")

# best CREMA-D probe layer per encoder (results/cremad/cremad_summary.csv)
ENCODER_LAYERS = {
    "HuBERT": 13, "wav2vec2": 10, "Whisper": 12,
    "WavLM": 14, "MERT": 9, "w2v-BERT": 17,
}
SSL = ["HuBERT", "wav2vec2", "WavLM", "w2v-BERT", "MERT"]
MATCH_THR = 0.5  # |corr| of per-utterance firing for a cross-encoder feature match


def load_encoder(enc, layer):
    tag = f"{enc}_CREMA-D_L{layer}"
    utt = np.load(f"{FEAT_DIR}/utt_mean_{tag}.npy")
    meta = np.load(f"{FRAMES_DIR}/utterances_{enc}_CREMA-D.npz", allow_pickle=True)
    return utt, meta


def main():
    # Common utterance order (by filename) across all encoders present.
    present = {e: l for e, l in ENCODER_LAYERS.items()
               if os.path.exists(f"{FEAT_DIR}/utt_mean_{e}_CREMA-D_L{l}.npy")}
    print(f"[step5] encoders present: {list(present)}")
    metas = {e: load_encoder(e, l) for e, l in present.items()}
    common = None
    for e, (_, meta) in metas.items():
        fn = set(meta["filename"].tolist())
        common = fn if common is None else (common & fn)
    common = sorted(common)
    print(f"[step5] {len(common)} common utterances")

    emo = {}        # encoder -> (activation matrix aligned to common, info)
    summary = []
    for e, (utt, meta) in metas.items():
        order = {f: i for i, f in enumerate(meta["filename"])}
        idx = [order[f] for f in common]
        utt_c = utt[idx]
        labels = meta["label"][idx]
        conf = {n: meta[n][idx] for n in ("sentence", "actor")
                if n in meta.files and len(set(meta[n].tolist())) > 1 and str(meta[n][0]) != ""}
        ids, info = A.select_emotion_features(utt_c, labels, conf)
        acts = utt_c[:, ids]
        # standardise columns for correlation
        acts_s = (acts - acts.mean(0)) / (acts.std(0) + 1e-9)
        emo[e] = {"ids": ids, "acts_s": acts_s, "labels": labels}
        dom = {}
        for fid in ids:
            per = {c: utt_c[labels == c, fid].sum() for c in np.unique(labels)}
            d = max(per, key=per.get); dom[d] = dom.get(d, 0) + 1
        summary.append({"encoder": e, "layer": present[e], "n_emotion_features": len(ids),
                        "median_mono": float(np.median(info["mono"][ids])) if len(ids) else 0.0,
                        **{f"n_{k}": v for k, v in dom.items()}})
    pd.DataFrame(summary).to_csv(f"{ART_DIR}/step5_encoder_summary.csv", index=False)

    # Cross-encoder overlap: fraction of e1's emotion features with a firing-pattern
    # match (|corr|>thr) among e2's emotion features.
    encs = list(emo)
    N = len(common)
    overlap = pd.DataFrame(index=encs, columns=encs, dtype=float)
    for e1 in encs:
        A1 = emo[e1]["acts_s"]
        for e2 in encs:
            A2 = emo[e2]["acts_s"]
            if A1.shape[1] == 0 or A2.shape[1] == 0:
                overlap.loc[e1, e2] = np.nan; continue
            corr = (A1.T @ A2) / N                      # (a, b)
            best = np.abs(corr).max(1) if e1 != e2 else _self_match(corr)
            overlap.loc[e1, e2] = float((best > MATCH_THR).mean())
    overlap.to_csv(f"{ART_DIR}/step5_overlap_matrix.csv")

    # Whisper vs SSL contrast
    whisper_to_ssl = float(np.nanmean([overlap.loc["Whisper", s] for s in SSL if s in encs])) \
        if "Whisper" in encs else None
    ssl_to_ssl = float(np.nanmean([overlap.loc[a, b] for a in SSL for b in SSL
                                   if a != b and a in encs and b in encs]))
    metrics = {
        "encoders": encs, "match_threshold": MATCH_THR,
        "n_common_utterances": N,
        "emotion_feature_counts": {e: int(emo[e]["acts_s"].shape[1]) for e in encs},
        "mean_whisper_to_ssl_overlap": whisper_to_ssl,
        "mean_ssl_to_ssl_overlap": ssl_to_ssl,
    }
    json.dump(metrics, open(f"{ART_DIR}/step5_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    M = overlap.astype(float).values
    im = ax[0].imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax[0].set_xticks(range(len(encs))); ax[0].set_xticklabels(encs, rotation=45, ha="right")
    ax[0].set_yticks(range(len(encs))); ax[0].set_yticklabels(encs)
    for i in range(len(encs)):
        for j in range(len(encs)):
            ax[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                       color="w" if M[i, j] < 0.6 else "k", fontsize=8)
    ax[0].set_title("Cross-encoder emotion-feature overlap\n(row matched into column)")
    fig.colorbar(im, ax=ax[0], fraction=.046)
    s = pd.DataFrame(summary).set_index("encoder")["n_emotion_features"]
    s.plot.bar(ax=ax[1], color=["crimson" if e == "Whisper" else "steelblue" for e in s.index])
    ax[1].set(ylabel="# confound-controlled emotion features", title="Emotion features per encoder (best layer)")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step5_cross_encoder.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 5 (cross-paradigm) ==========")
    print(json.dumps(metrics, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


def _self_match(corr):
    """For an encoder vs itself, ignore the diagonal (self-correlation = 1)."""
    c = np.abs(corr).copy()
    np.fill_diagonal(c, 0.0)
    return c.max(1)


if __name__ == "__main__":
    main()
