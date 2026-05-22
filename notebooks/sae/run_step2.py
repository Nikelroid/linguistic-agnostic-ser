#!/usr/bin/env python
"""Step 2 — emotion-feature identification.

For each SAE feature, compute mutual information (MI) between its utterance-pooled
activation and the categorical emotion label. A permutation null (label shuffles)
sets a significance threshold; features above it are flagged "emotion features".
For those, we measure monosemanticity (fraction of activation mass on the single
most-associated emotion) and count them.

CHECKPOINT (per SAE_CONTEXT.md): if no features clear the null, or all flagged
features are low-monosemanticity (entangled), that is itself a finding — it is
written to the metrics JSON with verdict=REVIEW and should be surfaced to a human
before Step 3.

    python notebooks/sae/run_step2.py --encoder HuBERT --dataset CREMA-D --layer 13
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.sae import analysis as A  # noqa: E402

FRAMES_DIR = "/scratch1/kelidari/ser-experiments/SAE_frames"
FEAT_DIR = "/scratch1/kelidari/ser-experiments/SAE_feats"
ART_DIR = os.path.join(_REPO, "results", "SAE")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="HuBERT")
    ap.add_argument("--dataset", default="CREMA-D")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--pool", default="mean", choices=["mean", "max"])
    ap.add_argument("--n-shuffles", type=int, default=10)
    ap.add_argument("--mono-min", type=float, default=0.5,
                    help="monosemanticity floor for a significant feature to count as an emotion feature")
    args = ap.parse_args()

    tag = f"{args.encoder}_{args.dataset}_L{args.layer}"
    os.makedirs(ART_DIR, exist_ok=True)

    utt = np.load(f"{FEAT_DIR}/utt_{args.pool}_{tag}.npy")
    labels = np.load(f"{FRAMES_DIR}/utterances_{args.encoder}_{args.dataset}.npz",
                     allow_pickle=True)["label"]
    print(f"[step2] {tag}: utt feats {utt.shape}, {len(np.unique(labels))} emotions")

    # MI on true labels + permutation null.
    mi = A.mutual_info_features(utt, labels)
    null = []
    rng = np.random.default_rng(0)
    for s in range(args.n_shuffles):
        null.append(A.mutual_info_features(utt, rng.permutation(labels), random_state=s))
    null = np.stack(null)
    thresh = float(null.mean() + 5 * null.std())   # global 5-sigma over the null
    classes = np.unique(labels)

    # An "emotion feature" must clear two bars, not just statistical significance:
    #   (1) significant association with emotion (MI > 5σ null), AND
    #   (2) monosemantic w.r.t. emotion (>= mono_min of its activation mass on one class).
    # Significance alone over-flags (~78% of features at N=7442 carry trivial MI);
    # monosemanticity is the effect-size/cleanliness criterion (per the brief).
    sig = np.where(mi > thresh)[0]
    mono_sig = A.monosemanticity(utt, labels, sig) if len(sig) else np.array([])
    keep = mono_sig >= args.mono_min
    emotion_feats = sig[keep]
    mono = mono_sig[keep]

    dom = []
    for fid in emotion_feats:
        acts = utt[:, fid]
        per_class = np.array([acts[labels == c].sum() for c in classes])
        dom.append(classes[per_class.argmax()])

    pd.DataFrame({
        "feature": emotion_feats,
        "mi": mi[emotion_feats],
        "monosemanticity": mono,
        "dominant_emotion": dom,
    }).sort_values("mi", ascending=False).to_csv(
        f"{ART_DIR}/step2_{tag}_emotion_features.csv", index=False)

    metrics = {
        "tag": tag, "pool": args.pool, "n_features": int(utt.shape[1]),
        "mi_threshold_5sigma": thresh, "mono_min": args.mono_min,
        "n_significant": int(len(sig)),
        "frac_significant": len(sig) / float(utt.shape[1]),
        "n_emotion_features": int(len(emotion_feats)),
        "frac_emotion_features": len(emotion_feats) / float(utt.shape[1]),
        "max_mi": float(mi.max()),
        "median_mono_emotion_feats": float(np.median(mono)) if len(mono) else None,
        "mono_counts": {f">{t}": int((mono_sig > t).sum()) for t in (0.4, 0.5, 0.6, 0.7)},
        "emotion_feature_counts_by_dominant": {c: int(np.sum(np.array(dom) == c)) for c in classes},
    }
    # CHECKPOINT verdict: clean emotion features must exist (count + cleanliness).
    clean = len(emotion_feats) >= 20 and (len(mono) and np.median(mono) >= args.mono_min)
    metrics["verdict"] = "PASS" if clean else "REVIEW"
    with open(f"{ART_DIR}/step2_{tag}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # plots: MI ranked + monosemanticity histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    order = np.argsort(mi)[::-1]
    ax[0].plot(mi[order], lw=1.2)
    ax[0].axhline(thresh, color="crimson", ls="--", label=f"5σ null = {thresh:.4f}")
    ax[0].set(xlabel="feature rank", ylabel="MI with emotion (nats)",
              title=f"Feature–emotion MI ({tag})")
    ax[0].legend(); ax[0].grid(alpha=.3)
    if len(mono_sig):
        ax[1].hist(mono_sig, bins=25, range=(0, 1), color="darkorange")
        ax[1].axvline(1.0 / len(classes), color="gray", ls=":", label="uniform")
        ax[1].axvline(args.mono_min, color="crimson", ls="--", label=f"emotion-feature cut={args.mono_min}")
    ax[1].set(xlabel="monosemanticity (mass on top emotion)", ylabel="# significant features",
              title=f"Monosemanticity of significant feats (emotion feats={len(emotion_feats)})")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step2_{tag}_emotion_features.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 2 GATE ==========")
    print(json.dumps(metrics, indent=2))
    print("VERDICT:", "PASS — clean emotion features exist, proceed to Step 3"
          if clean else "REVIEW — few/entangled emotion features; flag for human before Step 3")


if __name__ == "__main__":
    main()
