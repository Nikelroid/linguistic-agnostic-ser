#!/usr/bin/env python
"""Step 2 — confound-controlled emotion-feature identification.

For each SAE feature, mutual information (MI) between its firing and the emotion
label identifies significant, monosemantic features; we then REQUIRE the firing
to be more informative about emotion than about the available confounds
(sentence/lexical content, actor/speaker). On CREMA-D's actor×sentence×emotion
factorial this strips out phonetic and speaker-driven features that would
otherwise masquerade as emotion features (see SAE_CONTEXT risk note).

CHECKPOINT: if no features survive, that is itself a finding (verdict=REVIEW).

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


def build_confounds(meta) -> dict:
    """Confound factors available for a dataset (non-trivial categorical columns)."""
    conf = {}
    for name in ("sentence", "actor"):
        if name in meta.files:
            col = meta[name]
            if len(set(col.tolist())) > 1 and all(str(x) != "" for x in col[:50]):
                conf[name] = col
    return conf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="HuBERT")
    ap.add_argument("--dataset", default="CREMA-D")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--pool", default="mean", choices=["mean", "max"])
    ap.add_argument("--n-shuffles", type=int, default=10)
    ap.add_argument("--mono-min", type=float, default=0.5)
    args = ap.parse_args()

    tag = f"{args.encoder}_{args.dataset}_L{args.layer}"
    os.makedirs(ART_DIR, exist_ok=True)

    utt = np.load(f"{FEAT_DIR}/utt_{args.pool}_{tag}.npy")
    meta = np.load(f"{FRAMES_DIR}/utterances_{args.encoder}_{args.dataset}.npz", allow_pickle=True)
    labels = meta["label"]
    confounds = build_confounds(meta)
    classes = np.unique(labels)
    print(f"[step2] {tag}: utt feats {utt.shape}, {len(classes)} emotions, "
          f"confounds={list(confounds)}")

    ids, info = A.select_emotion_features(utt, labels, confounds,
                                          mono_min=args.mono_min, n_shuffles=args.n_shuffles)
    mi, mono = info["mi"], info["mono"]

    dom = []
    for fid in ids:
        acts = utt[:, fid]
        per_class = np.array([acts[labels == c].sum() for c in classes])
        dom.append(classes[per_class.argmax()])
    df = pd.DataFrame({
        "feature": ids, "mi": mi[ids], "monosemanticity": mono[ids], "dominant_emotion": dom,
    })
    for name, mc in info["confound_mi"].items():
        df[f"mi_{name}"] = mc[ids]
        df[f"margin_emo_minus_{name}"] = mi[ids] - mc[ids]
    df = df.sort_values("mi", ascending=False)
    df.to_csv(f"{ART_DIR}/step2_{tag}_emotion_features.csv", index=False)

    metrics = {
        "tag": tag, "pool": args.pool, "n_features": int(utt.shape[1]),
        "mi_threshold_5sigma": info["thresh"], "mono_min": args.mono_min,
        "confounds": list(confounds),
        "n_significant": info["n_significant"],
        "n_sig_monosemantic": info["n_sig_mono"],
        "n_emotion_features": int(len(ids)),
        "frac_emotion_features": len(ids) / float(utt.shape[1]),
        "max_mi": float(mi.max()),
        "median_mono_emotion_feats": float(np.median(mono[ids])) if len(ids) else None,
        "emotion_feature_counts_by_dominant": {c: int(np.sum(np.array(dom) == c)) for c in classes},
    }
    clean = len(ids) >= 20
    metrics["verdict"] = "PASS" if clean else "REVIEW"
    json.dump(metrics, open(f"{ART_DIR}/step2_{tag}_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    order = np.argsort(mi)[::-1]
    ax[0].plot(mi[order], lw=1.0, label="emotion")
    for name, mc in info["confound_mi"].items():
        ax[0].plot(np.sort(mc)[::-1], lw=0.8, alpha=.6, label=f"{name} (confound)")
    ax[0].axhline(info["thresh"], color="crimson", ls="--", label=f"5σ null={info['thresh']:.4f}")
    ax[0].set(xlabel="feature rank", ylabel="MI (nats)", title=f"Feature MI: emotion vs confounds ({tag})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    if len(ids):
        ax[1].hist(mono[ids], bins=20, range=(0, 1), color="darkorange")
        ax[1].axvline(1.0 / len(classes), color="gray", ls=":", label="uniform")
    ax[1].set(xlabel="monosemanticity", ylabel="# emotion features",
              title=f"Confound-controlled emotion features (n={len(ids)})")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step2_{tag}_emotion_features.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 2 GATE (confound-controlled) ==========")
    print(json.dumps(metrics, indent=2))
    print("VERDICT:", "PASS — clean emotion features survive confound control"
          if clean else "REVIEW — too few emotion features survive; flag for human")


if __name__ == "__main__":
    main()
