#!/usr/bin/env python
"""Step 4 — depth sweep (Q i): where does emotion *disentangle* vs *decode*?

For each of the 25 layers of one encoder (HuBERT here) we measure two curves on
CREMA-D:
  * decodability — 5-fold CV accuracy of a logistic probe on the mean-pooled
    raw activations (the classic "where is emotion readable" curve);
  * disentanglement — number of clean monosemantic SAE emotion features and
    their median monosemanticity (from the per-layer SAEs trained by the array
    job slurm/sae_train.sbatch).

Target finding: the disentanglement peak need not coincide with the accuracy peak.

    python notebooks/sae/run_step4.py --encoder HuBERT --dataset CREMA-D
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.sae import analysis as A  # noqa: E402

FRAMES_DIR = "/scratch1/kelidari/ser-experiments/SAE_frames"
FEAT_DIR = "/scratch1/kelidari/ser-experiments/SAE_feats"
ART_DIR = os.path.join(_REPO, "results", "SAE")


def pooled_probe_acc(frames_path, frame_index, labels, batch=200000, seed=42):
    """Mean-pool frames to utterance vectors, return 5-fold CV logistic accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fr = np.load(frames_path, mmap_mode="r")
    n_utt = int(frame_index.max()) + 1
    d = fr.shape[1]
    pooled = np.zeros((n_utt, d), dtype=np.float64)
    counts = np.bincount(frame_index, minlength=n_utt).astype(np.float64)
    for i in range(0, fr.shape[0], batch):
        np.add.at(pooled, frame_index[i:i + batch], np.asarray(fr[i:i + batch], dtype=np.float64))
    pooled /= np.clip(counts[:, None], 1, None)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-1))
    return float(np.mean(cross_val_score(pipe, pooled, labels, cv=5, scoring="accuracy")))


def sae_disentanglement(utt_path, labels, confounds=None, mono_min=0.5, n_shuffles=10):
    """Confound-controlled n emotion features + median monosemanticity for a layer."""
    utt = np.load(utt_path)
    ids, info = A.select_emotion_features(utt, labels, confounds,
                                          mono_min=mono_min, n_shuffles=n_shuffles)
    mono = info["mono"]
    return {
        "n_significant": info["n_significant"],
        "n_sig_monosemantic": info["n_sig_mono"],
        "n_emotion_features": int(len(ids)),
        "median_mono_emotion_feats": float(np.median(mono[ids])) if len(ids) else 0.0,
        "max_mi": float(info["mi"].max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="HuBERT")
    ap.add_argument("--dataset", default="CREMA-D")
    ap.add_argument("--layers", default=",".join(str(i) for i in range(25)))
    ap.add_argument("--mono-min", type=float, default=0.5)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    fi = np.load(f"{FRAMES_DIR}/frame_index_{args.encoder}_{args.dataset}.npy")
    meta = np.load(f"{FRAMES_DIR}/utterances_{args.encoder}_{args.dataset}.npz", allow_pickle=True)
    labels = meta["label"]
    confounds = {n: meta[n] for n in ("sentence", "actor")
                 if n in meta.files and len(set(meta[n].tolist())) > 1 and str(meta[n][0]) != ""}
    print(f"[step4] confounds: {list(confounds)}")

    rows = []
    for l in layers:
        tag = f"{args.encoder}_{args.dataset}_L{l}"
        frames_path = f"{FRAMES_DIR}/frames_{tag}.npy"
        utt_path = f"{FEAT_DIR}/utt_mean_{tag}.npy"
        if not (os.path.exists(frames_path) and os.path.exists(utt_path)):
            print(f"  ! layer {l}: missing frames/utt feats, skipping")
            continue
        acc = pooled_probe_acc(frames_path, fi, labels)
        dis = sae_disentanglement(utt_path, labels, confounds, args.mono_min)
        row = {"layer": l, "probe_acc": acc, **dis}
        rows.append(row)
        print(f"  L{l:2d}  acc={acc:.4f}  emo_feats={dis['n_emotion_features']:4d}  "
              f"med_mono={dis['median_mono_emotion_feats']:.3f}  max_mi={dis['max_mi']:.3f}")

    df = pd.DataFrame(rows).sort_values("layer")
    df.to_csv(f"{ART_DIR}/step4_{args.encoder}_{args.dataset}_depth.csv", index=False)

    acc_peak = int(df.loc[df.probe_acc.idxmax(), "layer"])
    disent_peak = int(df.loc[df.n_emotion_features.idxmax(), "layer"])
    mono_peak = int(df.loc[df.median_mono_emotion_feats.idxmax(), "layer"])
    metrics = {
        "encoder": args.encoder, "dataset": args.dataset, "n_layers": len(df),
        "accuracy_peak_layer": acc_peak, "max_probe_acc": float(df.probe_acc.max()),
        "disentanglement_peak_layer(n_emotion_features)": disent_peak,
        "monosemanticity_peak_layer": mono_peak,
        "peaks_coincide": acc_peak == disent_peak,
    }
    json.dump(metrics, open(f"{ART_DIR}/step4_{args.encoder}_{args.dataset}_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.layer, df.probe_acc, "-o", color="navy", label="probe accuracy (decodability)")
    ax1.axvline(acc_peak, color="navy", ls=":", alpha=.6)
    ax1.set(xlabel="layer", ylabel="probe accuracy", title=f"Depth sweep ({args.encoder} {args.dataset}): "
            f"decode peak L{acc_peak} vs disentangle peak L{disent_peak}")
    ax2 = ax1.twinx()
    ax2.plot(df.layer, df.n_emotion_features, "-s", color="crimson", label="# emotion features (disentanglement)")
    ax2.axvline(disent_peak, color="crimson", ls=":", alpha=.6)
    ax2.set_ylabel("# monosemantic emotion features", color="crimson")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step4_{args.encoder}_{args.dataset}_depth.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 4 (depth sweep) ==========")
    print(json.dumps(metrics, indent=2))
    print(f"VERDICT: decode peak L{acc_peak}, disentanglement peak L{disent_peak} "
          f"-> {'COINCIDE' if acc_peak == disent_peak else 'DIFFER (key finding for Q i)'}")


if __name__ == "__main__":
    main()
