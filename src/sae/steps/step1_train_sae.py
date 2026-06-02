#!/usr/bin/env python
"""Step 1 — first SAE (proof of life) on HuBERT CREMA-D L13.

Trains one TopK SAE on the per-frame L13 activations extracted in Step 0.5,
checks reconstruction quality + sparsity sanity, saves the checkpoint, the
utterance-pooled feature matrices (reused by Step 2+), and a feature dashboard.

CHECKPOINT (per SAE_CONTEXT.md): do not advance past Step 3 until clean emotion
features exist; this script's gate is "low reconstruction loss + sane sparsity".

    python notebooks/sae/run_step1.py --encoder HuBERT --dataset CREMA-D --layer 13
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

from src.sae import analysis as A           # noqa: E402
from src.sae.sae_model import train_sae     # noqa: E402

FRAMES_DIR = "/scratch1/kelidari/ser-experiments/SAE_frames"
CKPT_DIR = "/scratch1/kelidari/ser-experiments/SAE_ckpts"
FEAT_DIR = "/scratch1/kelidari/ser-experiments/SAE_feats"
ART_DIR = os.path.join(_REPO, "results", "SAE")    # committed plots/CSVs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="HuBERT")
    ap.add_argument("--dataset", default="CREMA-D")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()

    tag = f"{args.encoder}_{args.dataset}_L{args.layer}"
    for d in (CKPT_DIR, FEAT_DIR, ART_DIR):
        os.makedirs(d, exist_ok=True)

    frames_path = f"{FRAMES_DIR}/frames_{args.encoder}_{args.dataset}_L{args.layer}.npy"
    fi_path = f"{FRAMES_DIR}/frame_index_{args.encoder}_{args.dataset}.npy"
    utt_path = f"{FRAMES_DIR}/utterances_{args.encoder}_{args.dataset}.npz"
    print(f"[step1] loading frames {frames_path}")
    frames = np.load(frames_path)            # (M, d_in) into RAM (~3.85 GB)
    frame_index = np.load(fi_path)
    utts = np.load(utt_path, allow_pickle=True)
    labels = utts["label"]
    print(f"[step1] frames {frames.shape} from {len(labels)} utterances")

    # --- train ---------------------------------------------------------
    sae, history = train_sae(
        frames, expansion=args.expansion, k=args.k, steps=args.steps,
    )
    ckpt = f"{CKPT_DIR}/sae_{tag}.pt"
    sae.save(ckpt)
    print(f"[step1] saved SAE -> {ckpt}")

    # --- streaming feature summary + utterance pooling -----------------
    frames_mm = np.load(frames_path, mmap_mode="r")
    utt_mean, utt_max, stats = A.encode_pool_summarize(sae, frames_mm, frame_index)
    np.save(f"{FEAT_DIR}/utt_mean_{tag}.npy", utt_mean.astype(np.float32))
    np.save(f"{FEAT_DIR}/utt_max_{tag}.npy", utt_max.astype(np.float32))

    final = history[-1]
    metrics = {
        "tag": tag, "d_in": int(frames.shape[1]), "d_sae": int(sae.cfg.d_sae),
        "k": args.k, "expansion": args.expansion, "n_frames": int(frames.shape[0]),
        "n_utterances": int(len(labels)),
        "final_fvu": final["fvu"], "final_recon_mse": final["recon_mse"],
        "explained_variance": 1.0 - final["fvu"],
        "dead_features": stats["dead"], "alive_features": stats["alive"],
        "dead_frac": stats["dead"] / float(sae.cfg.d_sae),
        "median_density_alive": float(np.median(stats["density"][stats["density"] > 0])),
    }
    with open(f"{ART_DIR}/step1_{tag}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(f"{ART_DIR}/step1_{tag}_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    # per-feature stats CSV
    pd.DataFrame({
        "feature": np.arange(sae.cfg.d_sae),
        "density": stats["density"],
        "mean_act": stats["mean_act"],
        "max_act": stats["max_act"],
        "n_active_frames": stats["n_active_frames"],
    }).to_csv(f"{ART_DIR}/step1_{tag}_feature_stats.csv", index=False)

    A.plot_training(history, f"{ART_DIR}/step1_{tag}_training.png", title=f"({tag})")
    A.plot_feature_dashboard(stats, f"{ART_DIR}/step1_{tag}_dashboard.png", title=f"({tag})")

    # --- verdict -------------------------------------------------------
    print("\n========== STEP 1 GATE ==========")
    print(json.dumps(metrics, indent=2))
    ok_recon = metrics["final_fvu"] < 0.30
    ok_dead = metrics["dead_frac"] < 0.50
    print(f"reconstruction (FVU<0.30): {'PASS' if ok_recon else 'FAIL'} "
          f"(FVU={metrics['final_fvu']:.3f}, EV={metrics['explained_variance']:.3f})")
    print(f"sparsity/aliveness (dead<50%): {'PASS' if ok_dead else 'FAIL'} "
          f"(dead={metrics['dead_features']}/{sae.cfg.d_sae})")
    print("VERDICT:", "PASS — proceed to Step 2" if (ok_recon and ok_dead)
          else "REVIEW — tune k/expansion/steps before Step 2")


if __name__ == "__main__":
    main()
