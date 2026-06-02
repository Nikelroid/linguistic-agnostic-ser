#!/usr/bin/env python
"""Step 6 — cross-speaker generalization (Q ii.3) on CREMA-D's 91 actors.

Do the SAE emotion features fire consistently across *unseen speakers*, or are
they speaker-entangled (which would explain the published MESD leave-one-speaker-
out collapse of ~0.30)? We probe emotion from the confound-controlled emotion
features under (a) random stratified CV and (b) speaker-disjoint GroupKFold
(train/test never share an actor); the gap is the speaker-entanglement.

Cross-lingual (Q ii.2) needs SpeechCraft (EN/ZH/European), which is NOT on this
cluster (EXP20 holds only CSVs, no audio/embeddings). That arm is documented as
blocked in notebooks/sae/STEP6_CROSSLINGUAL_STUB.md.

    python notebooks/sae/run_step6.py --encoder HuBERT --dataset CREMA-D --layer 13
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.sae import analysis as A          # noqa: E402
from src.sae.sae_model import TopKSAE      # noqa: E402

FEAT_DIR = "/scratch1/kelidari/ser-experiments/SAE_feats"
FRAMES_DIR = "/scratch1/kelidari/ser-experiments/SAE_frames"
CKPT_DIR = "/scratch1/kelidari/ser-experiments/SAE_ckpts"
ART_DIR = os.path.join(_REPO, "results", "SAE")


def acc_random_and_grouped(X, y, groups, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rnd = float(np.mean(cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                        scoring="accuracy")))
    grp = float(np.mean(cross_val_score(pipe, X, y, groups=groups, cv=GroupKFold(5), scoring="accuracy")))
    return rnd, grp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="HuBERT")
    ap.add_argument("--dataset", default="CREMA-D")
    ap.add_argument("--layer", type=int, default=13)
    args = ap.parse_args()
    tag = f"{args.encoder}_{args.dataset}_L{args.layer}"

    utt = np.load(f"{FEAT_DIR}/utt_mean_{tag}.npy")
    meta = np.load(f"{FRAMES_DIR}/utterances_{args.encoder}_{args.dataset}.npz", allow_pickle=True)
    labels, actor, sentence = meta["label"], meta["actor"], meta["sentence"]
    print(f"[step6] {tag}: {utt.shape[0]} utts, {len(set(actor))} actors")

    emo_ids, info = A.select_emotion_features(utt, labels, {"sentence": sentence, "actor": actor})
    print(f"[step6] {len(emo_ids)} confound-controlled emotion features")

    # (1) emotion features subset; (2) full SAE reconstruction (standard-probe analogue)
    sae = TopKSAE.load(f"{CKPT_DIR}/sae_{tag}.pt")
    with torch.no_grad():
        full_recon = sae.decode(torch.from_numpy(utt.astype(np.float32))).cpu().numpy()

    rows = []
    for name, X in [("emotion_features", utt[:, emo_ids]), ("full_reconstruction", full_recon)]:
        rnd, grp = acc_random_and_grouped(X, labels, actor)
        rows.append({"representation": name, "acc_random_cv": rnd,
                     "acc_speaker_disjoint": grp, "loso_drop": rnd - grp})
        print(f"  {name:20s} random={rnd:.3f}  speaker-disjoint={grp:.3f}  drop={rnd-grp:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(f"{ART_DIR}/step6_{tag}_cross_speaker.csv", index=False)

    # per-emotion-feature actor coverage: of the actors who express the feature's
    # dominant emotion, what fraction have the feature fire at least once?
    classes = np.unique(labels)
    cov = []
    for fid in emo_ids:
        acts = utt[:, fid]
        per = {c: acts[labels == c].sum() for c in classes}
        dom = max(per, key=per.get)
        dom_actors = set(actor[labels == dom])
        fire_actors = set(actor[(labels == dom) & (acts > 0)])
        cov.append(len(fire_actors) / max(1, len(dom_actors)))
    cov = np.array(cov)

    ef = out.set_index("representation").loc["emotion_features"]
    metrics = {
        "tag": tag, "n_actors": int(len(set(actor))), "n_emotion_features": int(len(emo_ids)),
        "emotion_features_acc_random": float(ef.acc_random_cv),
        "emotion_features_acc_speaker_disjoint": float(ef.acc_speaker_disjoint),
        "emotion_features_loso_drop": float(ef.loso_drop),
        "full_recon_loso_drop": float(out.set_index("representation").loc["full_reconstruction"].loso_drop),
        "median_actor_coverage": float(np.median(cov)),
        "published_mesd_loso_drop_ref": 0.30,
    }
    json.dump(metrics, open(f"{ART_DIR}/step6_{tag}_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(out)); w = 0.35
    ax[0].bar(x - w / 2, out.acc_random_cv, w, label="random CV", color="steelblue")
    ax[0].bar(x + w / 2, out.acc_speaker_disjoint, w, label="speaker-disjoint", color="crimson")
    ax[0].set_xticks(x); ax[0].set_xticklabels(out.representation, rotation=15)
    ax[0].set(ylabel="emotion accuracy", title=f"Cross-speaker generalization ({tag})")
    ax[0].legend(); ax[0].grid(alpha=.3, axis="y")
    ax[1].hist(cov, bins=20, range=(0, 1), color="seagreen")
    ax[1].axvline(np.median(cov), color="k", ls="--", label=f"median {np.median(cov):.2f}")
    ax[1].set(xlabel="fraction of dominant-emotion actors that fire the feature",
              ylabel="# emotion features", title="Per-feature actor coverage")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step6_{tag}_cross_speaker.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 6 (cross-speaker, Q ii.3) ==========")
    print(json.dumps(metrics, indent=2))
    print(f"VERDICT: emotion-feature LOSO drop {ef.loso_drop:.3f} "
          f"(vs published MESD ~0.30); median actor coverage {np.median(cov):.2f}")


if __name__ == "__main__":
    main()
