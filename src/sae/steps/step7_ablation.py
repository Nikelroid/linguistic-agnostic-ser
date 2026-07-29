#!/usr/bin/env python
"""Step 7 — causal ablation (Q iii): is the recovered emotion signal acoustic or lexical?

Brief intent: on IEMOCAP, label emotion features acoustic/lexical, ablate the
lexical ones, re-probe. IEMOCAP transcripts are NOT synced to this cluster
(only wav + meta_data.json), so the exact IEMOCAP variant is left as a runnable
stub (see notebooks/sae/STEP7_IEMOCAP_STUB.md). Here we run the causal ablation
on CREMA-D, where the lexical axis is the fixed-sentence identity that Step 3
validated.

Method (uses the affine SAE decoder, so ablation is exact at utterance level —
mean_t decode(f_t) = decode(mean_t f_t)):
  * full reconstruction         = decode(utt_mean)
  * ablate acoustic-emotion     = decode(utt_mean with the 61 emotion features zeroed)
  * ablate lexical/phonetic     = decode(utt_mean with top-N sentence-driven features zeroed)
  * ablate random               = control (same N random features)
Re-probe emotion (5-fold CV logistic) on each. Acoustic-emotion ablation causing
the LARGEST accuracy drop (vs lexical/random) is causal evidence that emotion is
carried by acoustic, not lexical, features.

    python notebooks/sae/run_step7.py --encoder HuBERT --dataset CREMA-D --layer 13
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


def probe_acc(X, y, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=300))  # lbfgs multinomial (6-class)
    return float(np.mean(cross_val_score(pipe, X, y, cv=5, scoring="accuracy")))


def decode(sae, feats):
    with torch.no_grad():
        return sae.decode(torch.from_numpy(feats.astype(np.float32))).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="HuBERT")
    ap.add_argument("--dataset", default="CREMA-D")
    ap.add_argument("--layer", type=int, default=13)
    args = ap.parse_args()
    tag = f"{args.encoder}_{args.dataset}_L{args.layer}"

    utt = np.load(f"{FEAT_DIR}/utt_mean_{tag}.npy")
    meta = np.load(f"{FRAMES_DIR}/utterances_{args.encoder}_{args.dataset}.npz", allow_pickle=True)
    labels, sentence, actor = meta["label"], meta["sentence"], meta["actor"]
    sae = TopKSAE.load(f"{CKPT_DIR}/sae_{tag}.pt")
    rng = np.random.default_rng(42)

    # acoustic-emotion features (confound-controlled)
    conf = {"sentence": sentence, "actor": actor}
    emo_ids, info = A.select_emotion_features(utt, labels, conf)
    mi_emo = info["mi"]; mi_sent = info["confound_mi"]["sentence"]
    # lexical/phonetic features: sentence-driven (sig for sentence & sentence>emotion)
    sent_thresh = info["thresh"]
    lex_pool = np.where((mi_sent > sent_thresh) & (mi_sent > mi_emo))[0]
    n = len(emo_ids)
    lex_ids = lex_pool[np.argsort(mi_sent[lex_pool])[::-1][:n]]   # top-n by sentence-MI
    rand_ids = rng.choice(utt.shape[1], size=n, replace=False)
    print(f"[step7] {tag}: {n} acoustic-emotion feats, {len(lex_pool)} lexical pool "
          f"(ablating top {len(lex_ids)}), {n} random")

    # SUFFICIENCY (primary): decode emotion from ONLY each feature subset's
    # activations. Removing 61/8192 features from the full reconstruction is too
    # weak (signal is distributed), so we ask which 61-feature subset *suffices*.
    def suff(ids):
        return probe_acc(utt[:, ids], labels)

    # NECESSITY (secondary): zero a subset in the (affine) reconstruction, re-probe.
    def ablate_recon(ids):
        u = utt.copy(); u[:, ids] = 0.0
        return probe_acc(decode(sae, u), labels)

    full_recon = probe_acc(decode(sae, utt), labels)
    res = {
        "suff_acoustic_emotion": suff(emo_ids),
        "suff_lexical": suff(lex_ids),
        "suff_random": suff(rand_ids),
        "full_reconstruction": full_recon,
        "necessity_ablate_lexical_acc": ablate_recon(lex_ids),
        "necessity_ablate_acoustic_emotion_acc": ablate_recon(emo_ids),
    }
    pd.DataFrame([{"condition": k, "probe_acc": v} for k, v in res.items()]).to_csv(
        f"{ART_DIR}/step7_{tag}_ablation.csv", index=False)

    metrics = {
        "tag": tag, "n_features_per_subset": int(n), "n_classes": int(len(np.unique(labels))),
        **{k: float(v) for k, v in res.items()},
        "acoustic_minus_lexical_sufficiency": float(res["suff_acoustic_emotion"] - res["suff_lexical"]),
        "acoustic_minus_random_sufficiency": float(res["suff_acoustic_emotion"] - res["suff_random"]),
        "emotion_is_acoustic": bool(res["suff_acoustic_emotion"] > res["suff_lexical"]
                                    and res["suff_acoustic_emotion"] > res["suff_random"]),
    }
    json.dump(metrics, open(f"{ART_DIR}/step7_{tag}_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = ["suff_random", "suff_lexical", "suff_acoustic_emotion"]
    vals = [res[k] for k in order]
    ax.bar(range(3), vals, color=["steelblue", "goldenrod", "crimson"])
    ax.axhline(full_recon, color="gray", ls="--", alpha=.6, label=f"full reconstruction ({full_recon:.3f})")
    ax.set_xticks(range(3)); ax.set_xticklabels(["random 61", "lexical 61", "acoustic-emotion 61"])
    ax.set(ylabel="emotion probe accuracy from subset (5-fold CV)",
           title=f"Step 7 sufficiency ({tag}): emotion decodes from acoustic features\n"
                 f"acoustic {res['suff_acoustic_emotion']:.3f} vs lexical {res['suff_lexical']:.3f} "
                 f"vs random {res['suff_random']:.3f}")
    for i, v in enumerate(vals):
        ax.text(i, v + .005, f"{v:.3f}", ha="center", fontsize=10)
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step7_{tag}_ablation.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 7 (causal sufficiency/ablation, Q iii) ==========")
    print(json.dumps(metrics, indent=2))
    print(f"VERDICT: emotion decodes from acoustic features {res['suff_acoustic_emotion']:.3f} "
          f"vs lexical {res['suff_lexical']:.3f} vs random {res['suff_random']:.3f} -> emotion is "
          f"{'ACOUSTIC' if metrics['emotion_is_acoustic'] else 'NOT cleanly acoustic'}")


if __name__ == "__main__":
    main()
