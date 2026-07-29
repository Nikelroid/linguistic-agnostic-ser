#!/usr/bin/env python
"""Step 3 — acoustic-vs-lexical label validation on CREMA-D (positive control).

For each (confound-controlled) emotion feature we ask whether its FIRING is
better predicted by acoustic descriptors (eGeMAPS) or by lexical content (BERT
transcript embedding), using cross-validated ROC-AUC of a logistic classifier.
AUC is used rather than R² because the features are sparse/binary (their pooled
activation is ~96% zeros), which makes regression R² degenerate.

On CREMA-D the lexicon is fixed, so emotion features must come out acoustic —
the positive control that the labeling works before Step 7 trusts it on IEMOCAP.

    python notebooks/sae/run_step3.py --encoder HuBERT --dataset CREMA-D --layer 13
"""
from __future__ import annotations

import argparse
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

from src.sae import acoustic_lexical as AL  # noqa: E402
from src.sae import analysis as A           # noqa: E402
from src.data_ingestion.loader import load_cremad  # noqa: E402

FRAMES_DIR = os.path.join(SER_SCRATCH, "ser-experiments/SAE_frames")
FEAT_DIR = os.path.join(SER_SCRATCH, "ser-experiments/SAE_feats")
CACHE_DIR = os.path.join(SER_SCRATCH, "ser-experiments/SAE_aux")
ART_DIR = os.path.join(_REPO, "results", "SAE")
DATA_DIR = os.path.join(SER_SCRATCH, "ser_data/CREMA-D")


def load_egemaps(filenames):
    eg_path = f"{CACHE_DIR}/egemaps_CREMA-D.npy"
    fn_path = f"{CACHE_DIR}/egemaps_CREMA-D_filenames.npy"
    if os.path.exists(eg_path) and os.path.exists(fn_path):
        eg = np.load(eg_path); eg_fn = np.load(fn_path, allow_pickle=True)
        order = {f: i for i, f in enumerate(eg_fn)}
        return eg[[order[f] for f in filenames]]
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = load_cremad(DATA_DIR)
    by_fn = {d["file"]: d for d in data}
    X = AL.egemaps_functionals([by_fn[f] for f in filenames])
    np.save(eg_path, X); np.save(fn_path, np.array(filenames))
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="HuBERT")
    ap.add_argument("--dataset", default="CREMA-D")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--pool", default="mean")
    args = ap.parse_args()
    tag = f"{args.encoder}_{args.dataset}_L{args.layer}"
    os.makedirs(ART_DIR, exist_ok=True)

    utt = np.load(f"{FEAT_DIR}/utt_{args.pool}_{tag}.npy")
    meta = np.load(f"{FRAMES_DIR}/utterances_{args.encoder}_{args.dataset}.npz", allow_pickle=True)
    filenames, sentences = meta["filename"], meta["sentence"]
    feature_ids = pd.read_csv(f"{ART_DIR}/step2_{tag}_emotion_features.csv")["feature"].values
    print(f"[step3] {tag}: {len(feature_ids)} confound-controlled emotion features")

    X_ac = load_egemaps(filenames)
    print(f"[step3] eGeMAPS {X_ac.shape}")
    # Lexical predictor = sentence one-hot. On CREMA-D's fixed 12-sentence lexicon
    # this is exactly the lexical content (a BERT embedding of the transcript takes
    # only 12 distinct values, i.e. rank-12 ≡ sentence identity, but is 768-dim and
    # makes the logistic fits non-convergent/slow). Real varying-transcript BERT is
    # used in Step 7 (IEMOCAP) via src/sae/acoustic_lexical.bert_embed.
    uniq = sorted(set(sentences.tolist()))
    code2idx = {c: i for i, c in enumerate(uniq)}
    X_lex = np.eye(len(uniq), dtype=np.float32)[[code2idx[c] for c in sentences]]
    print(f"[step3] lexical = sentence one-hot {X_lex.shape} ({len(uniq)} fixed sentences)")

    rows = []
    for fid in feature_ids:
        firing = utt[:, fid] > 0
        ac = A.firing_auc(firing, X_ac)
        lex = A.firing_auc(firing, X_lex)
        rows.append({"feature": int(fid), "acoustic_auc": ac, "lexical_auc": lex,
                     "margin": ac - lex, "label": "acoustic" if ac >= lex else "lexical",
                     "fire_rate": float(firing.mean())})
    out = pd.DataFrame(rows).sort_values("margin", ascending=False)
    out.to_csv(f"{ART_DIR}/step3_{tag}_acoustic_vs_lexical.csv", index=False)

    n = len(out); n_ac = int((out.label == "acoustic").sum())
    metrics = {
        "tag": tag, "n_emotion_features": n, "n_acoustic": n_ac, "n_lexical": n - n_ac,
        "frac_acoustic": n_ac / n if n else None,
        "mean_acoustic_auc": float(out.acoustic_auc.mean()),
        "mean_lexical_auc": float(out.lexical_auc.mean()),
        "mean_margin": float(out.margin.mean()),
    }
    metrics["verdict"] = "PASS" if (n and n_ac / n >= 0.8
                                    and out.acoustic_auc.mean() > out.lexical_auc.mean()) else "REVIEW"
    json.dump(metrics, open(f"{ART_DIR}/step3_{tag}_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(out.lexical_auc, out.acoustic_auc, c=(out.label == "acoustic"),
               cmap="coolwarm_r", s=35, edgecolor="k", linewidth=.3)
    ax.plot([0.5, 1], [0.5, 1], "k--", alpha=.5)
    ax.set(xlabel="lexical AUC (sentence)", ylabel="acoustic AUC (eGeMAPS)",
           title=f"Emotion features acoustic vs lexical ({tag})\n{n_ac}/{n} acoustic",
           xlim=(0.45, 1), ylim=(0.45, 1))
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step3_{tag}_acoustic_vs_lexical.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 3 (positive control) ==========")
    print(json.dumps(metrics, indent=2))
    print("VERDICT:", "PASS — emotion features are acoustic on fixed-lexicon CREMA-D"
          if metrics["verdict"] == "PASS" else "REVIEW — unexpected lexical features")


if __name__ == "__main__":
    main()
