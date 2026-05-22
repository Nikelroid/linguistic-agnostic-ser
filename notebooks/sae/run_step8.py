#!/usr/bin/env python
"""Step 8 — EMIS audio-vs-text feature separation (Whisper-vs-SSL mechanism).

On EMIS the spoken-text emotion and the prosodic (audio) emotion are set
independently, so each SAE feature can be labeled AUDIO-aligned or TEXT-aligned
by whether its firing is better predicted (CV ROC-AUC) by the audio emotion or
the text emotion. Per encoder we report a feature-level text-bias
(mean text_auc - audio_auc over emotion-relevant features). Hypothesis tying to
the project's C3 finding: Whisper's features stay audio-grounded (negative bias)
while SSL encoders take the lexical shortcut (positive bias).

    python notebooks/sae/run_step8.py
"""
from __future__ import annotations

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

# Layer per encoder ~ where the project measured the strongest text-bias
# (HuBERT L20, wav2vec2 L15, WavLM L20, Whisper L24); deep default for the rest.
EMIS_LAYERS = {"HuBERT": 20, "wav2vec2": 15, "WavLM": 20, "Whisper": 24, "MERT": 20, "w2v-BERT": 20}
SSL = ["HuBERT", "wav2vec2", "WavLM", "w2v-BERT", "MERT"]


def onehot(labels):
    classes, y = np.unique(labels, return_inverse=True)
    M = np.zeros((len(labels), len(classes)), np.float32); M[np.arange(len(labels)), y] = 1
    return M


def sig_mask(utt, labels, n_shuffles=10):
    mi = A.mutual_info_features(utt, labels)
    rng = np.random.default_rng(0)
    null = np.stack([A.mutual_info_features(utt, rng.permutation(labels)) for _ in range(n_shuffles)])
    return mi > (null.mean() + 5 * null.std()), mi


def main():
    rows, per_feat = [], []
    for enc, layer in EMIS_LAYERS.items():
        tag = f"{enc}_EMIS_L{layer}"
        up = f"{FEAT_DIR}/utt_mean_{tag}.npy"
        if not os.path.exists(up):
            print(f"  ! {tag}: missing utt features, skipping")
            continue
        utt = np.load(up)
        meta = np.load(f"{FRAMES_DIR}/utterances_{enc}_EMIS.npz", allow_pickle=True)
        audio, text = meta["label"], meta["text_label"]
        sa, _ = sig_mask(utt, audio); st, _ = sig_mask(utt, text)
        ids = np.where(sa | st)[0]                       # emotion-relevant features
        Xa, Xt = onehot(audio), onehot(text)
        biases = []
        for fid in ids:
            firing = utt[:, fid] > 0
            aa = A.firing_auc(firing, Xa); ta = A.firing_auc(firing, Xt)
            biases.append(ta - aa)
            per_feat.append({"encoder": enc, "feature": int(fid), "audio_auc": aa,
                             "text_auc": ta, "text_bias": ta - aa})
        biases = np.array(biases)
        rows.append({
            "encoder": enc, "layer": layer, "n_emotion_features": int(len(ids)),
            "n_audio_aligned": int((biases < 0).sum()), "n_text_aligned": int((biases > 0).sum()),
            "mean_text_bias": float(biases.mean()) if len(biases) else 0.0,
            "median_text_bias": float(np.median(biases)) if len(biases) else 0.0,
        })
        print(f"  {enc:9s} L{layer}: {len(ids)} feats, mean text-bias {biases.mean():+.3f} "
              f"(audio-aligned {int((biases<0).sum())}, text-aligned {int((biases>0).sum())})")

    df = pd.DataFrame(rows)
    df.to_csv(f"{ART_DIR}/step8_EMIS_text_bias.csv", index=False)
    pd.DataFrame(per_feat).to_csv(f"{ART_DIR}/step8_EMIS_per_feature.csv", index=False)

    encs = df.encoder.tolist()
    whisper_bias = float(df[df.encoder == "Whisper"].mean_text_bias.iloc[0]) if "Whisper" in encs else None
    ssl_bias = float(df[df.encoder.isin(SSL)].mean_text_bias.mean()) if df.encoder.isin(SSL).any() else None
    metrics = {
        "encoders": encs, "layers": EMIS_LAYERS,
        "mean_text_bias_by_encoder": dict(zip(df.encoder, df.mean_text_bias.round(4))),
        "whisper_mean_text_bias": whisper_bias, "ssl_mean_text_bias": ssl_bias,
        "whisper_resists_shortcut": (whisper_bias is not None and ssl_bias is not None
                                     and whisper_bias < ssl_bias),
    }
    json.dump(metrics, open(f"{ART_DIR}/step8_EMIS_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    order = df.sort_values("mean_text_bias")
    colors = ["crimson" if e == "Whisper" else "steelblue" for e in order.encoder]
    ax[0].bar(order.encoder, order.mean_text_bias, color=colors)
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set(ylabel="feature text-bias  (text AUC - audio AUC)",
              title="EMIS: SSL features lean text, Whisper stays audio-grounded")
    ax[0].tick_params(axis="x", rotation=30)
    pf = pd.DataFrame(per_feat)
    if len(pf):
        w = pf[pf.encoder == "Whisper"]; s = pf[pf.encoder.isin(SSL)]
        ax[1].scatter(s.text_auc, s.audio_auc, s=12, alpha=.4, color="steelblue", label="SSL")
        ax[1].scatter(w.text_auc, w.audio_auc, s=14, alpha=.6, color="crimson", label="Whisper")
        lim = [0.45, 1.0]; ax[1].plot(lim, lim, "k--", alpha=.5)
        ax[1].set(xlabel="text AUC", ylabel="audio AUC", xlim=lim, ylim=lim,
                  title="Per-feature audio vs text alignment"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step8_EMIS_text_bias.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 8 (EMIS audio-vs-text) ==========")
    print(json.dumps(metrics, indent=2))
    print("VERDICT:", "Whisper resists the lexical shortcut (lower feature text-bias than SSL)"
          if metrics["whisper_resists_shortcut"] else "no clear Whisper-vs-SSL separation")


if __name__ == "__main__":
    main()
