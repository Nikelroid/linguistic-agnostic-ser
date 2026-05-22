#!/usr/bin/env python
"""Step 8 — EMIS text-bias via the congruent-train / incongruent-test probe.

This is the published EMIS protocol (Minoo's Task 7): train an emotion probe on
CONGRUENT clips (audio emotion == text emotion), then test on INCONGRUENT clips
and compare accuracy w.r.t. the AUDIO (prosodic) emotion (the target) vs the TEXT
emotion (the proxy). text_bias = proxy_acc - target_acc; positive => the probe
took the lexical shortcut, <=0 => it stayed audio-grounded.

We run it on (a) the SAE reconstruction at each encoder's text-bias layer
(directly comparable to the published per-encoder numbers) and (b) the SAE
feature activations (does the SAE feature basis carry the shortcut?). Hypothesis
from the project's C3: Whisper resists (text_bias <= 0), SSL encoders take it.

    python notebooks/sae/run_step8.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.sae.sae_model import TopKSAE  # noqa: E402

FRAMES_DIR = "/scratch1/kelidari/ser-experiments/SAE_frames"
FEAT_DIR = "/scratch1/kelidari/ser-experiments/SAE_feats"
CKPT_DIR = "/scratch1/kelidari/ser-experiments/SAE_ckpts"
ART_DIR = os.path.join(_REPO, "results", "SAE")

# Per-encoder layer ~ where the project measured the strongest text-bias.
EMIS_LAYERS = {"HuBERT": 20, "wav2vec2": 15, "WavLM": 20, "Whisper": 24, "MERT": 20, "w2v-BERT": 20}
SSL = ["HuBERT", "wav2vec2", "WavLM", "w2v-BERT", "MERT"]


def text_bias(X, audio, text, cong, incong, n_rep=5, seed=42):
    """Train emotion probe on congruent clips, test on incongruent; return
    (target_acc, proxy_acc, text_bias) averaged over n_rep regularised fits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tgt, prx = [], []
    Xc, yc = X[cong], audio[cong]
    Xi, ai, ti = X[incong], audio[incong], text[incong]
    rng = np.random.default_rng(seed)
    for r in range(n_rep):
        idx = rng.choice(len(Xc), size=len(Xc), replace=True)        # bootstrap congruent
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        clf.fit(Xc[idx], yc[idx])
        pred = clf.predict(Xi)
        tgt.append((pred == ai).mean()); prx.append((pred == ti).mean())
    tgt, prx = np.array(tgt), np.array(prx)
    return float(tgt.mean()), float(prx.mean()), float((prx - tgt).mean())


def main():
    rows = []
    for enc, layer in EMIS_LAYERS.items():
        tag = f"{enc}_EMIS_L{layer}"
        up = f"{FEAT_DIR}/utt_mean_{tag}.npy"
        if not os.path.exists(up):
            print(f"  ! {tag}: missing utt features, skipping"); continue
        utt = np.load(up)
        meta = np.load(f"{FRAMES_DIR}/utterances_{enc}_EMIS.npz", allow_pickle=True)
        audio, text = meta["label"], meta["text_label"]
        cong = audio == text; incong = ~cong
        sae = TopKSAE.load(f"{CKPT_DIR}/sae_{tag}.pt")
        with torch.no_grad():
            recon = sae.decode(torch.from_numpy(utt.astype(np.float32))).cpu().numpy()
        r_t, r_p, r_b = text_bias(recon, audio, text, cong, incong)   # reconstruction (~raw)
        f_t, f_p, f_b = text_bias(utt, audio, text, cong, incong)     # SAE features
        rows.append({"encoder": enc, "layer": layer,
                     "recon_target_acc": r_t, "recon_proxy_acc": r_p, "recon_text_bias": r_b,
                     "feat_target_acc": f_t, "feat_proxy_acc": f_p, "feat_text_bias": f_b})
        print(f"  {enc:9s} L{layer}: recon text-bias {r_b:+.3f} (audio {r_t:.2f} / text {r_p:.2f}) | "
              f"feat text-bias {f_b:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{ART_DIR}/step8_EMIS_text_bias.csv", index=False)
    encs = df.encoder.tolist()
    w = float(df[df.encoder == "Whisper"].recon_text_bias.iloc[0]) if "Whisper" in encs else None
    ssl = float(df[df.encoder.isin(SSL)].recon_text_bias.mean()) if df.encoder.isin(SSL).any() else None
    metrics = {
        "encoders": encs, "layers": EMIS_LAYERS, "protocol": "train-congruent / test-incongruent",
        "recon_text_bias_by_encoder": dict(zip(df.encoder, df.recon_text_bias.round(3))),
        "feat_text_bias_by_encoder": dict(zip(df.encoder, df.feat_text_bias.round(3))),
        "whisper_recon_text_bias": w, "ssl_mean_recon_text_bias": ssl,
        "whisper_resists_shortcut": (w is not None and ssl is not None and w < ssl),
    }
    json.dump(metrics, open(f"{ART_DIR}/step8_EMIS_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    o = df.sort_values("recon_text_bias")
    colors = ["crimson" if e == "Whisper" else "steelblue" for e in o.encoder]
    ax[0].bar(o.encoder, o.recon_text_bias, color=colors); ax[0].axhline(0, color="k", lw=.8)
    ax[0].set(ylabel="text-bias  (text acc - audio acc)", title="EMIS text-bias (train congruent / test incongruent)")
    ax[0].tick_params(axis="x", rotation=30)
    x = np.arange(len(df)); wd = .35
    ax[1].bar(x - wd / 2, df.recon_target_acc, wd, label="audio (target) acc", color="seagreen")
    ax[1].bar(x + wd / 2, df.recon_proxy_acc, wd, label="text (proxy) acc", color="goldenrod")
    ax[1].set_xticks(x); ax[1].set_xticklabels(df.encoder, rotation=30)
    ax[1].set(ylabel="accuracy on incongruent clips", title="Audio vs text accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step8_EMIS_text_bias.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 8 (EMIS text-bias) ==========")
    print(json.dumps(metrics, indent=2))
    print("VERDICT:", "Whisper resists the lexical shortcut (lower text-bias than SSL)"
          if metrics["whisper_resists_shortcut"] else "Whisper does NOT clearly resist at this layer/setup")


if __name__ == "__main__":
    main()
