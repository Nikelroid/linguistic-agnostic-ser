#!/usr/bin/env python
"""Step 8 investigation — is the EMIS Whisper text-bias layer- or TTS-specific?

The published result (Whisper L24 text-bias -0.19, "resists the lexical shortcut")
was not reproduced by run_step8.py on the Zenodo EMIS release (Whisper +0.88).
Before writing the paper we check, on mean-pooled RAW activations (matching the
published probe), (a) text-bias across ALL 25 layers per encoder, and (b) text-bias
split by TTS engine (COSY/F5TTS/STYLE) and by explicit/implicit text — to see
whether any layer or weaker TTS recovers Whisper's resistance.

Protocol: train an emotion probe on CONGRUENT clips, test on INCONGRUENT;
text_bias = text(proxy) acc - audio(target) acc.

    python notebooks/sae/run_step8_investigate.py
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

POOLED_DIR = "/scratch1/kelidari/ser-experiments/SAE_frames"
ART_DIR = os.path.join(_REPO, "results", "SAE")
ENCODERS = ["HuBERT", "wav2vec2", "WavLM", "Whisper", "MERT", "w2v-BERT"]
SSL = ["HuBERT", "wav2vec2", "WavLM", "w2v-BERT", "MERT"]


def text_bias(X, audio, text, cong, incong, n_rep=3, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if cong.sum() < 8 or incong.sum() < 8:
        return np.nan, np.nan, np.nan
    Xc, yc = X[cong], audio[cong]; Xi, ai, ti = X[incong], audio[incong], text[incong]
    rng = np.random.default_rng(seed); tg, px = [], []
    for _ in range(n_rep):
        idx = rng.choice(len(Xc), size=len(Xc), replace=True)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        clf.fit(Xc[idx], yc[idx]); pred = clf.predict(Xi)
        tg.append((pred == ai).mean()); px.append((pred == ti).mean())
    return float(np.mean(tg)), float(np.mean(px)), float(np.mean(px) - np.mean(tg))


def main():
    sweep_rows, tts_rows = [], []
    for enc in ENCODERS:
        p = f"{POOLED_DIR}/pooled_{enc}_EMIS.npy"
        if not os.path.exists(p):
            print(f"  ! {enc}: no pooled file, skipping"); continue
        pooled = np.load(p)                              # (N, L, H)
        m = np.load(f"{POOLED_DIR}/utterances_pooled_{enc}_EMIS.npz", allow_pickle=True)
        audio, text, tts, expl = m["label"], m["text_label"], m["tts"], m["explicit"]
        cong = audio == text; incong = ~cong
        L = pooled.shape[1]
        biases = []
        for l in range(L):
            _, _, b = text_bias(pooled[:, l, :], audio, text, cong, incong)
            sweep_rows.append({"encoder": enc, "layer": l, "text_bias": b})
            biases.append(b)
        biases = np.array(biases)
        best = int(np.nanargmin(biases))                 # most audio-grounded layer
        print(f"  {enc:9s}: min text-bias {np.nanmin(biases):+.3f} @ L{best}; "
              f"L24 {biases[min(24, L-1)]:+.3f}; max {np.nanmax(biases):+.3f}")
        # TTS / explicit breakdown at the most-resistant layer
        Xb = pooled[:, best, :]
        for tn in sorted(set(tts.tolist())):
            mtts = tts == tn
            _, _, b = text_bias(Xb, audio, text, cong & mtts, incong & mtts)
            tts_rows.append({"encoder": enc, "layer": best, "split": f"TTS={tn}", "text_bias": b})
        for name, mask in [("explicit", expl.astype(bool)), ("implicit", ~expl.astype(bool))]:
            _, _, b = text_bias(Xb, audio, text, cong & mask, incong & mask)
            tts_rows.append({"encoder": enc, "layer": best, "split": name, "text_bias": b})

    sweep = pd.DataFrame(sweep_rows); sweep.to_csv(f"{ART_DIR}/step8_layer_sweep.csv", index=False)
    pd.DataFrame(tts_rows).to_csv(f"{ART_DIR}/step8_tts_breakdown.csv", index=False)

    # summary: per-encoder min text-bias and the layer; does Whisper ever resist (<=0)?
    summ = {}
    for enc in sweep.encoder.unique():
        s = sweep[sweep.encoder == enc]
        summ[enc] = {"min_text_bias": float(s.text_bias.min()),
                     "min_layer": int(s.loc[s.text_bias.idxmin(), "layer"]),
                     "ever_resists(<=0)": bool((s.text_bias <= 0).any())}
    metrics = {"per_encoder": summ,
               "whisper_ever_resists": summ.get("Whisper", {}).get("ever_resists(<=0)"),
               "encoders_that_ever_resist": [e for e, v in summ.items() if v["ever_resists(<=0)"]]}
    json.dump(metrics, open(f"{ART_DIR}/step8_investigation_metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for enc in sweep.encoder.unique():
        s = sweep[sweep.encoder == enc].sort_values("layer")
        ax[0].plot(s.layer, s.text_bias, "-o", ms=3, lw=2 if enc == "Whisper" else 1,
                   color="crimson" if enc == "Whisper" else None, label=enc)
    ax[0].axhline(0, color="k", lw=.8, ls="--")
    ax[0].set(xlabel="layer", ylabel="text-bias (text - audio acc)",
              title="EMIS text-bias across depth (raw pooled activations)")
    ax[0].legend(fontsize=8, ncol=2)
    tb = pd.DataFrame(tts_rows)
    if len(tb):
        piv = tb[tb.split.str.startswith("TTS")].pivot(index="encoder", columns="split", values="text_bias")
        piv.plot.bar(ax=ax[1]); ax[1].axhline(0, color="k", lw=.8)
        ax[1].set(ylabel="text-bias", title="Text-bias by TTS engine (at min-bias layer)")
        ax[1].tick_params(axis="x", rotation=30); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{ART_DIR}/step8_investigation.png", dpi=120)
    plt.close(fig)

    print("\n========== STEP 8 INVESTIGATION ==========")
    print(json.dumps(metrics, indent=2))
    print("Whisper ever resists (text-bias<=0 at some layer):", metrics["whisper_ever_resists"])


if __name__ == "__main__":
    main()
