"""
Master plotting script — regenerate every report figure with one unified theme.

Run from project root:
    python figures/_regenerate_all.py

Outputs are written into figures/ (overwriting older PNGs).
All numbers come from CSVs in this repo or from Drive headline values
that have been hard-coded after manual verification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------------------------
# Unified theme
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
plt.rcParams.update({
    "axes.titleweight": "bold",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "axes.labelweight": "regular",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#444444",
    "axes.linewidth":    0.9,
    "grid.color":        "#DDDDDD",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "legend.fontsize":   9,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#CCCCCC",
    "figure.dpi":        180,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.facecolor":    "white",
})

# Encoder palette — fixed so identity is consistent across plots.
ENCODER_ORDER = ["Whisper", "HuBERT", "WavLM", "w2v-BERT", "wav2vec 2.0", "MERT"]
ENCODER_COLORS = {
    "Whisper":     "#D62728",   # crimson — supervised, distinct
    "HuBERT":      "#2E5EAA",   # deep blue
    "WavLM":       "#1F8A70",   # teal
    "w2v-BERT":    "#7B3FA0",   # purple
    "wav2vec 2.0": "#E27D2C",   # warm orange
    "MERT":        "#7F5539",   # brown
}
ENCODER_MARKERS = {
    "Whisper":     "s",
    "HuBERT":      "o",
    "WavLM":       "D",
    "w2v-BERT":    "^",
    "wav2vec 2.0": "v",
    "MERT":        "P",
}

# Heatmap palettes
CMAP_DIVERGING = "vlag"        # blue ↔ red, white at 0
CMAP_SEQUENTIAL = "rocket_r"   # warm sequential
CMAP_ACC = sns.color_palette("crest", as_cmap=True)
CMAP_ATTN = sns.color_palette("rocket_r", as_cmap=True)
CMAP_CKA = sns.color_palette("mako", as_cmap=True)

# Map encoder names used in CSVs ("wav2vec2") to display names ("wav2vec 2.0")
def to_display(name):
    return {"wav2vec2": "wav2vec 2.0"}.get(name, name)
def to_csv(name):
    return {"wav2vec 2.0": "wav2vec2"}.get(name, name)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "figures")
EXP20 = "/sessions/elegant-stoic-wozniak/mnt/EXP20"  # workspace path
# When run on the user's machine, EXP20 mount won't exist. Fall back.
if not os.path.isdir(EXP20):
    EXP20 = os.path.join(REPO, "..", "EXP20")

os.makedirs(FIG, exist_ok=True)


def style_axes(ax, *, ygrid=True, xgrid=False, ymin=None, ymax=None):
    if ygrid:
        ax.yaxis.grid(True)
    else:
        ax.yaxis.grid(False)
    ax.xaxis.grid(xgrid)
    ax.set_axisbelow(True)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)


# ---------------------------------------------------------------------------
# 1. ESC-50 noise robustness — best-layer accuracy vs SNR
# ---------------------------------------------------------------------------
def fig_noise_robustness():
    snr = ["Clean", "20 dB", "10 dB", "5 dB", "0 dB"]
    data = {  # source: Drive TASK 1/2/8 noise_analysis_report.txt
        "Whisper":     [0.864, 0.833, 0.812, 0.794, 0.768],
        "HuBERT":      [0.855, 0.807, 0.767, 0.722, 0.655],
        "WavLM":       [0.863, 0.812, 0.754, 0.685, 0.616],
        "w2v-BERT":    [0.841, 0.792, 0.760, 0.732, 0.695],
        "wav2vec 2.0": [0.834, 0.767, 0.713, 0.660, 0.579],
        "MERT":        [0.816, 0.706, 0.652, 0.614, 0.574],
    }

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for enc in ENCODER_ORDER:
        ax.plot(snr, data[enc],
                marker=ENCODER_MARKERS[enc],
                color=ENCODER_COLORS[enc],
                linewidth=2.0, markersize=7,
                label=enc, markeredgecolor="white", markeredgewidth=0.7)
    ax.set_xlabel("Signal-to-noise ratio (lower = noisier)")
    ax.set_ylabel("Best-layer accuracy")
    style_axes(ax, ymin=0.55, ymax=0.90)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "noise_robustness.png"))
    plt.close(fig)
    print("  noise_robustness.png")


# ---------------------------------------------------------------------------
# 2. V/A/D RMSE — two-panel: per encoder + per target
# ---------------------------------------------------------------------------
def fig_vad_rmse():
    # source: Drive TASK 10 statistical_analysis.txt
    mean_rmse = {
        "Whisper": 0.7536, "WavLM": 0.7606, "HuBERT": 0.7633,
        "MERT": 0.7751, "w2v-BERT": 0.7788, "wav2vec 2.0": 0.7885,
    }
    target_rmse = [("Dominance", 0.6993), ("Arousal", 0.7299), ("Valence", 0.8807)]

    # wider aspect for two-column figure*
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.0),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # Panel (a): mean RMSE per encoder (sorted ascending = best on top)
    enc_sorted = sorted(mean_rmse.keys(), key=lambda e: mean_rmse[e])
    ax = axes[0]
    bars = ax.barh(enc_sorted, [mean_rmse[e] for e in enc_sorted],
                   color=[ENCODER_COLORS[e] for e in enc_sorted],
                   edgecolor="white", linewidth=0.8)
    ax.set_xlabel("(a) Mean RMSE per encoder  (lower = better)")
    ax.set_xlim(0.74, 0.80)
    for b, v in zip(bars, [mean_rmse[e] for e in enc_sorted]):
        ax.text(v + 0.0008, b.get_y() + b.get_height()/2,
                f"{v:.3f}", va="center", fontsize=8.5, color="#333333")
    style_axes(ax, ygrid=False, xgrid=True)

    # Panel (b): per-target RMSE
    ax = axes[1]
    target_palette = ["#1F8A70", "#E27D2C", "#D62728"]
    bars = ax.bar([t[0] for t in target_rmse], [t[1] for t in target_rmse],
                  color=target_palette, edgecolor="white", linewidth=0.8, width=0.55)
    ax.set_ylabel("Mean RMSE across encoders")
    ax.set_xlabel("(b) Per-target RMSE averaged across encoders")
    ax.set_ylim(0.65, 0.94)
    for b, (_, v) in zip(bars, target_rmse):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=10, color="#333333")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vad_rmse.png"))
    plt.close(fig)
    print("  vad_rmse.png")


# ---------------------------------------------------------------------------
# 3. V/A/D noise resilience
# ---------------------------------------------------------------------------
def fig_vad_noise_resilience():
    # source: Drive TASK 11 statistical_analysis.txt
    delta = {
        "Whisper": 0.0279, "w2v-BERT": 0.0421, "MERT": 0.0603,
        "HuBERT": 0.0612, "WavLM": 0.0619, "wav2vec 2.0": 0.0710,
    }
    enc_sorted = sorted(delta.keys(), key=lambda e: delta[e])
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    bars = ax.barh(enc_sorted, [delta[e] for e in enc_sorted],
                   color=[ENCODER_COLORS[e] for e in enc_sorted],
                   edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Δ RMSE (0 dB − Clean)   lower = more resilient")
    for b, e in zip(bars, enc_sorted):
        v = delta[e]
        ax.text(v + 0.001, b.get_y() + b.get_height()/2,
                f"+{v:.3f}", va="center", fontsize=9, color="#333333")
    style_axes(ax, ygrid=False, xgrid=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vad_noise_resilience.png"))
    plt.close(fig)
    print("  vad_noise_resilience.png")


# ---------------------------------------------------------------------------
# Helpers for SpeechCraft (Task 9) figures
# ---------------------------------------------------------------------------
def _read_pooled(enc):
    csv_name = to_csv(enc)
    p = os.path.join(EXP20, "csv", f"results_{csv_name}_pooled.csv")
    return pd.read_csv(p)

def _read_lolo(enc):
    csv_name = to_csv(enc)
    p = os.path.join(EXP20, "csv", f"results_{csv_name}_LOLO.csv")
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# 4. SpeechCraft pooled vs LOLO (grouped bar)
# ---------------------------------------------------------------------------
def fig_speechcraft_pooled_vs_lolo():
    pooled = {}
    lolo_mean = {}
    for enc in ENCODER_ORDER:
        df_p = _read_pooled(enc)
        pooled[enc] = df_p["Accuracy"].max()
        df_l = _read_lolo(enc)
        lolo_mean[enc] = df_l["Accuracy"].mean()

    x = np.arange(len(ENCODER_ORDER))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    b1 = ax.bar(x - w/2, [pooled[e] for e in ENCODER_ORDER], w,
                color="#2E5EAA", label="Pooled best layer",
                edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w/2, [lolo_mean[e] for e in ENCODER_ORDER], w,
                color="#D62728", label="LOLO mean (held-out lang)",
                edgecolor="white", linewidth=0.8)
    ax.axhline(0.25, color="#888888", linestyle=":", linewidth=1.0)
    ax.text(len(ENCODER_ORDER)-0.5, 0.27, "4-cls chance", fontsize=8.5, color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels(ENCODER_ORDER, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    for bars in [b1, b2]:
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + 0.014,
                    f"{v:.2f}", ha="center", fontsize=7.5, color="#333333")
    style_axes(ax, ymin=0, ymax=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "speechcraft_pooled_vs_lolo.png"))
    plt.close(fig)
    print("  speechcraft_pooled_vs_lolo.png")


# ---------------------------------------------------------------------------
# 5. SpeechCraft pooled grid (per-encoder layer-wise curves)
# ---------------------------------------------------------------------------
def fig_speechcraft_pooled_grid():
    # wider aspect for two-column figure*
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 4.6), sharex=True, sharey=True)
    for ax, enc in zip(axes.flat, ENCODER_ORDER):
        df = _read_pooled(enc).copy()
        df["L"] = df["Layer_Num"]
        ax.plot(df["L"], df["Accuracy"],
                color=ENCODER_COLORS[enc], linewidth=2.0,
                marker=ENCODER_MARKERS[enc], markersize=4,
                markeredgecolor="white", markeredgewidth=0.5)
        # Mark best layer
        best = df.loc[df["Accuracy"].idxmax()]
        ax.axvline(best["L"], color=ENCODER_COLORS[enc], linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_title(f"{enc}   peak L{int(best['L'])} = {best['Accuracy']:.3f}",
                     fontsize=10, fontweight="regular", color="#333333", pad=4)
        ax.set_ylim(0.80, 1.0)
        style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Layer index")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "speechcraft_pooled_grid.png"))
    plt.close(fig)
    print("  speechcraft_pooled_grid.png")


# ---------------------------------------------------------------------------
# 6. SpeechCraft LOLO heatmap (encoders × held-out language)
# ---------------------------------------------------------------------------
def fig_speechcraft_lolo_heatmap():
    langs = ["German", "Greek", "English", "Spanish"]
    mat = np.zeros((len(ENCODER_ORDER), len(langs)))
    for i, enc in enumerate(ENCODER_ORDER):
        df = _read_lolo(enc)
        for j, lang in enumerate(langs):
            sub = df[df["split"] == f"LOLO_{lang}"]
            mat[i, j] = sub["Accuracy"].max()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sns.heatmap(mat, annot=True, fmt=".3f",
                xticklabels=langs, yticklabels=ENCODER_ORDER,
                cmap=CMAP_ACC, vmin=0.30, vmax=1.00,
                cbar_kws={"label": "Best-layer accuracy"},
                linewidths=0.6, linecolor="white",
                annot_kws={"fontsize": 9, "color": "#222222"},
                ax=ax)
    ax.set_xlabel("Held-out language")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "speechcraft_lolo_heatmap.png"))
    plt.close(fig)
    print("  speechcraft_lolo_heatmap.png")


# ---------------------------------------------------------------------------
# 7. SpeechCraft pooled-vs-LOLO gap (bar)
# ---------------------------------------------------------------------------
def fig_speechcraft_gap():
    gap = {}
    for enc in ENCODER_ORDER:
        pooled = _read_pooled(enc)["Accuracy"].max()
        lolo = _read_lolo(enc)["Accuracy"].mean()
        gap[enc] = pooled - lolo
    enc_sorted = sorted(gap.keys(), key=lambda e: gap[e])
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    bars = ax.barh(enc_sorted, [gap[e] for e in enc_sorted],
                   color=[ENCODER_COLORS[e] for e in enc_sorted],
                   edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Pooled − LOLO mean accuracy   (smaller = better transfer)")
    for b, e in zip(bars, enc_sorted):
        v = gap[e]
        ax.text(v + 0.005, b.get_y() + b.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9, color="#333333")
    style_axes(ax, ygrid=False, xgrid=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "speechcraft_pooled_vs_lolo_gap.png"))
    plt.close(fig)
    print("  speechcraft_pooled_vs_lolo_gap.png")


# ---------------------------------------------------------------------------
# 8. BERT CKA heatmap (encoders × layer, averaged across datasets)
# ---------------------------------------------------------------------------
def fig_bert_cka_heatmap():
    p = os.path.join(REPO, "results/bert_alignment/alignment_per_layer_wide.csv")
    df = pd.read_csv(p)
    # Keep CKA rows only, average across datasets
    df = df[df["Metric"] == "cka"]
    layer_cols = [c for c in df.columns if c.startswith("Layer_")]
    # CSV uses Layer_CNN, Layer_1, ..., Layer_24 — re-order numerically
    def lkey(c):
        s = c.replace("Layer_", "")
        return -1 if s == "CNN" else int(s)
    layer_cols = sorted(layer_cols, key=lkey)
    grouped = df.groupby("Model")[layer_cols].mean()
    grouped.index = [to_display(m) for m in grouped.index]
    grouped = grouped.reindex(ENCODER_ORDER)
    xlabels = [c.replace("Layer_", "") for c in layer_cols]

    # wider aspect for two-column figure*
    fig, ax = plt.subplots(figsize=(12.0, 2.8))
    sns.heatmap(grouped.values, ax=ax,
                xticklabels=xlabels, yticklabels=ENCODER_ORDER,
                cmap=CMAP_CKA, vmin=0, vmax=grouped.values.max(),
                cbar_kws={"label": "Linear CKA  (higher = more BERT-like)"},
                linewidths=0.0)
    ax.set_xlabel("Layer index (CNN front-end on the left)")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=7.5)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bert_cka_heatmap.png"))
    plt.close(fig)
    print("  bert_cka_heatmap.png")


# ---------------------------------------------------------------------------
# 9. Layer mixer attention heatmap (encoders × layer, averaged across datasets)
# ---------------------------------------------------------------------------
def fig_mixer_attention_heatmap():
    p = os.path.join(REPO, "results/layer_mixer/attention_weights_all.csv")
    df = pd.read_csv(p)
    layer_cols = [c for c in df.columns if c.startswith("Layer_")]
    def lkey(c):
        s = c.replace("Layer_", "")
        return -1 if s == "CNN" else int(s)
    layer_cols = sorted(layer_cols, key=lkey)
    grouped = df.groupby("Model")[layer_cols].mean()
    grouped.index = [to_display(m) for m in grouped.index]
    grouped = grouped.reindex(ENCODER_ORDER)
    xlabels = [c.replace("Layer_", "") for c in layer_cols]

    # wider aspect for two-column figure*
    fig, ax = plt.subplots(figsize=(12.0, 2.8))
    sns.heatmap(grouped.values, ax=ax,
                xticklabels=xlabels, yticklabels=ENCODER_ORDER,
                cmap=CMAP_ATTN, vmin=0, vmax=1.0,
                cbar_kws={"label": "Mixer attention weight"},
                linewidths=0.0)
    ax.set_xlabel("Layer index (CNN front-end on the left)")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=7.5)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mixer_attention_heatmap.png"))
    plt.close(fig)
    print("  mixer_attention_heatmap.png")


# ---------------------------------------------------------------------------
# 10. LoRA best-layer delta heatmap (6 encoders × 5 datasets)
# ---------------------------------------------------------------------------
def fig_lora_delta_heatmap():
    p = os.path.join(REPO, "results/lora/frozen_vs_tuned_comparison.csv")
    df = pd.read_csv(p)
    # Build encoder × dataset matrix of best_acc_delta
    df["Model_disp"] = df["Model"].apply(to_display)
    pivot = df.pivot(index="Model_disp", columns="Dataset", values="Best_Acc_Delta")
    pivot = pivot.reindex(ENCODER_ORDER)
    pivot = pivot[["RAVDESS", "EmoDB", "SAVEE", "AESDD", "MESD"]]

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    sns.heatmap(pivot.values, ax=ax, annot=True, fmt="+.3f",
                xticklabels=pivot.columns, yticklabels=pivot.index,
                cmap=CMAP_DIVERGING, center=0,
                vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "Δ best-layer accuracy (LoRA − frozen)"},
                linewidths=0.6, linecolor="white",
                annot_kws={"fontsize": 9, "color": "#222222"})
    ax.set_xlabel("Dataset")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lora_delta_heatmap.png"))
    plt.close(fig)
    print("  lora_delta_heatmap.png")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Regenerating figures into:", FIG)
    fig_noise_robustness()
    fig_vad_rmse()
    fig_vad_noise_resilience()
    fig_speechcraft_pooled_vs_lolo()
    fig_speechcraft_pooled_grid()
    fig_speechcraft_lolo_heatmap()
    fig_speechcraft_gap()
    fig_bert_cka_heatmap()
    fig_mixer_attention_heatmap()
    fig_lora_delta_heatmap()
    print("Done.")
