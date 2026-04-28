"""Task 7 summary plots — Part A bias-swing + Part B 6-model + cross-task headline.

Reads per-model CSVs from results/TASK7/ (the authoritative source — see
results/TASK7/README.md) and renders three presentation-grade figures:

1. task7B_disagreement_6models.png
   Part B IEMOCAP disagreement, 6 panels (one per model). Each panel shows
   high-agree, low-agree, and full-set accuracy curves over 25 layers.
   Source: agreement_probing_{Model}_IEMOCAP.csv x 6.

2. task7_cross_diagnostic.png
   Two-panel headline: left = Part A peak text_bias per (model, category),
   4 models x 3 categories. Right = Part B mean agreement gap per model,
   6 models. Lands the "two diagnostics, both find a problem" story.

3. task7A_bias_swing.png
   Part A only, 4 models x 3 categories grouped bar chart. Internal
   falsification: same model, different test slice -> bias flips sign.
   Source: probing_{Model}_{all,explicit,implicit}.csv x 12.

Sign convention (from scripts/task7_partA_EMIS.py:285):
    text_bias = proxy_acc - target_acc
    positive = probe took the text shortcut
    negative = probe is grounded in the audio

Run from repo root:
    python scripts/task7_summary_plots.py

Outputs:
    results/TASK7/plots_summary/task7B_disagreement_6models.png
    results/TASK7/plots_summary/task7_cross_diagnostic.png
    results/TASK7/plots_summary/task7A_bias_swing.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARTA_MODELS = ["HuBERT", "wav2vec2", "WavLM", "Whisper"]
PARTB_MODELS = ["HuBERT", "wav2vec2", "WavLM", "Whisper", "MERT", "w2v-BERT"]
CATEGORIES = ["all", "explicit", "implicit"]

# Visual conventions
COLOR_HIGH = "#2E7D32"   # green = clean labels
COLOR_LOW = "#C62828"    # red = noisy labels
COLOR_ALL = "#546E7A"    # gray = full set
COLOR_BIAS = {"all": "#5C6BC0", "explicit": "#E53935", "implicit": "#43A047"}


def load_partA(csv_dir: Path, model: str, category: str) -> pd.DataFrame | None:
    path = csv_dir / f"probing_{model}_{category}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_partB(csv_dir: Path, model: str) -> pd.DataFrame | None:
    path = csv_dir / f"agreement_probing_{model}_IEMOCAP.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_partB_6models(csv_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), sharey=True)
    axes = axes.flatten()
    for ax, model in zip(axes, PARTB_MODELS):
        df = load_partB(csv_dir, model)
        if df is None:
            ax.set_visible(False)
            continue
        layers = df["Layer_Num"].to_numpy()
        ax.plot(layers, df["HighAgree_Acc"], "o-", color=COLOR_HIGH, linewidth=2,
                markersize=3, label="High-agree (n=1308)")
        ax.plot(layers, df["LowAgree_Acc"], "s-", color=COLOR_LOW, linewidth=2,
                markersize=3, label="Low-agree (n=4152)")
        ax.plot(layers, df["All_Acc"], "^--", color=COLOR_ALL, linewidth=1.5,
                markersize=3, alpha=0.6, label="Full set")
        gap = df["HighAgree_Acc"] - df["LowAgree_Acc"]
        best_layer = int(layers[df["HighAgree_Acc"].idxmax()])
        peak_high = df["HighAgree_Acc"].max()
        peak_low = df.loc[df["HighAgree_Acc"].idxmax(), "LowAgree_Acc"]
        peak_gap = peak_high - peak_low
        ax.axvline(best_layer, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{model}\nL{best_layer}: high={peak_high:.2f}, low={peak_low:.2f}, gap=+{peak_gap:.2f}",
                     fontsize=10)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.45, 0.90)
        ax.axhline(0.309, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    axes[0].set_ylabel("Accuracy")
    axes[3].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Task 7B — IEMOCAP annotator disagreement: probes do better on clips humans agreed on\n"
        "(green > red across all layers, all 6 models. avg gap +0.15-0.19)",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cross_diagnostic(csv_dir: Path, out_path: Path) -> None:
    partA_peaks = {}  # model -> {category: peak_bias}
    for model in PARTA_MODELS:
        partA_peaks[model] = {}
        for cat in CATEGORIES:
            df = load_partA(csv_dir, model, cat)
            if df is None:
                continue
            idx = df["text_bias_mean"].idxmax()
            partA_peaks[model][cat] = float(df.loc[idx, "text_bias_mean"])

    partB_gaps = {}  # model -> peak-layer gap (gap at the best HighAgree layer)
    partB_best_layer = {}
    for model in PARTB_MODELS:
        df = load_partB(csv_dir, model)
        if df is None:
            continue
        idx = df["HighAgree_Acc"].idxmax()
        partB_gaps[model] = float(df.loc[idx, "HighAgree_Acc"] - df.loc[idx, "LowAgree_Acc"])
        partB_best_layer[model] = int(df.loc[idx, "Layer_Num"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    x = np.arange(len(PARTA_MODELS))
    width = 0.27
    for i, cat in enumerate(CATEGORIES):
        values = [partA_peaks[m].get(cat, np.nan) for m in PARTA_MODELS]
        offset = (i - 1) * width
        bars = axL.bar(x + offset, values, width, color=COLOR_BIAS[cat], label=cat,
                       edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            if np.isnan(v):
                continue
            va = "bottom" if v >= 0 else "top"
            offset_y = 0.02 if v >= 0 else -0.02
            axL.text(bar.get_x() + bar.get_width() / 2, v + offset_y, f"{v:+.2f}",
                     ha="center", va=va, fontsize=8)

    axL.axhline(0, color="black", linewidth=0.8)
    axL.set_xticks(x)
    axL.set_xticklabels(PARTA_MODELS)
    axL.set_ylabel("Peak text_bias  (proxy_acc - target_acc)")
    axL.set_title("Task 7A — text bias on EMIS\n(positive = probe took the text shortcut)")
    axL.set_ylim(-0.65, 0.95)
    axL.legend(title="Test slice", loc="upper right", fontsize=9)
    axL.grid(True, axis="y", alpha=0.25)

    models_b = list(partB_gaps.keys())
    gaps_b = [partB_gaps[m] for m in models_b]
    bars = axR.bar(range(len(models_b)), gaps_b, color="#7E57C2",
                   edgecolor="white", linewidth=0.5)
    for bar, v, m in zip(bars, gaps_b, models_b):
        layer = partB_best_layer.get(m)
        label = f"+{v:.3f}\nL{layer}" if layer is not None else f"+{v:.3f}"
        axR.text(bar.get_x() + bar.get_width() / 2, v + 0.005, label,
                 ha="center", va="bottom", fontsize=9)
    axR.set_xticks(range(len(models_b)))
    axR.set_xticklabels(models_b, rotation=15, ha="right")
    axR.set_ylabel("Peak-layer (HighAgree - LowAgree) accuracy")
    axR.set_title("Task 7B — IEMOCAP annotator-agreement gap\n(at the model's best HighAgree layer; positive = probe better on clips humans agreed on)")
    axR.set_ylim(0, 0.24)
    axR.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Two diagnostics of probe trustworthiness: text bias (left, 4 models) and "
        "annotator-disagreement gap (right, 6 models)",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_partA_bias_swing(csv_dir: Path, out_path: Path) -> None:
    rows = []
    for model in PARTA_MODELS:
        for cat in CATEGORIES:
            df = load_partA(csv_dir, model, cat)
            if df is None:
                continue
            idx = df["text_bias_mean"].idxmax()
            rows.append({
                "model": model,
                "category": cat,
                "peak_layer": int(df.loc[idx, "layer"]),
                "peak_bias": float(df.loc[idx, "text_bias_mean"]),
                "peak_target_acc": float(df.loc[idx, "target_acc_mean"]),
                "peak_proxy_acc": float(df.loc[idx, "proxy_acc_mean"]),
            })
    bias_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(PARTA_MODELS))
    width = 0.27
    for i, cat in enumerate(CATEGORIES):
        sub = bias_df[bias_df["category"] == cat].set_index("model")
        values = [sub.loc[m, "peak_bias"] if m in sub.index else np.nan for m in PARTA_MODELS]
        layers = [int(sub.loc[m, "peak_layer"]) if m in sub.index else None for m in PARTA_MODELS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, color=COLOR_BIAS[cat], label=cat,
                      edgecolor="white", linewidth=0.5)
        for bar, v, layer in zip(bars, values, layers):
            if np.isnan(v) or layer is None:
                continue
            va = "bottom" if v >= 0 else "top"
            offset_y = 0.02 if v >= 0 else -0.02
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset_y,
                    f"{v:+.2f}\nL{layer}", ha="center", va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(PARTA_MODELS)
    ax.set_ylabel("Peak text_bias  (proxy_acc - target_acc)")
    ax.set_title(
        "Task 7A — same model, different test slice -> bias flips sign\n"
        "(internal falsification: if probe used acoustics not text, all three bars would match)"
    )
    ax.set_ylim(-0.65, 0.95)
    ax.legend(title="Incongruent test slice", loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=Path, default=Path("results/TASK7"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/TASK7/plots_summary"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_partB_6models(args.csv_dir, args.out_dir / "task7B_disagreement_6models.png")
    plot_cross_diagnostic(args.csv_dir, args.out_dir / "task7_cross_diagnostic.png")
    plot_partA_bias_swing(args.csv_dir, args.out_dir / "task7A_bias_swing.png")


if __name__ == "__main__":
    main()
