"""Build publication-ready figures from EXP20 / Speechcraft CSVs.

Consumes:
  results/EXP20/csv/results_<Model>_pooled.csv
  results/EXP20/csv/results_<Model>_LOLO.csv

Writes:
  results/EXP20/plots/probing_<Model>_pooled.png            per-model pooled curve
  results/EXP20/plots/probing_<Model>_LOLO.png              per-model 4-language curves
  results/EXP20/plots/speechcraft_pooled_grid.png           all models pooled
  results/EXP20/plots/speechcraft_LOLO_grid.png             6x4 model x language matrix
  results/EXP20/plots/speechcraft_pooled_vs_lolo_gap.png    gap heatmap
  results/EXP20/plots/speechcraft_lolo_heatmap.png          best-layer per (model, lang) accuracy
  results/EXP20/plots/speechcraft_summary_bar.png           best accuracy per model x condition
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["wav2vec2", "HuBERT", "Whisper", "WavLM", "MERT", "w2v-BERT"]
LANG_ORDER = ["English", "German", "Greek", "Spanish"]
LANG_COLORS = {
    "English": "#1f77b4",
    "German":  "#d62728",
    "Greek":   "#2ca02c",
    "Spanish": "#ff7f0e",
}
MODEL_COLORS = {
    "wav2vec2": "#1f77b4",
    "HuBERT":   "#ff7f0e",
    "Whisper":  "#2ca02c",
    "WavLM":    "#d62728",
    "MERT":     "#9467bd",
    "w2v-BERT": "#8c564b",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def discover(csv_dir):
    """Return {model: {'pooled': df, 'lolo': df}}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(csv_dir, "results_*_pooled.csv"))):
        model = os.path.basename(f)[len("results_"):-len("_pooled.csv")]
        out.setdefault(model, {})["pooled"] = pd.read_csv(f)
    for f in sorted(glob.glob(os.path.join(csv_dir, "results_*_LOLO.csv"))):
        model = os.path.basename(f)[len("results_"):-len("_LOLO.csv")]
        out.setdefault(model, {})["lolo"] = pd.read_csv(f)
    return out


def plot_pooled_per_model(model, df, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = df["Layer_Num"].to_numpy()
    ax.plot(x, df["Accuracy"], marker="o", lw=2, color="#1f77b4", label="Accuracy")
    if "Std_Accuracy" in df.columns:
        lo = df["Accuracy"] - df["Std_Accuracy"]
        hi = df["Accuracy"] + df["Std_Accuracy"]
        ax.fill_between(x, lo, hi, alpha=0.15, color="#1f77b4")
    ax.plot(x, df["Weighted_F1"], marker="s", lw=2, color="#FF5722", label="Weighted F1")

    best = df.loc[df["Accuracy"].idxmax()]
    ax.annotate(f"best layer {int(best['Layer_Num'])}\n{best['Accuracy']:.3f}",
                xy=(best["Layer_Num"], best["Accuracy"]),
                xytext=(10, -25), textcoords="offset points",
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_title(f"Speechcraft pooled multilingual probe — {model}")
    ax.set_xlabel("Transformer layer (0 = CNN front-end)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lolo_per_model(model, df, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for lang in LANG_ORDER:
        sub = df[df["test_lang"] == lang].sort_values("Layer_Num")
        if sub.empty:
            continue
        ax.plot(sub["Layer_Num"], sub["Accuracy"],
                marker="o", lw=1.8, color=LANG_COLORS[lang],
                label=f"test = {lang}")
    ax.axhline(0.25, color="gray", lw=0.8, ls=":", label="chance (4 cls)")
    ax.set_title(f"Cross-lingual leave-one-language-out — {model}")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Accuracy on held-out language")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_grid(by_model, out_path):
    models = [m for m in MODEL_ORDER if m in by_model and "pooled" in by_model[m]]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for m in models:
        df = by_model[m]["pooled"]
        ax.plot(df["Layer_Num"], df["Accuracy"], marker="o", lw=2,
                color=MODEL_COLORS.get(m, None), label=m)
    ax.axhline(0.25, color="gray", lw=0.8, ls=":", label="chance")
    ax.set_title("Speechcraft pooled multilingual probe — all models")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lolo_grid(by_model, out_path):
    models = [m for m in MODEL_ORDER if m in by_model and "lolo" in by_model[m]]
    if not models:
        return
    nrows, ncols = len(models), 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = np.array([axes])
    for i, m in enumerate(models):
        df = by_model[m]["lolo"]
        for j, lang in enumerate(LANG_ORDER):
            ax = axes[i, j]
            sub = df[df["test_lang"] == lang].sort_values("Layer_Num")
            if not sub.empty:
                ax.plot(sub["Layer_Num"], sub["Accuracy"],
                        marker="o", lw=1.5, color=LANG_COLORS[lang])
                best = sub.loc[sub["Accuracy"].idxmax()]
                ax.scatter([best["Layer_Num"]], [best["Accuracy"]],
                           s=70, edgecolor="black", facecolor="white", zorder=5)
            ax.axhline(0.25, color="gray", lw=0.6, ls=":")
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(f"test = {lang}")
            if j == 0:
                ax.set_ylabel(m, fontweight="bold")
            if i == nrows - 1:
                ax.set_xlabel("layer")
    fig.suptitle("Cross-lingual probe accuracy: rows = model, cols = held-out language",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lolo_heatmap(by_model, out_path):
    models = [m for m in MODEL_ORDER if m in by_model and "lolo" in by_model[m]]
    if not models:
        return
    grid = np.full((len(models), len(LANG_ORDER)), np.nan)
    for i, m in enumerate(models):
        df = by_model[m]["lolo"]
        for j, lang in enumerate(LANG_ORDER):
            sub = df[df["test_lang"] == lang]
            if not sub.empty:
                grid[i, j] = sub["Accuracy"].max()

    fig, ax = plt.subplots(figsize=(0.9 * len(LANG_ORDER) + 3, 0.6 * len(models) + 2))
    im = ax.imshow(grid, cmap="viridis", vmin=0.2, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(LANG_ORDER))); ax.set_xticklabels(LANG_ORDER)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    ax.set_title("Best-layer cross-lingual accuracy (test = held-out language)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_vs_lolo_gap(by_model, out_path):
    """Heatmap: pooled best - LOLO mean best across languages."""
    models = [m for m in MODEL_ORDER if m in by_model
              and "pooled" in by_model[m] and "lolo" in by_model[m]]
    if not models:
        return
    pooled_best = []
    lolo_best = []
    for m in models:
        pooled_best.append(by_model[m]["pooled"]["Accuracy"].max())
        lolo = by_model[m]["lolo"].groupby("test_lang")["Accuracy"].max()
        lolo_best.append(lolo.mean())
    pooled_best = np.array(pooled_best); lolo_best = np.array(lolo_best)
    gap = pooled_best - lolo_best

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(models) + 1.8))
    x = np.arange(len(models))
    ax.bar(x - 0.18, pooled_best, width=0.36, color="#4c72b0", label="pooled (CV) best")
    ax.bar(x + 0.18, lolo_best, width=0.36, color="#dd8452", label="LOLO mean best")
    for i, g in enumerate(gap):
        ax.annotate(f"Δ {g:+.2f}", (x[i], max(pooled_best[i], lolo_best[i]) + 0.015),
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("best-layer accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Multilingual generalisation gap: pooled-CV vs leave-one-language-out")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(by_model, out_path):
    """Per-model bar: pooled best vs each LOLO-language best."""
    models = [m for m in MODEL_ORDER if m in by_model and "lolo" in by_model[m]]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    bw = 0.15
    x = np.arange(len(models))
    # pooled
    p_best = [by_model[m]["pooled"]["Accuracy"].max()
              if "pooled" in by_model[m] else np.nan for m in models]
    ax.bar(x - 2 * bw, p_best, width=bw, color="#444", label="pooled (CV)")
    for j, lang in enumerate(LANG_ORDER):
        vals = []
        for m in models:
            sub = by_model[m]["lolo"]
            sub = sub[sub["test_lang"] == lang]
            vals.append(sub["Accuracy"].max() if not sub.empty else np.nan)
        ax.bar(x + (j - 1) * bw, vals, width=bw,
               color=LANG_COLORS[lang], label=f"LOLO {lang}")
    ax.axhline(0.25, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("best-layer accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Per-model best-layer accuracy: pooled vs each held-out language")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="20")
    ap.add_argument("--csv_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    base = os.path.join("results", f"EXP{args.exp_id}")
    csv_dir = args.csv_dir or os.path.join(base, "csv")
    out_dir = args.out_dir or os.path.join(base, "plots")
    os.makedirs(out_dir, exist_ok=True)

    style()
    by_model = discover(csv_dir)
    if not by_model:
        print(f"No CSVs found in {csv_dir}")
        return
    print("Found models:", list(by_model.keys()))

    for model, dfs in by_model.items():
        if "pooled" in dfs:
            plot_pooled_per_model(model, dfs["pooled"],
                                  os.path.join(out_dir, f"probing_{model}_pooled.png"))
        if "lolo" in dfs:
            plot_lolo_per_model(model, dfs["lolo"],
                                os.path.join(out_dir, f"probing_{model}_LOLO.png"))

    plot_pooled_grid(by_model, os.path.join(out_dir, "speechcraft_pooled_grid.png"))
    plot_lolo_grid(by_model, os.path.join(out_dir, "speechcraft_LOLO_grid.png"))
    plot_lolo_heatmap(by_model, os.path.join(out_dir, "speechcraft_lolo_heatmap.png"))
    plot_pooled_vs_lolo_gap(by_model, os.path.join(out_dir, "speechcraft_pooled_vs_lolo_gap.png"))
    plot_summary_bar(by_model, os.path.join(out_dir, "speechcraft_summary_bar.png"))

    print(f"[done] plots written to {out_dir}")


if __name__ == "__main__":
    main()
