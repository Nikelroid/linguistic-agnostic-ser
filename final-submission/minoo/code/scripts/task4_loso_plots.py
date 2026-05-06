"""Per-layer 5-fold vs LOSO accuracy plots from the Task 4 CSVs.

For each loso_{Model}_{Dataset}.csv file produces one PNG with two panels:
  - Top:    Accuracy vs Layer (5F line + LOSO line + chance baseline)
  - Bottom: Per-layer Drop (5F - LOSO)

Also produces a single 6x6 grid of small accuracy plots covering all
balanced (model, dataset) combinations for paper-figure use, and a
matching 6x1 grid for MSP-Podcast.

Run from repo root:
    python scripts/task4_loso_plots.py
    python scripts/task4_loso_plots.py --no-grid              # skip grids, only individual plots
    python scripts/task4_loso_plots.py --out_dir results/LOSO/plots
"""
import argparse
import os
import sys
import math
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the chance baselines from the table script so values stay consistent.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from task4_loso_table import DATASET_CHANCE, MODELS, HEADLINE_DATASETS, SEPARATE_DATASETS
except ImportError:
    # Fallback if the file is missing — keep the script self-contained.
    MODELS = ["HuBERT", "wav2vec2", "Whisper", "WavLM", "MERT", "w2v-BERT"]
    HEADLINE_DATASETS = ["RAVDESS", "EmoDB", "IEMOCAP", "SAVEE", "AESDD", "MESD"]
    SEPARATE_DATASETS = ["MSP-Podcast"]
    DATASET_CHANCE = {
        "RAVDESS": {"baseline": 0.133}, "EmoDB": {"baseline": 0.237},
        "IEMOCAP": {"baseline": 0.309}, "SAVEE": {"baseline": 0.250},
        "AESDD": {"baseline": 0.200},   "MESD": {"baseline": 0.167},
        "MSP-Podcast": {"baseline": 0.507},
    }

C_5F = "#1f77b4"
C_LOSO = "#d62728"
C_CHANCE = "#888888"
C_DROP = "#7d3c98"


def load(csv_dir, model, dataset):
    path = os.path.join(csv_dir, f"loso_{model}_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).sort_values("Layer_Num").reset_index(drop=True)
    if df["LOSO_Accuracy"].max() == 0.0:
        return None
    return df


def plot_one(df, model, dataset, out_path):
    """Two-panel plot: accuracy curves on top, drop on bottom."""
    chance = DATASET_CHANCE.get(dataset, {}).get("baseline", float("nan"))
    n_speakers = int(df["Num_Speakers"].iloc[0])

    fig, (ax_acc, ax_drop) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    layers = df["Layer_Num"].values
    f5 = df["FiveFold_Accuracy"].values
    lo = df["LOSO_Accuracy"].values
    drop = df["Accuracy_Drop"].values
    lo_std = df["LOSO_Std"].values if "LOSO_Std" in df.columns else None

    ax_acc.plot(layers, f5, "o-", color=C_5F, lw=2, ms=4, label="5-fold accuracy")
    ax_acc.plot(layers, lo, "s-", color=C_LOSO, lw=2, ms=4, label="LOSO accuracy")
    if lo_std is not None:
        ax_acc.fill_between(layers, lo - lo_std, lo + lo_std,
                            color=C_LOSO, alpha=0.12, label="LOSO ±1σ across speakers")

    if not math.isnan(chance):
        ax_acc.axhline(chance, color=C_CHANCE, ls=":", lw=1.5,
                       label=f"Chance ({chance:.3f})")

    # Mark the best layers
    best_5f_i = int(df["FiveFold_Accuracy"].idxmax())
    best_loso_i = int(df["LOSO_Accuracy"].idxmax())
    ax_acc.axvline(layers[best_5f_i], color=C_5F, ls="--", alpha=0.4)
    ax_acc.axvline(layers[best_loso_i], color=C_LOSO, ls="--", alpha=0.4)
    ax_acc.annotate(f"5F best\nL{layers[best_5f_i]}\n{f5[best_5f_i]:.3f}",
                    xy=(layers[best_5f_i], f5[best_5f_i]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=C_5F)
    ax_acc.annotate(f"LOSO best\nL{layers[best_loso_i]}\n{lo[best_loso_i]:.3f}",
                    xy=(layers[best_loso_i], lo[best_loso_i]),
                    xytext=(5, -25), textcoords="offset points",
                    fontsize=8, color=C_LOSO)

    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.0)
    ax_acc.grid(alpha=0.25)
    ax_acc.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_acc.set_title(f"{model} on {dataset}  ({n_speakers} speakers, "
                     f"{len(layers)} layers)")

    ax_drop.bar(layers, drop, color=C_DROP, alpha=0.7, width=0.7)
    ax_drop.axhline(0, color="black", lw=0.5)
    ax_drop.set_xlabel("Layer (0 = CNN front-end)")
    ax_drop.set_ylabel("5F − LOSO\n(speaker leakage)", fontsize=9)
    ax_drop.grid(axis="y", alpha=0.25)
    # x-ticks: every layer if <=14, otherwise every other
    step = 1 if len(layers) <= 14 else 2
    ax_drop.set_xticks(layers[::step])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(csv_dir, out_path, models, datasets, title):
    """Compact grid: one small accuracy plot per (model, dataset) combo."""
    rows = len(models)
    cols = len(datasets)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.0 * rows),
                             sharey=True)
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for r, model in enumerate(models):
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            df = load(csv_dir, model, dataset)
            if df is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                layers = df["Layer_Num"].values
                ax.plot(layers, df["FiveFold_Accuracy"], "-", color=C_5F, lw=1.5)
                ax.plot(layers, df["LOSO_Accuracy"], "-", color=C_LOSO, lw=1.5)
                chance = DATASET_CHANCE.get(dataset, {}).get("baseline", float("nan"))
                if not math.isnan(chance):
                    ax.axhline(chance, color=C_CHANCE, ls=":", lw=0.8)
                ax.set_ylim(0, 1.0)
                ax.grid(alpha=0.2)
                ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(dataset, fontsize=9)
            if c == 0:
                ax.set_ylabel(model, fontsize=9)

    # Single legend for the figure
    handles = [
        plt.Line2D([0], [0], color=C_5F, lw=2, label="5-fold"),
        plt.Line2D([0], [0], color=C_LOSO, lw=2, label="LOSO"),
        plt.Line2D([0], [0], color=C_CHANCE, ls=":", lw=1.5, label="Chance"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=10, frameon=False)
    fig.suptitle(title, y=1.04, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.01, "Layer", ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Per-layer LOSO plots for Task 4")
    p.add_argument("--csv_dir", default="results/LOSO/csv")
    p.add_argument("--out_dir", default="results/LOSO/plots")
    p.add_argument("--no-grid", action="store_true",
                   help="Skip the multi-panel grid figures")
    p.add_argument("--no-individual", action="store_true",
                   help="Skip per-(model, dataset) PNGs")
    args = p.parse_args()

    if not os.path.isdir(args.csv_dir):
        print(f"ERROR: {args.csv_dir} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    written = 0
    skipped = 0

    if not args.no_individual:
        for model in MODELS:
            for dataset in HEADLINE_DATASETS + SEPARATE_DATASETS:
                df = load(args.csv_dir, model, dataset)
                if df is None:
                    skipped += 1
                    continue
                out_path = os.path.join(args.out_dir,
                                        f"loso_{model}_{dataset}.png")
                plot_one(df, model, dataset, out_path)
                written += 1
                print(f"  wrote {out_path}")
        print(f"\nIndividual plots: {written} written, {skipped} skipped (missing/stale).")

    if not args.no_grid:
        grid_path = os.path.join(args.out_dir, "loso_grid_balanced.png")
        plot_grid(args.csv_dir, grid_path, MODELS, HEADLINE_DATASETS,
                  title="Layer-wise 5-fold vs LOSO accuracy — 6 models × 6 balanced datasets")
        print(f"  wrote {grid_path}")

        msp_grid = os.path.join(args.out_dir, "loso_grid_msp_podcast.png")
        plot_grid(args.csv_dir, msp_grid, MODELS, SEPARATE_DATASETS,
                  title="Layer-wise 5-fold vs LOSO accuracy — MSP-Podcast (separate)")
        print(f"  wrote {msp_grid}")


if __name__ == "__main__":
    main()
