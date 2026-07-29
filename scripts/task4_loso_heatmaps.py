"""Task 4 LOSO summary heatmaps.

Reads per-model CSVs from results/LOSO/csv/ (the authoritative source — see
results/LOSO/README.md) and renders two presentation-grade heatmaps:

1. Accuracy-drop heatmap: 6 models x 6 balanced datasets, cell = (5F best acc) - (LOSO best acc)
2. Best-layer-shift heatmap: same axes, cell = (LOSO best layer) - (5F best layer)

MSP-Podcast is excluded from both (severe class imbalance, see report section 10).

Run from repo root:
    python scripts/task4_loso_heatmaps.py

Outputs:
    results/LOSO/plots/loso_heatmap_drop.png
    results/LOSO/plots/loso_heatmap_layer_shift.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["HuBERT", "wav2vec2", "Whisper", "WavLM", "MERT", "w2v-BERT"]
DATASETS_BALANCED = ["RAVDESS", "EmoDB", "IEMOCAP", "SAVEE", "AESDD", "MESD"]


def load_best_per_cell(csv_dir: Path, model: str, dataset: str) -> dict | None:
    csv_path = csv_dir / f"loso_{model}_{dataset}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    fivefold_idx = df["FiveFold_Accuracy"].idxmax()
    loso_idx = df["LOSO_Accuracy"].idxmax()
    return {
        "fivefold_layer": int(df.loc[fivefold_idx, "Layer_Num"]),
        "loso_layer": int(df.loc[loso_idx, "Layer_Num"]),
        "fivefold_acc": float(df.loc[fivefold_idx, "FiveFold_Accuracy"]),
        "loso_acc": float(df.loc[loso_idx, "LOSO_Accuracy"]),
        "avg_per_layer_drop": float(df["Accuracy_Drop"].mean()),
        "n_layers": int(len(df)),
    }


def write_avg_drop_csv(csv_dir: Path, out_path: Path) -> None:
    rows = []
    for model in MODELS:
        for dataset in DATASETS_BALANCED:
            cell = load_best_per_cell(csv_dir, model, dataset)
            if cell is None:
                continue
            rows.append({
                "Model": model,
                "Dataset": dataset,
                "Best_5F_Layer": cell["fivefold_layer"],
                "Best_5F_Acc": round(cell["fivefold_acc"], 4),
                "Best_LOSO_Layer": cell["loso_layer"],
                "Best_LOSO_Acc": round(cell["loso_acc"], 4),
                "BestVsBest_Drop": round(cell["fivefold_acc"] - cell["loso_acc"], 4),
                "AvgPerLayer_Drop": round(cell["avg_per_layer_drop"], 4),
                "Drop_Diff": round(
                    (cell["fivefold_acc"] - cell["loso_acc"]) - cell["avg_per_layer_drop"], 4
                ),
                "N_Layers": cell["n_layers"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def build_matrices(csv_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drop = np.full((len(MODELS), len(DATASETS_BALANCED)), np.nan)
    delta_layer = np.full_like(drop, np.nan)
    fivefold_layer = np.full_like(drop, np.nan)
    loso_layer = np.full_like(drop, np.nan)
    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS_BALANCED):
            cell = load_best_per_cell(csv_dir, model, dataset)
            if cell is None:
                continue
            drop[i, j] = cell["fivefold_acc"] - cell["loso_acc"]
            delta_layer[i, j] = cell["loso_layer"] - cell["fivefold_layer"]
            fivefold_layer[i, j] = cell["fivefold_layer"]
            loso_layer[i, j] = cell["loso_layer"]
    return drop, delta_layer, fivefold_layer, loso_layer


def order_columns_by_avg_drop(drop: np.ndarray) -> list[int]:
    avg = np.nanmean(drop, axis=0)
    return list(np.argsort(-avg))


def render_drop_heatmap(drop: np.ndarray, out_path: Path) -> None:
    col_order = order_columns_by_avg_drop(drop)
    drop_sorted = drop[:, col_order]
    datasets_sorted = [DATASETS_BALANCED[j] for j in col_order]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    im = ax.imshow(drop_sorted, cmap="Reds", vmin=0.0, vmax=0.36, aspect="auto")

    ax.set_xticks(range(len(datasets_sorted)))
    ax.set_xticklabels(datasets_sorted, rotation=0)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("Dataset (sorted by avg drop, worst → best)")
    ax.set_ylabel("Model")
    ax.set_title("Speaker leakage: 5-fold − LOSO accuracy drop\n(higher = more inflated by speaker memorization)")

    for i in range(drop_sorted.shape[0]):
        for j in range(drop_sorted.shape[1]):
            v = drop_sorted[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if v > 0.18 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Accuracy drop (5F − LOSO)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_layer_shift_heatmap(
    delta_layer: np.ndarray,
    fivefold_layer: np.ndarray,
    loso_layer: np.ndarray,
    drop: np.ndarray,
    out_path: Path,
) -> None:
    col_order = order_columns_by_avg_drop(drop)
    delta_sorted = delta_layer[:, col_order]
    f5_sorted = fivefold_layer[:, col_order]
    loso_sorted = loso_layer[:, col_order]
    datasets_sorted = [DATASETS_BALANCED[j] for j in col_order]

    vmax = float(np.nanmax(np.abs(delta_sorted)))
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    im = ax.imshow(delta_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(datasets_sorted)))
    ax.set_xticklabels(datasets_sorted, rotation=0)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("Dataset (matched ordering with drop heatmap)")
    ax.set_ylabel("Model")
    ax.set_title("Best-layer shift under LOSO\n(red = LOSO chose deeper layer; blue = shallower; white = no shift)")

    for i in range(delta_sorted.shape[0]):
        for j in range(delta_sorted.shape[1]):
            d = delta_sorted[i, j]
            if np.isnan(d):
                continue
            f5 = int(f5_sorted[i, j])
            lo = int(loso_sorted[i, j])
            text_color = "white" if abs(d) > 0.55 * vmax else "black"
            label = f"{f5}→{lo}\n(Δ{int(d):+d})"
            ax.text(j, i, label, ha="center", va="center",
                    color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("LOSO best layer − 5F best layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=Path, default=Path("results/LOSO/csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/LOSO/plots"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    drop, delta_layer, fivefold_layer, loso_layer = build_matrices(args.csv_dir)

    render_drop_heatmap(drop, args.out_dir / "loso_heatmap_drop.png")
    render_layer_shift_heatmap(
        delta_layer, fivefold_layer, loso_layer, drop,
        args.out_dir / "loso_heatmap_layer_shift.png",
    )
    write_avg_drop_csv(args.csv_dir, args.csv_dir.parent / "avg_drop_per_layer.csv")


if __name__ == "__main__":
    main()
