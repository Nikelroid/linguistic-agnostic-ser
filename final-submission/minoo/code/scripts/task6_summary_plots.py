"""Task 6 summary plots — two-regime bias swing, lambda-effect, two-attacks-fail.

Reads per-model CSVs from results/TASK6/csv/ and results/TASK6/projection/ and
renders three presentation-grade figures:

1. task6_two_regime_swing.png
   4-panel paired bar chart: RAV+SAV-trained vs EMIS-trained baseline (lambda=0)
   bias for each (model, best-bias-layer). Visualizes the core Task 6 finding:
   3 of 4 models flip from negative to strongly positive bias when training pool
   changes; Whisper L24 alone resists.

2. task6_lambda_sweep.png
   4-panel grid (one per model), bias vs lambda for the EMIS-trained regime.
   Shows GRL fails to reduce bias at safe lambda, only "works" at lambda=10
   when val_emo collapses.

3. task6_two_attacks.png
   Bar chart, 4 models x 2 attacks (GRL best safe Delta, CCA best Delta).
   Visualizes joint failure of v2 GRL + v3 CCA projection.

Sign convention: text_bias = proxy_acc - target_acc
(positive = probe took the text shortcut; matches scripts/task6_adversarial.py:534)

Run from repo root:
    python scripts/task6_summary_plots.py

Outputs:
    results/TASK6/plots_summary/task6_two_regime_swing.png
    results/TASK6/plots_summary/task6_lambda_sweep.png
    results/TASK6/plots_summary/task6_two_attacks.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["HuBERT", "wav2vec2", "WavLM", "Whisper"]
# Task 6 v2 GRL layer-sweep landed at L14 for wav2vec2; Task 6 v3 CCA followed
# Task 7A's peak text-bias layer (L15 for wav2vec2). Use the right layer per script.
BEST_LAYER = {"HuBERT": 20, "wav2vec2": 14, "WavLM": 20, "Whisper": 24}
CCA_LAYER = {"HuBERT": 20, "wav2vec2": 15, "WavLM": 20, "Whisper": 24}
COLOR_RAVSAV = "#1976D2"   # blue = neutral-text training (acoustic-forced)
COLOR_EMIS = "#E53935"     # red = EMIS-trained (text-shortcut available)
COLOR_GRL = "#7E57C2"
COLOR_CCA = "#43A047"


def load_baseline_bias(csv_dir: Path, model: str, regime: str) -> dict | None:
    """regime in {'ravsav', 'emis'}"""
    suffix = "_emistrain" if regime == "emis" else ""
    path = csv_dir / "csv" / f"adversarial_{model}{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    layer = BEST_LAYER[model]
    sub = df[(df["layer"] == layer) & (df["lambda"] == 0.0)]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return {
        "layer": layer,
        "bias": float(r["emis_incong_text_bias_mean"]),
        "target": float(r["emis_incong_target_acc_mean"]),
        "proxy": float(r["emis_incong_proxy_acc_mean"]),
        "val_emo": float(r["best_val_emo_acc_mean"]),
    }


def plot_two_regime_swing(csv_dir: Path, out_path: Path) -> None:
    rows = []
    for model in MODELS:
        for regime in ["ravsav", "emis"]:
            data = load_baseline_bias(csv_dir, model, regime)
            if data is None:
                continue
            rows.append({"model": model, "regime": regime, **data})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(MODELS))
    width = 0.36
    for i, regime in enumerate(["ravsav", "emis"]):
        sub = df[df["regime"] == regime].set_index("model")
        values = [sub.loc[m, "bias"] if m in sub.index else np.nan for m in MODELS]
        layers = [sub.loc[m, "layer"] if m in sub.index else None for m in MODELS]
        offset = (i - 0.5) * width
        color = COLOR_RAVSAV if regime == "ravsav" else COLOR_EMIS
        label = "RAV+SAV-trained" if regime == "ravsav" else "EMIS-trained"
        bars = ax.bar(x + offset, values, width, color=color, label=label,
                      edgecolor="white", linewidth=0.8)
        for bar, v, layer in zip(bars, values, layers):
            if np.isnan(v) or layer is None:
                continue
            va = "bottom" if v >= 0 else "top"
            offset_y = 0.025 if v >= 0 else -0.025
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset_y,
                    f"{v:+.2f}\nL{int(layer)}", ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("text_bias  (proxy_acc - target_acc)  at lambda=0")
    ax.set_title(
        "Task 6 — same encoder, two training pools, three of four models flip sign\n"
        "(only Whisper resists the text shortcut when EMIS_congruent training offers it)"
    )
    ax.set_ylim(-0.5, 1.10)
    ax.legend(title="Training pool", loc="upper left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_lambda_sweep(csv_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharey=True)
    axes = axes.flatten()

    legend_handles = None
    for ax, model in zip(axes, MODELS):
        path = csv_dir / "csv" / f"adversarial_{model}_emistrain.csv"
        if not path.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(path)
        layer = BEST_LAYER[model]
        sub = df[df["layer"] == layer].sort_values("lambda")
        sub_safe = sub[sub["lambda"] <= 1.0].reset_index(drop=True)
        if len(sub_safe) == 0:
            ax.set_visible(False)
            continue
        lambdas = sub_safe["lambda"].to_numpy()
        bias = sub_safe["emis_incong_text_bias_mean"].to_numpy()
        val_emo = sub_safe["best_val_emo_acc_mean"].to_numpy()
        adv_suppr = sub_safe["adv_suppression_mean"].to_numpy()

        x_pos = np.arange(len(lambdas))
        x_labels = [f"{l:g}" for l in lambdas]

        bars = ax.bar(x_pos, bias, color=COLOR_EMIS, alpha=0.85,
                      edgecolor="white", linewidth=0.5,
                      label="text_bias  (proxy − target)")
        for bar, v in zip(bars, bias):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:+.2f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(0, color="black", linewidth=0.6)
        ax2 = ax.twinx()
        line_emo, = ax2.plot(x_pos, val_emo, "o-", color="#2E7D32",
                             linewidth=1.5, markersize=5,
                             label="val_emo  (emotion acc, kept high)")
        line_adv, = ax2.plot(x_pos, adv_suppr, "s--", color="#1976D2",
                             linewidth=1.2, markersize=4, alpha=0.85,
                             label="adv_suppression  (sentence-ID → chance)")
        ax2.set_ylim(0.4, 1.05)
        ax2.tick_params(axis="y", labelsize=8, colors="#555555")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{model}  L{layer}", fontsize=11)
        ax.set_ylim(-0.5, 1.10)
        ax.grid(False)

        if legend_handles is None:
            legend_handles = [bars, line_emo, line_adv]

    for ax in axes[2:]:
        ax.set_xlabel("GRL λ")
    axes[0].set_ylabel("text_bias")
    axes[2].set_ylabel("text_bias")

    if legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
            loc="lower center", ncol=3, frameon=False, fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(
        "Task 6 v2 — GRL drives sentence-ID to chance while emotion accuracy stays high,\n"
        "yet text_bias barely moves  →  sentence-ID is not the lexical-emotion shortcut",
        fontsize=10.5, y=1.00,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def best_grl_safe_delta(csv_dir: Path, model: str) -> float | None:
    """Best GRL bias reduction at lambda <= 1.0 (val_emo not collapsed). EMIS-trained."""
    path = csv_dir / "csv" / f"adversarial_{model}_emistrain.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    layer = BEST_LAYER[model]
    sub = df[df["layer"] == layer].sort_values("lambda")
    if len(sub) == 0:
        return None
    baseline = sub[sub["lambda"] == 0.0]["emis_incong_text_bias_mean"].values
    safe = sub[(sub["lambda"] <= 1.0) & (sub["lambda"] > 0)]
    if len(baseline) == 0 or len(safe) == 0:
        return None
    base = float(baseline[0])
    best_safe_bias = float(safe["emis_incong_text_bias_mean"].min())
    return best_safe_bias - base  # negative = reduction


def best_cca_delta(csv_dir: Path, model: str) -> float | None:
    """Best CCA bias reduction across both corpora and all k values, on explicit slice."""
    layer = CCA_LAYER[model]
    deltas = []
    for corpus in ["", "_iemocap"]:
        path = csv_dir / "projection" / f"cca_proj_{model}_L{layer}{corpus}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "text_bias_explicit" not in df.columns or 0 not in df["k"].values:
            continue
        baseline = float(df[df["k"] == 0]["text_bias_explicit"].iloc[0])
        nonzero_k = df[df["k"] > 0]
        if len(nonzero_k) == 0:
            continue
        best_proj = float(nonzero_k["text_bias_explicit"].min())
        deltas.append(best_proj - baseline)
    if not deltas:
        return None
    return min(deltas)  # most negative reduction


def plot_two_attacks(csv_dir: Path, out_path: Path) -> None:
    rows = []
    for model in MODELS:
        grl = best_grl_safe_delta(csv_dir, model)
        cca = best_cca_delta(csv_dir, model)
        rows.append({"model": model, "grl": grl, "cca": cca})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    x = np.arange(len(MODELS))
    width = 0.36
    grl_vals = df["grl"].to_numpy()
    cca_vals = df["cca"].to_numpy()
    bars1 = ax.bar(x - width / 2, grl_vals, width, color=COLOR_GRL,
                   label="v2 GRL adversary  (best safe lambda)",
                   edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, cca_vals, width, color=COLOR_CCA,
                   label="v3 CCA projection  (best k, explicit slice)",
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars1, grl_vals):
        if v is None or np.isnan(v):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.005,
                f"{v:+.3f}", ha="center", va="top", fontsize=9)
    for bar, v in zip(bars2, cca_vals):
        if v is None or np.isnan(v):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.005,
                f"{v:+.3f}", ha="center", va="top", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Delta text_bias  (attack - baseline)\nnegative = bias reduction")
    ax.set_title(
        "Task 6 — two attacks pinpoint where the shortcut is NOT\n"
        "GRL (sentence-ID adversary): mechanically succeeds, bias barely moves → not in sentence-ID\n"
        "CCA projection: linear BERT-aligned subspace removed, bias barely moves → not low-rank linear"
    )
    ax.set_ylim(-0.12, 0.04)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=Path, default=Path("results/TASK6"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/TASK6/plots_summary"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_two_regime_swing(args.csv_dir, args.out_dir / "task6_two_regime_swing.png")
    plot_lambda_sweep(args.csv_dir, args.out_dir / "task6_lambda_sweep.png")
    plot_two_attacks(args.csv_dir, args.out_dir / "task6_two_attacks.png")


if __name__ == "__main__":
    main()
