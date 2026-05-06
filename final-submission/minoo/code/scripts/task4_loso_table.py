"""Generate Task 4 LOSO summary tables (Markdown) from per-model/dataset CSVs.

For each (model, dataset) reads results/LOSO/csv/loso_{Model}_{Dataset}.csv and pulls:
  - 5F Layer / 5F Acc  (argmax of FiveFold_Accuracy)
  - LOSO Layer / LOSO Acc  (argmax of LOSO_Accuracy)
  - Drop = best 5F Acc - best LOSO Acc
  - Δ Layer = LOSO Layer - 5F Layer
  - # Speakers (constant per file)
  - LOSO Std at the LOSO-best layer (across-speaker spread)
  - LOSO Min / Max speaker acc at the LOSO-best layer (worst/best speaker)

Emits two markdown tables sorted by Drop descending:
  1. Headline: 6 models x 6 balanced datasets (RAVDESS/EmoDB/IEMOCAP/SAVEE/AESDD/MESD)
  2. MSP-Podcast: 6 models (separate due to class imbalance)

Run from repo root:
    python scripts/task4_loso_table.py
    python scripts/task4_loso_table.py --extended   # adds Std + Min/Max speaker cols
    python scripts/task4_loso_table.py --out docs/task4_table.md
"""
import argparse
import os
import sys
import pandas as pd

MODELS = ["HuBERT", "wav2vec2", "Whisper", "WavLM", "MERT", "w2v-BERT"]
HEADLINE_DATASETS = ["RAVDESS", "EmoDB", "IEMOCAP", "SAVEE", "AESDD", "MESD"]
SEPARATE_DATASETS = ["MSP-Podcast"]

# Majority-class chance baselines, derived from midterm Table 4 (Appendix A).
# These are the fraction of the dataset taken by the largest emotion class,
# i.e. what a "always predict the most common class" classifier would score.
DATASET_CHANCE = {
    # RAVDESS: 192 neutral + 384 each of 7 others (max class = 384/2880)
    "RAVDESS":     {"n_classes": 8, "majority_class": "(7-way tie at 384)", "baseline": 0.133, "balanced": True},
    # EmoDB: angry 127, bored 81, neutral 79, happy 71, fearful 69, sad 62, disgust 46
    "EmoDB":       {"n_classes": 7, "majority_class": "anger",   "baseline": 0.237, "balanced": False},
    # IEMOCAP 4-class: neu 1708, hap 1636, ang 1103, sad 1084
    "IEMOCAP":     {"n_classes": 4, "majority_class": "neutral", "baseline": 0.309, "balanced": False},
    # SAVEE: neutral 120, others 60 each
    "SAVEE":       {"n_classes": 7, "majority_class": "neutral", "baseline": 0.250, "balanced": False},
    # AESDD: anger/disgust/fear/happiness 121, sadness 120
    "AESDD":       {"n_classes": 5, "majority_class": "(4-way tie at 121)", "baseline": 0.200, "balanced": True},
    # MESD: anger/disgust/fear/happiness 144, neutral/sadness 143
    "MESD":        {"n_classes": 6, "majority_class": "(4-way tie at 144)", "baseline": 0.167, "balanced": True},
    # MSP-Podcast (4-class): heavily imbalanced toward neutral (~50.7%)
    "MSP-Podcast": {"n_classes": 4, "majority_class": "neutral", "baseline": 0.507, "balanced": False},
}


def load_one(csv_dir, model, dataset):
    path = os.path.join(csv_dir, f"loso_{model}_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Treat all-zero LOSO columns (stale combined-CSV pattern) as missing.
    if df["LOSO_Accuracy"].max() == 0.0:
        return None
    best_5f = df.loc[df["FiveFold_Accuracy"].idxmax()]
    best_loso = df.loc[df["LOSO_Accuracy"].idxmax()]
    chance = DATASET_CHANCE.get(dataset, {}).get("baseline", float("nan"))
    return {
        "Model": model,
        "Dataset": dataset,
        "5F Layer": int(best_5f["Layer_Num"]),
        "LOSO Layer": int(best_loso["Layer_Num"]),
        "Δ Layer": int(best_loso["Layer_Num"]) - int(best_5f["Layer_Num"]),
        "5F Acc": float(best_5f["FiveFold_Accuracy"]),
        "LOSO Acc": float(best_loso["LOSO_Accuracy"]),
        "Drop": float(best_5f["FiveFold_Accuracy"]) - float(best_loso["LOSO_Accuracy"]),
        # Chance = majority-class baseline; Lift = how far above chance LOSO is.
        # If Lift is small, the probe is barely doing anything (relevant for MSP-Podcast).
        "Chance": chance,
        "LOSO Lift": float(best_loso["LOSO_Accuracy"]) - chance,
        "# Spk": int(best_loso["Num_Speakers"]),
        "LOSO Std@best": float(best_loso.get("LOSO_Std", float("nan"))),
        "LOSO Min Spk@best": float(best_loso.get("LOSO_Min_Speaker_Acc", float("nan"))),
        "LOSO Max Spk@best": float(best_loso.get("LOSO_Max_Speaker_Acc", float("nan"))),
    }


def gather(csv_dir, datasets):
    rows = []
    missing = []
    for model in MODELS:
        for dataset in datasets:
            row = load_one(csv_dir, model, dataset)
            if row is None:
                missing.append((model, dataset))
            else:
                rows.append(row)
    return pd.DataFrame(rows), missing


def fmt_signed(n):
    n = int(n)
    if n > 0:
        return f"+{n}"
    return str(n)


def render_table(df, extended=False):
    if df.empty:
        return "(no rows)"
    df = df.sort_values("Drop", ascending=False).reset_index(drop=True)

    if extended:
        cols = [
            "Model", "Dataset", "5F Layer", "LOSO Layer", "Δ Layer",
            "5F Acc", "LOSO Acc", "Drop", "Chance", "LOSO Lift",
            "LOSO Std@best", "LOSO Min Spk@best", "LOSO Max Spk@best", "# Spk",
        ]
        align = ["left", "left", "right", "right", "right",
                 "right", "right", "right", "right", "right",
                 "right", "right", "right", "right"]
        formatters = {
            "5F Acc": lambda v: f"{v:.4f}",
            "LOSO Acc": lambda v: f"{v:.4f}",
            "Drop": lambda v: f"{v:.3f}",
            "Chance": lambda v: f"{v:.3f}",
            "LOSO Lift": lambda v: f"{v:+.3f}",
            "Δ Layer": fmt_signed,
            "LOSO Std@best": lambda v: f"{v:.3f}",
            "LOSO Min Spk@best": lambda v: f"{v:.3f}",
            "LOSO Max Spk@best": lambda v: f"{v:.3f}",
        }
    else:
        cols = ["Model", "Dataset", "5F Layer", "LOSO Layer", "Δ Layer",
                "5F Acc", "LOSO Acc", "Drop", "Chance", "LOSO Lift", "# Spk"]
        align = ["left", "left", "right", "right", "right",
                 "right", "right", "right", "right", "right", "right"]
        formatters = {
            "5F Acc": lambda v: f"{v:.4f}",
            "LOSO Acc": lambda v: f"{v:.4f}",
            "Drop": lambda v: f"{v:.3f}",
            "Chance": lambda v: f"{v:.3f}",
            "LOSO Lift": lambda v: f"{v:+.3f}",
            "Δ Layer": fmt_signed,
        }

    def cell(row, col):
        v = row[col]
        if col in formatters:
            return formatters[col](v)
        return str(v)

    align_marker = {"left": ":---", "right": "---:", "center": ":---:"}

    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(align_marker[a] for a in align) + " |")
    for _, row in df.iterrows():
        out.append("| " + " | ".join(cell(row, c) for c in cols) + " |")
    return "\n".join(out)


def best_and_worst(df):
    """Pick the section-4 'four numbers' automatically."""
    if df.empty:
        return None
    df = df.copy()
    best_honest = df.sort_values(["LOSO Acc", "Drop"], ascending=[False, True]).iloc[0]
    worst_leak = df.sort_values("Drop", ascending=False).iloc[0]
    smallest_drop = df.sort_values("Drop").iloc[0]
    return {
        "best_honest": best_honest,
        "worst_leak": worst_leak,
        "smallest_drop": smallest_drop,
    }


def render_summary(df):
    picks = best_and_worst(df)
    if picks is None:
        return ""
    lines = [
        "### Quick-look summary (auto-computed)",
        "",
        f"- Best honest accuracy: **{picks['best_honest']['Model']}/{picks['best_honest']['Dataset']}** "
        f"— LOSO {picks['best_honest']['LOSO Acc']:.4f} at L{int(picks['best_honest']['LOSO Layer'])} "
        f"(drop {picks['best_honest']['Drop']:.3f}, {int(picks['best_honest']['# Spk'])} speakers)",
        f"- Worst leakage: **{picks['worst_leak']['Model']}/{picks['worst_leak']['Dataset']}** "
        f"— drop {picks['worst_leak']['Drop']:.3f} "
        f"(5F {picks['worst_leak']['5F Acc']:.3f} → LOSO {picks['worst_leak']['LOSO Acc']:.3f}, "
        f"{int(picks['worst_leak']['# Spk'])} speakers)",
        f"- Smallest drop: **{picks['smallest_drop']['Model']}/{picks['smallest_drop']['Dataset']}** "
        f"— drop {picks['smallest_drop']['Drop']:.3f} "
        f"(5F {picks['smallest_drop']['5F Acc']:.3f} → LOSO {picks['smallest_drop']['LOSO Acc']:.3f})",
    ]
    # Per-dataset average drop
    avg = df.groupby("Dataset")["Drop"].mean().sort_values(ascending=False)
    lines.append("")
    lines.append("**Avg drop per dataset (across 6 models):**")
    lines.append("")
    lines.append("| Dataset | Avg Drop |")
    lines.append("| :--- | ---: |")
    for ds, v in avg.items():
        lines.append(f"| {ds} | {v:.3f} |")

    # Largest |Δ Layer| shifts under LOSO — for the Finding 3 paragraph.
    # If best layer shifts a lot under speaker-aware eval, the 5F-favored layer
    # was relying on speaker-specific signal that doesn't transfer.
    shift = df.assign(absΔ=df["Δ Layer"].abs()).sort_values("absΔ", ascending=False).head(5)
    lines.append("")
    lines.append("**Top 5 largest layer shifts under LOSO (|Δ Layer|):**")
    lines.append("")
    for _, r in shift.iterrows():
        direction = "deeper" if r["Δ Layer"] > 0 else ("shallower" if r["Δ Layer"] < 0 else "same")
        lines.append(
            f"- **{r['Model']}/{r['Dataset']}**: 5F=L{int(r['5F Layer'])} → "
            f"LOSO=L{int(r['LOSO Layer'])} (Δ={fmt_signed(r['Δ Layer'])}, {direction}); "
            f"5F Acc {r['5F Acc']:.3f} → LOSO Acc {r['LOSO Acc']:.3f} "
            f"(drop {r['Drop']:.3f})"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Generate Task 4 LOSO summary tables")
    p.add_argument("--csv_dir", default="results/LOSO/csv",
                   help="Directory containing per-model CSVs")
    p.add_argument("--out", default=None,
                   help="Optional output path (.md). If omitted, prints to stdout.")
    p.add_argument("--extended", action="store_true",
                   help="Include LOSO Std and per-speaker min/max columns")
    args = p.parse_args()

    if not os.path.isdir(args.csv_dir):
        print(f"ERROR: {args.csv_dir} not found", file=sys.stderr)
        sys.exit(1)

    headline_df, headline_missing = gather(args.csv_dir, HEADLINE_DATASETS)
    msp_df, msp_missing = gather(args.csv_dir, SEPARATE_DATASETS)

    parts = []
    parts.append("# Task 4 — LOSO Summary Tables\n")
    parts.append(f"_Generated from `{args.csv_dir}` per-model CSVs._\n")
    parts.append(f"_Drop = (best 5F Acc) − (best LOSO Acc); Δ Layer = LOSO best − 5F best._\n")

    parts.append("## Headline matrix — 6 models × 6 balanced datasets")
    parts.append(f"_{len(headline_df)} of {len(MODELS)*len(HEADLINE_DATASETS)} runs available._")
    if headline_missing:
        parts.append(
            "_Missing/skipped: " +
            ", ".join(f"{m}/{d}" for m, d in headline_missing) + "_"
        )
    parts.append("")
    parts.append(render_table(headline_df, extended=args.extended))
    parts.append("")
    parts.append(render_summary(headline_df))

    parts.append("\n## MSP-Podcast (separate — see §10 of the report)")
    parts.append(f"_{len(msp_df)} of {len(MODELS)*len(SEPARATE_DATASETS)} runs available._")
    if msp_missing:
        parts.append(
            "_Missing/skipped: " +
            ", ".join(f"{m}/{d}" for m, d in msp_missing) + "_"
        )
    parts.append("")
    parts.append(render_table(msp_df, extended=args.extended))

    out_text = "\n".join(parts) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out_text)
        print(f"Wrote {args.out}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
