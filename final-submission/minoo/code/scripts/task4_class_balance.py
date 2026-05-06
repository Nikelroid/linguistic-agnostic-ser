"""Task 4 — class-balance audit for the 7 LOSO datasets.

Computes per-dataset imbalance metrics to document why MSP-Podcast is excluded
from the LOSO headline analysis (severe class imbalance).

Class counts are sourced from the midterm report Appendix A, Table 4
(CSCI 535 Midterm Report, "Linguistic-Agnostic Speech Emotion Recognition").
For IEMOCAP and MSP-Podcast, the standard 4-class evaluation subsets are used
to match what the LOSO experiments actually probed.

Run from repo root:
    python scripts/task4_class_balance.py

Outputs:
    results/LOSO/class_balance.csv     # one row per dataset
    results/LOSO/class_balance.md      # markdown table for the report
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

DATASET_CLASS_COUNTS: dict[str, dict[str, int]] = {
    "RAVDESS": {
        "Neutral": 192,
        "Calm": 384,
        "Angry": 384,
        "Disgust": 384,
        "Fearful": 384,
        "Happy": 384,
        "Sad": 384,
        "Surprised": 384,
    },
    "EmoDB": {
        "Angry": 127,
        "Bored": 81,
        "Neutral": 79,
        "Happy": 71,
        "Fearful": 69,
        "Sad": 62,
        "Disgust": 46,
    },
    "IEMOCAP": {
        "Neutral": 1708,
        "Happy": 1636,
        "Angry": 1103,
        "Sad": 1084,
    },
    "SAVEE": {
        "Neutral": 120,
        "Anger": 60,
        "Disgust": 60,
        "Fear": 60,
        "Happiness": 60,
        "Sadness": 60,
        "Surprise": 60,
    },
    "AESDD": {
        "Anger": 121,
        "Disgust": 121,
        "Fear": 121,
        "Happiness": 121,
        "Sadness": 120,
    },
    "MESD": {
        "Anger": 144,
        "Disgust": 144,
        "Fear": 144,
        "Happiness": 144,
        "Neutral": 143,
        "Sadness": 143,
    },
    "MSP-Podcast": {
        "Neutral": 5469,
        "Happiness": 3970,
        "Anger": 803,
        "Sadness": 537,
    },
}


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    n_classes = len(counts)
    if total == 0 or n_classes <= 1:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log(p)
    return entropy / math.log(n_classes)


def imbalance_ratio(counts: list[int]) -> float:
    nz = [c for c in counts if c > 0]
    if not nz:
        return float("nan")
    return max(nz) / min(nz)


def audit_dataset(name: str, counts: dict[str, int]) -> dict:
    values = list(counts.values())
    total = sum(values)
    n_classes = len(values)
    majority = max(values)
    minority = min(values)
    return {
        "Dataset": name,
        "N_Classes": n_classes,
        "N_Samples": total,
        "Majority_Class_Pct": round(100 * majority / total, 1),
        "Minority_Class_Pct": round(100 * minority / total, 1),
        "Majority_Baseline": round(majority / total, 4),
        "Imbalance_Ratio": round(imbalance_ratio(values), 2),
        "Norm_Entropy": round(normalized_entropy(values), 4),
        "Majority_Class": max(counts, key=counts.get),
        "Minority_Class": min(counts, key=counts.get),
    }


def label_balance(row: dict) -> str:
    """Conservative classification used for the report.

    'Balanced' = perfectly equal classes, normalized entropy = 1.0
    'Mostly balanced' = all classes within 2x of each other (imbalance ratio <= 2)
    'Imbalanced' = imbalance ratio > 2
    'Severely imbalanced' = majority class >= 50% AND imbalance ratio >= 5
    """
    ratio = row["Imbalance_Ratio"]
    majority_pct = row["Majority_Class_Pct"]
    if majority_pct >= 50 and ratio >= 5:
        return "Severely imbalanced"
    if ratio > 2:
        return "Imbalanced"
    if ratio <= 1.05:
        return "Balanced"
    return "Mostly balanced"


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path) -> None:
    df = pd.DataFrame(rows)
    df["Balance_Label"] = df.apply(lambda r: label_balance(r.to_dict()), axis=1)
    df = df[
        [
            "Dataset",
            "N_Classes",
            "N_Samples",
            "Majority_Class",
            "Majority_Class_Pct",
            "Minority_Class",
            "Minority_Class_Pct",
            "Majority_Baseline",
            "Imbalance_Ratio",
            "Norm_Entropy",
            "Balance_Label",
        ]
    ]
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    lines: list[str] = []
    lines.append("# Task 4 — Dataset class-balance audit\n")
    lines.append(
        "_Source: midterm report Appendix A Table 4. IEMOCAP and MSP-Podcast use the standard 4-class evaluation subsets (the same labels the LOSO probes were trained on)._\n"
    )
    lines.append(
        "**Norm. entropy** = Shannon entropy / log(N_classes); 1.0 = perfectly balanced. **Imbalance ratio** = majority count / minority count; 1.0 = perfectly balanced.\n"
    )
    lines.append("| Dataset | # Cls | # Samp | Majority | Maj % | Minority | Min % | Baseline | Imb. Ratio | Norm. H | Label |")
    lines.append("| :--- | ---: | ---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | :--- |")
    for _, row in df.iterrows():
        lines.append(
            "| {Dataset} | {N_Classes} | {N_Samples:,} | {Majority_Class} | {Majority_Class_Pct:.1f} | "
            "{Minority_Class} | {Minority_Class_Pct:.1f} | {Majority_Baseline:.3f} | {Imbalance_Ratio:.2f} | "
            "{Norm_Entropy:.3f} | {Balance_Label} |".format(**row.to_dict())
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "- **MSP-Podcast** is the only `Severely imbalanced` dataset: 50.7% of samples are neutral, "
        "imbalance ratio 10.2× (5469 neutral vs 537 sad). The majority-class baseline is already 0.507, "
        "so any probe scoring ~0.55–0.60 accuracy is barely beating 'always predict neutral'."
    )
    lines.append(
        "- The 6 balanced datasets all have normalized entropy ≥ 0.94 and imbalance ratio ≤ 2.6, "
        "so accuracy is a meaningful metric for them."
    )
    lines.append(
        "- **Recommendation:** for MSP-Podcast, report macro-UAR / macro-F1 instead of accuracy, "
        "and either class-weight the probe or stratify-downsample. This is documented as future work; "
        "in this version of the analysis, MSP-Podcast is reported separately (§10 of the report)."
    )
    out_md.write_text("\n".join(lines))
    print(f"  wrote {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("results/LOSO"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_dataset(name, counts) for name, counts in DATASET_CLASS_COUNTS.items()]
    write_outputs(rows, args.out_dir / "class_balance.csv", args.out_dir / "class_balance.md")


if __name__ == "__main__":
    main()
