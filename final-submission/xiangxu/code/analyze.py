from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    train_idx: np.ndarray
    test_idx: np.ndarray


def load_cache(cache_dir: str):
    df = pd.read_csv(os.path.join(cache_dir, "manifest_with_labels.csv"))
    hidden_states = np.load(os.path.join(cache_dir, "hidden_states.npy"))
    labels = np.load(os.path.join(cache_dir, "labels.npy"))
    return df, hidden_states, labels


def make_split(df: pd.DataFrame, protocol: str, heldout: Optional[str], random_state: int = 42) -> SplitData:
    idx = np.arange(len(df))

    if protocol == "pooled_random":
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=random_state,
            stratify=df["label_id"].to_numpy(),
        )
        return SplitData(np.array(train_idx), np.array(test_idx))

    if protocol == "leave_one_language_out":
        if not heldout:
            raise ValueError("--heldout is required")
        test_mask = (df["language"] == heldout).to_numpy()
        return SplitData(idx[~test_mask], idx[test_mask])

    if protocol == "leave_one_dataset_out":
        if not heldout:
            raise ValueError("--heldout is required")
        test_mask = (df["dataset"] == heldout).to_numpy()
        return SplitData(idx[~test_mask], idx[test_mask])

    raise ValueError(f"Unknown protocol: {protocol}")


def evaluate_groupwise(y_true: np.ndarray, y_pred: np.ndarray, meta_df: pd.DataFrame, key: str) -> Dict[str, Dict[str, float]]:
    out = {}
    for value, sub in meta_df.groupby(key):
        indices = sub.index.to_numpy()
        yt = y_true[indices]
        yp = y_pred[indices]
        out[str(value)] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "weighted_f1": float(f1_score(yt, yp, average="weighted")),
            "n": int(len(indices)),
        }
    return out


def run_probe(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray):
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    test_Xs = scaler.transform(test_X)

    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(train_Xs, train_y)
    pred = clf.predict(test_Xs)

    metrics = {
        "accuracy": float(accuracy_score(test_y, pred)),
        "weighted_f1": float(f1_score(test_y, pred, average="weighted")),
    }
    return pred, metrics


def simple_plot(results_df: pd.DataFrame, output_png: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(results_df["Layer_Num"], results_df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(results_df["Layer_Num"], results_df["Weighted_F1"], marker="s", label="Weighted F1")
    best_idx = results_df["Accuracy"].idxmax()
    best_layer = results_df.loc[best_idx, "Layer_Num"]
    best_acc = results_df.loc[best_idx, "Accuracy"]
    plt.annotate(
        f"Best {best_acc:.3f}",
        xy=(best_layer, best_acc),
        xytext=(best_layer + 0.3, min(best_acc + 0.03, 0.99)),
        arrowprops=dict(arrowstyle="->"),
    )
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def analyze(cache_dir: str, protocol: str, output_dir: str, heldout: Optional[str] = None):
    df, hidden_states, labels = load_cache(cache_dir)
    split = make_split(df, protocol=protocol, heldout=heldout)

    train_df = df.iloc[split.train_idx].reset_index(drop=True)
    test_df = df.iloc[split.test_idx].reset_index(drop=True)

    train_hidden = hidden_states[split.train_idx]
    test_hidden = hidden_states[split.test_idx]
    train_y = labels[split.train_idx]
    test_y = labels[split.test_idx]

    layer_rows = []
    detailed_results = []
    best_layer = None
    best_acc = -1.0

    for layer_idx in range(train_hidden.shape[1]):
        print(f"Probing layer {layer_idx}...", flush=True)
        train_X = train_hidden[:, layer_idx, :]
        test_X = test_hidden[:, layer_idx, :]
        pred, metrics = run_probe(train_X, train_y, test_X, test_y)

        group_dataset = evaluate_groupwise(test_y, pred, test_df.reset_index(drop=True), "dataset")
        group_language = evaluate_groupwise(test_y, pred, test_df.reset_index(drop=True), "language")

        layer_rows.append({
            "Layer": "CNN" if layer_idx == 0 else f"Layer {layer_idx}",
            "Layer_Num": int(layer_idx),
            "Accuracy": round(metrics["accuracy"], 4),
            "Weighted_F1": round(metrics["weighted_f1"], 4),
        })

        detailed_results.append({
            "layer": int(layer_idx),
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "per_dataset": group_dataset,
            "per_language": group_language,
        })

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_layer = layer_idx

    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(layer_rows)
    results_df.to_csv(os.path.join(output_dir, "task9_results.csv"), index=False)

    summary = {
        "protocol": protocol,
        "heldout": heldout,
        "num_train": int(len(train_df)),
        "num_test": int(len(test_df)),
        "best_layer": int(best_layer),
        "best_accuracy": float(best_acc),
        "results": detailed_results,
    }
    with open(os.path.join(output_dir, "task9_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    simple_plot(
        results_df,
        os.path.join(output_dir, "task9_plot.png"),
        f"Compiled Benchmark: {protocol}" + (f" ({heldout})" if heldout else ""),
    )

    print(f"Saved probe results to: {output_dir}")
    print(f"Best layer: {best_layer} | Best accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--protocol", choices=["pooled_random", "leave_one_language_out", "leave_one_dataset_out"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--heldout", type=str, default=None)
    args = parser.parse_args()

    analyze(
        cache_dir=args.cache_dir,
        protocol=args.protocol,
        output_dir=args.output_dir,
        heldout=args.heldout,
    )


if __name__ == "__main__":
    main()
