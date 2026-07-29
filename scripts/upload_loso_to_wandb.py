"""Upload per-(model,dataset) LOSO CSVs to W&B as one run each."""
import os

SER_SCRATCH = os.environ.get(
    "SER_SCRATCH", os.path.join("/scratch1", os.environ.get("USER", "unknown")))
import glob
import pandas as pd
import wandb

PROJECT = "linguistic-agnostic-ser"
ENTITY = "AGSER"
CSV_DIR = os.path.join(SER_SCRATCH, "linguistic-agnostic-ser/results/LOSO/csv")
SKIP_DATASETS = {"MESD", "MSP-Podcast"}  # rerunning with parser fix; upload separately later

csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "loso_*.csv")))
csv_files = [f for f in csv_files if os.path.basename(f) != "loso_all_combined.csv"]

uploaded = 0
skipped = 0
for path in csv_files:
    name = os.path.basename(path).replace("loso_", "").replace(".csv", "")
    # filename format: loso_<Model>_<Dataset>.csv
    parts = name.split("_", 1)
    if len(parts) != 2:
        print(f"  skip (bad name): {path}")
        continue
    model, dataset = parts

    if dataset in SKIP_DATASETS:
        print(f"  skip ({dataset} rerunning): {model}_{dataset}")
        skipped += 1
        continue

    df = pd.read_csv(path)

    best_5f_idx = df["FiveFold_Accuracy"].idxmax()
    best_loso_idx = df["LOSO_Accuracy"].idxmax()
    summary = {
        "best_5fold_acc": float(df.loc[best_5f_idx, "FiveFold_Accuracy"]),
        "best_5fold_layer": df.loc[best_5f_idx, "Layer"],
        "best_5fold_layer_num": int(df.loc[best_5f_idx, "Layer_Num"]),
        "best_loso_acc": float(df.loc[best_loso_idx, "LOSO_Accuracy"]),
        "best_loso_layer": df.loc[best_loso_idx, "Layer"],
        "best_loso_layer_num": int(df.loc[best_loso_idx, "Layer_Num"]),
        "best_loso_f1": float(df.loc[best_loso_idx, "LOSO_F1"]),
        "avg_drop": float(df["Accuracy_Drop"].mean()),
        "max_drop": float(df["Accuracy_Drop"].max()),
        "min_drop": float(df["Accuracy_Drop"].min()),
        "num_speakers": int(df["Num_Speakers"].iloc[0]),
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"{model}_{dataset}_loso",
        group="task4_loso",
        job_type="evaluation",
        tags=[model, dataset, "loso", "task4"],
        config={"model": model, "dataset": dataset, "task": "task4_loso"},
        reinit=True,
    )

    for _, row in df.iterrows():
        wandb.log({
            "layer": int(row["Layer_Num"]),
            "5fold/accuracy": float(row["FiveFold_Accuracy"]),
            "5fold/f1": float(row["FiveFold_F1"]),
            "loso/accuracy": float(row["LOSO_Accuracy"]),
            "loso/std": float(row["LOSO_Std"]),
            "loso/f1": float(row["LOSO_F1"]),
            "loso/drop": float(row["Accuracy_Drop"]),
            "loso/min_speaker_acc": float(row["LOSO_Min_Speaker_Acc"]),
            "loso/max_speaker_acc": float(row["LOSO_Max_Speaker_Acc"]),
        }, step=int(row["Layer_Num"]))

    wandb.log({"per_layer_table": wandb.Table(dataframe=df)})

    for k, v in summary.items():
        run.summary[k] = v

    artifact = wandb.Artifact(name=f"loso_csv_{model}_{dataset}", type="loso_results")
    artifact.add_file(path)
    run.log_artifact(artifact)

    run.finish()
    uploaded += 1
    print(f"  uploaded: {model}_{dataset}")

print(f"\nDone. Uploaded: {uploaded}, Skipped: {skipped}")
