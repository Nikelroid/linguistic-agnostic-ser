"""Upload any LOSO CSV not already in W&B (fills in MESD + MSP-Podcast after rerun)."""
import os
import glob
import pandas as pd
import wandb

PROJECT = "linguistic-agnostic-ser"
ENTITY = "AGSER"
CSV_DIR = "/scratch1/minooahm/linguistic-agnostic-ser/results/LOSO/csv"

api = wandb.Api()
existing = set()
for run in api.runs(f"{ENTITY}/{PROJECT}"):
    existing.add(run.name)
print(f"Found {len(existing)} existing runs")

csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "loso_*.csv")))
csv_files = [f for f in csv_files if os.path.basename(f) != "loso_all_combined.csv"]

uploaded, skipped = 0, 0
for path in csv_files:
    name = os.path.basename(path).replace("loso_", "").replace(".csv", "")
    parts = name.split("_", 1)
    if len(parts) != 2:
        continue
    model, dataset = parts
    run_name = f"{model}_{dataset}_loso"

    if run_name in existing:
        print(f"  skip (exists in W&B): {run_name}")
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
        name=run_name,
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
    print(f"  uploaded: {run_name}")

print(f"\nDone. Uploaded: {uploaded}, Skipped (already in W&B): {skipped}")
