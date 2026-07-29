import os

SER_SCRATCH = os.environ.get(
    "SER_SCRATCH", os.path.join("/scratch1", os.environ.get("USER", "unknown")))
"""Task 7 Part B: IEMOCAP annotator disagreement analysis.

Parse per-annotator categorical labels from EmoEvaluation files.
Split clips into:
  - high_agree: all categorical annotators (C-*) gave the same primary label
  - low_agree:  at least one annotator's primary label differed
Run layer-wise 5-fold probing on each subset and compare.
"""
import os, glob, re, json, sys
os.makedirs(os.path.join(SER_SCRATCH, "wandb_tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(SER_SCRATCH, "wandb_tmp")
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import wandb

MODEL = os.environ.get("MODEL", "HuBERT")
IEMOCAP_ROOT = "/project2/msoleyma_1026/IEMOCAP_full_release"
DATA_DIR = os.path.join(SER_SCRATCH, "ser_data")
RESULTS_DIR = os.path.join(SER_SCRATCH, "linguistic-agnostic-ser/results/TASK7")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- parse EmoEvaluation ----------
HEADER_RE = re.compile(r"^\[[\d\.\s\-]+\]\s+(\S+)\s+(\S+)\s+\[")

def parse_eval_files():
    out = {}  # turn -> {'final': str, 'annotators': [primary_label, ...]}
    for sess in range(1, 6):
        eval_dir = os.path.join(IEMOCAP_ROOT, f"Session{sess}", "dialog", "EmoEvaluation")
        for txt in glob.glob(os.path.join(eval_dir, "*.txt")):
            if os.path.basename(txt).startswith("."):
                continue
            current = None
            with open(txt) as f:
                for line in f:
                    line = line.rstrip()
                    m = HEADER_RE.match(line)
                    if m:
                        current = m.group(1)
                        out[current] = {"final": m.group(2).lower(), "annotators": []}
                    elif current and line.startswith("C-"):
                        # "C-E2:\tNeutral;\t()" or "C-F1:\tNeutral; Anger;\t()"
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            labels = [x.strip().lower() for x in parts[1].split(";") if x.strip()]
                            if labels:
                                out[current]["annotators"].append(labels[0])
    return out


# map full-word annotator labels → the short codes used in labels.npy
WORD2CODE = {
    "neutral": "neu", "neutral state": "neu",
    "happiness": "hap", "happy": "hap",
    "anger": "ang", "angry": "ang",
    "sadness": "sad", "sad": "sad",
    "frustration": "fru", "frustrated": "fru",
    "excited": "exc", "excitement": "exc",
    "surprise": "sur", "surprised": "sur",
    "disgust": "dis", "disgusted": "dis",
    "fear": "fea", "fearful": "fea",
    "other": "oth",
}
def norm(lbl):
    return WORD2CODE.get(lbl.lower(), lbl.lower())


# ---------- layer-wise 5-fold probing ----------
def probe_layer(X, y, n_splits=5, seed=42):
    if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return np.nan, np.nan
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in kf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        pr = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pr))
        f1s.append(f1_score(y[te], pr, average="weighted"))
    return float(np.mean(accs)), float(np.mean(f1s))


# ---------- main ----------
def main():
    print(f"Model: {MODEL}")
    print("Parsing EmoEvaluation...")
    eval_data = parse_eval_files()
    print(f"  parsed {len(eval_data)} turns")

    # Compute agreement per turn
    agreement = {}
    for turn, d in eval_data.items():
        annos = [norm(a) for a in d["annotators"]]
        if len(annos) < 2:
            continue
        uniq = set(annos)
        agreement[turn] = {
            "final": d["final"],
            "n_annotators": len(annos),
            "n_unique": len(uniq),
            "all_agree": len(uniq) == 1,
            "annos": annos,
        }
    print(f"  turns with ≥2 annotators: {len(agreement)}")
    all_agree_n = sum(1 for v in agreement.values() if v["all_agree"])
    print(f"  all_agree: {all_agree_n}  |  disagree: {len(agreement) - all_agree_n}")

    # Load npy
    hs = np.load(os.path.join(DATA_DIR, f"hidden_states_{MODEL}_IEMOCAP.npy"))
    lb = np.load(os.path.join(DATA_DIR, f"labels_{MODEL}_IEMOCAP.npy"), allow_pickle=True)
    fn = np.load(os.path.join(DATA_DIR, f"filenames_{MODEL}_IEMOCAP.npy"), allow_pickle=True)
    print(f"  hidden_states: {hs.shape}, labels: {len(lb)}, filenames: {len(fn)}")
    print(f"  label set: {sorted(set(lb.tolist()))}")

    # Match turns
    turn_names = [str(f).replace(".wav", "") for f in fn]
    idx_high, idx_low = [], []
    missing = 0
    for i, t in enumerate(turn_names):
        if t not in agreement:
            missing += 1
            continue
        (idx_high if agreement[t]["all_agree"] else idx_low).append(i)
    print(f"  matched: high={len(idx_high)}, low={len(idx_low)}, missing={missing}")

    # Class distribution in each split
    def class_dist(idxs):
        return dict(Counter(lb[idxs].tolist()))
    print(f"  high-agree class dist: {class_dist(idx_high)}")
    print(f"  low-agree class dist:  {class_dist(idx_low)}")

    # Encode labels
    le = LabelEncoder()
    y_all = le.fit_transform(lb)
    n_layers = hs.shape[1]  # 25
    layer_names = ["CNN"] + [f"Layer {i}" for i in range(1, n_layers)]

    # W&B
    run = wandb.init(
        entity="AGSER", project="linguistic-agnostic-ser",
        name=f"{MODEL}_IEMOCAP_agreement",
        group="task7_incongruent",
        job_type="agreement_probing",
        tags=[MODEL, "IEMOCAP", "task7", "agreement"],
        config={
            "model": MODEL, "dataset": "IEMOCAP",
            "n_high_agree": len(idx_high), "n_low_agree": len(idx_low),
            "n_classes": len(le.classes_), "classes": le.classes_.tolist(),
        },
        reinit=True,
    )

    rows = []
    for li in range(n_layers):
        Xh = hs[idx_high, li]
        yh = y_all[idx_high]
        Xl = hs[idx_low, li]
        yl = y_all[idx_low]
        acc_h, f1_h = probe_layer(Xh, yh)
        acc_l, f1_l = probe_layer(Xl, yl)
        # full set for reference
        Xa = hs[:, li]
        acc_a, f1_a = probe_layer(Xa, y_all)
        rows.append({
            "Layer": layer_names[li], "Layer_Num": li,
            "HighAgree_Acc": acc_h, "HighAgree_F1": f1_h,
            "LowAgree_Acc": acc_l, "LowAgree_F1": f1_l,
            "All_Acc": acc_a, "All_F1": f1_a,
            "HighAgree_N": len(idx_high), "LowAgree_N": len(idx_low),
        })
        wandb.log({
            "layer": li,
            "high_agree/accuracy": acc_h, "high_agree/f1": f1_h,
            "low_agree/accuracy": acc_l,  "low_agree/f1": f1_l,
            "all/accuracy": acc_a,        "all/f1": f1_a,
            "gap/acc_high_minus_low": acc_h - acc_l,
        }, step=li)
        print(f"  {layer_names[li]:<10s} | high {acc_h:.4f}  low {acc_l:.4f}  all {acc_a:.4f}  gap {acc_h-acc_l:+.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"agreement_probing_{MODEL}_IEMOCAP.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # summary
    best_h = df["HighAgree_Acc"].idxmax()
    best_l = df["LowAgree_Acc"].idxmax()
    summary = {
        "best_high_layer": df.loc[best_h, "Layer"],
        "best_high_acc": float(df.loc[best_h, "HighAgree_Acc"]),
        "best_low_layer": df.loc[best_l, "Layer"],
        "best_low_acc": float(df.loc[best_l, "LowAgree_Acc"]),
        "avg_gap_high_minus_low": float((df["HighAgree_Acc"] - df["LowAgree_Acc"]).mean()),
    }
    for k, v in summary.items():
        run.summary[k] = v
    wandb.log({"per_layer_table": wandb.Table(dataframe=df)})

    art = wandb.Artifact(f"agreement_csv_{MODEL}_IEMOCAP", type="task7_results")
    art.add_file(csv_path)
    run.log_artifact(art)

    # also save per-turn agreement map (once; not per model) — only on HuBERT run
    if MODEL == "HuBERT":
        agr_path = os.path.join(RESULTS_DIR, "iemocap_agreement_map.json")
        with open(agr_path, "w") as f:
            json.dump({k: {"final": v["final"], "n_annotators": v["n_annotators"],
                           "n_unique": v["n_unique"], "all_agree": v["all_agree"],
                           "annos": v["annos"]} for k, v in agreement.items()}, f, indent=2)
        print(f"Saved {agr_path}")

    run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
