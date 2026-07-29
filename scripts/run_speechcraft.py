"""
EXP20 / Speechcraft: Large-scale compiled benchmark stress-testing
cross-lingual emotion patterns across pooled datasets.

For one model (passed via --model_name) this script:
  1. Loads the four language groups
       - English  : RAVDESS + IEMOCAP + SAVEE
       - German   : EmoDB
       - Greek    : AESDD
       - Spanish  : MESD
  2. Normalises every label to the common 4-class set
       {happy, sad, fear, disgust}
  3. Runs the transformer once and caches per-layer mean-pooled
     hidden states for each language group.
  4. Computes two probe sweeps for every layer:
       - "pooled"   : 5-fold stratified CV on the union of all four
                      language groups (multilingual baseline).
       - "LOLO_<L>" : leave-one-language-out, train on the other
                      three groups and test on language L (this is
                      the part Henry didn't get to).
  5. Writes per-layer CSV files into
        results/EXP20/csv/                    (repo)
        /scratch1/$USER/ser-experiments/EXP20/csv/   (mirror)
     plus a tidy long-format summary CSV per model.
"""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_ingestion.loader import (
    load_aesdd,
    load_emodb,
    load_iemocap,
    load_mesd,
    load_ravdess,
    load_savee,
)
from src.models.feature_extractors import TransformerExtractor
from src.utils.config_parser import load_config


LABEL_NORM = {
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "fear": "fear",
    "fearful": "fear",
    "disgust": "disgust",
}
COMMON_LABELS = ["happy", "sad", "fear", "disgust"]

LANGUAGE_GROUPS = {
    "English": [("RAVDESS", load_ravdess), ("IEMOCAP", load_iemocap), ("SAVEE", load_savee)],
    "German":  [("EmoDB",  load_emodb)],
    "Greek":   [("AESDD",  load_aesdd)],
    "Spanish": [("MESD",   load_mesd)],
}


def clean_model_label(model_path: str) -> str:
    name = model_path.lower()
    if "wav2vec2" in name:
        return "wav2vec2"
    if "hubert" in name:
        return "HuBERT"
    if "whisper" in name:
        return "Whisper"
    if "wavlm" in name:
        return "WavLM"
    if "mert" in name:
        return "MERT"
    if "w2v-bert" in name or "w2v_bert" in name:
        return "w2v-BERT"
    return model_path.split("/")[-1]


class InMemoryDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "waveform": torch.from_numpy(item["audio"]),
            "label": item["label"],
            "file": item.get("file", f"sample_{idx}.wav"),
            "lang": item["lang"],
            "dataset": item["dataset"],
        }


def collate(batch):
    wavs = [b["waveform"] for b in batch]
    max_len = max(w.shape[0] for w in wavs)
    padded = []
    for w in wavs:
        if w.shape[0] < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
        padded.append(w)
    return {
        "waveform": torch.stack(padded),
        "labels": [b["label"] for b in batch],
        "files":  [b["file"] for b in batch],
        "langs":  [b["lang"] for b in batch],
        "datasets": [b["dataset"] for b in batch],
    }


def load_language_pool(data_root: str, sample_rate: int):
    """Returns a list of items with normalised labels, only those in COMMON_LABELS."""
    pool = []
    per_group_counts = {}
    for lang, datasets in LANGUAGE_GROUPS.items():
        kept_for_lang = 0
        for ds_name, loader in datasets:
            ds_dir = os.path.join(data_root, ds_name)
            if not os.path.isdir(ds_dir):
                print(f"[warn] {ds_name} not found at {ds_dir}, skipping.")
                continue
            print(f"[load] {ds_name} ({lang}) from {ds_dir}")
            ds = loader(ds_dir, sample_rate=sample_rate)
            for item in ds:
                norm = LABEL_NORM.get(str(item["label"]).lower())
                if norm is None:
                    continue
                pool.append({
                    "audio": item["audio"],
                    "label": norm,
                    "file": item.get("file", ""),
                    "lang": lang,
                    "dataset": ds_name,
                })
                kept_for_lang += 1
        per_group_counts[lang] = kept_for_lang
    print("[counts] per language:", per_group_counts)
    print("[counts] per (lang,label):", Counter((x["lang"], x["label"]) for x in pool))
    return pool


def extract_hidden_states(items, model_path: str, batch_size: int, sample_rate: int):
    extractor = TransformerExtractor(model_name=model_path)
    loader = DataLoader(InMemoryDataset(items), batch_size=batch_size, shuffle=False, collate_fn=collate)
    bucket = []
    labels, langs, files, datasets = [], [], [], []
    for batch in tqdm(loader, desc=f"extract {clean_model_label(model_path)}"):
        h = extractor.extract_from_waveform(batch["waveform"], sample_rate=sample_rate)
        # h is a list of length n_layers; each entry shape (batch, hidden_dim)
        stacked = np.stack(h, axis=1)  # (batch, n_layers, hidden_dim)
        bucket.extend(stacked)
        labels.extend(batch["labels"])
        langs.extend(batch["langs"])
        files.extend(batch["files"])
        datasets.extend(batch["datasets"])
    hidden = np.array(bucket)  # (N, n_layers, hidden_dim)
    return hidden, np.array(labels), np.array(langs), np.array(files), np.array(datasets)


def probe_pooled(hidden, labels, n_splits=5, seed=42):
    """Stratified K-fold over the whole pool, per layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_layers = hidden.shape[1]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for layer in range(n_layers):
        X = hidden[:, layer, :]
        accs, f1s = [], []
        for tr, te in skf.split(X, labels):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
            clf.fit(X_tr, labels[tr])
            preds = clf.predict(X_te)
            accs.append(accuracy_score(labels[te], preds))
            f1s.append(f1_score(labels[te], preds, average="weighted"))
        rows.append({
            "Layer": "CNN" if layer == 0 else f"Layer {layer}",
            "Layer_Num": layer,
            "Accuracy": round(float(np.mean(accs)), 4),
            "Std_Accuracy": round(float(np.std(accs)), 4),
            "Weighted_F1": round(float(np.mean(f1s)), 4),
            "split": "pooled",
            "n_samples": int(len(labels)),
        })
        print(f"  pooled layer {layer:>2d} | acc={np.mean(accs):.4f} | f1={np.mean(f1s):.4f}")
    return pd.DataFrame(rows)


def probe_lolo(hidden, labels, langs, seed=42):
    """Leave-one-language-out cross-lingual probe, per layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    n_layers = hidden.shape[1]
    unique_langs = sorted(set(langs.tolist()))
    rows = []
    for held in unique_langs:
        train_mask = langs != held
        test_mask = langs == held
        # Need every common class present in TRAIN for fairness; if not, drop
        if len(set(labels[train_mask])) < len(COMMON_LABELS):
            print(f"  [skip] {held}: train pool missing some classes")
            continue
        for layer in range(n_layers):
            X_tr = hidden[train_mask, layer, :]
            X_te = hidden[test_mask, layer, :]
            y_tr = labels[train_mask]
            y_te = labels[test_mask]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
            clf.fit(X_tr_s, y_tr)
            preds = clf.predict(X_te_s)
            acc = accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds, average="weighted")
            rows.append({
                "Layer": "CNN" if layer == 0 else f"Layer {layer}",
                "Layer_Num": layer,
                "Accuracy": round(float(acc), 4),
                "Weighted_F1": round(float(f1), 4),
                "split": f"LOLO_{held}",
                "test_lang": held,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            })
            print(f"  LOLO test={held} layer {layer:>2d} | acc={acc:.4f} | f1={f1:.4f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="HF model id (e.g. facebook/wav2vec2-large-960h)")
    ap.add_argument("--data_root", default=None, help="Parent directory containing dataset subdirs")
    ap.add_argument("--exp_id", default="20")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--save_embeddings", action="store_true", help="Persist hidden states (large)")
    args = ap.parse_args()

    config = load_config()
    sample_rate = config["training"]["sample_rate"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    user = os.environ.get("USER", "kelidari")
    data_root = args.data_root or f"/scratch1/{user}/ser_data"

    model_label = clean_model_label(args.model_name)
    print(f"=== Speechcraft EXP{args.exp_id} :: model={model_label} ({args.model_name}) ===")
    t0 = time.time()

    items = load_language_pool(data_root, sample_rate)
    if not items:
        raise SystemExit("Empty language pool — verify dataset paths.")

    hidden, labels, langs, files, datasets = extract_hidden_states(
        items, args.model_name, batch_size, sample_rate
    )
    print(f"[done] hidden states: {hidden.shape}, took {time.time() - t0:.1f}s")

    # Output dirs
    repo_base = os.path.join("results", f"EXP{args.exp_id}")
    scratch_base = f"/scratch1/{user}/ser-experiments/EXP{args.exp_id}"
    for base in (repo_base, scratch_base):
        os.makedirs(os.path.join(base, "csv"), exist_ok=True)
        os.makedirs(os.path.join(base, "summary"), exist_ok=True)

    pooled_df = probe_pooled(hidden, labels, n_splits=5, seed=config["training"]["random_state"])
    lolo_df = probe_lolo(hidden, labels, langs, seed=config["training"]["random_state"])

    pooled_path = os.path.join(repo_base, "csv", f"results_{model_label}_pooled.csv")
    lolo_path = os.path.join(repo_base, "csv", f"results_{model_label}_LOLO.csv")
    pooled_df.to_csv(pooled_path, index=False)
    lolo_df.to_csv(lolo_path, index=False)
    pooled_df.to_csv(os.path.join(scratch_base, "csv", os.path.basename(pooled_path)), index=False)
    lolo_df.to_csv(os.path.join(scratch_base, "csv", os.path.basename(lolo_path)), index=False)

    # Long-format combined summary
    pooled_long = pooled_df.assign(model=model_label)
    lolo_long = lolo_df.assign(model=model_label)
    summary = pd.concat([pooled_long, lolo_long], ignore_index=True, sort=False)
    summary_path = os.path.join(repo_base, "summary", f"summary_{model_label}.csv")
    summary.to_csv(summary_path, index=False)
    summary.to_csv(os.path.join(scratch_base, "summary", os.path.basename(summary_path)), index=False)

    # Manifest
    manifest = {
        "model": model_label,
        "model_path": args.model_name,
        "n_total_samples": int(hidden.shape[0]),
        "n_layers": int(hidden.shape[1]),
        "language_counts": {k: int((langs == k).sum()) for k in sorted(set(langs.tolist()))},
        "label_counts": {k: int((labels == k).sum()) for k in COMMON_LABELS},
        "wallclock_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(repo_base, "summary", f"manifest_{model_label}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(scratch_base, "summary", f"manifest_{model_label}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if args.save_embeddings:
        emb_dir = os.path.join(scratch_base, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(emb_dir, f"hidden_{model_label}.npz"),
            hidden=hidden, labels=labels, langs=langs, files=files, datasets=datasets,
        )
        print(f"[done] embeddings saved to {emb_dir}")

    print(f"[done] EXP{args.exp_id} model={model_label} CSVs in {repo_base}/csv (mirrored to {scratch_base})")
    print(f"[done] wallclock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
