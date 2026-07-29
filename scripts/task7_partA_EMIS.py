"""Task 7 Part A: Congruent vs Incongruent SER on EMIS.

Converted from notebooks/Task7_PartA_EMIS_clean.ipynb for Discovery execution.

Pipeline:
1) parse EMIS filenames (dual labels + metadata)
2) extract frozen-encoder hidden states for 4 models
3) train probe on congruent samples, test on incongruent (sentence-stratified k-fold)
4) report target vs proxy accuracy per layer + text-bias gap
5) also split incongruent by explicit/implicit (if metadata available)
6) log to W&B + save CSVs/plots
"""
import os, sys, glob, argparse
# Set TMPDIR before importing wandb so media files go to persistent scratch
os.makedirs("/scratch1/minooahm/wandb_tmp", exist_ok=True)
os.environ.setdefault("TMPDIR", "/scratch1/minooahm/wandb_tmp")

import numpy as np
import pandas as pd
import torch
import librosa
from collections import Counter
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

# Allow import of project's feature extractor
REPO_ROOT = "/scratch1/minooahm/linguistic-agnostic-ser"
sys.path.insert(0, REPO_ROOT)
from src.models.feature_extractors import TransformerExtractor

import wandb

# ============================================================
# CONFIG
# ============================================================
AUDIO_DIR   = "/scratch1/minooahm/ser_data/EMIS_audio"
RESULTS_DIR = "/scratch1/minooahm/linguistic-agnostic-ser/results/TASK7"
FEATURE_DIR = os.path.join(RESULTS_DIR, "features_EMIS")
PLOT_DIR    = os.path.join(RESULTS_DIR, "plots_EMIS")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Model paths match config/config.yaml
MODELS = {
    "HuBERT":   "facebook/hubert-large-ll60k",
    "wav2vec2": "facebook/wav2vec2-large-960h",
    "Whisper":  "openai/whisper-medium",
    "WavLM":    "microsoft/wavlm-large",
}

WANDB_ENTITY  = "AGSER"
WANDB_PROJECT = "linguistic-agnostic-ser"
WANDB_GROUP   = "task7_incongruent_EMIS"

EXPECTED_EMOTIONS = {"angry", "happy", "neutral", "sad"}
N_FOLDS = 5
RANDOM_STATE = 42
EXPECTED_LAYERS = 25
EXPECTED_DIM = 1024


# ============================================================
# EMIS METADATA
# ============================================================
def parse_emis_filename(filename):
    """Parse EMIS filename. Two observed formats:

    5-part (neutral text): {sid}_{text_emo}_{audio_emo}_{esd_spk}_{TTS}.wav
    6-part (emotional text): {sid}_{text_emo}_{category}_{audio_emo}_{esd_spk}_{TTS}.wav

    For 5-part files, category is set to "neutral" (since text_emotion is always neutral
    in this subset and has no explicit/implicit distinction).
    """
    fname = os.path.splitext(os.path.basename(str(filename)))[0]
    parts = fname.split("_")
    if len(parts) == 5:
        sid, text_emo, audio_emo, esd_spk, tts = parts
        category = "neutral"
    elif len(parts) == 6:
        sid, text_emo, category, audio_emo, esd_spk, tts = parts
        category = category.lower()
    else:
        return None
    if text_emo.lower() not in EXPECTED_EMOTIONS or audio_emo.lower() not in EXPECTED_EMOTIONS:
        return None
    return {
        "sentence_id":   sid,
        "text_emotion":  text_emo.lower(),
        "audio_emotion": audio_emo.lower(),
        "esd_speaker":   esd_spk,
        "tts_system":    tts,
        "category":      category,
        "congruent":     text_emo.lower() == audio_emo.lower(),
        "filename":      os.path.basename(str(filename)),
    }


def build_metadata():
    if not os.path.isdir(AUDIO_DIR):
        raise FileNotFoundError(
            f"EMIS directory not found: {AUDIO_DIR}. Download EMIS from "
            "https://ieee-dataport.org/documents/emotionally-incongruent-synthetic-speech-dataset-emis"
        )
    wavs = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")])
    parsed = [parse_emis_filename(f) for f in wavs]
    valid = [p for p in parsed if p]
    print(f"Total WAVs: {len(wavs)} | parsed: {len(valid)} | invalid: {len(wavs) - len(valid)}")
    assert valid, "No valid EMIS samples — check AUDIO_DIR and filename format."

    meta_df = pd.DataFrame(valid)
    print("Congruent/Incongruent:", dict(meta_df["congruent"].value_counts()))
    print("TTS systems:", sorted(meta_df["tts_system"].unique()))
    print("ESD speakers:", sorted(meta_df["esd_speaker"].unique()))
    print(f"Unique sentences: {meta_df['sentence_id'].nunique()}")

    # Explicit/implicit from sentences csv if available
    explicit_map = {}
    candidates = glob.glob(os.path.join(AUDIO_DIR, "*.csv")) + \
                 glob.glob(os.path.join(os.path.dirname(AUDIO_DIR), "*.csv"))
    for c in candidates:
        try:
            df = pd.read_csv(c)
            cols_lower = [col.lower() for col in df.columns]
            has_cat = "category" in cols_lower or "type" in cols_lower
            has_cat_values = any(
                "explicit" in str(v).lower() or "implicit" in str(v).lower()
                for v in df.values.flatten()[:50]
            )
            if has_cat or has_cat_values:
                id_col = next((col for col in df.columns if "id" in col.lower()), None)
                cat_col = next((col for col in df.columns if col.lower() in ("category", "type", "explicit")), None)
                if id_col and cat_col:
                    for _, row in df.iterrows():
                        sid = str(row[id_col])
                        cat = str(row[cat_col]).lower().strip()
                        if "explicit" in cat:
                            explicit_map[sid] = "explicit"
                        elif "implicit" in cat:
                            explicit_map[sid] = "implicit"
                    print(f"Loaded explicit/implicit from {c}: {len(explicit_map)} entries")
                    break
        except Exception:
            continue
    if not explicit_map:
        print("No sentences metadata CSV found — explicit/implicit split unavailable.")

    # Category already set by parser for each file; only overlay external CSV if parser category missing.
    if explicit_map:
        parser_cat = meta_df["category"]
        csv_cat = meta_df["sentence_id"].map(explicit_map)
        meta_df["category"] = parser_cat.where(parser_cat.notna() & (parser_cat != ""), csv_cat).fillna("unknown")
    else:
        meta_df["category"] = meta_df["category"].fillna("unknown")
    print("Category distribution:", dict(meta_df["category"].value_counts()))
    meta_df.to_csv(os.path.join(RESULTS_DIR, "emis_metadata.csv"), index=False)
    return meta_df


# ============================================================
# FEATURE EXTRACTION (using project's TransformerExtractor)
# ============================================================
def extract_features(meta_df):
    ordered = meta_df["filename"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Feature extraction device: {device}")

    for model_name, model_path in MODELS.items():
        hs_path = os.path.join(FEATURE_DIR, f"hidden_states_{model_name}_EMIS.npy")
        fn_path = os.path.join(FEATURE_DIR, f"filenames_{model_name}_EMIS.npy")
        if os.path.exists(hs_path) and os.path.exists(fn_path):
            cached = np.load(hs_path)
            print(f"[{model_name}] cached {cached.shape} — skipping")
            continue

        print(f"\n=== {model_name}: {model_path} ===")
        extractor = TransformerExtractor(model_name=model_path, device=device)

        all_hidden, kept, failures = [], [], []
        for fname in tqdm(ordered, desc=model_name):
            audio_path = os.path.join(AUDIO_DIR, fname)
            try:
                wav, _ = librosa.load(audio_path, sr=16000, mono=True)
                wav_t = torch.from_numpy(wav).float()
                if wav_t.dim() == 1:
                    wav_t = wav_t.unsqueeze(0)
                # list of (batch, hidden_dim) — one entry per layer (incl. embedding/CNN)
                pooled = extractor.extract_from_waveform(wav_t, sample_rate=16000)
                stacked = np.stack([p.squeeze(0) for p in pooled], axis=0)  # (L, D)
                all_hidden.append(stacked)
                kept.append(fname)
            except Exception as e:
                failures.append((fname, str(e)[:80]))

        if failures:
            print(f"  WARNING: {len(failures)} files failed. First 3: {failures[:3]}")

        hs = np.stack(all_hidden, axis=0)
        print(f"  shape: {hs.shape}")
        if hs.shape[1] != EXPECTED_LAYERS or hs.shape[2] != EXPECTED_DIM:
            print(f"  NOTE: expected (N, {EXPECTED_LAYERS}, {EXPECTED_DIM}); got {hs.shape}. Verify model variant.")

        np.save(hs_path, hs)
        np.save(fn_path, np.array(kept, dtype=object))
        print(f"  saved: {hs_path}")

        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll feature extraction complete.")


# ============================================================
# PROBING
# ============================================================
def run_probing_single_model(hidden_states, filenames, meta_df, model_name,
                             restrict_category=None):
    fn_to_idx = {fn: i for i, fn in enumerate(filenames)}
    meta = meta_df[meta_df["filename"].isin(fn_to_idx)].copy()
    meta["row_idx"] = meta["filename"].map(fn_to_idx)
    meta = meta.sort_values("row_idx").reset_index(drop=True)

    n_layers = hidden_states.shape[1]
    all_emotions = sorted(set(meta["audio_emotion"]).union(set(meta["text_emotion"])))
    le = LabelEncoder().fit(all_emotions)
    n_classes = len(all_emotions)
    chance = 1.0 / n_classes

    congruent_mask = meta["congruent"].values
    incongruent_mask = ~congruent_mask
    if restrict_category is not None:
        incongruent_mask = incongruent_mask & (meta["category"].values == restrict_category)

    cong_idx = np.where(congruent_mask)[0]
    incong_idx = np.where(incongruent_mask)[0]
    cong_sent = meta.loc[cong_idx, "sentence_id"].values
    incong_sent = meta.loc[incong_idx, "sentence_id"].values
    unique_sents = np.array(sorted(set(cong_sent).union(set(incong_sent))))

    tag = model_name + (f" [{restrict_category}]" if restrict_category else "")
    print(f"\n[{tag}] congruent train pool: {len(cong_idx)} | "
          f"incongruent test pool: {len(incong_idx)} | "
          f"unique sentences: {len(unique_sents)} | classes: {n_classes} (chance={chance:.3f})")

    if len(incong_idx) == 0:
        print(f"  [{tag}] empty test set — skipping.")
        return None

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sent_splits = list(kf.split(unique_sents))

    fold_records = []
    for fold_i, (tr_sent_i, te_sent_i) in enumerate(sent_splits):
        tr_sents = set(unique_sents[tr_sent_i])
        te_sents = set(unique_sents[te_sent_i])
        tr_rows = cong_idx[np.isin(cong_sent, list(tr_sents))]
        te_rows = incong_idx[np.isin(incong_sent, list(te_sents))]
        if len(tr_rows) < 2 * n_classes or len(te_rows) == 0:
            continue
        y_tr = le.transform(meta.loc[tr_rows, "audio_emotion"].values)
        y_target = le.transform(meta.loc[te_rows, "audio_emotion"].values)
        y_proxy = le.transform(meta.loc[te_rows, "text_emotion"].values)
        if len(np.unique(y_tr)) < 2:
            continue
        for li in range(n_layers):
            X_tr = hidden_states[tr_rows, li, :]
            X_te = hidden_states[te_rows, li, :]
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, n_jobs=1)
            clf.fit(X_tr_s, y_tr)
            pr = clf.predict(X_te_s)
            fold_records.append({
                "fold": fold_i, "layer": li,
                "target_acc": accuracy_score(y_target, pr),
                "proxy_acc":  accuracy_score(y_proxy, pr),
                "target_f1":  f1_score(y_target, pr, average="weighted", zero_division=0),
                "text_bias":  accuracy_score(y_proxy, pr) - accuracy_score(y_target, pr),
                "n_train": len(tr_rows), "n_test": len(te_rows),
            })

    if not fold_records:
        return None
    fold_df = pd.DataFrame(fold_records)
    summary = fold_df.groupby("layer").agg(
        target_acc_mean=("target_acc", "mean"), target_acc_std=("target_acc", "std"),
        proxy_acc_mean=("proxy_acc", "mean"),   proxy_acc_std=("proxy_acc", "std"),
        target_f1_mean=("target_f1", "mean"),
        text_bias_mean=("text_bias", "mean"),   text_bias_std=("text_bias", "std"),
        n_train=("n_train", "mean"), n_test=("n_test", "mean"),
    ).reset_index()
    summary["model"] = model_name
    summary["category"] = restrict_category or "all"
    summary["chance"] = chance

    print(f"  {'Layer':>5} | {'Target':>7}(std) | {'Proxy':>7}(std) | {'Bias':>6}")
    for _, row in summary.iterrows():
        ln = "CNN" if row["layer"] == 0 else f"L{int(row['layer'])}"
        print(f"  {ln:>5} | {row['target_acc_mean']:.3f}({row['target_acc_std']:.3f}) | "
              f"{row['proxy_acc_mean']:.3f}({row['proxy_acc_std']:.3f}) | "
              f"{row['text_bias_mean']:+.3f}")
    return summary


def run_all_probing(meta_df):
    have_category = (meta_df["category"] != "unknown").any()
    categories = [None] + (["explicit", "implicit"] if have_category else [])
    all_results = []
    for model_name in MODELS.keys():
        hs_path = os.path.join(FEATURE_DIR, f"hidden_states_{model_name}_EMIS.npy")
        fn_path = os.path.join(FEATURE_DIR, f"filenames_{model_name}_EMIS.npy")
        if not (os.path.exists(hs_path) and os.path.exists(fn_path)):
            print(f"[{model_name}] features missing — skipping")
            continue
        hs = np.load(hs_path)
        fn = np.load(fn_path, allow_pickle=True).tolist()
        print(f"\n{'='*60}\n  {model_name} — shape: {hs.shape}\n{'='*60}")
        for cat in categories:
            summary = run_probing_single_model(hs, fn, meta_df, model_name, restrict_category=cat)
            if summary is not None:
                all_results.append(summary)
                out_csv = os.path.join(RESULTS_DIR, f"probing_{model_name}_{cat or 'all'}.csv")
                summary.to_csv(out_csv, index=False)
                print(f"  saved: {out_csv}")
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "probing_all_combined.csv"), index=False)
        print(f"\nCombined: {combined.shape[0]} rows")
        return combined
    return None


# ============================================================
# PLOT
# ============================================================
def plot_text_bias(combined_df, category, out_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 9,
    })
    sub = combined_df[combined_df["category"] == category]
    models_present = [m for m in MODELS.keys() if m in sub["model"].unique()]
    if not models_present:
        print(f"No results for category={category}")
        return None
    n = len(models_present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models_present):
        d = sub[sub["model"] == m].sort_values("layer")
        layers = d["layer"].values
        chance = d["chance"].iloc[0]
        ax.plot(layers, d["target_acc_mean"], "o-", color="#2196F3", linewidth=2, markersize=4,
                label="Target (audio emotion)", zorder=3)
        ax.fill_between(layers, d["target_acc_mean"] - d["target_acc_std"],
                        d["target_acc_mean"] + d["target_acc_std"], color="#2196F3", alpha=0.15)
        ax.plot(layers, d["proxy_acc_mean"], "s--", color="#E53935", linewidth=2, markersize=4,
                label="Proxy (text emotion)", zorder=3)
        ax.fill_between(layers, d["proxy_acc_mean"] - d["proxy_acc_std"],
                        d["proxy_acc_mean"] + d["proxy_acc_std"], color="#E53935", alpha=0.15)
        ax.fill_between(layers, d["target_acc_mean"], d["proxy_acc_mean"],
                        where=d["proxy_acc_mean"] > d["target_acc_mean"],
                        color="#FFB74D", alpha=0.25, label="Text-bias region")
        ax.axhline(chance, color="gray", linestyle=":", alpha=0.6, label=f"Chance ({chance:.2f})")
        ax.set_xlabel("Layer"); ax.set_title(m, fontweight="bold")
        ax.grid(True, alpha=0.25); ax.legend(loc="best", framealpha=0.9); ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Accuracy on Incongruent Samples")
    cat_title = {"all": "all incongruent", "explicit": "explicit text only", "implicit": "implicit text only"}[category]
    fig.suptitle(f"Layer-Wise Text-Bias on EMIS ({cat_title})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ============================================================
# W&B
# ============================================================
def log_to_wandb(combined):
    for m in MODELS.keys():
        for cat in combined[combined["model"] == m]["category"].unique():
            d = combined[(combined["model"] == m) & (combined["category"] == cat)].sort_values("layer")
            if d.empty:
                continue
            run = wandb.init(
                entity=WANDB_ENTITY, project=WANDB_PROJECT, group=WANDB_GROUP,
                name=f"EMIS_{m}_{cat}", reinit=True,
                config={"model": m, "model_path": MODELS[m], "dataset": "EMIS",
                        "category": cat, "n_folds": N_FOLDS, "task": "task7_incongruent"},
            )
            for _, row in d.iterrows():
                wandb.log({
                    "layer": int(row["layer"]),
                    "target_acc_mean": row["target_acc_mean"],
                    "target_acc_std":  row["target_acc_std"],
                    "proxy_acc_mean":  row["proxy_acc_mean"],
                    "proxy_acc_std":   row["proxy_acc_std"],
                    "text_bias_mean":  row["text_bias_mean"],
                    "target_f1_mean":  row["target_f1_mean"],
                })
            run.summary["best_target_layer"] = int(d.loc[d["target_acc_mean"].idxmax(), "layer"])
            run.summary["best_target_acc"]   = float(d["target_acc_mean"].max())
            run.summary["max_text_bias"]     = float(d["text_bias_mean"].max())
            run.summary["max_text_bias_layer"] = int(d.loc[d["text_bias_mean"].idxmax(), "layer"])
            run.finish()
    print("W&B logging complete.")


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip feature extraction (use cached)")
    p.add_argument("--skip_wandb", action="store_true",
                   help="Skip W&B logging")
    args = p.parse_args()

    print("Task 7 Part A — EMIS pipeline")
    print(f"  AUDIO_DIR   = {AUDIO_DIR}")
    print(f"  FEATURE_DIR = {FEATURE_DIR}")
    print(f"  RESULTS_DIR = {RESULTS_DIR}")
    print(f"  MODELS      = {list(MODELS.keys())}")

    meta_df = build_metadata()

    if not args.skip_extract:
        extract_features(meta_df)
    else:
        print("Skipping extraction (--skip_extract).")

    combined = run_all_probing(meta_df)
    if combined is None:
        print("No probing results produced — exiting.")
        return

    have_category = (meta_df["category"] != "unknown").any()
    plot_text_bias(combined, "all", "text_bias_all.png")
    if have_category:
        plot_text_bias(combined, "explicit", "text_bias_explicit.png")
        plot_text_bias(combined, "implicit", "text_bias_implicit.png")

    # Best/worst summary table
    all_view = combined[combined["category"] == "all"]
    rows = []
    for m in MODELS.keys():
        d = all_view[all_view["model"] == m]
        if d.empty:
            continue
        best = d.loc[d["target_acc_mean"].idxmax()]
        worst = d.loc[d["target_acc_mean"].idxmin()]
        maxb = d.loc[d["text_bias_mean"].idxmax()]
        rows.append({
            "model": m,
            "best_target_layer":  int(best["layer"]),
            "best_target_acc":    round(best["target_acc_mean"], 4),
            "worst_target_layer": int(worst["layer"]),
            "worst_target_acc":   round(worst["target_acc_mean"], 4),
            "max_bias_layer":     int(maxb["layer"]),
            "max_bias_value":     round(maxb["text_bias_mean"], 4),
            "chance":             round(d["chance"].iloc[0], 4),
        })
    summary_tbl = pd.DataFrame(rows)
    print("\nBest/Worst summary:")
    print(summary_tbl.to_string(index=False))
    summary_tbl.to_csv(os.path.join(RESULTS_DIR, "best_worst_summary.csv"), index=False)

    if not args.skip_wandb:
        log_to_wandb(combined)

    print("\nDone.")


if __name__ == "__main__":
    main()
