"""
Task 4: Speaker-Independent Evaluation (Leave-One-Speaker-Out CV)
Replaces 5-fold stratified CV with LOSO to check for speaker identity leakage.

Usage:
  python scripts/task4_loso.py --model HuBERT --dataset SAVEE --data_dir /scratch1/$USER/ser_data
  python scripts/task4_loso.py --all --data_dir /scratch1/$USER/ser_data
"""

import os

SER_SCRATCH = os.environ.get(
    "SER_SCRATCH", os.path.join("/scratch1", os.environ.get("USER", "unknown")))
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

HAS_WANDB = False  # Set to True if you want W&B logging


# ============================================================
# SPEAKER ID EXTRACTION
# ============================================================

def get_speaker_id(filename, dataset_name):
    """Extract speaker ID from filename based on dataset conventions."""
    fname = os.path.basename(str(filename))
    
    if dataset_name == "RAVDESS":
        # Format: 03-01-06-01-02-01-12.wav → actor = "12" (last segment)
        parts = os.path.splitext(fname)[0].split("-")
        if len(parts) >= 7:
            return f"actor_{parts[6]}"
        return f"actor_{parts[-1]}"
    
    elif dataset_name == "EmoDB":
        # Format: 03a01Fa.wav → speaker = "03"
        return f"speaker_{fname[:2]}"
    
    elif dataset_name == "IEMOCAP":
        # Format: Ses01F_impro01_F000.wav or Ses01M_script01_1_M000.wav
        # Session + gender identifies unique speaker
        if "Ses" in fname:
            parts = fname.split("_")
            return parts[0]  # e.g., "Ses01F", "Ses01M"
        return fname.split("_")[0]
    
    elif dataset_name == "SAVEE":
        # Format: DC_a01.wav → speaker = "DC"
        return fname.split("_")[0]
    
    elif dataset_name == "AESDD":
        # Try common patterns for AESDD
        if "/" in str(filename):
            parts = str(filename).split("/")
            for p in parts:
                if p.startswith("s") and len(p) <= 3:
                    return p
        return fname.split("_")[0] if "_" in fname else fname[:2]
    
    elif dataset_name == "MESD":
        # Format: <Emotion>_<Sex>_<Letter>_<word>.wav -> speaker = <Sex>_<Letter> (e.g., M_A, F_B)
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
        return fname.split("_")[0]
    
    elif dataset_name == "MSP-Podcast":
        # Format: MSP-PODCAST_0001_0008.wav -> speaker = 0001 (podcast/speaker index)
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) >= 2:
            return f"speaker_{parts[1]}"
        return fname.split("-")[0] if "-" in fname else fname[:8]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================
# ORIGINAL 5-FOLD CV (for comparison)
# ============================================================

def original_5fold_layer(X, y):
    """Run original 5-fold stratified CV on one layer for comparison."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    scaler = StandardScaler()
    
    for train_idx, test_idx in kfold.split(X, y):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_test)
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average='weighted'))
    
    return np.mean(accs), np.mean(f1s)


# ============================================================
# LOSO EVALUATION
# ============================================================

def loso_evaluate_layer(X, y, speaker_ids, layer_idx, layer_name):
    """
    Leave-One-Speaker-Out CV for a single layer.
    Returns accuracy, std, f1, and per-speaker results.
    """
    unique_speakers = np.unique(speaker_ids)
    n_speakers = len(unique_speakers)
    
    speaker_accs = {}
    speaker_f1s = {}
    speaker_counts = {}
    all_preds = np.zeros_like(y, dtype=int)
    all_true = np.zeros_like(y, dtype=int)
    
    scaler = StandardScaler()
    
    for test_speaker in unique_speakers:
        test_mask = speaker_ids == test_speaker
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            continue
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
        speaker_accs[str(test_speaker)] = round(acc, 4)
        speaker_f1s[str(test_speaker)] = round(f1, 4)
        speaker_counts[str(test_speaker)] = int(np.sum(test_mask))
        
        all_preds[test_mask] = preds
        all_true[test_mask] = y_test
    
    global_acc = accuracy_score(all_true, all_preds)
    global_f1 = f1_score(all_true, all_preds, average='weighted')
    acc_values = list(speaker_accs.values())
    
    return {
        'Layer': layer_name,
        'Layer_Num': layer_idx,
        'LOSO_Accuracy': round(global_acc, 4),
        'LOSO_Std': round(np.std(acc_values), 4),
        'LOSO_F1': round(global_f1, 4),
        'LOSO_Min_Speaker_Acc': round(min(acc_values), 4),
        'LOSO_Max_Speaker_Acc': round(max(acc_values), 4),
        'Per_Speaker_Accs': speaker_accs,
        'Per_Speaker_F1s': speaker_f1s,
        'Per_Speaker_Counts': speaker_counts,
        'Num_Speakers': n_speakers
    }


def run_loso_all_layers(hidden_states, labels, filenames, dataset_name, model_name):
    """Run LOSO + 5-fold comparison across all layers."""
    num_layers = hidden_states.shape[1]
    
    # Convert labels to numeric
    if isinstance(labels[0], str) or isinstance(labels[0], np.str_):
        le = LabelEncoder()
        y = le.fit_transform(labels)
        label_map = dict(zip(le.classes_.astype(str), le.transform(le.classes_).tolist()))
        print(f"  Label mapping: {label_map}")
    else:
        y = np.array(labels)
        label_map = None
    
    # Extract speaker IDs
    speaker_ids = np.array([get_speaker_id(fn, dataset_name) for fn in filenames])
    unique_speakers = np.unique(speaker_ids)
    print(f"  Found {len(unique_speakers)} unique speakers: {list(unique_speakers)}")
    print(f"  Samples per speaker:")
    for sp in unique_speakers:
        print(f"    {sp}: {np.sum(speaker_ids == sp)} samples")
    
    # Probe each layer
    results = []
    per_speaker_details = []
    
    print(f"\n  {'Layer':<10} | {'5-Fold Acc':>10} | {'LOSO Acc':>10} | {'Drop':>8} | {'LOSO F1':>8} | {'Min Spk':>8} | {'Max Spk':>8}")
    print(f"  {'-'*76}")
    
    for layer_idx in range(num_layers):
        X = hidden_states[:, layer_idx, :]
        layer_name = 'CNN' if layer_idx == 0 else f'Layer {layer_idx}'
        
        # Original 5-fold
        fold5_acc, fold5_f1 = original_5fold_layer(X, y)
        
        # LOSO
        loso_result = loso_evaluate_layer(X, y, speaker_ids, layer_idx, layer_name)
        
        # Compute drop
        drop = fold5_acc - loso_result['LOSO_Accuracy']
        
        # Add 5-fold results
        loso_result['FiveFold_Accuracy'] = round(fold5_acc, 4)
        loso_result['FiveFold_F1'] = round(fold5_f1, 4)
        loso_result['Accuracy_Drop'] = round(drop, 4)
        
        results.append(loso_result)
        
        # Save per-speaker detail
        per_speaker_details.append({
            'Layer': layer_name,
            'Layer_Num': layer_idx,
            'Per_Speaker_Accs': loso_result['Per_Speaker_Accs'],
            'Per_Speaker_F1s': loso_result['Per_Speaker_F1s'],
            'Per_Speaker_Counts': loso_result['Per_Speaker_Counts']
        })
        
        print(f"  {layer_name:<10} | {fold5_acc:>10.4f} | {loso_result['LOSO_Accuracy']:>10.4f} | "
              f"{drop:>+8.4f} | {loso_result['LOSO_F1']:>8.4f} | "
              f"{loso_result['LOSO_Min_Speaker_Acc']:>8.4f} | {loso_result['LOSO_Max_Speaker_Acc']:>8.4f}")
    
    # Summary DataFrame
    results_df = pd.DataFrame([{
        'Layer': r['Layer'],
        'Layer_Num': r['Layer_Num'],
        'FiveFold_Accuracy': r['FiveFold_Accuracy'],
        'FiveFold_F1': r['FiveFold_F1'],
        'LOSO_Accuracy': r['LOSO_Accuracy'],
        'LOSO_Std': r['LOSO_Std'],
        'LOSO_F1': r['LOSO_F1'],
        'Accuracy_Drop': r['Accuracy_Drop'],
        'LOSO_Min_Speaker_Acc': r['LOSO_Min_Speaker_Acc'],
        'LOSO_Max_Speaker_Acc': r['LOSO_Max_Speaker_Acc'],
        'Num_Speakers': r['Num_Speakers']
    } for r in results])
    
    # Print summary
    best_5fold = results_df.loc[results_df['FiveFold_Accuracy'].idxmax()]
    best_loso = results_df.loc[results_df['LOSO_Accuracy'].idxmax()]
    
    print(f"\n  ========== SUMMARY ==========")
    print(f"  Best 5-Fold:  {best_5fold['Layer']} (Acc={best_5fold['FiveFold_Accuracy']:.4f})")
    print(f"  Best LOSO:    {best_loso['Layer']} (Acc={best_loso['LOSO_Accuracy']:.4f})")
    print(f"  Avg Drop:     {results_df['Accuracy_Drop'].mean():+.4f}")
    print(f"  Max Drop:     {results_df['Accuracy_Drop'].max():+.4f} at {results_df.loc[results_df['Accuracy_Drop'].idxmax(), 'Layer']}")
    
    return results_df, per_speaker_details


# ============================================================
# MAIN
# ============================================================

def run_single(model_name, dataset_name, data_dir, min_speaker_samples=0):
    """Run LOSO for one model-dataset combination."""
    print(f"\n{'='*60}")
    print(f"  LOSO Evaluation: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Load pre-extracted features
    hs_path = os.path.join(data_dir, f"hidden_states_{model_name}_{dataset_name}.npy")
    lb_path = os.path.join(data_dir, f"labels_{model_name}_{dataset_name}.npy")
    fn_path = os.path.join(data_dir, f"filenames_{model_name}_{dataset_name}.npy")
    
    for path, name in [(hs_path, "hidden_states"), (lb_path, "labels"), (fn_path, "filenames")]:
        if not os.path.exists(path):
            print(f"  SKIPPING: {name} file not found: {path}")
            return None, None
    
    hidden_states = np.load(hs_path)
    labels = np.load(lb_path, allow_pickle=True)
    filenames = np.load(fn_path, allow_pickle=True)
    
    print(f"  Loaded: hidden_states={hidden_states.shape}, labels={len(labels)}, filenames={len(filenames)}")
    print(f"  Sample filenames: {filenames[:3]}")

    if min_speaker_samples > 0:
        spk_ids = np.array([get_speaker_id(fn, dataset_name) for fn in filenames])
        from collections import Counter
        counts = Counter(spk_ids.tolist())
        keep_speakers = {s for s, c_ in counts.items() if c_ >= min_speaker_samples}
        keep_mask = np.array([s in keep_speakers for s in spk_ids])
        n_before = len(filenames)
        hidden_states = hidden_states[keep_mask]
        labels = labels[keep_mask]
        filenames = filenames[keep_mask]
        print(f"  Speaker filter (min>={min_speaker_samples}): "
              f"kept {len(keep_speakers)}/{len(counts)} speakers, "
              f"{keep_mask.sum()}/{n_before} samples")
    
    results_df, per_speaker_details = run_loso_all_layers(
        hidden_states, labels, filenames, dataset_name, model_name
    )
    
    # Save results
    out_dir = os.path.join("results", "LOSO", "csv")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, f"loso_{model_name}_{dataset_name}.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Save per-speaker details as JSON
    detail_dir = os.path.join("results", "LOSO", "per_speaker")
    os.makedirs(detail_dir, exist_ok=True)
    detail_path = os.path.join(detail_dir, f"per_speaker_{model_name}_{dataset_name}.json")
    with open(detail_path, 'w') as f:
        json.dump(per_speaker_details, f, indent=2)
    
    print(f"\n  CSV saved to {csv_path}")
    print(f"  Per-speaker details saved to {detail_path}")
    
    return results_df, per_speaker_details


def main():
    parser = argparse.ArgumentParser(description="Task 4: LOSO Speaker-Independent Evaluation")
    parser.add_argument("--model", type=str, default="HuBERT")
    parser.add_argument("--dataset", type=str, default="SAVEE")
    parser.add_argument("--data_dir", type=str, default=os.path.join(SER_SCRATCH, "ser_data"))
    parser.add_argument("--all", action="store_true", help="Run all available model-dataset combinations")
    parser.add_argument("--min_speaker_samples", type=int, default=0, help="Drop speakers with fewer than N samples (0=keep all)")
    args = parser.parse_args()
    
    if args.all:
        models = ["HuBERT", "wav2vec2", "Whisper", "WavLM", "MERT", "w2v-BERT"]
        datasets = ["RAVDESS", "EmoDB", "IEMOCAP", "SAVEE", "AESDD", "MESD", "MSP-Podcast"]
        
        all_results = []
        failed = []
        for model in models:
            for dataset in datasets:
                try:
                    result_df, _ = run_single(model, dataset, args.data_dir, args.min_speaker_samples)
                    if result_df is not None:
                        result_df['Model'] = model
                        result_df['Dataset'] = dataset
                        all_results.append(result_df)
                except Exception as e:
                    print(f"\n  ERROR: {model} on {dataset} failed: {e}")
                    print(f"  Skipping and continuing...\n")
                    failed.append(f"{model}_{dataset}: {e}")
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            out_path = os.path.join("results", "LOSO", "csv", "loso_all_combined.csv")
            combined.to_csv(out_path, index=False)
            print(f"\n{'='*60}")
            print(f"  ALL RESULTS COMBINED: {out_path}")
            print(f"  Total experiments: {len(all_results)}")
            print(f"{'='*60}")
            
            # Print summary comparison table
            print(f"\n  {'Model':<10} | {'Dataset':<12} | {'5-Fold Best':>11} | {'LOSO Best':>10} | {'Drop':>8}")
            print(f"  {'-'*60}")
            for df in all_results:
                model = df['Model'].iloc[0]
                dataset = df['Dataset'].iloc[0]
                best5 = df['FiveFold_Accuracy'].max()
                bestL = df['LOSO_Accuracy'].max()
                print(f"  {model:<10} | {dataset:<12} | {best5:>11.4f} | {bestL:>10.4f} | {best5-bestL:>+8.4f}")
        
        if failed:
            print(f"\n  FAILED EXPERIMENTS ({len(failed)}):")
            for f in failed:
                print(f"    - {f}")
    else:
        run_single(args.model, args.dataset, args.data_dir, args.min_speaker_samples)


if __name__ == "__main__":
    main()
