"""
Task 6: Adversarial Text Debiasing via Gradient Reversal (v2)

Learns text-invariant emotion readouts from frozen SSL speech representations.
Uses sentence identity as the adversarial target to force the shared projection
to be uninformative about linguistic content.

Training: pooled congruent samples from RAVDESS + SAVEE (English-only baseline)
Testing: EMIS congruent (sanity) + EMIS incongruent (target acc vs proxy acc)

Architecture:
    h_l (frozen, 1024-dim) → MLP (1024→256→256) → z
    z → Emotion Head (256→128→4)
    z → GRL → Sentence Head (256→128→N_sentences)  [adversarially suppressed]

v2 changes from v1:
- Removed StandardScaler (leaks train statistics onto EMIS; MLP handles scale)
- Restricted training emotions to EMIS 4-class subset (angry/happy/neutral/sad)
- GPU by default when available
- Added train/val split with early stopping (10-epoch patience on val emo acc)
- Reports congruent + incongruent EMIS accuracy separately
- Explicit adversary-success metric reported
- Per-seed detail files saved
- Plot layer selection made proportional to n_layers
- Cleaned dead loop structure in load_training_data
"""

import os

SER_SCRATCH = os.environ.get(
    "SER_SCRATCH", os.path.join("/scratch1", os.environ.get("USER", "unknown")))
import json
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

os.makedirs(os.path.join(SER_SCRATCH, "wandb_tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(SER_SCRATCH, "wandb_tmp")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# The four emotion classes that exist in EMIS. Training labels are filtered
# to this set so train-time and test-time label spaces match.
EMIS_EMOTIONS = {"angry", "happy", "neutral", "sad"}

# Map raw dataset labels (lowercased) to the EMIS canonical set.
# Anything not in this map is dropped from training.
EMOTION_LABEL_MAP = {
    # RAVDESS: angry, calm, disgust, fearful, happy, neutral, sad, surprised
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "happiness": "happy",
    "anger": "angry",
    # EmoDB German labels sometimes show up translated; include common variants
    "traurig": "sad",
    "wut": "angry",
    "freude": "happy",
    "froh": "happy",
    "neutral_de": "neutral",
    # Explicitly rejected (mapped to None -> dropped):
    # calm, disgust, fearful, surprised, bored
}


# ============================================================
# GRADIENT REVERSAL LAYER
# ============================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_val = 0.0  # start at 0; Ganin schedule will ramp up

    def set_lambda(self, val):
        self.lambda_val = val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


# ============================================================
# MODEL
# ============================================================

class AdversarialProber(nn.Module):
    """
    Shared MLP + emotion head + sentence adversarial head.

    Note: input normalization is handled by the first Linear layer's weights
    rather than an external scaler, so this module can be applied directly to
    raw frozen hidden states from different distributions (training pool vs EMIS).
    """
    def __init__(self, input_dim=1024, shared_dim=256, n_emotions=4, n_sentences=2,
                 dropout=0.1):
        super().__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim // 2, n_emotions),
        )

        self.grl = GradientReversalLayer()
        self.sentence_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim // 2, n_sentences),
        )

    def forward(self, x):
        z = self.shared_mlp(x)
        emotion_logits = self.emotion_head(z)
        sentence_logits = self.sentence_head(self.grl(z))
        return emotion_logits, sentence_logits

    def set_lambda(self, val):
        self.grl.set_lambda(val)


def ganin_lambda_schedule(progress, lambda_max):
    """Sigmoid schedule from Ganin et al. 2016. progress in [0, 1]."""
    return lambda_max * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


# ============================================================
# SENTENCE ID EXTRACTION
# ============================================================

def get_sentence_id(filename, dataset_name):
    fname = os.path.splitext(os.path.basename(str(filename)))[0]

    if dataset_name == "RAVDESS":
        parts = fname.split("-")
        if len(parts) >= 6:
            return f"ravdess_sent_{parts[4]}"
        return "ravdess_sent_unknown"
    elif dataset_name == "SAVEE":
        parts = fname.split("_")
        if len(parts) >= 2:
            return f"savee_sent_{parts[1]}"
        return "savee_sent_unknown"
    elif dataset_name == "EmoDB":
        if len(fname) >= 5:
            return f"emodb_sent_{fname[2:5]}"
        return "emodb_sent_unknown"
    elif dataset_name == "AESDD":
        return f"aesdd_sent_{fname}"
    elif dataset_name == "MESD":
        parts = fname.split("_")
        if len(parts) >= 4:
            return f"mesd_sent_{parts[3]}"
        return f"mesd_sent_{fname}"
    elif dataset_name.startswith("EMIS"):
        # EMIS filenames: 5-part "{sid}_{text_emo}_{audio_emo}_{esd_spk}_{TTS}.wav"
        # or 6-part "{sid}_{text_emo}_{category}_{audio_emo}_{esd_spk}_{TTS}.wav"
        parts = fname.split("_")
        return f"emis_sent_{parts[0]}" if parts else "emis_sent_unknown"
    else:
        return f"unknown_sent_{fname}"


# ============================================================
# DATA LOADING
# ============================================================

def canonicalize_emotion(raw_label):
    """Map a raw emotion label to the EMIS 4-class subset, or None to drop."""
    if raw_label is None:
        return None
    key = str(raw_label).strip().lower()
    return EMOTION_LABEL_MAP.get(key, None)


def load_training_data(data_dir, datasets, model_name, layer_idx):
    """
    Load congruent training data for ONE model across multiple datasets,
    filtered to the EMIS 4-class emotion subset.

    Returns: X, emotions, sentences, dataset_tags
    """
    all_features = []
    all_emotions = []
    all_sentences = []
    all_dataset_tags = []

    for dataset_name in datasets:
        hs_path = os.path.join(data_dir, f"hidden_states_{model_name}_{dataset_name}.npy")
        lb_path = os.path.join(data_dir, f"labels_{model_name}_{dataset_name}.npy")
        fn_path = os.path.join(data_dir, f"filenames_{model_name}_{dataset_name}.npy")

        if not all(os.path.exists(p) for p in [hs_path, lb_path, fn_path]):
            print(f"    [load] missing files for {model_name}/{dataset_name} - skipping")
            continue

        hidden_states = np.load(hs_path)
        labels_raw = np.load(lb_path, allow_pickle=True)
        filenames = np.load(fn_path, allow_pickle=True)

        X_layer = hidden_states[:, layer_idx, :]

        # Canonicalize emotions, drop samples outside the 4-class subset
        kept_rows = []
        kept_emotions = []
        kept_sentences = []
        for i, raw in enumerate(labels_raw):
            canon = canonicalize_emotion(raw)
            if canon is None:
                continue
            kept_rows.append(i)
            kept_emotions.append(canon)
            kept_sentences.append(get_sentence_id(filenames[i], dataset_name))

        if not kept_rows:
            print(f"    [load] {dataset_name}: no samples matched 4-class subset")
            continue

        kept_rows = np.array(kept_rows)
        all_features.append(X_layer[kept_rows])
        all_emotions.extend(kept_emotions)
        all_sentences.extend(kept_sentences)
        all_dataset_tags.extend([dataset_name] * len(kept_rows))

    if not all_features:
        return None, None, None, None

    X = np.concatenate(all_features, axis=0)
    return X, all_emotions, all_sentences, all_dataset_tags


def load_emis_test_data(data_dir, model_name, layer_idx):
    """Load EMIS features + parsed labels for a given model and layer."""
    hs_path = os.path.join(data_dir, f"hidden_states_{model_name}_EMIS.npy")
    fn_path = os.path.join(data_dir, f"filenames_{model_name}_EMIS.npy")

    if not all(os.path.exists(p) for p in [hs_path, fn_path]):
        return None

    hidden_states = np.load(hs_path)
    filenames = np.load(fn_path, allow_pickle=True)

    X = hidden_states[:, layer_idx, :]

    audio_emotions = []
    text_emotions = []
    congruent_flags = []

    for fn in filenames:
        fname = os.path.splitext(os.path.basename(str(fn)))[0]
        parts = fname.split("_")
        if len(parts) >= 5:
            text_emotions.append(parts[1].lower())
            audio_emotions.append(parts[2].lower())
            congruent_flags.append(parts[1].lower() == parts[2].lower())
        else:
            text_emotions.append("unknown")
            audio_emotions.append("unknown")
            congruent_flags.append(False)

    return {
        "X": X,
        "audio_emotion": audio_emotions,
        "text_emotion": text_emotions,
        "congruent": congruent_flags,
    }


# ============================================================
# TRAINING WITH EARLY STOPPING
# ============================================================

def train_adversarial(X_train, y_emotion, y_sentence,
                      lambda_max, n_epochs=50, lr=1e-3, batch_size=64,
                      val_frac=0.2, patience=10, seed=42, device="cpu",
                      verbose=False):
    """
    Train the adversarial prober with a train/val split and early stopping
    on validation emotion accuracy. Returns the best-val model and training
    trajectory metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Encode labels
    emotion_le = LabelEncoder()
    sentence_le = LabelEncoder()
    y_emo_all = emotion_le.fit_transform(y_emotion)
    y_sent_all = sentence_le.fit_transform(y_sentence)

    n_emotions = len(emotion_le.classes_)
    n_sentences = len(sentence_le.classes_)
    sent_chance = 1.0 / max(n_sentences, 1)

    # Train/val split stratified by emotion
    try:
        tr_idx, val_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=val_frac,
            random_state=seed,
            stratify=y_emo_all,
        )
    except ValueError:
        # Fallback if some class has <2 samples
        tr_idx, val_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=val_frac,
            random_state=seed,
        )

    X_tr = torch.FloatTensor(X_train[tr_idx]).to(device)
    y_emo_tr = torch.LongTensor(y_emo_all[tr_idx]).to(device)
    y_sent_tr = torch.LongTensor(y_sent_all[tr_idx]).to(device)

    X_val = torch.FloatTensor(X_train[val_idx]).to(device)
    y_emo_val = torch.LongTensor(y_emo_all[val_idx]).to(device)
    y_sent_val = torch.LongTensor(y_sent_all[val_idx]).to(device)

    dataset = TensorDataset(X_tr, y_emo_tr, y_sent_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    input_dim = X_train.shape[1]
    model = AdversarialProber(
        input_dim=input_dim,
        shared_dim=256,
        n_emotions=n_emotions,
        n_sentences=n_sentences,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_steps = n_epochs * len(loader)
    step = 0

    best_val_emo = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improvement = 0

    trajectory = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_seen = 0

        for X_batch, y_emo_batch, y_sent_batch in loader:
            progress = step / max(total_steps, 1)
            current_lambda = ganin_lambda_schedule(progress, lambda_max)
            model.set_lambda(current_lambda)

            emo_logits, sent_logits = model(X_batch)
            loss_emo = criterion(emo_logits, y_emo_batch)
            loss_sent = criterion(sent_logits, y_sent_batch)
            loss = loss_emo + loss_sent  # GRL flips sign of sent grad internally

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)
            n_seen += len(X_batch)
            step += 1

        # Validation
        model.eval()
        with torch.no_grad():
            e_logits, s_logits = model(X_val)
            val_emo_acc = (e_logits.argmax(1) == y_emo_val).float().mean().item()
            val_sent_acc = (s_logits.argmax(1) == y_sent_val).float().mean().item()

            e_logits_tr, s_logits_tr = model(X_tr)
            tr_emo_acc = (e_logits_tr.argmax(1) == y_emo_tr).float().mean().item()
            tr_sent_acc = (s_logits_tr.argmax(1) == y_sent_tr).float().mean().item()

        trajectory.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_seen, 1),
            "lambda": current_lambda,
            "train_emo_acc": tr_emo_acc,
            "train_sent_acc": tr_sent_acc,
            "val_emo_acc": val_emo_acc,
            "val_sent_acc": val_sent_acc,
        })

        if val_emo_acc > best_val_emo + 1e-4:
            best_val_emo = val_emo_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"      ep {epoch:02d}: lam={current_lambda:.3f} "
                  f"tr_emo={tr_emo_acc:.3f} tr_sent={tr_sent_acc:.3f} "
                  f"val_emo={val_emo_acc:.3f} val_sent={val_sent_acc:.3f}")

        if epochs_since_improvement >= patience:
            if verbose:
                print(f"      early stop at epoch {epoch} (best={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute adversary-success metric at final model
    model.eval()
    with torch.no_grad():
        _, s_tr = model(X_tr)
        final_tr_sent_acc = (s_tr.argmax(1) == y_sent_tr).float().mean().item()
    # Normalized: 0 = perfect adversary suppression, 1 = no suppression
    if 1.0 - sent_chance > 1e-6:
        adv_suppression = 1.0 - max(
            0.0, (final_tr_sent_acc - sent_chance) / (1.0 - sent_chance)
        )
    else:
        adv_suppression = 0.0

    return {
        "model": model,
        "emotion_le": emotion_le,
        "sentence_le": sentence_le,
        "best_val_emo_acc": best_val_emo,
        "best_epoch": best_epoch,
        "final_train_sent_acc": final_tr_sent_acc,
        "sent_chance": sent_chance,
        "adv_suppression": adv_suppression,  # higher = better adversarial debiasing
        "trajectory": trajectory,
    }


# ============================================================
# EVALUATION ON EMIS (congruent + incongruent)
# ============================================================

def evaluate_on_emis(trained, emis_data, device="cpu"):
    """
    Report emotion-head accuracy on:
      - congruent EMIS samples (sanity check: does emotion recognition work on EMIS distribution?)
      - incongruent EMIS samples, target label (audio emotion)
      - incongruent EMIS samples, proxy label (text emotion)

    Only EMIS samples whose audio_emotion is in the training label space are evaluated.
    """
    model = trained["model"]
    emotion_le = trained["emotion_le"]

    X = emis_data["X"]
    audio_emo = emis_data["audio_emotion"]
    text_emo = emis_data["text_emotion"]
    congruent = emis_data["congruent"]

    known_labels = set(emotion_le.classes_)

    # Valid = audio_emotion is in training label space AND text_emotion too
    # (if text_emotion isn't in training space, proxy_acc is ill-defined)
    def valid_row(i):
        return audio_emo[i] in known_labels and text_emo[i] in known_labels

    cong_idx = [i for i, c in enumerate(congruent) if c and valid_row(i)]
    incong_idx = [i for i, c in enumerate(congruent) if (not c) and valid_row(i)]

    results = {
        "n_congruent": len(cong_idx),
        "n_incongruent": len(incong_idx),
    }

    def _predict(idx_list):
        if not idx_list:
            return None
        X_sub = torch.FloatTensor(X[idx_list]).to(device)
        model.eval()
        with torch.no_grad():
            e_logits, _ = model(X_sub)
            preds = e_logits.argmax(1).cpu().numpy()
        return emotion_le.inverse_transform(preds)

    # Congruent sanity check
    if cong_idx:
        pred_cong = _predict(cong_idx)
        audio_cong = [audio_emo[i] for i in cong_idx]
        results["congruent_target_acc"] = round(
            accuracy_score(audio_cong, pred_cong), 4
        )
    else:
        results["congruent_target_acc"] = None

    # Incongruent: target vs proxy
    if incong_idx:
        pred_incong = _predict(incong_idx)
        audio_incong = [audio_emo[i] for i in incong_idx]
        text_incong = [text_emo[i] for i in incong_idx]

        target_acc = accuracy_score(audio_incong, pred_incong)
        proxy_acc = accuracy_score(text_incong, pred_incong)
        results["incong_target_acc"] = round(target_acc, 4)
        results["incong_proxy_acc"] = round(proxy_acc, 4)
        results["incong_text_bias"] = round(proxy_acc - target_acc, 4)
    else:
        results["incong_target_acc"] = None
        results["incong_proxy_acc"] = None
        results["incong_text_bias"] = None

    return results


# ============================================================
# PER-MODEL SWEEP
# ============================================================

def run_single_model(model_name, data_dir, train_datasets, lambdas,
                     n_seeds=5, n_epochs=50, patience=10, device="cpu",
                     use_wandb=False, out_root="results/TASK6", run_suffix=""):
    """Run adversarial training for one model across all layers and lambdas."""
    print(f"\n{'='*70}")
    print(f"  Task 6: Adversarial Debiasing - {model_name}")
    print(f"  Train: {'+'.join(train_datasets)} | lambdas: {lambdas} | seeds: {n_seeds}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    # Probe layer count and dim
    sample_hs = os.path.join(
        data_dir, f"hidden_states_{model_name}_{train_datasets[0]}.npy"
    )
    if not os.path.exists(sample_hs):
        print(f"  ERROR: {sample_hs} not found")
        return None
    sample_arr = np.load(sample_hs)
    n_layers, input_dim = sample_arr.shape[1], sample_arr.shape[2]
    print(f"  Layers: {n_layers} | Input dim: {input_dim}")

    has_emis = os.path.exists(
        os.path.join(data_dir, f"hidden_states_{model_name}_EMIS.npy")
    )
    if not has_emis:
        print(f"  WARNING: EMIS features not found for {model_name} - "
              f"will skip EMIS evaluation")

    out_csv_dir = os.path.join(out_root, "csv")
    out_seed_dir = os.path.join(out_root, "per_seed")
    out_traj_dir = os.path.join(out_root, "trajectories")
    for d in [out_csv_dir, out_seed_dir, out_traj_dir]:
        os.makedirs(d, exist_ok=True)

    all_agg_results = []
    all_seed_results = []

    for layer_idx in range(n_layers):
        layer_name = "CNN" if layer_idx == 0 else f"L{layer_idx}"

        X_train, emotions, sentences, ds_tags = load_training_data(
            data_dir, train_datasets, model_name, layer_idx
        )
        if X_train is None or len(X_train) < 2 * len(EMIS_EMOTIONS):
            print(f"  [{layer_name}] insufficient training data - skipping")
            continue

        # Filter rare sentences (need >=2 per sentence for stratification)
        sent_counts = Counter(sentences)
        keep = [sent_counts[s] >= 2 for s in sentences]
        X_train = X_train[keep]
        emotions = [e for e, k in zip(emotions, keep) if k]
        sentences = [s for s, k in zip(sentences, keep) if k]
        ds_tags = [d for d, k in zip(ds_tags, keep) if k]

        n_unique_emotions = len(set(emotions))
        n_unique_sentences = len(set(sentences))
        if n_unique_emotions < 2:
            print(f"  [{layer_name}] <2 emotion classes after filter - skipping")
            continue

        # Load EMIS test data once per layer
        emis_data = load_emis_test_data(data_dir, model_name, layer_idx) if has_emis else None

        print(f"\n  [{layer_name}] n_train={len(X_train)} "
              f"emotions={n_unique_emotions} sentences={n_unique_sentences}")

        for lambda_max in lambdas:
            per_seed = []
            for seed in range(n_seeds):
                trained = train_adversarial(
                    X_train, emotions, sentences,
                    lambda_max=lambda_max,
                    n_epochs=n_epochs,
                    patience=patience,
                    seed=seed + 42,
                    device=device,
                    verbose=False,
                )

                seed_record = {
                    "model": model_name,
                    "layer": layer_idx,
                    "layer_name": layer_name,
                    "lambda": lambda_max,
                    "seed": seed,
                    "n_train": len(X_train),
                    "n_sentences": n_unique_sentences,
                    "best_val_emo_acc": round(trained["best_val_emo_acc"], 4),
                    "best_epoch": trained["best_epoch"],
                    "final_train_sent_acc": round(trained["final_train_sent_acc"], 4),
                    "sent_chance": round(trained["sent_chance"], 4),
                    "adv_suppression": round(trained["adv_suppression"], 4),
                }

                if emis_data is not None:
                    emis_res = evaluate_on_emis(trained, emis_data, device=device)
                    seed_record.update({
                        "emis_n_congruent": emis_res["n_congruent"],
                        "emis_n_incongruent": emis_res["n_incongruent"],
                        "emis_congruent_acc": emis_res["congruent_target_acc"],
                        "emis_incong_target_acc": emis_res["incong_target_acc"],
                        "emis_incong_proxy_acc": emis_res["incong_proxy_acc"],
                        "emis_incong_text_bias": emis_res["incong_text_bias"],
                    })

                per_seed.append(seed_record)
                all_seed_results.append(seed_record)

                # Save trajectory for one representative seed (seed=0) at a few lambdas
                if seed == 0 and lambda_max in (0.0, lambdas[len(lambdas) // 2], lambdas[-1]):
                    traj_df = pd.DataFrame(trained["trajectory"])
                    traj_df["model"] = model_name
                    traj_df["layer"] = layer_idx
                    traj_df["lambda"] = lambda_max
                    traj_path = os.path.join(
                        out_traj_dir,
                        f"traj_{model_name}_L{layer_idx:02d}_lam{lambda_max}.csv",
                    )
                    traj_df.to_csv(traj_path, index=False)

            # Aggregate over seeds for this (layer, lambda)
            agg = {
                "model": model_name,
                "layer": layer_idx,
                "layer_name": layer_name,
                "lambda": lambda_max,
                "n_train": len(X_train),
                "n_sentences": n_unique_sentences,
            }
            float_cols = [
                "best_val_emo_acc", "final_train_sent_acc", "adv_suppression",
                "emis_congruent_acc", "emis_incong_target_acc",
                "emis_incong_proxy_acc", "emis_incong_text_bias",
            ]
            for col in float_cols:
                vals = [r.get(col) for r in per_seed if r.get(col) is not None]
                if vals:
                    agg[f"{col}_mean"] = round(float(np.mean(vals)), 4)
                    agg[f"{col}_std"] = round(float(np.std(vals)), 4)
            all_agg_results.append(agg)

            emis_str = ""
            if "emis_incong_target_acc_mean" in agg:
                emis_str = (
                    f" | EMIS cong={agg.get('emis_congruent_acc_mean', 0):.3f} "
                    f"incong_tgt={agg['emis_incong_target_acc_mean']:.3f} "
                    f"incong_prox={agg['emis_incong_proxy_acc_mean']:.3f} "
                    f"bias={agg['emis_incong_text_bias_mean']:+.3f}"
                )
            adv_str = agg.get('adv_suppression_mean', None)
            print(f"    lam={lambda_max:<5} | val_emo={agg['best_val_emo_acc_mean']:.3f} "
                  f"adv_suppress={adv_str}{emis_str}")

    # Save all aggregates
    agg_df = pd.DataFrame(all_agg_results)
    agg_path = os.path.join(out_csv_dir, f"adversarial_{model_name}{run_suffix}.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"\n  Aggregated results: {agg_path}")

    seed_df = pd.DataFrame(all_seed_results)
    seed_path = os.path.join(out_seed_dir, f"adversarial_{model_name}{run_suffix}_per_seed.csv")
    seed_df.to_csv(seed_path, index=False)
    print(f"  Per-seed results:   {seed_path}")

    if use_wandb and HAS_WANDB:
        try:
            run = wandb.init(
                entity="AGSER",
                project="linguistic-agnostic-ser",
                group="task6_adversarial_v2",
                name=f"adversarial_{model_name}{run_suffix}",
                reinit=True,
                config={
                    "model": model_name,
                    "train_datasets": train_datasets,
                    "lambdas": lambdas,
                    "n_seeds": n_seeds,
                    "n_epochs": n_epochs,
                    "patience": patience,
                    "task": "adversarial_debiasing_v2",
                },
            )
            for _, row in agg_df.iterrows():
                log_dict = {
                    k: v for k, v in row.items()
                    if isinstance(v, (int, float, np.integer, np.floating))
                    and not (isinstance(v, float) and np.isnan(v))
                }
                wandb.log(log_dict)
            run.finish()
        except Exception as e:
            print(f"  W&B logging failed: {e}")

    return agg_df


# ============================================================
# PLOTS
# ============================================================

def _representative_layers(n_layers, n_points=7):
    """Pick ~n_points layers spread across depth, including CNN (0) and last."""
    if n_layers <= n_points:
        return list(range(n_layers))
    idx = np.linspace(0, n_layers - 1, n_points, dtype=int)
    return sorted(set(int(i) for i in idx))


def plot_pareto_frontier(results_df, model_name, save_dir="results/TASK6/plots", run_suffix=""):
    """Pareto curve: target_acc vs proxy_acc as lambda varies, per layer."""
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    if "emis_incong_target_acc_mean" not in results_df.columns:
        print("  No EMIS results to plot - skipping Pareto")
        return

    n_layers = int(results_df["layer"].max()) + 1
    layers = _representative_layers(n_layers)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    for layer_idx, color in zip(layers, colors):
        sub = results_df[results_df["layer"] == layer_idx].sort_values("lambda")
        if sub.empty:
            continue
        label = "CNN" if layer_idx == 0 else f"L{layer_idx}"
        ax.plot(
            sub["emis_incong_proxy_acc_mean"],
            sub["emis_incong_target_acc_mean"],
            "o-", color=color, label=label, markersize=6, linewidth=1.5,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"λ={row['lambda']}",
                (row["emis_incong_proxy_acc_mean"], row["emis_incong_target_acc_mean"]),
                fontsize=6, alpha=0.7,
            )

    ax.set_xlabel("Proxy Accuracy (text emotion) - lower is better")
    ax.set_ylabel("Target Accuracy (audio emotion) - higher is better")
    ax.set_title(f"{model_name}: EMIS Pareto (incongruent samples)")
    ax.axhline(0.25, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.25, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"pareto_{model_name}{run_suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Pareto: {path}")


def plot_lambda_effect(results_df, model_name, save_dir="results/TASK6/plots", run_suffix=""):
    """Four panels: val emo acc, adv suppression, EMIS congruent acc, EMIS text bias."""
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    lambdas = sorted(results_df["lambda"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for lam in lambdas:
        sub = results_df[results_df["lambda"] == lam].sort_values("layer")
        if sub.empty:
            continue
        axes[0].plot(sub["layer"], sub["best_val_emo_acc_mean"], "o-",
                     label=f"λ={lam}", markersize=3)
        if "adv_suppression_mean" in sub.columns:
            axes[1].plot(sub["layer"], sub["adv_suppression_mean"], "o-",
                         label=f"λ={lam}", markersize=3)
        if "emis_congruent_acc_mean" in sub.columns:
            axes[2].plot(sub["layer"], sub["emis_congruent_acc_mean"], "o-",
                         label=f"λ={lam}", markersize=3)
        if "emis_incong_text_bias_mean" in sub.columns:
            axes[3].plot(sub["layer"], sub["emis_incong_text_bias_mean"], "o-",
                         label=f"λ={lam}", markersize=3)

    axes[0].set_title("Val Emotion Acc (best epoch)")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Adversary Suppression (1 = fully suppressed)")
    axes[2].set_title("EMIS Congruent Acc (sanity)")
    axes[3].set_title("EMIS Incongruent Text Bias (proxy - target)")
    axes[3].axhline(0, color="gray", linestyle="--", alpha=0.5)

    for a in axes:
        a.set_xlabel("Layer")
        a.legend(fontsize=7)
        a.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name}: Effect of λ Across Layers",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, f"lambda_effect_{model_name}{run_suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Lambda effect: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Task 6: Adversarial Text Debiasing (v2)")
    parser.add_argument("--model", type=str, default="HuBERT")
    parser.add_argument("--data_dir", type=str, default=os.path.join(SER_SCRATCH, "ser_data"))
    parser.add_argument("--train_datasets", type=str, default="RAVDESS,SAVEE")
    parser.add_argument("--lambdas", type=str, default="0,0.01,0.1,1.0,10.0")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="Run HuBERT, wav2vec2, Whisper, WavLM")
    parser.add_argument("--out_root", type=str, default="results/TASK6")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="Appended to W&B run name and CSV basename (to disambiguate re-runs)")
    args = parser.parse_args()

    train_datasets = args.train_datasets.split(",")
    lambdas = [float(x) for x in args.lambdas.split(",")]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  [device] CUDA requested but not available, falling back to CPU")
        device = "cpu"
    print(f"  [device] using {device}")

    models_to_run = ["HuBERT", "wav2vec2", "Whisper", "WavLM"] if args.all else [args.model]

    for model_name in models_to_run:
        try:
            agg_df = run_single_model(
                model_name=model_name,
                data_dir=args.data_dir,
                train_datasets=train_datasets,
                lambdas=lambdas,
                n_seeds=args.n_seeds,
                n_epochs=args.n_epochs,
                patience=args.patience,
                device=device,
                use_wandb=args.wandb,
                out_root=args.out_root,
            run_suffix=args.run_suffix,
)
            if agg_df is not None and len(agg_df) > 0:
                plot_pareto_frontier(agg_df, model_name, run_suffix=args.run_suffix,
                                     save_dir=os.path.join(args.out_root, "plots"))
                plot_lambda_effect(agg_df, model_name, run_suffix=args.run_suffix,
                                   save_dir=os.path.join(args.out_root, "plots"))
        except Exception as e:
            print(f"  ERROR on {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
