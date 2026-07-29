"""Task 6 V3: CCA subspace projection on EMIS for text-bias removal.

Pipeline:
1. Load model's hidden states for RAV+SAV at chosen layer + matching BERT embeddings
2. Standardize speech features; fit sklearn CCA (n_components ≤ 16)
3. Report canonical correlations (sanity); SVD-orthonormalize x_rotations_ → U_ortho
4. Load EMIS hidden states + filenames + parse to (sid, text_emo, audio_emo, category)
5. Sentence-stratified 5-fold CV (matches Task 7A protocol):
   - For each k ∈ {0, 1, 2, 5, 10, 16}:
     - Project speech features orthogonal to U_ortho[:, :k] (k=0 = no projection / baseline)
     - For each fold: train probe on CONGRUENT in train sentences, test on INCONGRUENT in test sentences
     - Aggregate per category (all / explicit / implicit) across folds
6. Save CSV + log to W&B group task6_projection.

No GPU needed. Run on login node.
"""
import argparse
import os

SER_SCRATCH = os.environ.get(
    "SER_SCRATCH", os.path.join("/scratch1", os.environ.get("USER", "unknown")))
import json
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold

SER_DATA = os.path.join(SER_SCRATCH, "ser_data")
REPO_ROOT = os.path.join(SER_SCRATCH, "linguistic-agnostic-ser")


def parse_emis_filename(fn):
    base = str(fn).replace(".wav", "")
    parts = base.split("_")
    if len(parts) == 5:
        sid, text_emo, audio_emo, esd_spk, tts = parts
        category = "neutral"
    elif len(parts) == 6:
        sid, text_emo, category, audio_emo, esd_spk, tts = parts
    else:
        return None
    return {"sid": sid, "text_emo": text_emo.lower(), "audio_emo": audio_emo.lower(),
            "category": category.lower(),
            "is_congruent": text_emo.lower() == audio_emo.lower()}


def fit_cca_orthonormal(H_speech, H_text, n_components):
    """Fit CCA, return (U_ortho speech-side, canonical correlations)."""
    n_components = min(n_components, H_speech.shape[1] - 1, H_text.shape[1] - 1)
    cca = CCA(n_components=n_components, max_iter=2000)
    cca.fit(H_speech, H_text)

    Xc, Yc = cca.transform(H_speech, H_text)
    corrs = np.array([np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(n_components)])

    # x_rotations_: speech-side projection matrix (d_speech × n_components)
    U_raw = cca.x_rotations_
    # SVD-orthonormalize
    U_ortho, _, _ = np.linalg.svd(U_raw, full_matrices=False)
    return U_ortho, corrs


def project_orthogonal(H, U_ortho, k):
    if k == 0:
        return H
    Uk = U_ortho[:, :k]
    return H - H @ Uk @ Uk.T


def evaluate_kfold(H_emis, parsed, U_ortho, k, n_folds=5, random_state=42):
    """Sentence-stratified k-fold: train CONGRUENT(in train sent), test INCONGRUENT(in test sent)."""
    sids = np.array([p["sid"] for p in parsed])
    is_cong = np.array([p["is_congruent"] for p in parsed])
    audio_emos = np.array([p["audio_emo"] for p in parsed])
    text_emos = np.array([p["text_emo"] for p in parsed])
    categories = np.array([p["category"] for p in parsed])

    unique_sids = np.array(sorted(set(sids.tolist())))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Project once per k (independent of fold)
    H_proj = project_orthogonal(H_emis, U_ortho, k)

    # Per-category aggregators across folds
    agg = {cat: {"target": [], "proxy": [], "n": 0} for cat in ["all", "explicit", "implicit"]}
    val_emo_accs = []

    for tr_si, te_si in kf.split(unique_sids):
        tr_sents = set(unique_sids[tr_si])
        te_sents = set(unique_sids[te_si])

        train_mask = np.array([s in tr_sents for s in sids]) & is_cong
        test_mask = np.array([s in te_sents for s in sids]) & ~is_cong

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        sc = StandardScaler()
        H_tr = sc.fit_transform(H_proj[train_mask])
        H_te = sc.transform(H_proj[test_mask])

        le = LabelEncoder().fit(np.concatenate([audio_emos, text_emos]))
        y_tr = le.transform(audio_emos[train_mask])

        # val_emo on a small held-out portion of train (within-fold)
        n_tr = len(y_tr)
        if n_tr > 20:
            val_idx = np.random.RandomState(random_state).permutation(n_tr)[:max(1, n_tr // 5)]
            tr_idx = np.setdiff1d(np.arange(n_tr), val_idx)
            lr_v = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)
            lr_v.fit(H_tr[tr_idx], y_tr[tr_idx])
            val_emo_accs.append((lr_v.predict(H_tr[val_idx]) == y_tr[val_idx]).mean())

        lr = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)
        lr.fit(H_tr, y_tr)
        preds = le.inverse_transform(lr.predict(H_te))
        target = audio_emos[test_mask]
        proxy = text_emos[test_mask]
        cats = categories[test_mask]

        for cat in ["all", "explicit", "implicit"]:
            sub = np.ones_like(cats, dtype=bool) if cat == "all" else (cats == cat)
            if sub.sum() == 0:
                continue
            agg[cat]["target"].append((preds[sub] == target[sub]).mean())
            agg[cat]["proxy"].append((preds[sub] == proxy[sub]).mean())
            agg[cat]["n"] += int(sub.sum())

    out = {"k": k, "val_emo_acc_mean": float(np.mean(val_emo_accs)) if val_emo_accs else None}
    for cat in ["all", "explicit", "implicit"]:
        if agg[cat]["target"]:
            t = np.array(agg[cat]["target"])
            p = np.array(agg[cat]["proxy"])
            out[f"target_acc_{cat}"] = float(t.mean())
            out[f"target_acc_{cat}_std"] = float(t.std())
            out[f"proxy_acc_{cat}"] = float(p.mean())
            out[f"proxy_acc_{cat}_std"] = float(p.std())
            # text_bias = proxy_acc - target_acc, matching Task 7A convention.
            # Positive = probe predicts TEXT emotion more than AUDIO emotion (text shortcut).
            # Negative = probe predicts AUDIO emotion more than TEXT (audio-grounded).
            out[f"text_bias_{cat}"] = float(p.mean() - t.mean())
            out[f"n_{cat}"] = agg[cat]["n"]
    return out


def main(args):
    print(f"\n=== Task 6 V3 — CCA projection — model={args.model} layer={args.layer} ===\n")

    # ---- 1. Load CCA training data ----
    if args.cca_corpus == "ravsav":
        H_rav = np.load(f"{SER_DATA}/hidden_states_{args.model}_RAVDESS.npy")[:, args.layer, :].astype(np.float32)
        H_sav = np.load(f"{SER_DATA}/hidden_states_{args.model}_SAVEE.npy")[:, args.layer, :].astype(np.float32)
        H_train_cca = np.concatenate([H_rav, H_sav], axis=0)
        bert_rav = np.load(f"{SER_DATA}/bert_embeddings_RAVDESS.npy").astype(np.float32)
        bert_sav = np.load(f"{SER_DATA}/bert_embeddings_SAVEE.npy").astype(np.float32)
        bert_all = np.concatenate([bert_rav, bert_sav], axis=0)
    elif args.cca_corpus == "iemocap":
        H_train_cca = np.load(f"{SER_DATA}/hidden_states_{args.model}_IEMOCAP.npy")[:, args.layer, :].astype(np.float32)
        bert_all = np.load(f"{SER_DATA}/bert_embeddings_IEMOCAP.npy").astype(np.float32)
    else:
        raise ValueError(f"Unknown cca_corpus: {args.cca_corpus}")

    print(f"  CCA corpus: {args.cca_corpus}")
    print(f"  CCA training: speech {H_train_cca.shape}, text {bert_all.shape}")

    # Standardize speech (CCA assumes centered features for its variance computations)
    sc_speech = StandardScaler()
    H_train_cca_s = sc_speech.fit_transform(H_train_cca)

    # ---- 2. Fit CCA, get U_ortho ----
    U_ortho, corrs = fit_cca_orthonormal(H_train_cca_s, bert_all, n_components=args.k_max)
    print(f"  Canonical correlations (top {len(corrs)}): {np.round(corrs, 3)}")
    print(f"  U_ortho shape: {U_ortho.shape}")

    # ---- 3. Load EMIS data + parse ----
    H_emis_full = np.load(f"{SER_DATA}/hidden_states_{args.model}_EMIS.npy")[:, args.layer, :].astype(np.float32)
    fns_emis = np.load(f"{SER_DATA}/filenames_{args.model}_EMIS.npy", allow_pickle=True)
    parsed = [parse_emis_filename(fn) for fn in fns_emis]

    # Filter out unparseable rows
    valid = [i for i, p in enumerate(parsed) if p is not None]
    H_emis = H_emis_full[valid]
    parsed = [parsed[i] for i in valid]

    n_cong = sum(1 for p in parsed if p["is_congruent"])
    n_incong = sum(1 for p in parsed if not p["is_congruent"])
    print(f"  EMIS: {len(parsed)} clips total ({n_cong} congruent, {n_incong} incongruent)")
    cats = [p["category"] for p in parsed if not p["is_congruent"]]
    from collections import Counter
    print(f"  Incongruent category breakdown: {dict(Counter(cats))}")

    # Standardize EMIS using SAME scaler as CCA training (otherwise projection geometry doesnt match)
    H_emis_s = sc_speech.transform(H_emis)

    # ---- 4. Sweep k (k=0 = baseline, no projection) ----
    rows = []
    for k in args.ks:
        print(f"\n--- k={k} ---")
        result = evaluate_kfold(H_emis_s, parsed, U_ortho, k, n_folds=args.n_folds)
        result["model"] = args.model
        result["layer"] = args.layer
        result["cca_corr_topk_mean"] = float(corrs[:max(k, 1)].mean()) if k > 0 else None
        rows.append(result)
        ve = result.get("val_emo_acc_mean")
        print(f"  val_emo_acc: {ve:.4f}" if ve is not None else "  val_emo_acc: n/a")
        for cat in ["all", "explicit", "implicit"]:
            if f"text_bias_{cat}" in result:
                print(f"  {cat:9s}: target={result[f'target_acc_{cat}']:.4f} "
                      f"proxy={result[f'proxy_acc_{cat}']:.4f} "
                      f"bias={result[f'text_bias_{cat}']:+.4f}  n={result[f'n_{cat}']}")

    # ---- 5. Save CSV ----
    out_dir = f"{REPO_ROOT}/results/TASK6/projection"
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if args.cca_corpus == "ravsav" else f"_{args.cca_corpus}"
    out_csv = f"{out_dir}/cca_proj_{args.model}_L{args.layer}{suffix}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    np.save(f"{out_dir}/cca_corrs_{args.model}_L{args.layer}{suffix}.npy", corrs)
    print(f"\nSaved: {out_csv}")
    print(f"Saved canonical correlations: cca_corrs_{args.model}_L{args.layer}.npy")

    # ---- 6. Optional W&B ----
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project="linguistic-agnostic-ser", entity="AGSER",
                       group="task6_projection",
                       name=f"projection_{args.model}_L{args.layer}",
                       config=vars(args))
            for r in rows:
                wandb.log({k: v for k, v in r.items() if isinstance(v, (int, float))})
            wandb.finish()
            print("Logged to W&B group task6_projection")
        except Exception as e:
            print(f"W&B logging skipped: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuBERT", choices=["HuBERT", "wav2vec2", "WavLM", "Whisper"])
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--k_max", type=int, default=16)
    p.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 5, 10, 16])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--cca_corpus", default="ravsav", choices=["ravsav", "iemocap"])
    args = p.parse_args()
    main(args)
