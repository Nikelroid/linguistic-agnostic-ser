#!/usr/bin/env python
"""Shared analysis helpers for the SAE study.

Used across steps: encode frames into SAE feature activations, pool frame-level
features up to utterance level, compute per-feature statistics, mutual
information between features and the emotion label, and a few plots. Kept free
of step-specific logic so each step script stays thin.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Encoding / pooling
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_frames(sae, frames: np.ndarray, batch_size: int = 16384, device=None) -> np.ndarray:
    """Encode a ``(M, d_in)`` frame matrix into ``(M, d_sae)`` TopK feature acts."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device).eval()
    out = np.empty((frames.shape[0], sae.cfg.d_sae), dtype=np.float32)
    for i in range(0, frames.shape[0], batch_size):
        xb = torch.from_numpy(frames[i:i + batch_size]).to(device).float()
        out[i:i + batch_size] = sae.encode(xb).cpu().numpy()
    return out


@torch.no_grad()
def encode_pool_summarize(sae, frames, frame_index: np.ndarray, batch_size: int = 16384, device=None):
    """One streaming pass: encode frames, pool to utterance level, gather stats.

    Avoids ever holding the full ``(M, d_sae)`` feature matrix in memory (which
    is ~30 GB for CREMA-D). ``frames`` may be a memmapped ``np.load(..., mmap_mode='r')``.

    Returns ``(utt_mean, utt_max, stats)`` where utt_* are ``(N_utt, d_sae)`` and
    stats has per-feature density / mean_act / max_act / dead / alive.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device).eval()
    M, d_sae = frames.shape[0], sae.cfg.d_sae
    n_utt = int(frame_index.max()) + 1
    utt_sum = np.zeros((n_utt, d_sae), dtype=np.float32)
    utt_max = np.zeros((n_utt, d_sae), dtype=np.float32)
    counts = np.bincount(frame_index, minlength=n_utt).astype(np.float32)
    feat_sum = np.zeros(d_sae, dtype=np.float64)
    feat_active = np.zeros(d_sae, dtype=np.int64)
    feat_max = np.zeros(d_sae, dtype=np.float32)
    for i in range(0, M, batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch_size])).to(device).float()
        fb = sae.encode(xb).cpu().numpy()                 # (b, d_sae) TopK acts
        idx = frame_index[i:i + batch_size]
        np.add.at(utt_sum, idx, fb)
        np.maximum.at(utt_max, idx, fb)
        feat_sum += fb.sum(0)
        feat_active += (fb > 0).sum(0)
        feat_max = np.maximum(feat_max, fb.max(0))
    utt_mean = utt_sum / np.clip(counts[:, None], 1, None)
    mean_act = np.divide(feat_sum, feat_active, out=np.zeros_like(feat_sum), where=feat_active > 0)
    stats = {
        "density": feat_active / float(M),
        "mean_act": mean_act.astype(np.float32),
        "max_act": feat_max,
        "n_active_frames": feat_active,
        "dead": int((feat_active == 0).sum()),
        "alive": int((feat_active > 0).sum()),
    }
    return utt_mean, utt_max, stats


def pool_to_utterance(feat_frames: np.ndarray, frame_index: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Pool frame-level feature acts ``(M, d_sae)`` to ``(N_utt, d_sae)``.

    ``mean`` averages each feature over the utterance's frames; ``max`` takes the
    peak. Mean reflects how *pervasively* a feature fires; max how *strongly*.
    """
    n_utt = int(frame_index.max()) + 1
    d = feat_frames.shape[1]
    out = np.zeros((n_utt, d), dtype=np.float32)
    counts = np.bincount(frame_index, minlength=n_utt).astype(np.float32)
    if mode == "mean":
        np.add.at(out, frame_index, feat_frames)
        out /= np.clip(counts[:, None], 1, None)
    elif mode == "max":
        for u in range(n_utt):
            sel = feat_frames[frame_index == u]
            if len(sel):
                out[u] = sel.max(0)
    else:
        raise ValueError(mode)
    return out


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------
def feature_stats(feat_frames: np.ndarray) -> dict:
    """Per-feature density / mean-when-active / max, plus dead/alive counts."""
    active = feat_frames > 0
    density = active.mean(0)                       # fraction of frames firing
    sums = feat_frames.sum(0)
    n_active = active.sum(0)
    mean_act = np.divide(sums, n_active, out=np.zeros_like(sums), where=n_active > 0)
    max_act = feat_frames.max(0)
    return {
        "density": density,
        "mean_act": mean_act,
        "max_act": max_act,
        "n_active_frames": n_active,
        "dead": int((n_active == 0).sum()),
        "alive": int((n_active > 0).sum()),
    }


def mutual_info_features(utt_feats: np.ndarray, labels: np.ndarray, random_state: int = 42) -> np.ndarray:
    """MI (nats) between each feature's *firing* and the categorical emotion label.

    TopK SAE features are sparse, so we use the activation **indicator**
    (fires / does not fire on the utterance) and compute MI from the 2×C
    contingency table. This is fully vectorised (two matmuls) — seconds for
    8192 features vs. tens of minutes for a per-feature k-NN estimator — and is
    the natural signal for sparse monosemantic features. ``random_state`` is
    accepted for call-site compatibility but unused.
    """
    N, F = utt_feats.shape
    _, y = np.unique(labels, return_inverse=True)
    C = int(y.max()) + 1
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    A = (utt_feats > 0).astype(np.float64)            # (N, F) firing indicator
    pc = Y.mean(0)[None, :]                            # (1, C) class priors
    ac_c = (A.T @ Y) / N                               # (F, C) p(fire, class)
    in_c = (Y.sum(0)[None, :] / N) - ac_c              # (F, C) p(not-fire, class)
    p_ac = A.mean(0)[:, None]                          # (F, 1) p(fire)
    p_in = 1.0 - p_ac                                  # (F, 1) p(not-fire)
    eps = 1e-12
    mi = (ac_c * np.log((ac_c + eps) / (p_ac * pc + eps))).sum(1)
    mi += (in_c * np.log((in_c + eps) / (p_in * pc + eps))).sum(1)
    return np.clip(mi, 0.0, None)


def select_emotion_features(utt_feats: np.ndarray, labels: np.ndarray,
                            confounds: "dict | None" = None, mono_min: float = 0.5,
                            n_shuffles: int = 10, seed: int = 0):
    """Confound-controlled emotion-feature selection.

    A feature is an *emotion feature* iff its firing is (1) significantly
    associated with emotion (MI > 5σ permutation null), (2) monosemantic
    (>= mono_min mass on one emotion), and (3) MORE informative about emotion
    than about each supplied confound (e.g. sentence/lexical, actor/speaker) —
    so phonetic/lexical and speaker-driven features are filtered out. Returns
    ``(feature_ids, info)`` where info carries mi, mono, threshold and per-
    confound MI arrays.
    """
    mi = mutual_info_features(utt_feats, labels)
    rng = np.random.default_rng(seed)
    null = np.stack([mutual_info_features(utt_feats, rng.permutation(labels))
                     for _ in range(n_shuffles)])
    thresh = float(null.mean() + 5 * null.std())
    sig = mi > thresh
    mono = np.zeros(utt_feats.shape[1])
    sig_ids = np.where(sig)[0]
    if len(sig_ids):
        mono[sig_ids] = monosemanticity(utt_feats, labels, sig_ids)
    keep = sig & (mono >= mono_min)
    conf_mi = {}
    if confounds:
        for name, c in confounds.items():
            mc = mutual_info_features(utt_feats, np.asarray(c))
            conf_mi[name] = mc
            keep &= mi > mc
    return np.where(keep)[0], {
        "mi": mi, "mono": mono, "thresh": thresh, "confound_mi": conf_mi,
        "n_significant": int(sig.sum()),
        "n_sig_mono": int((sig & (mono >= mono_min)).sum()),
    }


def firing_auc(firing: np.ndarray, X: np.ndarray, cv: int = 5, seed: int = 42) -> float:
    """CV ROC-AUC of logistic regression predicting a binary firing indicator
    from X. Robust to the class imbalance of sparse features (unlike R²)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = firing.astype(int)
    if y.sum() < cv or (len(y) - y.sum()) < cv:    # too few in a class for CV
        return 0.5
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=500, class_weight="balanced"))
    try:
        return float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")))
    except Exception:
        return 0.5


def monosemanticity(utt_feats: np.ndarray, labels: np.ndarray, feature_ids: np.ndarray) -> np.ndarray:
    """For each given feature, the fraction of its total activation mass that
    falls on its single most-associated emotion class (1.0 = perfectly
    class-specific = monosemantic w.r.t. emotion; 1/num_classes = uniform).
    """
    classes = np.unique(labels)
    out = np.zeros(len(feature_ids))
    for i, fid in enumerate(feature_ids):
        acts = utt_feats[:, fid]
        total = acts.sum()
        if total <= 0:
            continue
        per_class = np.array([acts[labels == c].sum() for c in classes])
        out[i] = per_class.max() / total
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training(history: list[dict], path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(steps, [h["fvu"] for h in history], "-o", ms=3)
    ax[0].set(xlabel="step", ylabel="FVU", title=f"Fraction of variance unexplained {title}")
    ax[0].grid(alpha=.3)
    ax[1].plot(steps, [h["dead"] for h in history], "-o", ms=3, color="crimson")
    ax[1].set(xlabel="step", ylabel="dead features", title="Dead-feature count")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_dashboard(stats: dict, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    density = stats["density"]
    alive = density[density > 0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(np.log10(alive + 1e-9), bins=60, color="steelblue")
    ax[0].set(xlabel="log10(firing density)", ylabel="# features",
              title=f"Feature density {title}  (alive={len(alive)}/{len(density)})")
    ax[0].grid(alpha=.3)
    ax[1].hist(stats["max_act"][stats["max_act"] > 0], bins=60, color="seagreen")
    ax[1].set(xlabel="max activation", ylabel="# features", title="Per-feature peak activation")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
