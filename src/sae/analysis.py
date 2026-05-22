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
    """MI between each (utterance-pooled) feature and the categorical emotion label.

    Returns an array of length d_sae. Uses sklearn's k-NN MI estimator for a
    continuous feature vs. a discrete target.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(labels)
    # mutual_info_classif treats all-zero (dead) columns as MI 0 automatically.
    return mutual_info_classif(utt_feats, y, random_state=random_state)


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
