#!/usr/bin/env python
"""A compact TopK Sparse Autoencoder for frozen-encoder speech activations.

We use a TopK SAE (Gao et al., 2024) rather than an L1 SAE: TopK fixes the L0
(exactly ``k`` active features per input), giving direct, interpretable sparsity
control and avoiding L1 activation shrinkage. The implementation is
self-contained (no dependence on the SAELens training runner, which is built
around TransformerLens language-model hooks) but follows the same conventions —
unit-norm decoder columns, a pre-encoder bias tied to the decoder bias, and
input standardisation — so the learned features are directly comparable to
standard SAE work and the module exposes ``encode``/``decode``.

Activations are standardised to E[||x||^2] ≈ d_in before training; the constants
are stored on the module so ``encode`` works on raw activations at inference.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import torch
import torch.nn as nn


@dataclasses.dataclass
class SAEConfig:
    d_in: int
    expansion: int = 8
    k: int = 32                 # active features per input (TopK)
    aux_k: int = 256            # features used by the AuxK dead-feature loss
    aux_coef: float = 1.0 / 32  # weight of the AuxK auxiliary reconstruction
    dead_steps_threshold: int = 400  # steps without firing before "dead"

    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_in, d_sae = cfg.d_in, cfg.d_sae
        # Decoder columns unit-normalised; encoder tied to decoder transpose at init.
        W_dec = torch.randn(d_sae, d_in)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Input standardisation (filled by fit_normalizer); buffers travel with state_dict.
        self.register_buffer("x_mean", torch.zeros(d_in))
        self.register_buffer("x_scale", torch.ones(1))
        # Dead-feature tracking.
        self.register_buffer("steps_since_fired", torch.zeros(d_sae, dtype=torch.long))

    # -- normalisation ----------------------------------------------------
    def fit_normalizer(self, x: torch.Tensor):
        mean = x.mean(0)
        scale = (x - mean).pow(2).sum(1).mean().sqrt() / (self.cfg.d_in ** 0.5)
        self.x_mean.copy_(mean)
        self.x_scale.copy_(scale.reshape(1))

    def norm(self, x):
        return (x - self.x_mean) / self.x_scale

    def denorm(self, x):
        return x * self.x_scale + self.x_mean

    # -- core -------------------------------------------------------------
    def _preacts(self, x_normed):
        return (x_normed - self.b_dec) @ self.W_enc + self.b_enc

    @staticmethod
    def _topk(z, k):
        vals, idx = z.topk(k, dim=-1)
        vals = torch.relu(vals)
        out = torch.zeros_like(z)
        out.scatter_(-1, idx, vals)
        return out

    def encode(self, x, normalize: bool = True):
        """Return TopK feature activations for raw activations ``x``."""
        x_n = self.norm(x) if normalize else x
        return self._topk(self._preacts(x_n), self.cfg.k)

    def decode(self, f, denormalize: bool = True):
        x_n = f @ self.W_dec + self.b_dec
        return self.denorm(x_n) if denormalize else x_n

    def forward(self, x_normed):
        """Training forward on already-normalised input. Returns dict of tensors."""
        pre = self._preacts(x_normed)
        feats = self._topk(pre, self.cfg.k)
        recon = feats @ self.W_dec + self.b_dec
        # AuxK: reconstruct the residual using top-aux_k *dead* features (keeps
        # features alive without L1). Following Gao et al. 2024.
        dead = self.steps_since_fired >= self.cfg.dead_steps_threshold
        aux_loss = x_normed.new_zeros(())
        if dead.any():
            pre_dead = pre.masked_fill(~dead, float("-inf"))
            kk = int(min(self.cfg.aux_k, int(dead.sum())))
            if kk > 0:
                aux_feats = self._topk(pre_dead, kk)
                residual = (x_normed - recon).detach()
                aux_recon = aux_feats @ self.W_dec
                aux_loss = (aux_recon - residual).pow(2).mean()
        # update dead-feature counters
        fired = (feats.abs() > 0).any(0)
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0
        return {"recon": recon, "feats": feats, "aux_loss": aux_loss}

    @torch.no_grad()
    def normalize_decoder_(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    # -- persistence ------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        with open(path.replace(".pt", ".json"), "w") as fh:
            json.dump(dataclasses.asdict(self.cfg), fh, indent=2)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        with open(path.replace(".pt", ".json")) as fh:
            cfg = SAEConfig(**json.load(fh))
        sae = cls(cfg)
        sae.load_state_dict(torch.load(path, map_location=map_location))
        return sae


def fvu(x: torch.Tensor, recon: torch.Tensor) -> float:
    """Fraction of variance unexplained (lower is better; 0 = perfect)."""
    return ((x - recon).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()


def train_sae(
    activations: np.ndarray,
    expansion: int = 8,
    k: int = 32,
    steps: int = 4000,
    batch_size: int = 4096,
    lr: float = 4e-4,
    device: str | None = None,
    seed: int = 42,
    log_every: int = 500,
):
    """Train a TopK SAE on a ``(M, d_in)`` activation matrix. Returns (sae, history)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X = torch.from_numpy(np.ascontiguousarray(activations, dtype=np.float32))
    d_in = X.shape[1]
    cfg = SAEConfig(d_in=d_in, expansion=expansion, k=k)
    sae = TopKSAE(cfg).to(device)
    sae.fit_normalizer(X.to(device))
    Xn = sae.norm(X.to(device))            # standardise once (fits in GPU mem for our sizes)
    sae.b_dec.data.copy_(Xn.mean(0))

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    M = Xn.shape[0]
    history = []
    for step in range(steps):
        idx = rng.integers(0, M, size=batch_size)
        xb = Xn[idx]
        out = sae(xb)
        recon_loss = (out["recon"] - xb).pow(2).mean()
        loss = recon_loss + cfg.aux_coef * out["aux_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder_()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                # Batched FVU over the full set — never materialise (M, d_sae).
                mu = Xn.mean(0)
                sse = sst = 0.0
                for j in range(0, M, 65536):
                    xb2 = Xn[j:j + 65536]
                    feats = sae._topk(sae._preacts(xb2), cfg.k)
                    recon = feats @ sae.W_dec + sae.b_dec
                    sse += (xb2 - recon).pow(2).sum().item()
                    sst += (xb2 - mu).pow(2).sum().item()
                metrics = {
                    "step": step,
                    "recon_mse": recon_loss.item(),
                    "fvu": sse / sst,
                    "L0": cfg.k,
                    "dead": int((sae.steps_since_fired >= cfg.dead_steps_threshold).sum()),
                }
            history.append(metrics)
            print(f"  step {step:5d}  mse {metrics['recon_mse']:.4f}  "
                  f"fvu {metrics['fvu']:.4f}  dead {metrics['dead']}/{cfg.d_sae}")
    return sae, history
