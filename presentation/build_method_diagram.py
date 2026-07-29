#!/usr/bin/env python
"""Methodology flow diagram. Detailed top-to-bottom flow; every arrow is a short
straight vertical line that sits inside the gap between boxes (it touches the edges
but never enters a box)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

NAVY = "#1f3a6e"; SOFT = "#eaf0fb"; RED = "#c0392b"; GREEN = "#1e7d4f"; INK = "#222"
fig, ax = plt.subplots(figsize=(10, 8.7)); ax.set_xlim(0, 12); ax.set_ylim(0.0, 8.6); ax.axis("off")
M = 0.07  # arrow inset from a box edge, so the head/tail never overlaps the box


def box(x, y, w, h, title, body, fc=SOFT, ec=NAVY, tcol=NAVY, body_fs=8.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10",
                                fc=fc, ec=ec, lw=1.7))
    ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="top", fontsize=11.5, fontweight="bold", color=tcol)
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=body_fs, color=INK)


def vdown(x, upper_box_bottom, lower_box_top):
    """Vertical arrow that lives entirely in the gap between two boxes."""
    ax.add_patch(FancyArrowPatch((x, upper_box_bottom - M), (x, lower_box_top + M),
                                 arrowstyle="-|>", mutation_scale=17, lw=2.0, color=NAVY,
                                 shrinkA=0, shrinkB=0))


ax.text(6, 8.42, "Methodology: from a frozen speech model to interpretable emotion features",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)

# centered stack, width 8 (x 2..10)
box(2.0, 7.10, 8.0, 0.92, "1.  Input audio (4 corpora)",
    "CREMA-D (same 12 sentences = control),  EMIS (text emotion != voice emotion),\nIEMOCAP,  MSP-Podcast")
box(2.0, 5.62, 8.0, 1.02, "2.  Frozen encoders, per-frame activations",
    "6 models, 25 layers each (same encoders as the probing project);\nactivations kept un-pooled: 1024-d, about 940k frames per layer")
box(2.0, 4.28, 8.0, 0.92, "3.  Train a TopK SAE",
    "dictionary of 8192 features, 32 active per frame;  rebuilds 93% of the signal")
box(2.0, 2.50, 8.0, 1.30, "4.  Pick the emotion features  (three filters)", "",
    ec=RED, fc="#fff6e6", tcol=RED)
ax.text(6.0, 3.18, "keep a feature only if:  (1) it clearly relates to emotion,\n"
        "(2) it fires for mostly one emotion,   (3) it beats the sentence and the speaker\n"
        "394 candidates   ->   61 clean emotion features",
        ha="center", va="top", fontsize=8.9, color=INK)

# two children, both under box 4's span -> straight vertical arrows
box(1.30, 0.55, 4.40, 1.42, "5a.  Label each feature",
    "acoustic (eGeMAPS) vs\nlexical (sentence), by AUC.\nResult: 60 of 61 acoustic", ec=GREEN, tcol=GREEN)
box(6.30, 0.55, 4.40, 1.42, "5b.  Five analyses",
    "depth (decode vs disentangle),\ncross-model overlap, cross-speaker,\ncausal sufficiency, EMIS text-bias")

# all arrows: short vertical, centered, inside the gaps
vdown(6.0, 7.10, 6.64)     # 1 -> 2
vdown(6.0, 5.62, 5.20)     # 2 -> 3
vdown(6.0, 4.28, 3.80)     # 3 -> 4
vdown(3.5, 2.50, 1.97)     # 4 -> 5a
vdown(8.5, 2.50, 1.97)     # 4 -> 5b

ax.text(6.0, 0.18, "Reuses the probing project's encoders, layers, logistic probe, and cached embeddings; "
        "the SAE and the feature analysis are the new part.",
        ha="center", va="center", fontsize=8.8, style="italic", color="#555")

fig.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(HERE, "story_figures", "figure_0_method.png"),
          os.path.join(HERE, "latex", "images", "fig_method.png")]:
    fig.savefig(p, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote method diagram")
