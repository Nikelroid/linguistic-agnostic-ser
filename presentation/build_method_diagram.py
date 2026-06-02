#!/usr/bin/env python
"""Methodology flow diagram. Pure top-to-bottom flow: every arrow is straight
vertical, centered under a wide box, so no arrow ever crosses a box or text."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

NAVY = "#1f3a6e"; SOFT = "#eaf0fb"; RED = "#c0392b"; GREEN = "#1e7d4f"; INK = "#222"
fig, ax = plt.subplots(figsize=(11, 7.4)); ax.set_xlim(0, 12); ax.set_ylim(0.4, 8.2); ax.axis("off")


def box(x, y, w, h, title, body, fc=SOFT, ec=NAVY, tcol=NAVY, body_fs=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.7))
    ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="top", fontsize=11.5, fontweight="bold", color=tcol)
    ax.text(x + w / 2, y + h - 0.63, body, ha="center", va="top", fontsize=body_fs, color=INK)


def varrow(x, y_top, y_bot):
    ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bot), arrowstyle="-|>", mutation_scale=18,
                                 lw=2.0, color=NAVY, shrinkA=0, shrinkB=0))


ax.text(6, 7.98, "Methodology: from a frozen speech model to interpretable emotion features",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)

# wide centered boxes, stacked; center x = 6.0 (span 2.0..10.0)
box(2.0, 6.55, 8.0, 1.00, "1.  Frozen encoders and activations",
    "6 models, 25 layers (same as the probing project);\nkeep the activations per frame, un-pooled (~940k per layer)")
box(2.0, 5.05, 8.0, 1.00, "2.  Train a TopK SAE",
    "8192 features, 32 active per frame; rebuilds 93% of the signal")
box(2.0, 3.25, 8.0, 1.30, "3.  Pick the emotion features", "", ec=RED, fc="#fff6e6", tcol=RED)
ax.text(6.0, 3.92, "keep a feature only if:  (1) it clearly relates to emotion,\n"
        "(2) it fires for mostly one emotion,  (3) it beats the sentence and the speaker\n"
        "394 candidates  ->  61 clean emotion features",
        ha="center", va="top", fontsize=8.8, color=INK)

# two children, both under the span of box 3 -> straight vertical arrows
box(1.3, 1.40, 4.4, 1.45, "4a.  Label each feature",
    "acoustic (eGeMAPS) vs\nlexical (sentence), by AUC.\nResult: 60/61 acoustic", ec=GREEN, tcol=GREEN)
box(6.3, 1.40, 4.4, 1.45, "4b.  Five analyses",
    "depth, cross-model overlap,\ncross-speaker, causal sufficiency,\nEMIS text-bias")

# all arrows: straight vertical, centered in clear gaps
varrow(6.0, 6.55, 6.05)        # box1 -> box2
varrow(6.0, 5.05, 4.55)        # box2 -> box3
varrow(3.5, 3.25, 2.85)        # box3 -> 4a  (x=3.5 is under box3 span)
varrow(8.5, 3.25, 2.85)        # box3 -> 4b  (x=8.5 is under box3 span)

ax.text(6.0, 0.95, "Reuses the probing project's encoders, layers, logistic probe, and cached embeddings; "
        "the SAE and the feature analysis are the new part.",
        ha="center", va="center", fontsize=9, style="italic", color="#555")

fig.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(HERE, "story_figures", "figure_0_method.png"),
          os.path.join(HERE, "latex", "images", "fig_method.png")]:
    fig.savefig(p, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote method diagram")
