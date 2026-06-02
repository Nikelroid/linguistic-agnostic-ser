#!/usr/bin/env python
"""Draw a clean methodology flow diagram for the SAE study (no arrows over boxes/text)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

NAVY = "#1f3a6e"; SOFT = "#eaf0fb"; RED = "#c0392b"; GREEN = "#1e7d4f"; INK = "#222"; GREY = "#f0f0f0"
fig, ax = plt.subplots(figsize=(12, 6.6)); ax.set_xlim(0, 12); ax.set_ylim(1.7, 7.8); ax.axis("off")


def box(x, y, w, h, title, body, fc=SOFT, ec=NAVY, tcol=NAVY, body_fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top", fontsize=11, fontweight="bold", color=tcol)
    if body:
        ax.text(x + w / 2, y + h - 0.60, body, ha="center", va="top", fontsize=body_fs, color=INK)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.8, color=NAVY, shrinkA=0, shrinkB=0))


ax.text(6, 7.55, "Methodology: from a frozen speech model to interpretable emotion features",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)

# Row 1: the pipeline (left to right)
y1, h1 = 6.3, 1.05
box(0.30, y1, 2.30, h1, "Audio clip", "CREMA-D, IEMOCAP,\nMSP-Podcast, EMIS", fc="#ffffff")
box(2.95, y1, 2.55, h1, "Frozen encoder", "6 models, 25 layers\n(reused from probing)")
box(5.85, y1, 2.75, h1, "Per-frame activations", "un-pooled, 1024-d\n~940k vectors / layer")
box(8.95, y1, 2.75, h1, "TopK SAE", "8192 features,\n32 active, 93% var")
ya = y1 + h1 / 2
arrow(2.60, ya, 2.95, ya); arrow(5.50, ya, 5.85, ya); arrow(8.60, ya, 8.95, ya)

# Row 2: prior-work note (left) and the feature-selection box (right)
y2, h2 = 4.05, 1.55
box(0.30, y2, 5.90, h2, "Built on the probing project",
    "same encoders, same 25 layers,\nsame logistic probe (5-fold CV),\nsame cached embeddings.\nNew part: read un-pooled, then the SAE.",
    fc=GREY, ec="#888", tcol="#444", body_fs=8.4)
box(6.55, y2, 5.15, h2, "Pick the emotion features",
    "keep a feature only if:\n(1) clearly relates to emotion\n(2) fires for mostly one emotion\n(3) beats the sentence and the speaker\n394 candidates  ->  61 clean features",
    fc="#fff6e6", ec=RED, tcol=RED, body_fs=8.4)

# SAE (center x = 10.325) straight down into the Pick box top
arrow(10.325, y1, 10.325, y2 + h2)

# Row 3: the two downstream uses
y3, h3 = 2.15, 1.45
box(1.30, y3, 4.10, h3, "Label each feature",
    "acoustic (eGeMAPS) vs\nlexical (sentence), by AUC.\nResult: 60/61 acoustic", fc=SOFT, ec=GREEN, tcol=GREEN)
box(6.55, y3, 5.15, h3, "Five analyses",
    "depth (decode vs disentangle)\ncross-model overlap | cross-speaker\ncausal sufficiency | EMIS text-bias", fc=SOFT)

# Pick box (bottom y=y2, spans x 6.55..11.70) fans down to the two boxes
arrow(9.10, y2, 9.10, y3 + h3)          # down to Five analyses (vertical, clean)
arrow(7.10, y2, 3.45, y3 + h3)          # down-left to Label (through empty space)

fig.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(HERE, "story_figures", "figure_0_method.png"),
          os.path.join(HERE, "latex", "images", "fig_method.png")]:
    fig.savefig(p, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote method diagram")
