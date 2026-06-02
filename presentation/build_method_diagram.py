#!/usr/bin/env python
"""Draw a clean methodology flow diagram for the SAE study."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

NAVY = "#1f3a6e"; SOFT = "#eaf0fb"; RED = "#c0392b"; GREEN = "#1e7d4f"; INK = "#222"
fig, ax = plt.subplots(figsize=(12, 7.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 7.6); ax.axis("off")


def box(x, y, w, h, title, body, fc=SOFT, ec=NAVY, tcol=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="top", fontsize=11, fontweight="bold", color=tcol)
    if body:
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.6, color=INK)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.8, color=NAVY))


# top row: extraction -> SAE
box(0.2, 5.6, 2.5, 1.3, "Audio clip", "CREMA-D, IEMOCAP,\nMSP-Podcast, EMIS", fc="#ffffff")
box(3.1, 5.6, 2.7, 1.3, "Frozen encoder", "6 models, 25 layers\n(reused from probing)")
box(6.2, 5.6, 2.9, 1.3, "Per-frame activations", "un-pooled, 1024-d\n~940k vectors / layer")
box(9.5, 5.6, 2.3, 1.3, "TopK SAE", "8192 features,\n32 active, 93% var")
arrow(2.7, 6.25, 3.1, 6.25); arrow(5.8, 6.25, 6.2, 6.25); arrow(9.1, 6.25, 9.5, 6.25)

# down to selection
arrow(10.6, 5.6, 10.6, 4.55)
box(7.6, 3.2, 4.2, 1.35, "Pick the emotion features", "keep a feature only if:\n(1) MI with emotion > 5 sigma   (2) monosemantic >= 0.5\n(3) MI(emotion) > MI(sentence) AND > MI(speaker)", fc="#fff6e6", ec=RED, tcol=RED)
ax.text(9.7, 2.95, "394 candidates  ->  61 clean emotion features", ha="center", fontsize=9,
        style="italic", color=RED)

# arrow from SAE area to selection box (curved via the down arrow target)
arrow(9.7, 4.55, 9.7, 4.55)  # placeholder no-op for spacing

# from selection to two analyses
arrow(8.4, 3.2, 6.2, 2.2)
arrow(9.0, 3.2, 9.0, 2.25)

box(2.2, 0.7, 4.0, 1.5, "Label each feature", "acoustic (eGeMAPS) vs\nlexical (sentence), by AUC.\nResult: 60/61 acoustic", fc=SOFT, ec=GREEN, tcol=GREEN)
box(6.7, 0.7, 5.1, 1.5, "Five analyses", "depth (decode vs disentangle) | cross-model overlap\ncross-speaker (leave-one-out) | causal sufficiency\nEMIS text-bias by layer", fc=SOFT)

# left column: prior work tie-in
box(0.2, 3.25, 6.9, 1.25, "Built on the probing project", "same encoders, same 25 layers, same logistic probe (5-fold CV),\nsame cached embeddings. New part: read activations un-pooled, then factor with the SAE.", fc="#f0f0f0", ec="#888", tcol="#444")

ax.text(6, 7.4, "Methodology: from a frozen speech model to interpretable emotion features",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)

fig.tight_layout()
HERE = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(HERE, "story_figures", "figure_0_method.png"),
          os.path.join(HERE, "latex", "images", "fig_method.png")]:
    fig.savefig(p, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote method diagram")
