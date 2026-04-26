"""
Fig. 6: Per-class performance heatmap for MCANet on RescueNet (v2).
Displays AP, Precision, Recall, F1-score for each damage category.
Data: v2 canonical results (lam=0.1, epoch=30, seed=42).
Source log: logs/TEST_rescuenet_res2net101_csra_1_2026-04-04_11-08-36/
Colormap: viridis (colorblind-safe, print-safe, Nature-recommended).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ── Font: Arial preferred (Nature), fallback to DejaVu Sans ──────────────────
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['pdf.fonttype'] = 42   # embeds fonts properly for journal submission

# ── Data ─────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Water",
    "Bldg-No Damage",
    "Bldg-Minor Damage",
    "Bldg-Major Damage",
    "Bldg-Total Destr.",
    "Vehicle",
    "Road-Clear",
    "Road-Blocked",
    "Tree",
    "Pool",
]

METRIC_NAMES = ["AP (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]

# v2 MCANet h=1 test-set results (450 images)
DATA = np.array([
    [95.80, 93.25, 86.86, 89.94],  # Water
    [94.61, 85.99, 91.75, 88.78],  # Bldg_No_Damage
    [88.85, 86.21, 82.42, 84.27],  # Bldg_Minor_Damage
    [85.94, 83.96, 77.39, 80.54],  # Bldg_Major_Damage
    [88.88, 81.61, 84.52, 83.04],  # Bldg_Total_Destruction
    [94.85, 88.71, 92.18, 90.41],  # Vehicle
    [98.15, 94.12, 91.95, 93.02],  # Road-Clear
    [72.90, 67.61, 64.86, 66.21],  # Road-Blocked
    [96.00, 86.94, 91.03, 88.94],  # Tree
    [97.72, 88.89, 88.89, 88.89],  # Pool
])

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_heatmap(out_path="figures/fig6_perclass_heatmap_v2.png", dpi=300):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    vmin, vmax = 60, 100

    fig, ax = plt.subplots(figsize=(8, 5.5))

    if HAS_SEABORN:
        df = pd.DataFrame(DATA, index=CLASS_NAMES, columns=METRIC_NAMES)
        sns.heatmap(
            df,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor='white',
            annot_kws={"size": 10, "weight": "bold"},
            cbar_kws={"label": "Score (%)", "shrink": 0.85},
            ax=ax,
        )
        # Move x-axis labels to top
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', labelsize=11, labeltop=True, labelbottom=False)
        ax.tick_params(axis='y', labelsize=10, rotation=0)
        # Colorbar font
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("Score (%)", fontsize=10)
    else:
        # Fallback: pure matplotlib
        im = ax.imshow(DATA, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(METRIC_NAMES)))
        ax.set_xticklabels(METRIC_NAMES, fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        for r in range(DATA.shape[0]):
            for c in range(DATA.shape[1]):
                val = DATA[r, c]
                norm_val = (val - vmin) / (vmax - vmin)
                text_color = 'white' if norm_val < 0.6 else 'black'
                ax.text(c, r, f"{val:.1f}", ha='center', va='center',
                        fontsize=10, color=text_color, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.85)
        cbar.set_label("Score (%)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Fig. 6. Heatmap of per-class performance of MCANet on the RescueNet dataset\n"
        "displaying precision, recall, F1-score, and AP for each damage category",
        fontsize=10, pad=16, loc='left'
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    # Also save PDF for journal submission
    pdf_path = out_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PNG: {out_path}")
    print(f"Saved PDF: {pdf_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/fig6_perclass_heatmap_v2.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    plot_heatmap(out_path=args.out, dpi=args.dpi)
