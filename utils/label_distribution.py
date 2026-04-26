"""
utils/label_distribution.py
-----------------------------
Reads the RescueNet trainval annotation JSON and produces:
  - A horizontal bar chart sorted by class frequency
  - A printed table of per-class counts and percentages

Usage (from repo root):
    python utils/label_distribution.py \
        [--ann_path data/rescuenet/trainval_rescuenet.json] \
        [--out_path figures/label_distribution.png] \
        [--dpi 300]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = (
    "Water",
    "Building_No_Damage",
    "Building_Minor_Damage",
    "Building_Major_Damage",
    "Building_Total_Destruction",
    "Vehicle",
    "Road-Clear",
    "Road-Blocked",
    "Tree",
    "Pool",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ann_path",
        default="data/rescuenet/trainval_rescuenet.json",
        help="Path to annotation JSON (relative to repo root or absolute)",
    )
    p.add_argument(
        "--out_path",
        default="figures/label_distribution.png",
        help="Output figure path",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to repo root (one level above utils/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ann_path = args.ann_path if os.path.isabs(args.ann_path) else os.path.join(repo_root, args.ann_path)
    out_path = args.out_path if os.path.isabs(args.out_path) else os.path.join(repo_root, args.out_path)

    if not os.path.exists(ann_path):
        sys.exit(f"Annotation file not found: {ann_path}")

    # Load annotations
    with open(ann_path, "r") as f:
        data = json.load(f)

    n_images = len(data)
    n_classes = len(CLASS_NAMES)
    counts = np.zeros(n_classes, dtype=int)

    for record in data:
        for i, v in enumerate(record["target"]):
            if v == 1:
                counts[i] += 1

    # Print table
    print(f"\nRescueNet label distribution  (n={n_images} images)\n")
    print(f"{'Class':<35} {'Count':>6}  {'% images':>9}")
    print("-" * 56)
    order = np.argsort(counts)[::-1]
    for i in order:
        print(f"{CLASS_NAMES[i]:<35} {counts[i]:>6}  {100*counts[i]/n_images:>8.1f}%")
    print("-" * 56)
    print(f"{'Total images':<35} {n_images:>6}")

    # Plot horizontal bar chart (sorted ascending so largest is at top)
    sort_idx = np.argsort(counts)
    sorted_names = [CLASS_NAMES[i] for i in sort_idx]
    sorted_counts = counts[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_names, sorted_counts, color="#4C72B0", edgecolor="white", linewidth=0.5)

    # Annotate bars with counts
    for bar, cnt in zip(bars, sorted_counts):
        ax.text(
            bar.get_width() + n_images * 0.005,
            bar.get_y() + bar.get_height() / 2,
            str(cnt),
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Number of Images", fontsize=11)
    ax.set_title("RescueNet Training Set — Class Label Distribution", fontsize=12)
    ax.set_xlim(0, max(sorted_counts) * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
