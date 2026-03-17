#!/usr/bin/env python3
"""Nepes Chart Showcase — SECOM semiconductor manufacturing data.

Generates PNG screenshots for README documentation.

Data source: UCI Machine Learning Repository — SECOM Dataset
  https://archive.ics.uci.edu/ml/datasets/SECOM
  Semi-conductor manufacturing process data (1567 samples, 590 features).
  This is PUBLIC SAMPLE DATA, not production data.

Usage: python showcase-secom.py
Output: docs/light.png, docs/dark.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -- Load SECOM data --
DATA_DIR = "../../fdc-analytics/data/secom"
data = np.loadtxt(f"{DATA_DIR}/secom.data")
labels = np.loadtxt(f"{DATA_DIR}/secom_labels.data", usecols=0)

# Select 6 features with low NaN count and good variance
feature_indices = [0, 1, 2, 3, 5, 6]
feature_names = ["Sensor A", "Sensor B", "Sensor C", "Sensor D", "Sensor E", "Sensor F"]
features = data[:, feature_indices]

# Handle NaN: replace with column mean
for i in range(features.shape[1]):
    col = features[:, i]
    mask = np.isfinite(col)
    if not mask.all():
        features[~mask, i] = col[mask].mean()

# SPC colors
SPC_LIGHT = {
    "center_line": "#23438E", "data_points": "#5A7EB0",
    "control_limit": "#6A6A6A", "spec_limit": "#C25609", "violation": "#C4181F",
}
SPC_DARK = {
    "center_line": "#5C8CFF", "data_points": "#6B8AD8",
    "control_limit": "#8A9199", "spec_limit": "#FEA413", "violation": "#FF5C5C",
}


def generate(theme: str):
    try:
        plt.style.use(f"nepes-{theme}.mplstyle")
    except OSError:
        plt.style.use(f"nepes-{theme}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    spc = SPC_LIGHT if theme == "light" else SPC_DARK

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Nepes {theme.title()} — SECOM Semiconductor Data",
        fontsize=16, fontweight="bold", y=0.98
    )
    fig.text(0.5, 0.955,
        "Data: UCI ML Repository SECOM Dataset (public sample, not production)",
        ha="center", fontsize=9, style="italic",
        color="#7A7A84" if theme == "light" else "#8A9199"
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.3, top=0.92, bottom=0.06)

    # 1. Multi-series time series (first 200 samples)
    ax1 = fig.add_subplot(gs[0, 0])
    n = 200
    for i, name in enumerate(feature_names):
        std = features[:n, i].std()
        vals = (features[:n, i] - features[:n, i].mean()) / (std if std > 0 else 1)
        ax1.plot(vals, linewidth=0.9, color=colors[i], label=name, alpha=0.85)
    ax1.set_title("Multi-Sensor Time Series")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Normalized Value")
    ax1.legend(fontsize=7, ncol=2)

    # 2. Box plot by pass/fail
    ax2 = fig.add_subplot(gs[0, 1])
    pass_data = [features[labels == -1, i] for i in range(6)]
    bp = ax2.boxplot(pass_data, tick_labels=feature_names, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors[:6]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax2.set_title("Feature Distribution (Pass Only)")
    ax2.set_ylabel("Value")
    ax2.tick_params(axis="x", rotation=30)

    # 3. PCA scatter (using 6 selected features, not all 590)
    ax3 = fig.add_subplot(gs[0, 2])
    from numpy.linalg import svd
    centered = features - features.mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)
    X_pca = U[:, :2] * S[:2]
    var_ratio = S[:2]**2 / (S**2).sum()
    pass_mask = labels == -1
    fail_mask = labels == 1
    ax3.scatter(X_pca[pass_mask, 0], X_pca[pass_mask, 1],
                c=colors[0], s=12, alpha=0.5, label="Pass")
    ax3.scatter(X_pca[fail_mask, 0], X_pca[fail_mask, 1],
                c=colors[3], s=20, alpha=0.7, label="Fail", marker="x")
    ax3.set_title("PCA — Pass vs Fail")
    ax3.set_xlabel(f"PC1 ({var_ratio[0]:.1%})")
    ax3.set_ylabel(f"PC2 ({var_ratio[1]:.1%})")
    ax3.legend(fontsize=8)

    # 4. Bar chart — feature variance
    ax4 = fig.add_subplot(gs[1, 0])
    variances = [features[:, i].std() for i in range(6)]
    ax4.bar(feature_names, variances, color=colors[:6])
    ax4.set_title("Feature Std Dev")
    ax4.set_ylabel("Standard Deviation")
    ax4.tick_params(axis="x", rotation=30)

    # 5. SPC control chart (Sensor A)
    ax5 = fig.add_subplot(gs[1, 1])
    sensor = features[:100, 0]
    cl = sensor.mean()
    sigma = sensor.std()
    ucl, lcl = cl + 3 * sigma, cl - 3 * sigma
    ax5.plot(sensor, color=spc["data_points"], linewidth=0.8, marker="o", markersize=2)
    ax5.axhline(cl, color=spc["center_line"], linewidth=1.5, label="CL")
    ax5.axhline(ucl, color=spc["control_limit"], linestyle="--", linewidth=0.8, label="UCL/LCL")
    ax5.axhline(lcl, color=spc["control_limit"], linestyle="--", linewidth=0.8)
    violations = np.where((sensor > ucl) | (sensor < lcl))[0]
    if len(violations):
        ax5.scatter(violations, sensor[violations], color=spc["violation"],
                    s=50, marker="D", zorder=5, label="Violation")
    ax5.set_title("SPC — Sensor A")
    ax5.set_xlabel("Sample")
    ax5.legend(fontsize=7)

    # 6. Histogram — Pass vs Fail
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(features[pass_mask, 0], bins=30, alpha=0.6, color=colors[0], label="Pass")
    ax6.hist(features[fail_mask, 0], bins=30, alpha=0.6, color=colors[3], label="Fail")
    ax6.set_title("Sensor A — Pass vs Fail")
    ax6.set_xlabel("Value")
    ax6.set_ylabel("Count")
    ax6.legend(fontsize=8)

    plt.savefig(f"docs/{theme}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Generated docs/{theme}.png")


if __name__ == "__main__":
    import os
    os.makedirs("docs", exist_ok=True)
    generate("light")
    generate("dark")
    print("Done!")
