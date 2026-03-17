#!/usr/bin/env python3
"""Nepes Chart Palette Showcase — generates PDF with all chart types.

Usage: python chart-showcase.py
Output: nepes-charts-light.pdf, nepes-charts-dark.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

np.random.seed(42)

# ── Sample data ──
n = 100
x = np.arange(n)
series_data = {
    "Power": np.cumsum(np.random.randn(n) * 0.5) + 50,
    "Temperature": np.cumsum(np.random.randn(n) * 0.3) + 25,
    "Flow": np.cumsum(np.random.randn(n) * 0.2) + 10,
    "Voltage": np.cumsum(np.random.randn(n) * 0.4) + 220,
    "Current": np.cumsum(np.random.randn(n) * 0.1) + 5,
    "Gas": np.cumsum(np.random.randn(n) * 0.15) + 8,
}

chambers = [f"CH{i}" for i in range(1, 7)]
box_data = [np.random.randn(30) * s + m for m, s in
            [(50, 5), (52, 4), (48, 6), (51, 3), (49, 7), (53, 4)]]

# SPC data
spc_n = 50
spc_values = np.random.randn(spc_n) * 3 + 50
spc_values[14] = 62  # violation
spc_values[37] = 41  # violation
cl = spc_values.mean()
sigma = spc_values.std()
ucl, lcl = cl + 3 * sigma, cl - 3 * sigma
usl, lsl = 63, 37

# SPC colors (light theme)
spc_light = {
    "center_line": "#23438E",
    "data_points": "#5A7EB0",
    "control_limit": "#6A6A6A",
    "spec_limit": "#C25609",
    "violation": "#C4181F",
}
spc_dark = {
    "center_line": "#5C8CFF",
    "data_points": "#6B8AD8",
    "control_limit": "#8A9199",
    "spec_limit": "#FEA413",
    "violation": "#FF5C5C",
}

# Scatter data
scatter_groups = 6
scatter_n = 20
scatter_x = [np.random.randn(scatter_n) + i * 0.5 for i in range(scatter_groups)]
scatter_y = [np.random.randn(scatter_n) + i * 0.3 for i in range(scatter_groups)]

# Bar data
steps = [f"Step {i}" for i in range(1, 7)]
yields = [85, 72, 93, 68, 88, 76]


def generate_showcase(theme_name: str):
    style_path = f"nepes-{theme_name}.mplstyle"
    spc = spc_light if theme_name == "light" else spc_dark

    try:
        plt.style.use(style_path)
    except OSError:
        plt.style.use(f"nepes-{theme_name}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    with PdfPages(f"nepes-charts-{theme_name}.pdf") as pdf:
        # 1. Multi-series time series (6 colors)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (name, vals) in enumerate(series_data.items()):
            ax.plot(x, vals, label=name, linewidth=1.2, color=colors[i % len(colors)])
        ax.set_title(f"Multi-Series Time Series — Nepes {theme_name.title()}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        ax.legend(loc="upper left")
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(box_data, labels=chambers, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors[:6]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title("Chamber Comparison (Box Plot)")
        ax.set_xlabel("Chamber")
        ax.set_ylabel("Measurement")
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Scatter plot (6 groups)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(scatter_groups):
            ax.scatter(scatter_x[i], scatter_y[i], label=f"Group {i+1}",
                       color=colors[i % len(colors)], alpha=0.7, s=40)
        ax.set_title("PCA-style Scatter (6 groups)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(steps, yields, color=colors[:6])
        ax.set_title("Process Step Yield")
        ax.set_xlabel("Step")
        ax.set_ylabel("Yield (%)")
        pdf.savefig(fig)
        plt.close(fig)

        # 5. SPC Control Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(spc_n), spc_values, color=spc["data_points"],
                linewidth=0.8, marker="o", markersize=3)
        ax.axhline(cl, color=spc["center_line"], linewidth=1.5, label="CL")
        ax.axhline(ucl, color=spc["control_limit"], linestyle="--",
                   linewidth=0.8, label="UCL/LCL")
        ax.axhline(lcl, color=spc["control_limit"], linestyle="--", linewidth=0.8)
        ax.axhline(usl, color=spc["spec_limit"], linestyle=":",
                   linewidth=0.8, label="USL/LSL")
        ax.axhline(lsl, color=spc["spec_limit"], linestyle=":", linewidth=0.8)
        violations = np.where((spc_values > ucl) | (spc_values < lcl))[0]
        ax.scatter(violations, spc_values[violations], color=spc["violation"],
                   s=80, marker="D", zorder=5, label="Violation")
        ax.set_title("SPC X-bar Control Chart")
        ax.set_xlabel("Subgroup")
        ax.set_ylabel("Value")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # 6. Histogram overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, data in enumerate(box_data):
            ax.hist(data, bins=15, alpha=0.5, color=colors[i % len(colors)],
                    label=chambers[i])
        ax.set_title("Distribution Overlay (6 chambers)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  Generated nepes-charts-{theme_name}.pdf (6 charts)")


if __name__ == "__main__":
    generate_showcase("light")
    generate_showcase("dark")
    print("Done!")
