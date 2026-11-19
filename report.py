"""Save a headless training summary and a held-out 3D prediction example."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_comparison(directory, metrics, variants, cases):
    methods = [f"{variant} top-1" for variant in variants] + ["CV", "CA"]
    colors = {"TMM-F": "#337cb4", "TMM-N": "#128769", "CV": "#8796a8", "CA": "#ac6757"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True, layout="constrained")
    fig.suptitle(f"Held-out comparison ({cases:,} trajectories)", fontweight="bold")
    for axis, key, label in zip(axes, ("ADE", "ED", "DF"), ("ADE", "ED / FDE", "Discrete Frechet (DF)")):
        values = [metrics[method][key] for method in methods]
        bars = axis.barh(methods, values, color=[colors[method.split()[0]] for method in methods])
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set(title=label, xlabel="Error (m, lower is better)", xlim=(0, max(max(values) * 1.23, 0.01)))
        axis.grid(axis="x", alpha=0.23)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)


def save_plots(directory, history, preview, variant):
    epochs = [record["epoch"] for record in history]
    fig, (loss_axis, distance_axis) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    loss_axis.plot(epochs, [record["train_nll"] for record in history], label="Train")
    loss_axis.plot(epochs, [record["val_nll"] for record in history], label="Validation")
    loss_axis.set(xlabel="Epoch", ylabel="Mixture NLL", title="Coefficient fit")
    loss_axis.legend()
    distance_axis.plot(epochs, [record["val_mean_df"] for record in history], color="#128769")
    distance_axis.set(xlabel="Epoch", ylabel="Weighted discrete Frechet (m)", title=f"{variant} validation error")
    for axis in (loss_axis, distance_axis):
        axis.grid(alpha=0.25)
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)

    obs, truth, modes, weights, cv, ca, kind, index = preview
    fig, axis = plt.subplots(figsize=(10, 7), subplot_kw={"projection": "3d"}, layout="constrained")
    axis.plot(*obs.T, color="#49566d", label="Observation", linewidth=2)
    for track, color, line, label in (
        (truth, "#17191c", "-", "Ground truth"), (cv, "#c64727", ":", "CV"), (ca, "#7c439b", "--", "CA")
    ):
        axis.plot(*np.vstack((obs[-1], track)).T, linestyle=line, color=color, label=label, linewidth=2)
    shades = ("#086d9c", "#2395a6", "#3673b9", "#36947b", "#4d91d5")
    for number, (mode, probability, color) in enumerate(zip(modes, weights, shades), 1):
        axis.plot(*np.vstack((obs[-1], mode)).T, color=color, linewidth=2.5 if number == 1 else 1.7,
                  label=f"{variant} mode {number} ({probability:.1%})")
    axis.scatter(*obs[-1], color="#49566d", s=28)
    plotted = np.vstack((obs, truth, cv, ca, modes.reshape(-1, 3)))
    axis.set_box_aspect(np.maximum(np.ptp(plotted, axis=0), 0.1), zoom=0.86)
    axis.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
             title=f"{variant} | Test trajectory {index} | {kind.replace('_', ' ')} | {len(truth)} future steps")
    axis.legend(loc="upper left", fontsize=9)
    fig.savefig(directory / "trajectory.png", dpi=160)
    plt.close(fig)
