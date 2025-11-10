import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

def plot_box_with_points(
    metrics,
    names,
    x_values,
    *,
    x_label="X",
    y_label="Value",
    title=None,
    log_scale_x=False,
    log_scale_y=False,
    palette="Set2", #"Set2", "deep", "tab10"
    save_path=None,
    figsize=(7, 4),
):
    """
    Draw side-by-side boxplots (with seed-level points) for one or multiple metrics
    measured over discrete x-values (e.g., checkpoints, batch sizes, penalty values).

    Args:
        metrics: list of 2D numpy arrays, each shape = (num_x, num_seeds)
        names: list of metric names (same length as metrics)
        x_values: list or array of x-axis values (length = num_x)
        x_label: label for the x-axis
        y_label: label for the y-axis
        title: optional string for figure title
        log_scale_x: bool, whether to use log scale on x-axis
        log_scale_y: bool, whether to use log scale on y-axis
        palette: seaborn color palette name or list of colors
        save_path: optional path (folder or full file) to save the figure
        figsize: tuple, size of the figure
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    colors = sns.color_palette(palette, n_colors=len(metrics))

    # --- Convert all metrics into one long-form DataFrame ---
    data = []
    for metric, name in zip(metrics, names):
        for xi, x_val in enumerate(x_values):
            for val in np.ravel(metric[xi]):
                data.append((x_val, name, val))
    df = pd.DataFrame(data, columns=["X", "Metric", "Value"])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df, x="X", y="Value", hue="Metric",
        palette=colors, width=0.6, ax=ax
    )

    # Overlay seed-level points in matching colors
    sns.stripplot(
        data=df, x="X", y="Value", hue="Metric",
        dodge=True, palette=colors, size=4, jitter=0.15, linewidth=0.3, ax=ax, legend=False
    )

    # --- Style ---
    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, weight="bold")

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()

    # --- Save (optional) ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.isdir(save_path):
            fname = (title or "plot").lower().replace(" ", "_") + ".png"
            save_file = os.path.join(save_path, fname)
        else:
            save_file = save_path
        plt.savefig(save_file, dpi=300, bbox_inches="tight")

    plt.show()

    
def plot_metrics_grid(
    metrics,
    checkpoints,
    *,
    nrows=2,
    ncols=3,
    figsize=(16, 9),
    x_label="Checkpoint",
    y_label="Value",
    font_scale=1.1,
    style="whitegrid",
    tight_layout=True,
    suptitle=None,
):
    """
    Plot multiple metrics (mean/std per checkpoint) as a grid of boxplots with scatter points.

    Args:
        metrics: list of tuples (metric_data, metric_name, logy, palette)
            - metric_data: np.ndarray of shape (num_checkpoints, num_seeds)
            - metric_name: str, title for subplot
            - logy: bool, whether to use log scale on y-axis
            - palette: list or str, single color or palette
        checkpoints: list or np.ndarray of checkpoint x-values
        nrows, ncols: grid layout
        figsize: figure size (width, height)
        x_label, y_label: axis labels
        font_scale: seaborn font scaling
        style: seaborn style (e.g., 'whitegrid', 'ticks')
        tight_layout: whether to apply plt.tight_layout()
        suptitle: optional figure title
    """

    sns.set_theme(style=style, font_scale=font_scale, rc={
        "axes.edgecolor": "0.3",
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for idx, (metric, name, logy, palette) in enumerate(metrics):
        if idx >= len(axes):  # avoid overflow
            break
        ax = axes[idx]

        # flatten metric data into long-form DataFrame
        data = [(ckpt, val) for i, ckpt in enumerate(checkpoints)
                for val in np.ravel(metric[i])]
        df = pd.DataFrame(data, columns=[x_label, y_label])

        # choose color
        color = palette[0] if isinstance(palette, list) else palette

        sns.boxplot(data=df, x=x_label, y=y_label,
                    color=color, linewidth=1.0, fliersize=2.5, width=0.8, ax=ax)
        sns.stripplot(data=df, x=x_label, y=y_label,
                      color="black", size=4, jitter=0.15, alpha=0.6, ax=ax)

        if logy:
            ax.set_yscale("log")

        ax.set_title(name, fontsize=11, weight="bold", pad=5)
        ax.set_xlabel(x_label, fontsize=11 )
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_facecolor("white")

    # hide unused subplots if any
    for ax in axes[len(metrics):]:
        ax.remove()

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, weight="bold", y=1.02)

    if tight_layout:
        plt.tight_layout()
    plt.show()
    

def plot_metric_over_epochs(
    metric_array,
    baselines,
    *,
    epochs_per_val=20,
    metric_name="Metric",
    label="Baseline",
    title=None,
    log_scale_y=False,
    log_scale_x=False,
    linthresh=1e-2,
    figsize=(8, 5),
    alpha_fill=0.1,
    colors=None,
):
    """
    Plot the evolution of a metric over validation epochs for multiple checkpoints.

    Args:
        metric_array: np.ndarray of shape (num_ckpts, num_seeds, num_epochs)
        baselines: list of baseline identifiers (same length as first axis)
        epochs_per_val: multiply index by this to get actual epoch count
        metric_name: y-axis label
        title: optional plot title (default based on metric_name)
        log_scale_y: use symlog y-scale for better visibility
        log_scale_x: use log scale on x-axis
        linthresh: linear threshold for symlog
        figsize: figure size
        alpha_fill: transparency for std shading
        colors: either a colormap name (str) or list of custom colors
    """
    num_ckpts = metric_array.shape[0]
    num_epochs = metric_array.shape[2]
    epochs = np.arange(num_epochs) * epochs_per_val

    # --- Handle colors ---
    if colors is None:
        # default to tab10
        colors = plt.cm.tab10(np.linspace(0, 1, num_ckpts))
    elif isinstance(colors, str):
        # interpret as a colormap name
        cmap = plt.get_cmap(colors)
        colors = cmap(np.linspace(0, 1, num_ckpts))
    else:
        # assume user passed a list of RGB/hex colors
        if len(colors) < num_ckpts:
            raise ValueError(f"Need at least {num_ckpts} colors, got {len(colors)}")

    # --- Plot ---
    plt.figure(figsize=figsize)
    for i, ckpt in enumerate(baselines):
        mean_ = metric_array[i].mean(axis=0)
        std_  = metric_array[i].std(axis=0)

        plt.plot(
            epochs, mean_,
            label=f"{label} {ckpt}",
            color=colors[i],
            lw=2.0,
            alpha=1.0,
        )
        plt.fill_between(
            epochs,
            mean_ - std_,
            mean_ + std_,
            color=colors[i],
            alpha=alpha_fill,
            linewidth=0,
        )

    # --- Styling ---
    if log_scale_x:
        plt.xscale("log")
    if log_scale_y:
        plt.yscale("symlog", linthresh=linthresh)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(title or f"Validation {metric_name} over Epochs", fontsize=13, weight="bold")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()



# sns.set_theme(style="whitegrid", font_scale=1.2)

# # assume you already have these arrays (num_ckpts × num_seeds)
# metrics = {
#     "Opt Gap (Mean)": opt_gap_mean,
#     "Opt Gap (Max)": opt_gap_max,
#     "Opt Gap (Std)": opt_gap_std,
#     "Eq Violation L1 (Mean)": eq_violation_l1_mean,
#     "Eq Violation L1 (Max)": eq_violation_l1_max,
#     "Ineq Violation L1 (Mean)": ineq_violation_l1_mean,
# }

# checkpoints = np.array([0, 20, 60, 100, 200])

# def plot_metric(ax, data, title, ylabel, color):
#     mean_ = data.mean(axis=1)
#     std_  = data.std(axis=1)
    
#     ax.plot(checkpoints, mean_, marker='o', lw=2.5, color=color, label='Mean across seeds')
#     ax.fill_between(checkpoints, mean_-std_, mean_+std_, color=color, alpha=0.15, label='±1 std')
    
#     ax.set_title(title, fontsize=13, weight='bold')
#     ax.set_xlabel("Checkpoint")
#     ax.set_ylabel(ylabel)
#     ax.spines[['top', 'right']].set_visible(False)
#     ax.legend(frameon=False, fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)

# # --- make a grid of subplots ---
# fig, axs = plt.subplots(2, 3, figsize=(15, 8))
# axs = axs.flatten()
# colors = sns.color_palette("deep", len(metrics))

# for i, (title, data) in enumerate(metrics.items()):
#     plot_metric(axs[i], data, title, "Value", colors[i])

# fig.suptitle("Performance Across Checkpoints", fontsize=16, weight='bold')
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()