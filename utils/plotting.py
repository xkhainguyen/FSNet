import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from pathlib import Path


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
    tight_layout=True,
    suptitle=None,
):
    """
    Plot multiple metrics as mean ± std bar plots in a grid (ggplot-style).

    Args:
        metrics: list of tuples (metric_data, metric_name, logy, palette)
            - metric_data: array-like, shape (num_checkpoints, num_seeds)
            - metric_name: str
            - logy: bool
            - palette: str or list (uses first if list)
        checkpoints: list/array of checkpoint labels
    """

    checkpoints = list(checkpoints)
    x = np.arange(len(checkpoints))
    width = 0.6

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.ravel(axes)

    panel_grey = "#EBEBEB"

    for idx, (metric, name, logy, palette) in enumerate(metrics):
        if idx >= len(axes):
            break
        ax = axes[idx]

        color = palette[0] if isinstance(palette, (list, tuple)) else palette

        metric = np.asarray(metric)

        means = np.nanmean(metric, axis=1)
        stds  = np.nanstd(metric, axis=1)

        # Panel style
        ax.set_facecolor(panel_grey)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="white", linewidth=1.2)
        ax.grid(False, axis="x")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

        # Bars with error bars
        ax.bar(
            x, means, width,
            yerr=stds,
            color=color,
            edgecolor="black",
            linewidth=1.6,
            capsize=6,
            error_kw=dict(ecolor="black", lw=1.6, capthick=1.6)
        )

        if logy:
            ax.set_yscale("log")

        # Labels
        ax.set_title(name, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in checkpoints], fontsize=10)
        ax.tick_params(axis="y", labelsize=10)

    # Remove unused axes
    for ax in axes[len(metrics):]:
        ax.remove()

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

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



def plot_metrics_grid_noseeds_hline(
    metrics,
    checkpoints,
    group_names=None,
    group_colors=None,
    group_styles=None,
    *,
    nrows=2,
    ncols=3,
    figsize=(16, 9),
    x_label="Labels",
    y_label="Value",
    font_scale=1.1,
    style="whitegrid",
    tight_layout=True,
    suptitle=None,
):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------
    # Basic setup
    # -------------------------
    num_groups = metrics[0][0].shape[1]

    if group_names is None:
        group_names = {g: f"Group {g}" for g in range(num_groups)}

    DEFAULT_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    if group_colors is None:
        group_colors = {g: DEFAULT_COLORS[g % len(DEFAULT_COLORS)] for g in range(num_groups)}

    if group_styles is None:
        group_styles = {}

    # enforce a deterministic group ordering
    sorted_groups = sorted(group_names.keys())
    hue_order = [group_names[g] for g in sorted_groups]

    # baseline is the first checkpoint (e.g., 0 labels)
    baseline_ckpt = checkpoints[0]
    x_values = list(checkpoints[1:])  # only these have bars

    sns.set_theme(style=style, font_scale=font_scale)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    handles_for_legend = None
    labels_for_legend = None

    # -------------------------
    # Loop over metrics
    # -------------------------
    for idx, (metric, name, logy, _) in enumerate(metrics):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # metric shape: (num_checkpoints, num_groups)
        assert metric.shape[0] == len(checkpoints), "metric/checkpoints mismatch"
        assert metric.shape[1] == num_groups, "metric/group mismatch"

        baseline_row = metric[0, :]  # values at baseline_ckpt (all groups)

        # Build dataframe for non-baseline checkpoints
        rows = []
        for ci, ckpt in enumerate(checkpoints[1:], start=1):  # skip index 0 (baseline)
            for g in sorted_groups:
                rows.append((ckpt, group_names[g], metric[ci, g]))

        df = pd.DataFrame(rows, columns=[x_label, "Group", y_label])
        # ensure x axis order is [100, 1000, 10000] etc.
        df[x_label] = pd.Categorical(df[x_label], categories=x_values, ordered=True)

        # -------------------------
        # Draw bars
        # -------------------------
        bar = sns.barplot(
            data=df,
            x=x_label,
            y=y_label,
            hue="Group",
            hue_order=hue_order,  # IMPORTANT: fixes seaborn reordering
            ax=ax,
            palette=[group_colors[g] for g in sorted_groups],
            edgecolor="black",
            linewidth=0.5,
        )

        # -------------------------
        # Apply group_styles correctly
        # -------------------------
        patches = bar.patches
        n_x = len(x_values)
        n_g = len(sorted_groups)

        # seaborn bar order: for each x, for each hue in hue_order
        # index -> x_idx = i // n_g, group_idx = i % n_g
        for i, patch in enumerate(patches):
            g_local_idx = i % n_g                 # 0..n_g-1  within sorted_groups
            g = sorted_groups[g_local_idx]        # actual group index

            if g in group_styles:
                style = group_styles[g]
                if "hatch" in style:
                    patch.set_hatch(style["hatch"])
                if "alpha" in style:
                    patch.set_alpha(style["alpha"])
                if "edgecolor" in style:
                    patch.set_edgecolor(style["edgecolor"])
                if "linewidth" in style:
                    patch.set_linewidth(style["linewidth"])

        # -------------------------
        # Baseline horizontal line
        # -------------------------
        # all 4 at ckpt=0 are the same run → choose any, or mean
        baseline_val = float(baseline_row.mean())
        baseline_line = ax.axhline(
            baseline_val,
            linestyle="--",
            linewidth=2.0,
            color="tab:blue",
            label=f"Baseline (Labels={baseline_ckpt})",
        )

        # -------------------------
        # Legend (only once)
        # -------------------------
        if handles_for_legend is None:
            h_bars, l_bars = ax.get_legend_handles_labels()  # groups only
            handles_for_legend = h_bars + [baseline_line]
            labels_for_legend = l_bars + [f"Baseline (Labels={baseline_ckpt})"]

        ax.get_legend().remove()

        # Formatting
        ax.set_title(name, fontsize=12, weight="bold")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_facecolor("white")

    # remove unused subplots
    for ax in axes[len(metrics):]:
        ax.remove()

    # -------------------------
    # Global legend
    # -------------------------
    fig.legend(
        handles_for_legend,
        labels_for_legend,
        loc="upper center",
        ncol=len(sorted_groups) + 1,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, weight="bold", y=1.08)

    if tight_layout:
        plt.tight_layout()

    plt.show()


def load_results(file_path):
    """
    Load results from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Dictionary containing the loaded results.
    """
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None 

def get_train_metrics(results):
    """
    Extract training metrics from the results dictionary.

    Args:
        results (dict): Dictionary containing training history.

    Returns:
        dict: Dictionary containing lists of training metrics.
    """
    train_metrics = {
        'obj': [],
        'loss': [],
        # 'opt_gap': [],
        'eq_violation_l1': [],
        'ineq_violation_l1': [],
        # 'distance': []    
    }

    for k in range(len(results['train_history'])):
        for key in train_metrics.keys():
            train_metrics[key].append(results['train_history'][k][key])
    
    return train_metrics

# dict_keys(['val_history', 'test_results'])
# dict_keys(['epoch', 'opt_gap_mean', 'opt_gap_std', 'opt_gap_max', 
#            'eq_violation_l1_mean', 'eq_violation_l1_max', 
#            'ineq_violation_l1_mean', 'ineq_violation_l1_max'])

def get_val_metrics(results):
    """
    Extract validation metrics from the results dictionary.

    Args:
            results (dict): Dictionary containing validation history.

    Returns:
            dict: Dictionary containing lists of validation metrics.
    """
    val_metrics = {
            'opt_gap_mean': [],
            'opt_gap_std': [],
            'opt_gap_max': [],
            'eq_violation_l1_mean': [],
            'eq_violation_l1_max': [],
            'ineq_violation_l1_mean': [],
            'ineq_violation_l1_max': [],
            'merit_mean': [],
            'merit_max': []
    }

    for k in range(len(results['val_history'])):
        for key in val_metrics.keys():
            val_metrics[key].append(results['val_history'][k][key])
    return val_metrics

# Get batch comparison data from the loaded results
def show_table(results, batch_size=512):
    batch_comparison = results['test_results']['batch_size_comparison']
    # Create a more readable DataFrame by expanding the metrics
    readable_data = []
    data = batch_comparison[batch_size]
    metrics = data['metrics']
    row = {
        'Batch Size': batch_size,
        'Opt Gap Mean': f"{metrics['opt_gap_mean']:.2%}",
        # 'Opt Gap Std': f"{metrics['opt_gap_std']:.4f}",
        # 'Opt Gap Max': f"{metrics['opt_gap_max']:.4f}",
        'Eq Violation Mean': f"{metrics['eq_violation_l1_mean']:.2e}",
        # 'Eq Violation Max': f"{metrics['eq_violation_l1_max']:.2e}",
        'Ineq Violation Mean': f"{metrics['ineq_violation_l1_mean']:.2e}",
        # 'Ineq Violation Max': f"{metrics['ineq_violation_l1_max']:.2e}",
        # 'Average Batch Time (s)': f"{metrics['avg_inference_time']:.4f}",
    }
    readable_data.append(row)

    df_readable = pd.DataFrame(readable_data)
    df_readable_styled = df_readable.style.set_properties(**{'text-align': 'center'})
    display(df_readable_styled)