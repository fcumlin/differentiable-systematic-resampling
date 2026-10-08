"""Reads and plots gradient norms from train.log files over multiple runs."""

import re
import numpy as np
import matplotlib.pyplot as plt


def read_gradient_norms(folder_paths, required_epochs=500):
    """
    Read mean and within-epoch std gradient norm per epoch from train.log files.
    Folders whose run did not reach required_epochs are skipped.

    Returns:
        means_per_run: list of lists of per-epoch mean gradient norms
        stds_per_run:  list of lists of per-epoch within-epoch std gradient norms
    """
    means_per_run, stds_per_run = [], []
    for folder in folder_paths:
        log_path = f'{folder}/train.log'
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            means = [float(v) for v in re.findall(r'Mean gradient norm=([\d.]+)', content)]
            stds  = [float(v) for v in re.findall(r'Mean gradient norm=[\d.]+ \(std=([\d.]+)\)', content)]
            if len(means) < required_epochs:
                print(f"Skipping {folder}: only {len(means)} epochs (need {required_epochs})")
                continue
            means_per_run.append(means)
            stds_per_run.append(stds)
        except FileNotFoundError:
            print(f"Log file not found: {log_path}")
    return means_per_run, stds_per_run


def read_gradient_norms_single(folder_path):
    """
    Read mean and std gradient norm per epoch from a single train.log.

    Returns:
        means: list of per-epoch mean gradient norms
        stds:  list of per-epoch std gradient norms (from the log file)
    """
    log_path = f'{folder_path}/train.log'
    with open(log_path, 'r') as f:
        content = f.read()
    means = [float(v) for v in re.findall(r'Mean gradient norm=([\d.]+)', content)]
    stds  = [float(v) for v in re.findall(r'Mean gradient norm=[\d.]+ \(std=([\d.]+)\)', content)]
    return means, stds


def compute_stats(means_per_run, start_epoch=1):
    """Return (epochs, mean, std) arrays aligned to the shortest run."""
    n_epochs = min(len(r) for r in means_per_run)
    array = np.array([r[start_epoch:n_epochs] for r in means_per_run])  # (n_runs, n_epochs)
    return np.arange(start_epoch + 1, n_epochs + 1), array.mean(axis=0), array.std(axis=0)


if __name__ == "__main__":
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'text.usetex': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.2,
        'lines.markersize': 4.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    base_dir = "runs/lorenz63"
    experiments = {
        "Hard": f"{base_dir}/hard_10db",
        "Soft": f"{base_dir}/soft_10db",
        "EROT": f"{base_dir}/erot_10db",
        "DSR":  f"{base_dir}/dsr_10db",
    }

    PLOT_SINGLE_SEED = True   # set False to show only the multi-run plot

    colors  = ["#0072B2", "#CC79A7", "#E69F00", "#009E73"]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    print(f"\nAll-runs summary (converged runs, epoch 50 onwards):")
    print(f"{'Method':<10}  {'Runs':>5}  {'Mean norm':>12}  {'Mean std':>10}")
    print("-" * 44)

    for (label, base), color in zip(experiments.items(), colors):
        folders = [f"{base}_{i}" for i in range(10)]
        means_per_run, stds_per_run = read_gradient_norms(folders)
        n_runs = len(means_per_run)
        start = 0
    
        all_means = np.concatenate([r[start:] for r in means_per_run])
        all_stds  = np.concatenate([r[start:] for r in stds_per_run])
        print(f"{label:<10}  {n_runs:>5}  {np.mean(all_means):>12.2f}  {np.mean(all_stds):>10.2f}")

        if PLOT_SINGLE_SEED:
            num = 4 if label == "Soft" else 5  # plot the single run for SOFT, which is more stable            
            means, stds = read_gradient_norms_single(f"{base}_{num}")
            means, stds = np.array(means), np.array(stds)
            epochs = np.arange(1, len(means) + 1)
            ax.semilogy(epochs, means, label=label, color=color)
            ax.fill_between(epochs, np.maximum(means - stds, 1e-9), means + stds,
                            alpha=0.15, color=color, linewidth=0)
        else:
            epochs, mean, std = compute_stats(means_per_run)
            ax.semilogy(epochs, mean, label=label, color=color)
            ax.fill_between(epochs, np.maximum(mean - std, 1e-9), mean + std,
                            alpha=0.15, color=color, linewidth=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean gradient norm")
    ax.set_ylim(bottom=1.0)
    ax.legend()

    plt.tight_layout()
    for out_path in ["gradient_norms.png", "gradient_norms.pdf"]:
        plt.savefig(out_path)
        print(f"\nPlot saved to {out_path}")
    plt.show()
