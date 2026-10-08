"""Creates Fig. 6 of the appendix (effective ancestors and L2 distance).

Plots the output of relaxation_metrics.py; by default the published run
(10 sequences of length 100). From the repository root:

    python -m appendix.empirical_analysis.plot_relaxation_metrics
    python -m appendix.empirical_analysis.plot_relaxation_metrics \
        --results relaxation_metrics.json
"""

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.lines
import matplotlib.pyplot as plt

PUBLISHED_RESULTS = os.path.join(
    os.path.dirname(__file__), 'published_relaxation_metrics.json'
)
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7')
MARKERS = ('o', 's', '^', 'D')
FIGSIZE = (3.25, 2.6)


def series(results, num_particles, method, metric):
    """One curve of the results.

    Args:
        results: Output of relaxation_metrics.py loaded from JSON, as
            results[N][tau][method][metric]["mean" | "std"].
        num_particles: Number of particles of the curve.
        method: Resampling method: 'morton', 'unsorted', 'erot' or 'random'.
        metric: Metric: 'eff_ancestors', 'correlation' or 'l2'.

    Returns:
        Taus, means and stds.
    """
    by_tau = results[str(num_particles)]
    taus = sorted(by_tau, key=float)
    values = [by_tau[tau][method][metric] for tau in taus]
    return (
        [float(tau) for tau in taus],
        [v['mean'] for v in values],
        [v['std'] for v in values],
    )


def format_tau_axis(ax, taus):
    """Log-scale tau axis with one tick per tau.

    Args:
        ax: Axes whose x-axis is the temperature.
        taus: Temperatures, one tick each.
    """
    ax.set_xscale('log')
    ax.set_xlabel(r'Temperature $\tau$')
    ax.set_xticks(taus)
    ax.set_xticklabels([str(tau) for tau in taus])
    ax.set_xlim(0.007, 1.3)
    ax.grid(True, alpha=0.2, linewidth=0.4)


def save(fig, name):
    """Saves the figure as <name>.pdf and <name>.png.

    Args:
        fig: Figure to save.
        name: File name without extension.
    """
    fig.tight_layout(pad=0.3)
    for extension in ('pdf', 'png'):
        fig.savefig(f'{name}.{extension}', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}.pdf, {name}.png')


def plot_effective_ancestors(results, particle_counts):
    """Fig. "eff_ancestors": N_eff of DSR with Morton sorting.

    Args:
        results: Output of relaxation_metrics.py loaded from JSON, as
            results[N][tau][method][metric]["mean" | "std"].
        particle_counts: Numbers of particles, one curve each.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for k, n in enumerate(particle_counts):
        taus, mean, std = series(results, n, 'morton', 'eff_ancestors')
        ax.errorbar(
            taus,
            mean,
            yerr=std,
            color=COLORS[k % len(COLORS)],
            marker=MARKERS[k % len(MARKERS)],
            label=f'$N={n}$',
            capsize=2.5,
            capthick=0.7,
            zorder=3,
        )
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.annotate(
        'Hard SR', xy=(0.008, 1), xytext=(0.008, 1.10), fontsize=7, color='0.4'
    )
    format_tau_axis(ax, taus)
    ax.set_yscale('log')
    ax.set_ylabel(r'$N_{\mathrm{eff}}$')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, edgecolor='0.8')
    save(fig, 'eff_ancestors')


def plot_l2_distance(results, particle_counts):
    """Fig. "l2_distance": L2 to hard SR, with and without Morton sorting.

    Args:
        results: Output of relaxation_metrics.py loaded from JSON, as
            results[N][tau][method][metric]["mean" | "std"].
        particle_counts: Numbers of particles, one pair of curves each.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for k, n in enumerate(particle_counts):
        taus, mean, std = series(results, n, 'morton', 'l2')
        ax.errorbar(
            taus,
            mean,
            yerr=std,
            color=COLORS[k % len(COLORS)],
            marker=MARKERS[k % len(MARKERS)],
            capsize=2.5,
            capthick=0.7,
            zorder=3,
        )
        taus, mean, std = series(results, n, 'unsorted', 'l2')
        ax.errorbar(
            taus,
            mean,
            yerr=std,
            color=COLORS[k % len(COLORS)],
            marker=MARKERS[k % len(MARKERS)],
            linestyle='--',
            fillstyle='none',
            capsize=2.5,
            capthick=0.7,
            zorder=3,
            alpha=0.55,
        )
    handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            color=COLORS[k % len(COLORS)],
            marker=MARKERS[k % len(MARKERS)],
            label=f'$N={n}$',
        )
        for k, n in enumerate(particle_counts)
    ]
    handles += [
        matplotlib.lines.Line2D(
            [0], [0], color='0.4', linestyle='-', label='Morton'
        ),
        matplotlib.lines.Line2D(
            [0], [0], color='0.4', linestyle='--', alpha=0.8, label='Random'
        ),
    ]
    ax.legend(
        handles=handles,
        fontsize=6.5,
        loc='lower right',
        ncol=2,
        framealpha=0.9,
        edgecolor='0.8',
    )
    format_tau_axis(ax, taus)
    ax.set_ylabel(r'$\mathrm{L2}_{\mathrm{particles}}$')
    save(fig, 'l2_distance')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=PUBLISHED_RESULTS)
    args = parser.parse_args()
    with open(args.results) as f:
        results = json.load(f)
    particle_counts = sorted(int(n) for n in results)

    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 8.5,
            'ytick.labelsize': 8.5,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.linewidth': 0.6,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'lines.linewidth': 1.2,
            'lines.markersize': 4.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )
    plot_effective_ancestors(results, particle_counts)
    plot_l2_distance(results, particle_counts)


if __name__ == '__main__':
    main()
