"""Transport plans of hard SR, DSR and EROT for the same particles and offset.

Paper Fig. 1 (introduction, run with --taus 0.1) and Fig. 5 (appendix, all
temperatures). From the repository root:

    python -m appendix.empirical_analysis.transport_maps
    python -m appendix.empirical_analysis.transport_maps --taus 0.1 \
        --output T_heatmap_combined_N25.png
"""

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch

from appendix.empirical_analysis import oracle_filter as oracle_filter_lib


@torch.no_grad()
def transport_plans(
    num_particles: int,
    step: int,
    seq_length: int,
    smnr_db: float,
    sigma_e2_db: float,
    taus: list[float],
) -> list[tuple[str, np.ndarray]]:
    """Transport plans of hard SR, DSR at each tau and EROT.

    Args:
        num_particles: Number of particles of the particle filter.
        step: Time step of the filter at which the plans are computed.
        seq_length: Number of time steps of the simulated sequence; it affects
            the particles at `step`.
        smnr_db: Signal-to-measurement noise ratio of the observations in dB.
        sigma_e2_db: Process noise variance of the Lorenz-63 dynamics, in dB.
        taus: Temperatures of the DSR relaxation, one panel each.

    Returns:
        (label, T) per plan.
    """
    steps = oracle_filter_lib.run_filter(
        num_particles, seq_length, smnr_db, sigma_e2_db
    )
    state = next(itertools.islice(steps, step, None))

    hard = torch.zeros(num_particles, num_particles)
    hard[torch.arange(num_particles), state.ancestors] = 1.0
    cdf = torch.cumsum(state.weights, dim=0)
    plans = [('Hard SR', hard)]
    plans += [
        (
            f'DSR ($\\tau={tau}$)',
            oracle_filter_lib.dsr_transport(cdf, state.positions, tau),
        )
        for tau in taus
    ]
    plans.append(
        (
            rf'EROT ($\epsilon={oracle_filter_lib.EROT_EPS}$)',
            oracle_filter_lib.erot_transport(state.particles, state.weights),
        )
    )
    return [(label, plan.numpy()) for label, plan in plans]


def plot_transport_plans(
    plans: list[tuple[str, np.ndarray]], path: str
) -> None:
    """Saves the plans side by side.

    Args:
        plans: Panel title and transport plan (N, N) for each panel.
        path: Path of the output image.
    """
    matrices = [plan for _, plan in plans]
    vmin = min(m.min() for m in matrices)
    vmax = max(m.max() for m in matrices)
    num_particles = len(matrices[0])

    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.size': 9,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'mathtext.fontset': 'cm',
        }
    )
    fig, axes = plt.subplots(
        1,
        len(plans),
        figsize=(6.75 / 3 * len(plans), 2.4),
        gridspec_kw={'wspace': 0.04},
        constrained_layout=False,
    )
    for ax, (label, plan) in zip(axes, plans):
        image = ax.imshow(
            plan, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax
        )
        ax.set_xlabel('Ancestor particle $i$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, pad=4)
    axes[0].set_yticks(range(0, num_particles, max(1, num_particles // 5)))
    axes[0].set_ylabel('Resampled particle $j$')
    fig.colorbar(
        image, ax=axes, shrink=0.92, pad=0.015, aspect=28, label='$T_{ji}$'
    )
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_particles', type=int, default=25)
    parser.add_argument(
        '--taus', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.5, 1.0]
    )
    parser.add_argument('--step', type=int, default=150)
    # The simulated sequence, so the particles at `step`, depend on its length.
    parser.add_argument('--seq_length', type=int, default=200)
    parser.add_argument('--smnr_db', type=float, default=10.0)
    parser.add_argument('--sigma_e2_db', type=float, default=-10.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Defaults to T_heatmap_combined_N=<num_particles>.png.',
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    plans = transport_plans(
        args.num_particles,
        args.step,
        args.seq_length,
        args.smnr_db,
        args.sigma_e2_db,
        args.taus,
    )
    plot_transport_plans(
        plans, args.output or f'T_heatmap_combined_N={args.num_particles}.png'
    )


if __name__ == '__main__':
    main()
