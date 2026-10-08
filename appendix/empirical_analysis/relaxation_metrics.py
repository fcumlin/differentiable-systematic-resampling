"""How closely DSR and EROT transport plans follow hard systematic resampling.

Paper appendix "Empirical Analysis of the DSR Relaxation": the effective number
of ancestors, the correlation of offspring counts and the per-particle L2
distance to hard SR, for DSR with and without Morton sorting and for EROT.
Used by plot_relaxation_metrics.py. From the repository root:

    python -m appendix.empirical_analysis.relaxation_metrics \
        --output relaxation_metrics.json
"""

import argparse
import collections
import json

import numpy as np
import torch

from appendix.empirical_analysis import oracle_filter as oracle_filter_lib

TAUS = (0.01, 0.05, 0.1, 0.5, 1.0)
# EROT does not depend on tau, so it is evaluated once, during this tau's pass.
EROT_PASS_TAU = 0.5


def sequence_metrics(
    steps: list[oracle_filter_lib.ResamplingStep],
    tau: float,
    with_erot: bool,
) -> dict[tuple[str, str], float]:
    """Means over the steps of one sequence.

    Args:
        steps: Filter states at every step of one sequence.
        tau: Temperature of the DSR relaxation.
        with_erot: Whether to also evaluate the EROT transport plan.

    Returns:
        (method, metric) -> mean.
    """
    values = collections.defaultdict(list)
    for step in steps:
        cdf = torch.cumsum(step.weights, dim=0)
        plans = {
            'unsorted': (
                oracle_filter_lib.dsr_transport(cdf, step.positions, tau),
                step.particles,
                step.ancestors,
            )
        }
        particles, cdf, ancestors = oracle_filter_lib.morton_order(step)
        plans['morton'] = (
            oracle_filter_lib.dsr_transport(cdf, step.positions, tau),
            particles,
            ancestors,
        )
        if with_erot:
            plans['erot'] = (
                oracle_filter_lib.erot_transport(step.particles, step.weights),
                step.particles,
                step.ancestors,
            )
        for method, plan in plans.items():
            for metric, value in oracle_filter_lib.plan_metrics(*plan).items():
                values[method, metric].append(value)

        values['random', 'l2'].append(
            oracle_filter_lib.mean_distance(
                step.particles[step.random_ancestors],
                step.particles[step.ancestors],
            )
        )
    return {key: np.mean(v) for key, v in values.items()}


@torch.no_grad()
def relaxation_metrics(
    num_particles: int,
    num_sequences: int,
    seq_length: int,
    smnr_db: float,
    sigma_e2_db: float,
    taus: tuple[float, ...] = TAUS,
) -> dict[float, dict[str, dict[str, dict[str, float]]]]:
    """Mean and sample std over sequences of every method and metric.

    EROT appears only under tau = EROT_PASS_TAU.

    Args:
        num_particles: Number of particles of the particle filter.
        num_sequences: Number of simulated sequences to average over.
        seq_length: Number of time steps of each simulated sequence.
        smnr_db: Signal-to-measurement noise ratio of the observations in dB.
        sigma_e2_db: Process noise variance of the Lorenz-63 dynamics, in dB.
        taus: Temperatures of the DSR relaxation.

    Returns:
        results[tau][method][metric]['mean' | 'std'].
    """
    results = {}
    for tau in taus:
        per_sequence = collections.defaultdict(list)
        for _ in range(num_sequences):
            steps = list(
                oracle_filter_lib.run_filter(
                    num_particles, seq_length, smnr_db, sigma_e2_db
                )
            )
            metrics = sequence_metrics(
                steps, tau, with_erot=tau == EROT_PASS_TAU
            )
            for key, value in metrics.items():
                per_sequence[key].append(value)
        results[tau] = collections.defaultdict(dict)
        for (method, metric), values in per_sequence.items():
            results[tau][method][metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
            }
    return results


def print_results(num_particles: int, results) -> None:
    """Prints the output of relaxation_metrics.

    Args:
        num_particles: Number of particles of the particle filter.
        results: Output of relaxation_metrics for that number of particles.
    """
    print(f'\nN = {num_particles}')
    print(
        f"{'tau':>6}  {'method':<9}{'N_eff':>16}{'correlation':>18}{'L2':>18}"
    )
    for tau, methods in results.items():
        for method, metrics in methods.items():
            cells = [
                f"{metrics[m]['mean']:8.4f} +- {metrics[m]['std']:.4f}"
                if m in metrics
                else ''
                for m in ('eff_ancestors', 'correlation', 'l2')
            ]
            print(f'{tau:>6}  {method:<9}' + ''.join(f'{c:>18}' for c in cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_particles', type=int, nargs='+', default=[25, 50, 100, 200]
    )
    parser.add_argument('--num_sequences', type=int, default=10)
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--smnr_db', type=float, default=10.0)
    parser.add_argument('--sigma_e2_db', type=float, default=-10.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='JSON file for plot_relaxation_metrics.py.',
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_results = {}
    for num_particles in args.num_particles:
        results = relaxation_metrics(
            num_particles,
            args.num_sequences,
            args.seq_length,
            args.smnr_db,
            args.sigma_e2_db,
        )
        print_results(num_particles, results)
        all_results[num_particles] = results

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=1)
        print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
