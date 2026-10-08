"""Lorenz-63 particle filter with the true dynamics, and transport-plan metrics.

Shared by the scripts of the paper appendix "Empirical Analysis of the DSR
Relaxation". The filter resamples with hard systematic resampling (SR); at
each step the DSR and EROT transport plans are built from the same particles,
weights and offset u_0, so they can be compared to the hard SR assignment.
"""

from collections.abc import Iterator
import dataclasses

import numpy as np
import torch

import lorenz_attractor as lorenz_attractor_lib
import resampling as resampling_lib

STATE_DIM = 3
OBSERVATION_DIM = 64
EROT_EPS = 0.5


@dataclasses.dataclass
class ResamplingStep:
    """Filter state at one resampling step, before resampling.

    Attributes:
        particles: (N, D) particles.
        weights: (N,) normalized weights.
        positions: (N,) systematic comb u_0 + j / N.
        ancestors: (N,) hard SR ancestors; the filter continues with these.
        random_ancestors: (N,) uniformly random permutation, a no-information
            baseline for the particle distance.
    """

    particles: torch.Tensor
    weights: torch.Tensor
    positions: torch.Tensor
    ancestors: torch.Tensor
    random_ancestors: torch.Tensor


def run_filter(
    num_particles: int,
    seq_length: int,
    smnr_db: float,
    sigma_e2_db: float,
) -> Iterator[ResamplingStep]:
    """Simulates one Lorenz-63 sequence and filters it.

    Args:
        num_particles: Number of particles of the particle filter.
        seq_length: Number of time steps of each simulated sequence.
        smnr_db: Signal-to-measurement noise ratio of the observations in dB.
        sigma_e2_db: Process noise variance of the Lorenz-63 dynamics, in dB.

    Yields:
        The filter state at every step, before resampling.
    """
    observation_fn = lorenz_attractor_lib.camera_observation
    model = lorenz_attractor_lib.LorenzSSM(
        n_states=STATE_DIM,
        n_obs=OBSERVATION_DIM,
        observation_fn=observation_fn,
    )
    _, observations = model.generate_single_sequence(
        seq_length,
        sigma_e2_dB=sigma_e2_db,
        smnr_dB=smnr_db,
    )
    observations = torch.from_numpy(observations).float()
    observation_cov = torch.from_numpy(model.Cw).float()
    state_cov = torch.from_numpy(model.Ce).float()

    particles = torch.randn(num_particles, STATE_DIM) / 10**0.5
    for observation in observations:
        means = np.stack([model.f_linearize(x.numpy()) for x in particles])
        particles = torch.distributions.MultivariateNormal(
            torch.from_numpy(means).float(),
            state_cov,
        ).sample()
        # numpy, as in the published runs; the torch branch rounds differently.
        predicted = torch.from_numpy(observation_fn(particles.numpy())).float()
        log_weights = torch.distributions.MultivariateNormal(
            predicted,
            observation_cov.unsqueeze(0).expand(num_particles, -1, -1),
        ).log_prob(observation)
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))

        u0 = torch.rand(1) / num_particles
        positions = (
            u0
            + torch.arange(num_particles, dtype=torch.float32) / num_particles
        )
        ancestors = hard_assignment(torch.cumsum(weights, dim=0), positions)
        yield ResamplingStep(
            particles,
            weights,
            positions,
            ancestors,
            torch.randperm(num_particles),
        )
        particles = particles[ancestors]


def hard_assignment(cdf: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Hard SR ancestors of the comb `positions`.

    Args:
        cdf: Cumulative sum of the normalized weights (N,), the last entry is 1.
        positions: Systematic resampling comb (N,), u_j = u_0 + j / N.

    Returns:
        Ancestor indices (N,).
    """
    return torch.clamp(torch.searchsorted(cdf, positions), 0, len(cdf) - 1)


def dsr_transport(
    cdf: torch.Tensor, positions: torch.Tensor, tau: float
) -> torch.Tensor:
    """DSR transport plan.

    Args:
        cdf: Cumulative sum of the normalized weights (N,), the last entry is 1.
        positions: Systematic resampling comb (N,), u_j = u_0 + j / N.
        tau: Temperature of the DSR relaxation.

    Returns:
        T (N, N); T[j, i] is the weight of ancestor i in particle j.
    """
    cdf_prev = torch.cat([torch.zeros(1, device=cdf.device), cdf[:-1]])
    soft = torch.sigmoid(
        (positions[:, None] - cdf_prev[None, :]) / tau
    ) - torch.sigmoid((positions[:, None] - cdf[None, :]) / tau)
    return soft / (soft.sum(dim=1, keepdim=True) + 1e-8)


def erot_transport(
    particles: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """EROT transport plan (eps = 0.5, at most 100 Sinkhorn iterations).

    Args:
        particles: Particles (N, D) before resampling.
        weights: Normalized particle weights (N,).

    Returns:
        T (N, N).
    """
    return (
        resampling_lib.transport_function(
            particles.unsqueeze(0),
            torch.log(weights).unsqueeze(0),
            eps=EROT_EPS,
            scaling=0.75,
            threshold=1e-3,
            max_iter=100,
            n=len(weights),
            device='cpu',
        )
        .squeeze(0)
        .detach()
        .float()
    )


def morton_order(
    step: ResamplingStep,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The step's particles after Morton sorting.

    Args:
        step: Filter state at one step, before resampling.

    Returns:
        Sorted particles (N, D), their CDF (N,) and hard SR ancestors (N,).
    """
    _, order = resampling_lib.morton_sort(step.particles.unsqueeze(0))
    order = order[0].long()
    weights = step.weights[order]
    cdf = torch.cumsum(weights / weights.sum(), dim=0)
    return step.particles[order], cdf, hard_assignment(cdf, step.positions)


def effective_ancestors(transport: torch.Tensor) -> float:
    """Mean over resampled particles of exp(entropy of their ancestors).

    Args:
        transport: Transport plan (N, N); row j holds the ancestor weights of
            the resampled particle j.

    Returns:
        The mean effective number of ancestors.
    """
    transport = transport.clamp(min=1e-10)
    entropy = -(transport * transport.log()).sum(dim=1)
    return entropy.exp().mean().item()


def count_correlation(
    ancestors: torch.Tensor, transport: torch.Tensor
) -> float:
    """Pearson correlation of hard SR and DSR offspring counts per ancestor.

    Args:
        ancestors: Hard systematic resampling ancestor indices (N,).
        transport: Transport plan (N, N); row j holds the ancestor weights of
            the resampled particle j.

    Returns:
        The correlation.
    """
    n = len(ancestors)
    hard_counts = torch.zeros(n).scatter_add_(0, ancestors, torch.ones(n))
    soft_counts = transport.sum(dim=0)
    hc = hard_counts - hard_counts.mean()
    sc = soft_counts - soft_counts.mean()
    return (
        (hc * sc).sum()
        / (torch.sqrt((hc**2).sum()) * torch.sqrt((sc**2).sum()) + 1e-10)
    ).item()


def mean_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean Euclidean distance between corresponding rows.

    Args:
        x: Points (N, D), e.g. resampled particles.
        y: Points (N, D) compared row by row with x.

    Returns:
        The mean distance.
    """
    return ((x - y) ** 2).sum(dim=-1).sqrt().mean().item()


def plan_metrics(
    transport: torch.Tensor, particles: torch.Tensor, ancestors: torch.Tensor
) -> dict[str, float]:
    """How closely a transport plan follows hard SR.

    Args:
        transport: Transport plan (N, N); row j holds the ancestor weights of
            the resampled particle j.
        particles: Particles (N, D) before resampling, in the order of the plan.
        ancestors: Hard SR ancestor indices (N,), in the same order.

    Returns:
        {'correlation', 'l2', 'eff_ancestors'}.
    """
    return {
        'correlation': count_correlation(ancestors, transport),
        'l2': mean_distance(transport @ particles, particles[ancestors]),
        'eff_ancestors': effective_ancestors(transport),
    }
