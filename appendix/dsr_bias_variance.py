"""Bias, variance and MSE of resampled estimates.

Paper appendix "Bias and Variance of DSR Estimates", Table 6.

For a fixed particle set and weights, compares the estimator
I_hat = sum_j w_new_j f(x_new_j) after resampling with the weighted target
I = sum_i w_i f(x_i), over many resampling draws. Hard systematic and
multinomial resampling satisfy E[N_i] = N w_i, so both are unbiased and differ
only in variance; systematic having the lower variance provides "soft"
jusification of relaxing systematic resampling. DSR at finite tau does not
preserve expected replication counts, so it trades bias for variance and MSE is
the informative quantity.

From the repository root (run the self-test first):

    python -m appendix.dsr_bias_variance --self-test
    python -m appendix.dsr_bias_variance
"""

import argparse
from collections.abc import Callable
import math

import torch

import resampling as resampling_lib

DTYPE = torch.float64
TAUS = (0.01, 0.05, 0.1, 0.5, 1.0)

Resampler = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def hard_systematic(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard systematic resampling of a batch.

    Args:
        x: Particle sets (B, N, d): B sets of N particles in d dimensions.
        w: Normalized particle weights (B, N), summing to one over N.

    Returns:
        Resampled particles (B, N, d) and uniform weights (B, N).
    """
    B, N, d = x.shape
    cdf = torch.cumsum(w, 1)
    u = (
        torch.rand(B, 1, dtype=x.dtype) / N
        + torch.arange(N, dtype=x.dtype).view(1, N) / N
    )
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp(max=N - 1)
    return (
        torch.gather(x, 1, idx.unsqueeze(-1).expand(B, N, d)),
        torch.full_like(w, 1.0 / N),
    )


def multinomial(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multinomial resampling of a batch.

    Args:
        x: Particle sets (B, N, d): B sets of N particles in d dimensions.
        w: Normalized particle weights (B, N), summing to one over N.

    Returns:
        Resampled particles (B, N, d) and uniform weights (B, N).
    """
    B, N, d = x.shape
    idx = torch.multinomial(w, N, replacement=True)
    return (
        torch.gather(x, 1, idx.unsqueeze(-1).expand(B, N, d)),
        torch.full_like(w, 1.0 / N),
    )


def make_particle_set(
    n_draws: int,
    N: int,
    d: int,
    sigma_prior: float = 1.0,
    sigma_lik: float = 0.5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One particle set with likelihood weights, replicated n_draws times.

    Particles come from N(0, sigma_prior^2 I) and are weighted by a Gaussian
    likelihood at the origin. Replicating the set along the batch dimension
    makes the resampling draw the only source of randomness.

    Args:
        n_draws: Number of resampling draws of the fixed particle set.
        N: Number of particles in each particle set.
        d: Dimension of each particle.
        sigma_prior: Standard deviation of the particle distribution.
        sigma_lik: Standard deviation of the likelihood giving the weights.
        seed: Random seed for drawing the particle set.

    Returns:
        Particles (n_draws, N, d) and weights (n_draws, N).
    """
    torch.manual_seed(seed)
    x = sigma_prior * torch.randn(1, N, d, dtype=DTYPE)
    w = torch.softmax(-0.5 * (x**2).sum(-1) / sigma_lik**2, dim=1)
    return (
        x.expand(n_draws, N, d).contiguous(),
        w.expand(n_draws, N).contiguous(),
    )


def estimator_error(
    x: torch.Tensor,
    w: torch.Tensor,
    resample: Resampler,
    f: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, float]:
    """Bias, variance and MSE of sum_j w_new_j f(x_new_j) vs sum_i w_i f(x_i).

    The estimates are vectors: bias is the norm of the mean error, variance
    the expected squared distance to the mean estimate, and MSE = bias^2 +
    variance.

    Args:
        x: Particle set (B, N, d), the same set repeated over the B draws.
        w: Normalized weights (B, N), the same over the B draws.
        resample: Resampling scheme mapping (x, w) to resampled (x, w).
        f: Test function applied elementwise to the particles.

    Returns:
        {'bias', 'var', 'mse'}.
    """
    target = (w.unsqueeze(-1) * f(x)).sum(1)
    with torch.no_grad():
        x_new, w_new = resample(x, w)
    w_new = w_new / w_new.sum(1, keepdim=True)
    estimate = (w_new.unsqueeze(-1) * f(x_new)).sum(1)

    mean_estimate = estimate.mean(0, keepdim=True)
    bias2 = float(((mean_estimate - target[:1]) ** 2).sum())
    return {
        'bias': math.sqrt(bias2),
        'var': float(((estimate - mean_estimate) ** 2).sum(-1).mean()),
        'mse': float(((estimate - target) ** 2).sum(-1).mean()),
    }


def build_methods(taus: tuple[float, ...], eps: float) -> dict[str, Resampler]:
    """Hard SR and multinomial, then DSR at each tau, soft and EROT.

    Args:
        taus: Temperatures of the DSR relaxation, one table row each.
        eps: Entropic regularization of EROT.

    Returns:
        Method name -> resampling scheme.
    """

    def detached(resampled):
        return tuple(v.detach() for v in resampled[:2])

    methods = {'hard SR': hard_systematic, 'multinomial': multinomial}
    for tau in taus:
        methods[f'DSR tau={tau}'] = lambda x, w, tau=tau: detached(
            resampling_lib.resampler_systematic_morton(
                x, w, tau=tau, requires_grad=False
            )
        )
    methods['soft (a=0.5)'] = lambda x, w: detached(
        resampling_lib.soft_resampling(
            x, w, alpha=0.5, num_particles=x.shape[1]
        )
    )
    methods[f'EROT eps={eps}'] = lambda x, w: detached(
        resampling_lib.resampler_ot(
            x, w, eps=eps, max_iter=100, threshold=1e-3, device='cpu'
        )
    )
    return methods


def run(
    N: int = 100,
    d: int = 3,
    n_draws: int = 20000,
    taus: tuple[float, ...] = TAUS,
    eps: float = 0.5,
    seed: int = 0,
) -> None:
    """Prints the table for f(x) = x and f(x) = x^2.

    Args:
        N: Number of particles in each particle set.
        d: Dimension of each particle.
        n_draws: Number of resampling draws of the fixed particle set.
        taus: Temperatures of the DSR relaxation, one table row each.
        eps: Entropic regularization of EROT.
        seed: Random seed for drawing the particle set.
    """
    x, w = make_particle_set(n_draws, N, d, seed=seed)
    test_functions = {'f(x)=x': lambda z: z, 'f(x)=x^2': lambda z: z**2}
    methods = build_methods(taus, eps)

    for f_name, f in test_functions.items():
        print(f'\n=== {f_name} ===  N={N}, d={d}, draws={n_draws}')
        print(f"{'method':<16}{'bias':>12}{'variance':>14}{'MSE':>14}")
        print('-' * 56)
        for name, resample in methods.items():
            r = estimator_error(x, w, resample, f)
            print(
                f"{name:<16}{r['bias']:>12.3e}{r['var']:>14.3e}"
                f"{r['mse']:>14.3e}"
            )
        print('-' * 56)
    print('\nhard SR and multinomial are unbiased (E[N_i] = N w_i); the bias')
    print('column for those rows is Monte-Carlo error and should shrink as')
    print('1/sqrt(draws). DSR is not unbiased at finite tau.')


def self_test(N: int = 100, d: int = 3, n_draws: int = 40000, seed: int = 0):
    """Checks the harness on the classical systematic-vs-multinomial result.

    Args:
        N: Number of particles in each particle set.
        d: Dimension of each particle.
        n_draws: Number of resampling draws of the fixed particle set.
        seed: Random seed for drawing the particle set.
    """
    x, w = make_particle_set(n_draws, N, d, seed=seed)
    systematic = estimator_error(x, w, hard_systematic, lambda z: z)
    multi = estimator_error(x, w, multinomial, lambda z: z)
    print(f"{'':<14}{'bias':>12}{'variance':>14}")
    print(
        f"{'hard SR':<14}{systematic['bias']:>12.3e}{systematic['var']:>14.3e}"
    )
    print(f"{'multinomial':<14}{multi['bias']:>12.3e}{multi['var']:>14.3e}")
    print(
        '\nvariance ratio multinomial / systematic: '
        f"{multi['var'] / systematic['var']:.2f}"
    )
    print(
        'Expected: both biases at Monte-Carlo level, ratio > 1 (systematic has'
    )
    print(
        'the lower variance). If the ratio is near 1 or below, the harness is'
    )
    print('wrong -- most likely the particle set is varying across draws.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--eps', type=float, default=0.5)
    args = parser.parse_args()

    torch.set_default_dtype(DTYPE)
    if args.self_test:
        self_test(N=args.N, d=args.dim)
    else:
        run(N=args.N, d=args.dim, n_draws=args.draws, eps=args.eps)


if __name__ == '__main__':
    main()
