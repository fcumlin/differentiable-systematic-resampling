"""Evaluation-time marginal likelihood on the linear Gaussian SSM.

Paper appendix "Marginal likelihood at evaluation time", Table 14.

The learned LinearTransitionModel is itself linear Gaussian, so a Kalman filter
gives the exact evidence of the learned model, which splits the gap to the true
evidence into

    log p_true(y)  - log p_hat(y)       model error (the learned transition)
    log p_hat(y)   - E[log p_hat_N(y)]  Jensen gap (the N-particle filter)

where the filter uses hard systematic resampling for every method. The learned
covariance is diag(softplus(log_diag)), and nn.Linear(bias=False) computes
x @ W.T, so A_hat = A.weight without a transpose.

From the repository root (run the self-test first):

    python -m appendix.marginal_likelihood --self-test
    python -m appendix.marginal_likelihood --runs DSR=runs/linear/dsr_10db,...
"""

from __future__ import annotations

import argparse
import math
import pathlib

import torch
import torch.nn.functional as F

import dataset as dataset_lib

DTYPE = torch.float64
# Initial-state covariance scale of LinearSSMRotational.kalman_filter, which
# produced the dataset's posterior labels.
DATASET_P0_SCALE = 0.1


def kalman_loglik(
    y: torch.Tensor,
    A: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    m0: torch.Tensor | None = None,
    P0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact log p(y_1:T) of x_t = A x_(t-1) + e_t, y_t = x_t + w_t.

    Args:
        y: Observation sequences (B, T, d): B sequences of T steps each.
        A: State transition matrix (d, d), x_t = A x_(t-1) + e_t.
        Q: Covariance of the process noise e_t (d, d).
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        m0: Mean of the initial state (d,); zero if None.
        P0: Covariance of the initial state (d, d); DATASET_P0_SCALE * I if
            None, as in the dataset.

    Returns:
        Log-evidence per sequence (B,).
    """
    B, T, d = y.shape
    m = (
        torch.zeros(B, d, dtype=y.dtype)
        if m0 is None
        else m0.expand(B, d).clone()
    )
    if P0 is None:
        P0 = DATASET_P0_SCALE * torch.eye(d, dtype=y.dtype)
    P = P0.expand(B, d, d).clone()
    ll = torch.zeros(B, dtype=y.dtype)

    for t in range(T):
        m = m @ A.T
        P = A @ P @ A.T + Q
        S = P + R
        v = y[:, t] - m
        L = torch.linalg.cholesky(S)
        sol = torch.cholesky_solve(v.unsqueeze(-1), L).squeeze(-1)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        ll = ll - 0.5 * ((v * sol).sum(-1) + logdet + d * math.log(2 * math.pi))
        K = torch.cholesky_solve(P.transpose(-1, -2), L).transpose(-1, -2)
        m = m + (K @ v.unsqueeze(-1)).squeeze(-1)
        P = P - K @ S @ K.transpose(-1, -2)
        P = 0.5 * (P + P.transpose(-1, -2))
    return ll


def _systematic(
    x: torch.Tensor, W: torch.Tensor, gen: torch.Generator
) -> torch.Tensor:
    """Hard systematic resampling of a batch.

    Args:
        x: Batch of particle sets (B, N, d).
        W: Normalized particle weights (B, N).
        gen: Random generator for the resampling offsets.

    Returns:
        Resampled particles (B, N, d).
    """
    B, N, d = x.shape
    cdf = torch.cumsum(W, 1)
    u = (
        torch.rand(B, 1, generator=gen, dtype=x.dtype) / N
        + torch.arange(N, dtype=x.dtype).view(1, N) / N
    )
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp(max=N - 1)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(B, N, d))


def smc_loglik(
    y: torch.Tensor,
    A: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    N: int = 1000,
    ess_frac: float = 0.5,
    resample_every_step: bool = False,
    seed: int = 0,
    m0: torch.Tensor | None = None,
    P0: torch.Tensor | None = None,
) -> torch.Tensor:
    """log p_hat(y_1:T) of a bootstrap PF with hard systematic resampling.

    With the bootstrap proposal the incremental weight is p(y_t | x_t).

    Args:
        y: Observation sequences (B, T, d): B sequences of T steps each.
        A: State transition matrix (d, d), x_t = A x_(t-1) + e_t.
        Q: Covariance of the process noise e_t (d, d).
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        N: Number of particles of the particle filter.
        ess_frac: Resample when the ESS drops below ess_frac * N.
        resample_every_step: Resample at every step regardless of the ESS.
        seed: Random seed of the particle filter.
        m0: Mean of the initial state (d,); zero if None.
        P0: Covariance of the initial state (d, d); DATASET_P0_SCALE * I if
            None, as in the dataset.

    Returns:
        Estimated log-evidence per sequence (B,).
    """
    B, T, d = y.shape
    gen = torch.Generator().manual_seed(seed)
    Lq = torch.linalg.cholesky(Q)
    Lr = torch.linalg.cholesky(R)
    logdet_r = 2.0 * torch.log(torch.diagonal(Lr)).sum()
    r_inv = torch.cholesky_inverse(Lr)

    if m0 is None:
        m0 = torch.zeros(d, dtype=y.dtype)
    if P0 is None:
        P0 = DATASET_P0_SCALE * torch.eye(d, dtype=y.dtype)
    x = (
        m0
        + torch.randn(B, N, d, generator=gen, dtype=y.dtype)
        @ torch.linalg.cholesky(P0).T
    )
    log_W = torch.full((B, N), -math.log(N), dtype=y.dtype)
    ll = torch.zeros(B, dtype=y.dtype)

    for t in range(T):
        x = x @ A.T + torch.randn(B, N, d, generator=gen, dtype=y.dtype) @ Lq.T
        v = y[:, t].unsqueeze(1) - x
        log_w = -0.5 * (
            ((v @ r_inv) * v).sum(-1) + logdet_r + d * math.log(2 * math.pi)
        )
        # The evidence needs the unnormalized incremental weights, so it is
        # accumulated before normalizing.
        ll = ll + torch.logsumexp(log_W + log_w, dim=1)
        log_W = log_W + log_w
        log_W = log_W - torch.logsumexp(log_W, dim=1, keepdim=True)

        W = log_W.exp()
        ess = 1.0 / (W**2).sum(1)
        if resample_every_step or bool((ess < ess_frac * N).any()):
            x = _systematic(x, W, gen)
            log_W = torch.full((B, N), -math.log(N), dtype=y.dtype)
    return ll


def load_transition(ckpt_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Transition parameters of a LinearTransitionModel checkpoint.

    Args:
        ckpt_path: Path of a saved LinearTransitionModel state dict.

    Returns:
        A_hat (d, d) and Q_hat (d, d).
    """
    state = torch.load(ckpt_path, map_location='cpu')
    state = state.state_dict() if hasattr(state, 'state_dict') else state
    state = state.get('state_dict', state)
    key_a = next(k for k in state if k.endswith('A.weight'))
    key_d = next(k for k in state if k.endswith('log_diag'))
    A_hat = state[key_a].detach().to(DTYPE)
    Q_hat = torch.diag(F.softplus(state[key_d].detach().to(DTYPE)))
    return A_hat, Q_hat


def load_test_data(
    num_samples: int = 100, signal_length: int = 100, smnr_db: float = 10.0
) -> tuple[torch.Tensor, ...]:
    """Test observations and the true model of LinearSSMRotationalDataset.

    Args:
        num_samples: Number of test sequences to generate.
        signal_length: Number of time steps of each test sequence.
        smnr_db: Signal-to-measurement noise ratio of the observations in dB.

    Returns:
        y, A, Ce, Cw, mu0, P0, where (mu0, P0) is the initial state of the
        dataset's own Kalman filter, so evaluation uses the same prior as
        the training labels.
    """
    ds = dataset_lib.LinearSSMRotationalDataset(
        num_samples=num_samples, signal_length=signal_length, smnr_db=smnr_db
    )
    model = ds._linear_model
    return tuple(
        torch.as_tensor(v).to(DTYPE)
        for v in (
            ds._noisy_observations,
            model.A,
            model.Ce,
            model.Cw,
            model.mu0,
            model.P0,
        )
    )


def _demo_model(
    d: int = 4, smnr_db: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The linear model of the self-test.

    Args:
        d: Dimension of the state and the observations.
        smnr_db: Signal-to-measurement noise ratio of the observations in dB.

    Returns:
        A, Q and R, each (d, d).
    """

    def rot(th):
        return torch.tensor(
            [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]],
            dtype=DTYPE,
        )

    A = 0.9 * torch.block_diag(rot(0.3), rot(0.5))
    Q = 0.1 * torch.eye(d, dtype=DTYPE)
    P = torch.eye(d, dtype=DTYPE)
    for _ in range(2000):
        P = A @ P @ A.T + Q
    R = (float(torch.trace(P)) / (d * 10 ** (smnr_db / 10.0))) * torch.eye(
        d, dtype=DTYPE
    )
    return A, Q, R


def self_test(T: int = 100, B: int = 20, seed: int = 0) -> None:
    """Checks that the filter's evidence converges to the Kalman value in N.

    Args:
        T: Number of time steps of each simulated sequence.
        B: Number of simulated sequences.
        seed: Random seed for the sequences and the particle filter.
    """
    A, Q, R = _demo_model()
    gen = torch.Generator().manual_seed(seed)
    Lq, Lr = torch.linalg.cholesky(Q), torch.linalg.cholesky(R)
    x = torch.randn(B, 4, generator=gen, dtype=DTYPE) @ Lq.T
    ys = []
    for _ in range(T):
        x = x @ A.T + torch.randn(B, 4, generator=gen, dtype=DTYPE) @ Lq.T
        ys.append(x + torch.randn(B, 4, generator=gen, dtype=DTYPE) @ Lr.T)
    y = torch.stack(ys, 1)

    exact = kalman_loglik(y, A, Q, R)
    print(f'exact log p(y_1:T)   {exact.mean():9.3f} ± {exact.std():<7.3f}\n')
    print(f"{'N':>6}{'E[log p_hat]':>19}{'gap':>19}{'gap/T':>10}")
    for N in (25, 100, 1000):
        estimate = smc_loglik(y, A, Q, R, N=N, seed=seed)
        gap = exact - estimate
        print(
            f'{N:>6}{estimate.mean():>9.3f} ± {estimate.std():<7.3f}'
            f'{gap.mean():>9.3f} ± {gap.std():<7.3f}{gap.mean() / T:>10.4f}'
        )
    print(
        '\nThe gap is the Jensen slack: it must be POSITIVE and shrink with N.'
    )
    print('Negative or non-shrinking means the weights were normalised before')
    print('being accumulated into the evidence.')


def compare(
    run_bases: dict[str, str],
    N: int = 100,
    num_samples: int = 10,
    signal_length: int = 100,
    smnr_db: float = 10.0,
    seed: int = 0,
    train_seeds: tuple[int, ...] = (0, 1, 2),
) -> None:
    """Prints the evidence decomposition for each method.

    Every checkpoint '<base>_<seed>/transition_model_epoch_last.pt' is
    evaluated on the same test sequences. Each training seed's value is first
    averaged over the sequences; mean ± std are then over training seeds.
    Missing checkpoints are skipped with a warning.

    Args:
        run_bases: Method label -> run prefix, e.g. 'runs/linear/dsr_10db'; the
            checkpoints are <prefix>_<seed>/.
        N: Number of particles of the evaluation particle filter.
        num_samples: Number of test sequences.
        signal_length: Length of the test sequences.
        smnr_db: SMNR of the test sequences.
        seed: Random seed of the evaluation particle filter.
        train_seeds: Run indices (training seeds) to average over.
    """
    y, A, Ce, Cw, mu0, P0 = load_test_data(num_samples, signal_length, smnr_db)
    exact_true = kalman_loglik(y, A, Ce, Cw, m0=mu0, P0=P0)
    print(
        f'true model, exact     {exact_true.mean():9.3f} ±'
        f' {exact_true.std():<7.3f}'
        f'({y.shape[0]} sequences, T={y.shape[1]}, N={N})\n'
    )

    def cell(values: list[torch.Tensor]) -> str:
        v = torch.stack(values)
        return f'{v.mean():>9.3f} ± {v.std():<7.3f}'

    print(
        f"{'method':<12}{'exact(learned)':>19}{'E[log p_hat]':>19}"
        f"{'model err':>19}{'SMC slack':>19}"
    )
    print('-' * 104)
    for name, base in run_bases.items():
        exact_hats, estimates, model_errors, smc_slacks = [], [], [], []
        for train_seed in train_seeds:
            ckpt_path = (
                pathlib.Path(f'{base}_{train_seed}')
                / 'transition_model_epoch_last.pt'
            )
            if not ckpt_path.exists():
                print(
                    f'  [warn] {name}: missing {ckpt_path}, skipping seed'
                    f' {train_seed}'
                )
                continue
            A_hat, Q_hat = load_transition(str(ckpt_path))
            exact_hat = kalman_loglik(y, A_hat, Q_hat, Cw, m0=mu0, P0=P0)
            estimate = smc_loglik(
                y, A_hat, Q_hat, Cw, N=N, seed=seed, m0=mu0, P0=P0
            )
            exact_hats.append(exact_hat.mean())
            estimates.append(estimate.mean())
            model_errors.append((exact_true - exact_hat).mean())
            smc_slacks.append((exact_hat - estimate).mean())
        if not exact_hats:
            print(f'{name:<12}  no checkpoints found for seeds {train_seeds}')
            continue
        print(
            f'{name:<12}{cell(exact_hats):>19}{cell(estimates):>19}'
            f'{cell(model_errors):>19}{cell(smc_slacks):>19}'
        )
    print('-' * 104)
    print(f"mean ± std is over training seeds {train_seeds}; each seed's value")
    print('is itself averaged over the fixed test sequences first.')
    print('model err: evidence lost by the learned transition vs truth (lower')
    print('           is better).')
    print(
        'SMC slack: Jensen gap; small at N=1000 and comparable across methods,'
    )
    print('           otherwise model err is not interpretable.')
    print("\nlog p_hat is unbiased for the LEARNED model's evidence, so it can")
    print('exceed the true-model value on a given sample. Not a bug.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument(
        '--runs',
        type=str,
        default='',
        help=(
            'Comma-separated label=run_prefix pairs, e.g.'
            " 'DSR=runs/linear/dsr_10db'; each is expanded to"
            " '<prefix>_<seed>/transition_model_epoch_last.pt' for"
            ' --train-seeds.'
        ),
    )
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument(
        '--train-seeds',
        type=str,
        default='0,1,2',
        help='Comma-separated training seeds to average over.',
    )
    args = parser.parse_args()

    torch.set_default_dtype(DTYPE)
    if args.self_test:
        self_test()
    elif args.runs:
        run_bases = dict(kv.split('=', 1) for kv in args.runs.split(','))
        train_seeds = tuple(int(s) for s in args.train_seeds.split(','))
        compare(run_bases, N=args.N, train_seeds=train_seeds)
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
