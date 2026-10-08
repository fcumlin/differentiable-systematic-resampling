"""
Gradient bias of DSR against the unbiased VSMC gradient (paper appendix,
"Comparison against unbiased VSMC gradients", Table 15).

Reference
---------
Naesseth, Linderman, Ranganath & Blei, "Variational Sequential Monte Carlo",
AISTATS 2018, eqs. (7)-(9):

    grad L  = g_rep + g_score
    g_rep   = E[ grad log Zhat ]
    g_score = E[ log Zhat * grad log phi(a_{1:T} | eps_{1:T}) ]

with the Rao-Blackwellised form (8) replacing log Zhat by the *future*
log-average weights log( Zhat_T / Zhat_{t-1} ), since the ancestor variables
drawn at time t cannot influence any weight before time t.

Experiment
----------
Linear Gaussian SSM at SMNR = 10 dB, N = 4, T = 4, for d = 2 (one rotation
block) and d = 4.  Gradients are w.r.t. all d^2 entries of A and the d
entries of l, where Q = diag(softplus(l)), at the data-generating parameters.

  * grad L : exact, by enumerating every ancestor vector, averaged over
             1000 bootstrap noise draws.
  * g_hat  : average of n = 1600 independent SMC runs (100/4 x 64).
  * E[g_hat] from the same 1000 noise draws x 500 offset sequences
    (5e5 runs); the variance from 200 independent realisations of g_hat.

    Bias = ||E[g_hat] - grad L||,  Variance = E||g_hat - E[g_hat]||^2,
    MSE  = Bias^2 + Variance.

Validation
----------
--self-test runs, before any number from the table should be trusted:
  [0] unit test of the pmf: sum_a P(a) = 1, sum_a grad P(a) = 0, and
      grad log P checked against finite differences in W.
  [1] on the table's model, the Monte Carlo g_rep, g_score and their sum
      against the exact enumeration, and central differences under common
      random numbers against the same target.

Usage (from the repository root):
    python -m appendix.vsmc_gradient_bias [--self-test] [--quick]
"""

from __future__ import annotations

import argparse
import math
import time

import torch

DTYPE = torch.float64
TINY = torch.finfo(DTYPE).tiny
LOG2PI = math.log(2 * math.pi)

TAUS = (1.0, 0.5, 0.1, 0.05, 0.01)


def true_params(
    d: int = 4, smnr_db: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(A, Q, R) of the d-dimensional linear Gaussian SSM.

    A is 0.9 times d/2 rotation blocks (angles 0.3, 0.5, ...), Q = 0.1 I,
    and R is isotropic with the stationary signal power set by `smnr_db`.

    Args:
        d: Dimension of the state and the observations; even, as A is built from
            2x2 rotation blocks.
        smnr_db: Signal-to-measurement noise ratio of the observations in dB.

    Returns:
        A, Q and R, each (d, d).
    """

    def rot(t):
        return torch.tensor(
            [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]],
            dtype=DTYPE,
        )

    A = 0.9 * torch.block_diag(*[rot(0.3 + 0.2 * k) for k in range(d // 2)])
    Q = 0.1 * torch.eye(d, dtype=DTYPE)
    P = torch.eye(d, dtype=DTYPE)
    for _ in range(2000):
        P = A @ P @ A.T + Q
    R = (torch.trace(P) / (d * 10.0 ** (smnr_db / 10.0))) * torch.eye(
        d, dtype=DTYPE
    )
    return A, Q, R


def simulate(
    A: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, T: int, seed: int = 0
) -> torch.Tensor:
    """One observation sequence of the SSM.

    Args:
        A: State transition matrix (d, d), x_t = A x_(t-1) + e_t.
        Q: Covariance of the process noise e_t (d, d).
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        T: Number of time steps of the observation sequence.
        seed: Random seed of the simulated sequence.

    Returns:
        Observations (T, d).
    """
    d = A.shape[0]
    g = torch.Generator().manual_seed(seed)
    Lq = torch.linalg.cholesky(Q)
    Lr = torch.linalg.cholesky(R)
    x = torch.randn(1, d, generator=g, dtype=DTYPE) @ Lq.T
    ys = []
    for _ in range(T):
        x = x @ A.T + torch.randn(1, d, generator=g, dtype=DTYPE) @ Lq.T
        ys.append(
            (x + torch.randn(1, d, generator=g, dtype=DTYPE) @ Lr.T).squeeze(0)
        )
    return torch.stack(ys)


def draw_noise(
    B: int, T: int, N: int, d: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap noise and resampling offsets of B filter runs.

    Args:
        B: Number of independent particle filter runs.
        T: Number of time steps of the observation sequence.
        N: Number of particles of each run.
        d: Dimension of the state x_t and the observation y_t.
        seed: Random seed of the noise.

    Returns:
        eps (B, T + 1, N, d) and u0 (B, T), uniform on [0, 1/N).
    """
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn(B, T + 1, N, d, generator=g, dtype=DTYPE)
    u0 = torch.rand(B, T, generator=g, dtype=DTYPE) / N
    return eps, u0


def setup(
    d: int, T: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """True parameters and the observation sequence of the experiment.

    Args:
        d: Dimension of the state x_t and the observation y_t.
        T: Number of time steps of the observation sequence.
        seed: Random seed of the observation sequence.

    Returns:
        A0, logd0 = softplus^-1(diag(Q)), y (T, d) and R.
    """
    A0, Q0, R = true_params(d=d)
    y = simulate(A0, Q0, R, T, seed)
    logd0 = torch.log(torch.expm1(torch.diagonal(Q0)))
    return A0, logd0, y, R


def _cdf(W):
    """cumsum(W) with the final entry pinned to exactly 1.

    W is normalised, so the last entry is the constant 1 and carries zero
    derivative. Pinning it stops round-off from (a) pushing u past the end
    of the grid and (b) breaking the identity b_N = 0 used below.

    Args:
        W: Normalized particle weights (B, N) of B particle sets.

    Returns:
        The CDF (B, N).
    """
    F = torch.cumsum(W, dim=1)
    return torch.cat([F[:, :-1], torch.ones_like(F[:, -1:])], dim=1)


def systematic_resample(
    W: torch.Tensor, u0: torch.Tensor, return_upper: bool = False
) -> tuple[torch.Tensor, ...]:
    """Systematic resampling together with log P(idx | W).

    Args:
        W: Normalized particle weights (B, N), differentiable.
        u0: Resampling offsets (B,), uniform on [0, 1/N); carry no gradient.
        return_upper: Whether to also return U.

    Returns:
        idx: Ancestor indices (B, N).
        logp: Log-probability of the whole ancestor vector (B,).
        U: Right end of the u0-interval giving idx (B,); if return_upper.
    """
    B, N = W.shape
    F = _cdf(W)
    F_prev = torch.cat([torch.zeros_like(F[:, :1]), F[:, :-1]], dim=1)
    grid = torch.arange(N, device=W.device, dtype=W.dtype) / N
    u = u0[:, None] + grid[None, :]

    # a_n = min{ i : F_i > u_n } -- strict, hence right=True.  This makes the
    # interval half-open, [L, U), which the enumeration walk relies on.
    idx = torch.searchsorted(
        F.detach().contiguous(), u.detach().contiguous(), right=True
    ).clamp(max=N - 1)
    F_a = torch.gather(F, 1, idx)
    F_prev_a = torch.gather(F_prev, 1, idx)

    # widest interval of u0 that reproduces exactly this ancestor vector;
    # a clipped endpoint is a constant, so clamp correctly kills its gradient
    L = (F_prev_a - grid[None, :]).max(dim=1).values.clamp_min(0.0)
    U = (F_a - grid[None, :]).min(dim=1).values.clamp_max(1.0 / N)
    logp = torch.log((N * (U - L)).clamp_min(TINY))
    return (idx, logp, U) if return_upper else (idx, logp)


def enumerate_intervals(W: torch.Tensor) -> torch.Tensor:
    """Endpoints of every u0-interval on which the ancestor vector is constant.

    The vector changes when u0 crosses F_i - n/N for the unique n with
    F_i - n/N in [0, 1/N), i.e. at b_i = F_i mod (1/N). F_N = 1 gives
    b_N = 0, so the sorted b's already supply the left endpoint.

    Args:
        W: Normalized particle weights (B, N) of B particle sets.

    Returns:
        Sorted endpoints s (B, N + 1), with s[:, 0] = 0 and s[:, N] = 1/N.
    """
    B, N = W.shape
    F = _cdf(W)
    b = F - torch.floor(N * F).detach() / N
    s, _ = torch.sort(b, dim=1)
    top = torch.full((B, 1), 1.0 / N, dtype=W.dtype, device=W.device)
    return torch.cat([s, top], dim=1)


def dsr_resampling(
    W: torch.Tensor, u0: torch.Tensor, tau: float
) -> torch.Tensor:
    """DSR transport matrix, the relaxed systematic resampling.

    Same relaxation as resampling.resampler_systematic, without Morton
    sorting and without the gradient-clamp hook.

    Args:
        W: Normalized particle weights (B, N) of B particle sets.
        u0: Resampling offsets (B,), uniform on [0, 1/N).
        tau: Temperature of the DSR relaxation.

    Returns:
        T (B, N, N); T[b, i, j] is the weight of ancestor i in particle j.
    """
    F = _cdf(W)
    F_prev = torch.cat([torch.zeros_like(F[:, :1]), F[:, :-1]], dim=1)
    N = W.shape[1]
    grid = torch.arange(N, dtype=W.dtype, device=W.device) / N
    u = u0[:, None] + grid[None, :]
    Tmat = torch.sigmoid(
        (u[:, None, :] - F_prev[:, :, None]) / tau
    ) - torch.sigmoid((u[:, None, :] - F[:, :, None]) / tau)
    return Tmat / Tmat.sum(dim=1, keepdim=True).clamp_min(1e-30)


def _model_terms(logd, R):
    """Factors of the filter's Gaussian densities.

    Args:
        logd: Log-diagonal parameters (d,); Q = diag(softplus(logd)).
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.

    Returns:
        Cholesky factor of Q, R^-1 and log det R.
    """
    Q = torch.diag(torch.nn.functional.softplus(logd))
    Lq = torch.linalg.cholesky(Q)
    Lr = torch.linalg.cholesky(R)
    return (
        Lq,
        torch.cholesky_inverse(Lr),
        2 * torch.log(torch.diagonal(Lr)).sum(),
    )


def run_filter(
    A: torch.Tensor,
    logd: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    eps: torch.Tensor,
    u0: torch.Tensor,
    mode: str = "hard",
    tau: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Bootstrap particle filter with resampling at every step.

    Args:
        A: State transition matrix (d, d), x_t = A x_(t-1) + e_t.
        logd: Log-diagonal parameters (d,); Q = diag(softplus(logd)).
        y: The observation sequence (T, d) that every filter run targets.
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        eps: Bootstrap noise of the B filter runs (B, T + 1, N, d): initial
            state and the process noise of each of the T steps.
        u0: Resampling offsets of the B runs (B, T), uniform on [0, 1/N).
        mode: "hard" for systematic resampling, "dsr" for DSR.
        tau: Temperature of the DSR relaxation; unused for "hard".

    Returns:
        incs: log(mean_i w_t^i) (B, T); log Zhat = incs.sum(1).
        logps: log phi(a_t | W_t) (B, T) if mode="hard", else None.
    """
    B, _, N, _ = eps.shape
    T, d = y.shape[0], A.shape[0]
    Lq, Rinv, logdetR = _model_terms(logd, R)
    x = eps[:, 0] @ Lq.T
    incs, logps = [], []
    for t in range(T):
        x = x @ A.T + eps[:, t + 1] @ Lq.T
        v = y[t].view(1, 1, d) - x
        logw = -0.5 * ((v @ Rinv) * v).sum(dim=-1) - 0.5 * (
            logdetR + d * LOG2PI
        )

        # weights are uniform after each resampling step
        incs.append(torch.logsumexp(logw - math.log(N), dim=1))
        W = torch.softmax(logw, dim=1)
        if mode == "hard":
            idx, logp = systematic_resample(W, u0[:, t])
            logps.append(logp)

            # NO .detach() here: g_rep must propagate through the genealogy,
            # only the indices are frozen.
            x = torch.gather(x, 1, idx[:, :, None].expand(B, N, d))
        elif mode == "dsr":
            x = torch.einsum(
                "bij,bid->bjd", dsr_resampling(W, u0[:, t], tau), x
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return (torch.stack(incs, 1), torch.stack(logps, 1) if logps else None)


def flat(grads: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Concatenation of the flattened gradients.

    Args:
        grads: Gradients w.r.t. A and logd.

    Returns:
        One flat vector.
    """
    return torch.cat([g.reshape(-1) for g in grads])


def _params(A0, logd0):
    """Copies of the parameters as leaves that require gradients.

    Args:
        A0: State transition matrix (d, d).
        logd0: Log-diagonal parameters (d,) of Q.

    Returns:
        A and logd.
    """
    return A0.clone().requires_grad_(), logd0.clone().requires_grad_()


def reward_to_go(
    incs: torch.Tensor, rao_blackwell: bool = True
) -> torch.Tensor:
    """Reward multiplying log phi(a_t | .).

    Rao-Blackwellised (eq. 8): only the future increments, because a_t is
    independent of everything before t and E[grad log phi_t | F_t] = 0.

    Args:
        incs: Log-weight increments (B, T) of the B filter runs.
        rao_blackwell: If False, use the full log Zhat at every step.

    Returns:
        Rewards (B, T).
    """
    if not rao_blackwell:
        return incs.sum(1, keepdim=True).expand_as(incs).contiguous()
    return incs.sum(1, keepdim=True) - torch.cumsum(incs, dim=1)


def apply_baseline(rew: torch.Tensor, baseline: str = "rloo") -> torch.Tensor:
    """Control variate; 'rloo' is the leave-one-out batch mean (unbiased).

    Args:
        rew: Rewards (B, T) of the score-function term.
        baseline: 'none', 'mean' or 'rloo' (leave-one-out batch mean).

    Returns:
        Rewards minus the baseline (B, T).
    """
    if baseline == "none":
        return rew
    if baseline == "mean":
        return rew - rew.mean(0, keepdim=True)
    if baseline == "rloo":
        B = rew.shape[0]
        if B < 2:
            return rew
        return rew - (rew.sum(0, keepdim=True) - rew) / (B - 1)
    raise ValueError(baseline)


def vsmc_gradient(
    A0: torch.Tensor,
    logd0: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    eps: torch.Tensor,
    u0: torch.Tensor,
    rao_blackwell: bool = True,
    baseline: str = "rloo",
) -> dict[str, torch.Tensor]:
    """g_rep and g_score of eqs. (7)/(8), averaged over the runs.

    Args:
        A0: State transition matrix (d, d) at which the gradient is evaluated.
        logd0: Log-diagonal parameters (d,) at which the gradient is evaluated;
            Q = diag(softplus(logd0)).
        y: The observation sequence (T, d) that every filter run targets.
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        eps: Bootstrap noise of the B filter runs (B, T + 1, N, d): initial
            state and the process noise of each of the T steps.
        u0: Resampling offsets of the B runs (B, T), uniform on [0, 1/N).
        rao_blackwell: Score reward from the future increments only (eq. 8).
        baseline: Control variate of the score term, see apply_baseline.

    Returns:
        {"g_rep", "g_score", "g_vsmc" = g_rep + g_score, "ll" = log Zhat}.
    """
    A, logd = _params(A0, logd0)
    incs, logps = run_filter(A, logd, y, R, eps, u0, mode="hard")
    ll = incs.sum(1)
    g_rep = flat(torch.autograd.grad(ll.mean(), [A, logd], retain_graph=True))
    rew = apply_baseline(reward_to_go(incs.detach(), rao_blackwell), baseline)
    g_score = flat(torch.autograd.grad((rew * logps).sum(1).mean(), [A, logd]))
    return {
        "g_rep": g_rep.detach(),
        "g_score": g_score.detach(),
        "g_vsmc": (g_rep + g_score).detach(),
        "ll": ll.detach(),
    }


def hard_gradient(
    A0: torch.Tensor,
    logd0: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    eps: torch.Tensor,
    u0: torch.Tensor,
) -> torch.Tensor:
    """Gradient with hard systematic resampling: g_rep, no score term.

    Args:
        A0: State transition matrix (d, d) at which the gradient is evaluated.
        logd0: Log-diagonal parameters (d,) at which the gradient is evaluated;
            Q = diag(softplus(logd0)).
        y: The observation sequence (T, d) that every filter run targets.
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        eps: Bootstrap noise of the B filter runs (B, T + 1, N, d): initial
            state and the process noise of each of the T steps.
        u0: Resampling offsets of the B runs (B, T), uniform on [0, 1/N).

    Returns:
        Flat gradient w.r.t. (A, logd).
    """
    return vsmc_gradient(A0, logd0, y, R, eps, u0)["g_rep"]


def dsr_gradient(
    A0: torch.Tensor,
    logd0: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    eps: torch.Tensor,
    u0: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Gradient of the mean log Zhat with DSR.

    Args:
        A0: State transition matrix (d, d) at which the gradient is evaluated.
        logd0: Log-diagonal parameters (d,) at which the gradient is evaluated;
            Q = diag(softplus(logd0)).
        y: The observation sequence (T, d) that every filter run targets.
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        eps: Bootstrap noise of the B filter runs (B, T + 1, N, d): initial
            state and the process noise of each of the T steps.
        u0: Resampling offsets of the B runs (B, T), uniform on [0, 1/N).
        tau: Temperature of the DSR relaxation.

    Returns:
        Flat gradient w.r.t. (A, logd).
    """
    A, logd = _params(A0, logd0)
    incs, _ = run_filter(A, logd, y, R, eps, u0, mode="dsr", tau=tau)
    return flat(torch.autograd.grad(incs.sum(1).mean(), [A, logd])).detach()


def exact_objective(
    A: torch.Tensor,
    logd: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    eps: torch.Tensor,
    part: str = "total",
) -> torch.Tensor:
    r"""E_{u0}[ log Zhat ] with eps fixed, by exhaustive enumeration.

    Each step splits every surviving branch into the N intervals of
    [0, 1/N) on which the ancestor vector is constant, so the frontier is
    B * N^t wide. The result is differentiable and, by construction,

        d/dth E_{u0}[log Zhat] = E[grad log Zhat]  +  E[log Zhat grad log phi]
                                 \_____ "rep" ____/   \______ "score" _______/

    so `part` selects either component or their sum -- a noise-free target.

    Args:
        A: State transition matrix (d, d), x_t = A x_(t-1) + e_t.
        logd: Log-diagonal parameters (d,); Q = diag(softplus(logd)).
        y: The observation sequence (T, d) that every filter run targets.
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        eps: Bootstrap noise (B, T + 1, N, d) of B noise draws; the resampling
            randomness is enumerated.
        part: "rep", "score" or "total" (their sum).

    Returns:
        The selected part, averaged over the B noise draws.
    """
    B, _, N, d = eps.shape
    T = y.shape[0]
    Lq, Rinv, logdetR = _model_terms(logd, R)
    src = torch.arange(B)
    x = eps[:, 0] @ Lq.T
    ll = torch.zeros(B, dtype=DTYPE)
    logp_path = torch.zeros(B, dtype=DTYPE)
    for t in range(T):
        x = x @ A.T + eps[src, t + 1] @ Lq.T
        v = y[t].view(1, 1, d) - x
        logw = -0.5 * ((v @ Rinv) * v).sum(dim=-1) - 0.5 * (
            logdetR + d * LOG2PI
        )
        ll = ll + torch.logsumexp(logw - math.log(N), dim=1)
        W = torch.softmax(logw, dim=1)
        M = W.shape[0]

        # Walk the partition of [0, 1/N) from the left: the interval
        # containing u is [L, U), so probing at u = U_prev lands exactly in
        # the next interval.  Degenerate (zero-width) intervals are skipped
        # automatically and the probabilities telescope to 1 by construction,
        # unlike midpoint probes, which can round into a neighbour.
        u = torch.zeros(M, dtype=W.dtype, device=W.device)
        b_idx, b_logp = [], []
        dead = torch.full((M,), -float("inf"), dtype=W.dtype, device=W.device)
        for _ in range(N):
            # A branch is exhausted once the walk leaves [0, 1/N): there are
            # at most N distinct intervals, but there can be fewer.  The
            # tolerance matters: the final endpoint is min_n(F_{a_n} - n/N),
            # which equals 1/N only up to round-off (1 - 4/5 = 0.1999...96
            # for N = 5), and without it the last interval is probed twice.
            alive = u < 1.0 / N - 1e-12
            idx_k, logp_k, U_k = systematic_resample(
                W,
                torch.where(alive, u, torch.zeros_like(u)),
                return_upper=True,
            )
            b_idx.append(idx_k)
            b_logp.append(torch.where(alive, logp_k, dead))
            u = torch.where(alive, U_k.detach(), u)
        idx = torch.stack(b_idx, dim=1).reshape(M * N, N)
        logp = torch.stack(b_logp, dim=1).reshape(M * N)
        x = x.repeat_interleave(N, dim=0)
        x = torch.gather(x, 1, idx[:, :, None].expand(M * N, N, d))
        ll = ll.repeat_interleave(N, dim=0)
        logp_path = logp_path.repeat_interleave(N, dim=0) + logp
        src = src.repeat_interleave(N, dim=0)
    p = logp_path.exp()
    mass = torch.zeros(B, dtype=DTYPE).index_add(0, src, p.detach())
    assert torch.allclose(
        mass, torch.ones_like(mass), atol=1e-9
    ), f"enumeration lost mass: {(mass - 1).abs().max().item():.3e}"
    if part == "rep":
        contrib = p.detach() * ll
    elif part == "score":
        contrib = p * ll.detach()
    elif part == "total":
        contrib = p * ll
    else:
        raise ValueError(part)
    return torch.zeros(B, dtype=DTYPE).index_add(0, src, contrib).mean()


def exact_gradient(
    A0: torch.Tensor,
    logd0: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    eps: torch.Tensor,
    part: str = "total",
) -> torch.Tensor:
    """Gradient of exact_objective; grad L itself for part="total".

    Args:
        A0: State transition matrix (d, d) at which the gradient is evaluated.
        logd0: Log-diagonal parameters (d,) at which the gradient is evaluated;
            Q = diag(softplus(logd0)).
        y: The observation sequence (T, d) that every filter run targets.
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        eps: Bootstrap noise (B, T + 1, N, d) of B noise draws; the resampling
            randomness is enumerated.
        part: "rep", "score" or "total" (their sum).

    Returns:
        Flat gradient w.r.t. (A, logd).
    """
    A, logd = _params(A0, logd0)
    return flat(
        torch.autograd.grad(
            exact_objective(A, logd, y, R, eps, part), [A, logd]
        )
    ).detach()


def finite_difference(
    A0: torch.Tensor,
    logd0: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    eps: torch.Tensor,
    u0: torch.Tensor,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences of mean log Zhat at fixed eps, u0.

    With CRN the estimator is piecewise smooth with jumps wherever an
    ancestor index flips, and a jump is crossed with probability O(h) while
    contributing O(1/h). Hence E[FD] -> g_rep + g_score, NOT g_rep, while
    Var(FD) grows like 1/h.

    Args:
        A0: State transition matrix (d, d) at which the gradient is evaluated.
        logd0: Log-diagonal parameters (d,) at which the gradient is evaluated;
            Q = diag(softplus(logd0)).
        y: The observation sequence (T, d) that every filter run targets.
        R: Covariance of the observation noise w_t (d, d), y_t = x_t + w_t.
        eps: Bootstrap noise of the B filter runs (B, T + 1, N, d): initial
            state and the process noise of each of the T steps.
        u0: Resampling offsets of the B runs (B, T), uniform on [0, 1/N).
        h: Step size of the central differences.

    Returns:
        Mean and standard error of the differences, each (d^2 + d,).
    """

    def obj(Ai, di):
        incs, _ = run_filter(Ai, di, y, R, eps, u0, mode="hard")
        return incs.sum(1)

    perturbs = [
        ("A", i, j) for i in range(A0.shape[0]) for j in range(A0.shape[1])
    ] + [("d", i, None) for i in range(logd0.numel())]
    vals, ses = [], []
    for kind, i, j in perturbs:
        Ap, Am = A0.clone(), A0.clone()
        dp, dm = logd0.clone(), logd0.clone()
        if kind == "A":
            Ap[i, j] += h
            Am[i, j] -= h
        else:
            dp[i] += h
            dm[i] -= h
        with torch.no_grad():
            diff = (obj(Ap, dp) - obj(Am, dm)) / (2 * h)
        vals.append(diff.mean())
        ses.append(diff.std(unbiased=True) / math.sqrt(diff.numel()))
    return torch.stack(vals), torch.stack(ses)


def compare(
    name: str,
    g: torch.Tensor,
    ref: torch.Tensor,
    se: torch.Tensor | None = None,
) -> float:
    """Prints how an estimate agrees with a reference.

    Args:
        name: Label of the printed row.
        g: Estimated gradient (P,).
        ref: Reference gradient (P,).
        se: Standard error of g (P,); if given, the largest z-score is printed.

    Returns:
        The relative error ||g - ref|| / ||ref||.
    """
    rel = ((g - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    cos = (torch.dot(g, ref) / (g.norm() * ref.norm() + 1e-30)).item()
    line = (
        f"  {name:<26}||g||={g.norm().item():11.4e}  "
        f"rel.err={rel:9.3e}  cos={cos:9.6f}"
    )
    if se is not None:
        line += f"  max|z|={((g - ref).abs() / se.clamp_min(1e-30)).max():7.2f}"
    print(line)
    return rel


def mean_se(chunks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard error of the mean.

    Args:
        chunks: Independent estimates of the same quantity, each (P,).

    Returns:
        Mean and standard error, each (P,).
    """
    z = torch.stack(chunks)
    return z.mean(0), z.std(0, unbiased=True) / math.sqrt(z.shape[0])


def test_pmf(N: int = 6, B: int = 4, seed: int = 0) -> None:
    """Self-test [0]: the pmf of the systematic ancestor vector.

    Args:
        N: Number of particles.
        B: Number of random weight vectors to test.
        seed: Random seed of the weights.
    """
    print("\n" + "=" * 78)
    print("[0] SYSTEMATIC-RESAMPLING PMF UNIT TEST")
    print("=" * 78)
    g = torch.Generator().manual_seed(seed)
    logits = (
        torch.randn(B, N, generator=g, dtype=DTYPE) * 1.5
    ).requires_grad_()
    W = torch.softmax(logits, dim=1)
    s = enumerate_intervals(W)
    assert (
        s[:, 0].abs().max() < 1e-14
    ), "left endpoint of the partition must be 0"
    mid = (0.5 * (s[:, :-1] + s[:, 1:])).detach()
    total = torch.zeros(B, dtype=DTYPE)
    grad_total = torch.zeros_like(logits)
    for k in range(N):
        _, logp = systematic_resample(W, mid[:, k])
        p = logp.exp()
        total = total + p
        grad_total = (
            grad_total
            + torch.autograd.grad(p.sum(), logits, retain_graph=True)[0]
        )
    print(f"  max |sum_a P(a) - 1|        : {(total - 1).abs().max():.3e}")
    print(
        f"  max |sum_a grad P(a)|       : {grad_total.abs().max():.3e}"
        f"   (exactly 0 <=> score is mean-zero)"
    )
    u = torch.full((B,), 0.5 / N, dtype=DTYPE)
    _, logp = systematic_resample(torch.softmax(logits, 1), u)
    ana = torch.autograd.grad(logp.sum(), logits)[0]
    h, num = 1e-6, torch.zeros_like(logits)
    for i in range(B):
        for k in range(N):
            lp, lm = logits.detach().clone(), logits.detach().clone()
            lp[i, k] += h
            lm[i, k] -= h
            _, a = systematic_resample(torch.softmax(lp, 1), u)
            _, b = systematic_resample(torch.softmax(lm, 1), u)
            num[i, k] = (a[i] - b[i]) / (2 * h)
    print(
        f"  grad log P vs FD (relative) : "
        f"{((ana - num).abs().max() / num.abs().max()):.3e}"
    )


def test_exact(
    N: int = 4,
    T: int = 4,
    d: int = 2,
    B_eps: int = 8,
    chunk: int = 4000,
    n_chunks: int = 40,
    h_fd: float = 1e-2,
    seed: int = 0,
) -> None:
    """Self-test [1]: Monte Carlo and finite differences against enumeration.

    Args:
        N: Number of particles of each filter run.
        T: Number of time steps of the observation sequence.
        d: Dimension of the state x_t and the observation y_t.
        B_eps: Number of bootstrap noise draws.
        chunk: Resampling offsets per noise draw in one Monte Carlo chunk.
        n_chunks: Number of Monte Carlo chunks.
        h_fd: Step size of the central differences.
        seed: Random seed of the sequence and the noise.
    """
    print("\n" + "=" * 78)
    print(
        f"[1] EXACT ENUMERATION   d={d}  N={N}  T={T}   "
        f"({B_eps} eps-paths x {N ** T} resampling outcomes)"
    )
    print("=" * 78)
    A0, logd0, y, R = setup(d, T, seed)
    eps, _ = draw_noise(B_eps, T, N, d, seed + 1)
    ex_rep = exact_gradient(A0, logd0, y, R, eps, "rep")
    ex_sc = exact_gradient(A0, logd0, y, R, eps, "score")
    ex_tot = exact_gradient(A0, logd0, y, R, eps, "total")
    print(f"  exact ||g_rep||   = {ex_rep.norm():.6f}")
    print(f"  exact ||g_score|| = {ex_sc.norm():.6f}")
    print(
        f"  exact ||sum||     = {ex_tot.norm():.6f}   "
        f"(large cancellation: |sum| << |parts|)"
    )
    eps_t = eps.repeat(chunk, 1, 1, 1)
    reps, scores, vs = [], [], []
    t0 = time.time()
    for c in range(n_chunks):
        g = torch.Generator().manual_seed(seed + 1000 + c)
        u0 = torch.rand(B_eps * chunk, T, generator=g, dtype=DTYPE) / N
        v = vsmc_gradient(A0, logd0, y, R, eps_t, u0)
        reps.append(v["g_rep"])
        scores.append(v["g_score"])
        vs.append(v["g_vsmc"])
        print(f"  MC chunk {c + 1}/{n_chunks}", end="\r", flush=True)
    print(" " * 40, end="\r")
    print(
        f"\n  Monte Carlo estimator, {B_eps * chunk * n_chunks:,} paths "
        f"({time.time() - t0:.0f}s), same eps (CRN):"
    )
    m_rep, se_rep = mean_se(reps)
    m_sc, se_sc = mean_se(scores)
    m_v, se_v = mean_se(vs)
    compare("g_rep      vs exact", m_rep, ex_rep, se_rep)
    compare("g_score    vs exact", m_sc, ex_sc, se_sc)
    compare("g_rep+g_score vs exact", m_v, ex_tot, se_v)

    # same thing without Rao-Blackwellisation and without a control variate
    g = torch.Generator().manual_seed(seed + 77)
    u0 = torch.rand(B_eps * chunk, T, generator=g, dtype=DTYPE) / N
    v_rb = vsmc_gradient(A0, logd0, y, R, eps_t, u0)
    v_raw = vsmc_gradient(
        A0, logd0, y, R, eps_t, u0, rao_blackwell=False, baseline="none"
    )
    print(f"\n  one batch of {B_eps * chunk:,} paths, g_score error:")
    compare("Rao-Blackwell + RLOO", v_rb["g_score"], ex_sc)
    compare("vanilla  (no RB, no CV)", v_raw["g_score"], ex_sc)
    print(f"\n  central differences, h={h_fd}, same eps:")
    fd, se_fd = finite_difference(A0, logd0, y, R, eps_t, u0, h_fd)
    compare("FD vs exact g_rep+g_score", fd, ex_tot, se_fd)
    compare("FD vs exact g_rep alone", fd, ex_rep)


def bias_variance_table(
    d: int,
    N: int = 4,
    T: int = 4,
    n_eps: int = 1000,
    eps_chunk: int = 50,
    offsets_per_rep: int = 50,
    n_rep: int = 10,
    n_runs: int = 1600,
    n_real: int = 200,
    taus: tuple[float, ...] = TAUS,
    seed: int = 0,
) -> None:
    """Prints one block of the table.

    grad L   : exact gradient averaged over n_eps bootstrap noise draws.
    E[g_hat] : the same n_eps draws x (offsets_per_rep * n_rep) offsets,
               in n_rep repetitions; Bias +- is the norm of the standard
               error of the mean over those repetitions.
    Variance : trace of the covariance of g_hat over n_real realisations,
               each averaging n_runs fresh (noise, offset) runs; +- is the
               standard error of that trace estimate.
    MSE      : Bias^2 + Variance, +- by error propagation.

    Args:
        d: Dimension of the state x_t and the observation y_t.
        N: Number of particles of each filter run.
        T: Number of time steps of the observation sequence.
        n_eps: Bootstrap noise draws for grad L and E[g_hat].
        eps_chunk: Noise draws per call of the exact enumeration.
        offsets_per_rep: Resampling offsets per noise draw in one repetition.
        n_rep: Repetitions for E[g_hat] and its standard error.
        n_runs: Filter runs averaged in one g_hat.
        n_real: Independent realisations of g_hat for the variance.
        taus: Temperatures of the DSR relaxation, one table row each.
        seed: Random seed of the sequence and the noise.
    """
    A0, logd0, y, R = setup(d, T, seed)
    eps_all, _ = draw_noise(n_eps, T, N, d, seed + 1)
    grad_L = sum(
        exact_gradient(A0, logd0, y, R, eps_all[i : i + eps_chunk], "total")
        for i in range(0, n_eps, eps_chunk)
    ) / (n_eps // eps_chunk)
    print(
        f"\n### d={d}  N={N}  T={T}   ||grad L|| = {grad_L.norm():.2f}"
        f"   (exact, over {n_eps} noise draws)"
    )
    print(f"{'Method':<16}{'Bias':>18}{'Variance':>22}{'MSE':>22}")
    eps_rep = eps_all.repeat(offsets_per_rep, 1, 1, 1)

    def row(name, grad_fn):
        t0 = time.time()
        means = []
        for c in range(n_rep):
            g = torch.Generator().manual_seed(seed + 5000 + c)
            u0 = (
                torch.rand(n_eps * offsets_per_rep, T, generator=g, dtype=DTYPE)
                / N
            )
            means.append(grad_fn(eps_rep, u0))
        mean, se = mean_se(means)
        bias = (mean - grad_L).norm().item()
        bias_std = se.norm().item()
        g_hat = torch.stack(
            [
                grad_fn(*draw_noise(n_runs, T, N, d, seed + 9000 + k))
                for k in range(n_real)
            ]
        )
        sq_dev = ((g_hat - g_hat.mean(0)) ** 2).sum(1)
        var = sq_dev.sum().item() / (n_real - 1)
        var_std = (
            sq_dev.std(unbiased=True).item()
            / math.sqrt(n_real)
            * n_real
            / (n_real - 1)
        )
        mse = bias**2 + var
        mse_std = math.hypot(2 * bias * bias_std, var_std)
        print(
            f"{name:<16}"
            f"{bias:10.3f} +- {bias_std:<6.3f}"
            f"{var:12.3f} +- {var_std:<7.3f}"
            f"{mse:12.3f} +- {mse_std:<7.3f}"
            f"  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    row("Hard SR", lambda e, u: hard_gradient(A0, logd0, y, R, e, u))
    for tau in taus:
        row(
            f"DSR (tau={tau})",
            lambda e, u, tau=tau: dsr_gradient(A0, logd0, y, R, e, u, tau),
        )


def main() -> None:
    torch.set_default_dtype(DTYPE)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="validate the estimators instead of making the table",
    )
    ap.add_argument(
        "--quick", action="store_true", help="tiny sizes, for a smoke test only"
    )
    args = ap.parse_args()
    if args.self_test:
        test_pmf()
        test_exact(n_chunks=4 if args.quick else 40)
        return
    sizes = {}
    if args.quick:
        sizes = dict(
            n_eps=100, offsets_per_rep=5, n_rep=3, n_runs=64, n_real=20
        )
    for d in (2, 4):
        bias_variance_table(d=d, **sizes)


if __name__ == "__main__":
    main()
