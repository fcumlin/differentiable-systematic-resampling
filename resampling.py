"""Resampling schemes of the particle filter.

* Hard systematic resampling (SR), whose gradient ignores the resampling.
* Soft resampling: multinomial resampling from a mixture of the weights and
  the uniform distribution, with importance weights.
* DSR: differentiable systematic resampling, with or without Morton sorting.
* EROT: entropy-regularized optimal transport, solved with Sinkhorn.

`conditional_resample` applies one of them to the particle sets whose
effective sample size is low.
"""

from __future__ import annotations

import enum

import torch
import torch.nn.functional as F


class ResamplingMethod(enum.Enum):
    """Resampling method, configured by its string value."""

    SOFT = 'soft'
    OT = 'ot'
    SYSTEMATIC = 'systematic'
    SYSTEMATIC_MORTON = 'systematic_morton'
    SYSTEMATIC_HARD_IGNORE_GRAD = 'hard_ignore_grad'


# -----------------------------------------------------------------------------
# Morton sorting.
# -----------------------------------------------------------------------------


def morton_sort(
    particles: torch.Tensor, bits: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorts particles along a Morton (Z-order) space-filling curve.

    Each coordinate is discretized to 2^bits cells over the range of the
    particle set and the bits are interleaved. When bits * D > 62, the Morton
    index is split into 62-bit chunks that are sorted with an LSD radix sort,
    so there is no limit on bits or D.

    Args:
        particles: Particle sets (B, N, D).
        bits: Grid resolution per dimension, capped at 62.

    Returns:
        The sorted particles (B, N, D) and the sorting permutation (B, N).
    """
    B, N, D = particles.shape
    device = particles.device
    bits = min(bits, 62)  # So that the grid size 2^bits - 1 fits in int64.

    mins = particles.amin(dim=1, keepdim=True)
    maxs = particles.amax(dim=1, keepdim=True)
    grid_max = (1 << bits) - 1
    int_coords = (
        ((particles - mins) / (maxs - mins) * grid_max)
        .long()
        .clamp(0, grid_max)
    )
    bit_pos = torch.arange(bits, dtype=torch.int64, device=device)
    bit_vals = (int_coords.unsqueeze(-1) >> (bits - 1 - bit_pos)) & 1
    interleaved = bit_vals.permute(0, 1, 3, 2).reshape(B, N, bits * D)

    chunk = 62
    num_chunks = (bits * D + chunk - 1) // chunk
    pad = num_chunks * chunk - bits * D
    if pad > 0:
        interleaved = F.pad(interleaved, (pad, 0))
    chunk_weights = torch.tensor(
        [1 << (chunk - 1 - k) for k in range(chunk)],
        dtype=torch.int64,
        device=device,
    )
    keys = (interleaved.reshape(B, N, num_chunks, chunk) * chunk_weights).sum(
        -1
    )

    sort_indices = (
        torch.arange(N, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(B, N)
        .clone()
    )
    for i in range(num_chunks - 1, -1, -1):
        current_keys = torch.gather(keys[:, :, i], 1, sort_indices)
        perm = torch.argsort(current_keys, dim=1, stable=True)
        sort_indices = torch.gather(sort_indices, 1, perm)

    sorted_particles = torch.gather(
        particles, 1, sort_indices.unsqueeze(-1).expand_as(particles)
    )
    return sorted_particles, sort_indices


# -----------------------------------------------------------------------------
# Hard and soft resampling.
# -----------------------------------------------------------------------------


def systematic_resample_torch(weights: torch.Tensor) -> torch.Tensor:
    """Ancestor indices of hard systematic resampling.

    The comb is u_j = (j + U) / N with a single U ~ U[0, 1), i.e. the offset
    U / N is uniform on [0, 1/N).

    Args:
        weights: Normalized particle weights (B, N).

    Returns:
        Ancestor indices (B, N).
    """
    B, N = weights.shape
    device = weights.device
    cdf = torch.cumsum(weights, dim=1)
    u = torch.rand(B, 1, device=device)
    positions = (
        torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0) + u
    ) / N
    return torch.clamp(torch.searchsorted(cdf, positions), 0, N - 1)


def hard_resampling_ignore_grad(
    particles: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard systematic resampling; gradients flow through the copied particles.

    Args:
        particles: Particle sets (B, N, D).
        weights: Normalized particle weights (B, N).

    Returns:
        Resampled particles (B, N, D) and uniform weights (B, N).
    """
    B, N, _ = particles.shape
    indices = systematic_resample_torch(weights)
    batch_indices = torch.arange(B, device=particles.device)[:, None].expand(
        B, N
    )
    return (
        particles[batch_indices, indices],
        torch.ones(B, N, device=particles.device) / N,
    )


def soft_resampling(
    particles: torch.Tensor,
    weights: torch.Tensor,
    alpha: float = 0.5,
    num_particles: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft resampling of Karkus et al. (2018).

    Draws ancestors from q = alpha * w + (1 - alpha) / N and corrects with the
    importance weights w / q, so gradients reach the weights.

    Args:
        particles: Particle sets (B, N, D).
        weights: Particle weights (B, N), normalized here.
        alpha: Weight of the particle weights in the sampling mixture.
        num_particles: Number of resampled particles; N if None.

    Returns:
        Resampled particles (B, M, D) and their normalized importance weights
        (B, M).
    """
    _, N, state_dim = particles.shape
    M = num_particles if num_particles is not None else N
    weights = weights / weights.sum(dim=-1, keepdim=True)
    q = alpha * weights + (1 - alpha) * torch.full_like(weights, 1.0 / N)
    indices = torch.multinomial(q, M, replacement=True)
    resampled_particles = torch.gather(
        particles, 1, indices.unsqueeze(-1).expand(-1, -1, state_dim)
    )
    log_w = torch.log(torch.gather(weights, 1, indices) + 1e-12) - torch.log(
        torch.gather(q, 1, indices) + 1e-12
    )
    return resampled_particles, torch.softmax(log_w, dim=-1)


# -----------------------------------------------------------------------------
# Differentiable systematic resampling (DSR).
# -----------------------------------------------------------------------------


def _dsr(
    particles: torch.Tensor,
    weights: torch.Tensor,
    tau: float,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSR of the particles in their given order.

    With CDF F and comb u_j = u_0 + j / N, particle j is the convex combination
    with weights T[j, i] proportional to
    sigmoid((u_j - F_(i-1)) / tau) - sigmoid((u_j - F_i) / tau).
    """
    B, N, _ = particles.shape
    device = particles.device
    cdf = torch.cumsum(weights, dim=1)
    cdf_prev = torch.cat([torch.zeros(B, 1, device=device), cdf[:, :-1]], dim=1)
    u0 = torch.rand(B, 1, device=device) / N
    u = u0 + (torch.arange(N, device=device, dtype=torch.float32) / N).view(
        1, N
    )
    soft = torch.sigmoid(
        (u.unsqueeze(2) - cdf_prev.unsqueeze(1)) / tau
    ) - torch.sigmoid((u.unsqueeze(2) - cdf.unsqueeze(1)) / tau)
    transport = soft / (soft.sum(dim=2, keepdim=True) + 1e-8)
    if requires_grad:
        transport.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
    return torch.bmm(transport, particles), torch.ones(B, N, device=device) / N


def resampler_systematic(
    particles: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 0.1,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSR without sorting the particles.

    Args:
        particles: Particle sets (B, N, D).
        weights: Normalized particle weights (B, N).
        tau: Temperature of the relaxation.
        requires_grad: Clamp the gradient of the transport plan to [-1, 1];
            requires that the weights carry gradients.

    Returns:
        Resampled particles (B, N, D) and uniform weights (B, N).
    """
    return _dsr(particles, weights, tau, requires_grad)


def resampler_systematic_morton(
    particles: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 0.1,
    bits: int = 20,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSR after sorting the particles along a Morton curve.

    Sorting makes CDF neighbours spatial neighbours, so the soft assignments
    mix nearby particles. The sort is a hard permutation without gradient.

    Args:
        particles: Particle sets (B, N, D).
        weights: Normalized particle weights (B, N).
        tau: Temperature of the relaxation.
        bits: Grid resolution per dimension of the Morton curve.
        requires_grad: Clamp the gradient of the transport plan to [-1, 1];
            requires that the weights carry gradients.

    Returns:
        Resampled particles (B, N, D), in Morton order, and uniform weights
        (B, N).
    """
    with torch.no_grad():
        _, sort_indices = morton_sort(particles.detach(), bits=bits)
    sorted_particles = torch.gather(
        particles, 1, sort_indices.unsqueeze(-1).expand_as(particles)
    )
    sorted_weights = torch.gather(weights, 1, sort_indices)
    sorted_weights = sorted_weights / sorted_weights.sum(dim=1, keepdim=True)
    return _dsr(sorted_particles, sorted_weights, tau, requires_grad)


# -----------------------------------------------------------------------------
# Entropy-regularized optimal transport (EROT).
# -----------------------------------------------------------------------------


def _diameter(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diameter_x = x.std(dim=1, unbiased=False).max(dim=-1)[0]
    diameter_y = y.std(dim=1, unbiased=False).max(dim=-1)[0]
    res = torch.maximum(diameter_x, diameter_y)
    return torch.where(res == 0.0, 1.0, res.double())


def _cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (torch.cdist(x, y, p=2.0) ** 2) / 2.0


def _max_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Initial scale of the epsilon-scaling.

    The minimum uses x.max; kept as is, since the results were computed with it.
    """
    max_max = torch.maximum(
        x.max(dim=1)[0].max(dim=1)[0], y.max(dim=1)[0].max(dim=1)[0]
    )
    min_min = torch.minimum(
        x.max(dim=1)[0].min(dim=1)[0], y.min(dim=1)[0].min(dim=1)[0]
    )
    return max_max - min_min


def _softmin(
    epsilon: torch.Tensor, cost_matrix: torch.Tensor, f: torch.Tensor
) -> torch.Tensor:
    """Soft minimum -epsilon log sum_j exp(f_j - C_ij / epsilon)."""
    b, _, n = cost_matrix.shape
    temp_val = f.reshape([b, 1, n]) - cost_matrix / epsilon.reshape([-1, 1, 1])
    return -epsilon.reshape([-1, 1]) * torch.logsumexp(temp_val, dim=2)


def sinkhorn_loop(
    log_alpha: torch.Tensor,
    log_beta: torch.Tensor,
    cost_xy: torch.Tensor,
    cost_yx: torch.Tensor,
    cost_xx: torch.Tensor,
    cost_yy: torch.Tensor,
    epsilon: torch.Tensor,
    particles_diameter: torch.Tensor,
    scaling: float,
    threshold: float,
    max_iter: int,
    device: str | torch.device = 'cuda',
) -> tuple[torch.Tensor, ...]:
    """Symmetric Sinkhorn iterations with epsilon-scaling.

    Epsilon starts at particles_diameter^2 and shrinks by scaling^2 per
    iteration down to `epsilon`; the iterations stop when the potentials
    change less than `threshold`, or after max_iter.

    Args:
        log_alpha: Log-weights of the source (B, N).
        log_beta: Log-weights of the target (B, N).
        cost_xy: Cost matrices (B, N, N) between source and target, and
        cost_yx: target and source,
        cost_xx: source and source,
        cost_yy: and target and target.
        epsilon: Final regularization.
        particles_diameter: Scale of the initial regularization (B,).
        scaling: Factor of the epsilon-scaling.
        threshold: Convergence threshold of the potentials.
        max_iter: Maximum number of iterations.
        device: Device of the iteration flags.

    Returns:
        Potentials a_y, b_x, a_x, b_y (B, N) and the number of iterations.
    """
    batch_size = log_alpha.shape[0]
    continue_flag = torch.ones([batch_size], dtype=bool).to(device)
    epsilon_0 = particles_diameter**2
    scaling_factor = scaling**2

    def apply_one(a_y, b_x, a_x, b_y, continue_, running_epsilon):
        running_epsilon_ = running_epsilon.reshape([-1, 1])
        continue_reshaped = continue_.reshape([-1, 1])
        at_y = torch.where(
            continue_reshaped,
            _softmin(
                running_epsilon, cost_yx, log_alpha + b_x / running_epsilon_
            ),
            a_y,
        )
        bt_x = torch.where(
            continue_reshaped,
            _softmin(
                running_epsilon, cost_xy, log_beta + a_y / running_epsilon_
            ),
            b_x,
        )
        at_x = torch.where(
            continue_reshaped,
            _softmin(
                running_epsilon, cost_xx, log_alpha + a_x / running_epsilon_
            ),
            a_x,
        )
        bt_y = torch.where(
            continue_reshaped,
            _softmin(
                running_epsilon, cost_yy, log_beta + b_y / running_epsilon_
            ),
            b_y,
        )
        a_y_new = (a_y + at_y) / 2
        b_x_new = (b_x + bt_x) / 2
        a_x_new = (a_x + at_x) / 2
        b_y_new = (b_y + bt_y) / 2
        a_y_diff = (torch.abs(a_y_new - a_y)).max(dim=1)[0]
        b_x_diff = (torch.abs(b_x_new - b_x)).max(dim=1)[0]
        local_continue = torch.logical_or(
            a_y_diff > threshold, b_x_diff > threshold
        )
        return a_y_new, b_x_new, a_x_new, b_y_new, local_continue

    a_y = _softmin(epsilon_0, cost_yx, log_alpha)
    b_x = _softmin(epsilon_0, cost_xy, log_beta)
    a_x = _softmin(epsilon_0, cost_xx, log_alpha)
    b_y = _softmin(epsilon_0, cost_yy, log_beta)
    running_epsilon = epsilon_0
    total_iter = 0
    while torch.logical_and(
        torch.tensor(total_iter < max_iter - 1, dtype=bool).to(device),
        torch.all(continue_flag.bool()),
    ):
        a_y, b_x, a_x, b_y, local_continue = apply_one(
            a_y, b_x, a_x, b_y, continue_flag, running_epsilon
        )
        new_epsilon = torch.maximum(running_epsilon * scaling_factor, epsilon)
        continue_flag = torch.logical_or(
            new_epsilon < running_epsilon, local_continue
        )
        running_epsilon = new_epsilon
        total_iter += 1

    a_y, b_x, a_x, b_y = (
        a_y.detach().clone(),
        b_x.detach().clone(),
        a_x.detach().clone(),
        b_y.detach().clone(),
    )
    epsilon_ = epsilon.reshape([-1, 1])
    final_a_y = _softmin(epsilon, cost_yx, log_alpha + b_x / epsilon_)
    final_b_x = _softmin(epsilon, cost_xy, log_beta + a_y / epsilon_)
    final_a_x = _softmin(epsilon, cost_xx, log_alpha + a_x / epsilon_)
    final_b_y = _softmin(epsilon, cost_yy, log_beta + b_y / epsilon_)
    return final_a_y, final_b_x, final_a_x, final_b_y, total_iter + 2


def sinkhorn_potentials(
    log_alpha: torch.Tensor,
    x: torch.Tensor,
    log_beta: torch.Tensor,
    y: torch.Tensor,
    epsilon: torch.Tensor,
    scaling: float,
    threshold: float,
    max_iter: int,
    device: str | torch.device = 'cuda',
) -> tuple[torch.Tensor, ...]:
    """Sinkhorn potentials between weighted point clouds x and y.

    Args:
        log_alpha: Log-weights of x (B, N).
        x: Source points (B, N, D).
        log_beta: Log-weights of y (B, N).
        y: Target points (B, N, D).
        epsilon: Regularization.
        scaling: Factor of the epsilon-scaling.
        threshold: Convergence threshold of the potentials.
        max_iter: Maximum number of iterations.
        device: Device of the iteration flags.

    Returns:
        As sinkhorn_loop.
    """
    return sinkhorn_loop(
        log_alpha,
        log_beta,
        _cost(x, y.detach().clone()),
        _cost(y, x.detach().clone()),
        _cost(x, x.detach().clone()),
        _cost(y, y.detach().clone()),
        epsilon,
        _max_min(x, y).detach().clone(),
        scaling,
        threshold,
        max_iter,
        device=device,
    )


def transport_from_potentials(
    x: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    eps: torch.Tensor,
    logw: torch.Tensor,
    n: torch.Tensor,
    device: str | torch.device = 'cuda',
) -> torch.Tensor:
    """Transport matrix from the Sinkhorn potentials.

    Args:
        x: Points (B, N, D).
        f: Potential of the source (B, N).
        g: Potential of the target (B, N).
        eps: Regularization.
        logw: Log-weights of the source (B, N).
        n: Number of points, as a float tensor.
        device: Device of log(n).

    Returns:
        Transport matrix (B, N, N), whose columns sum to the weights.
    """
    log_n = torch.log(n).to(device)
    temp = (torch.unsqueeze(f, 2) + torch.unsqueeze(g, 1) - _cost(x, x)) / eps
    temp = temp - torch.logsumexp(temp, dim=1, keepdims=True) + log_n
    return torch.exp(temp + torch.unsqueeze(logw, 1))


def transport_function(
    x: torch.Tensor,
    logw: torch.Tensor,
    eps: float,
    scaling: float,
    threshold: float,
    max_iter: int,
    n: int,
    device: str | torch.device = 'cuda',
) -> torch.Tensor:
    """EROT transport matrix from weighted particles to uniform weights.

    The particles are centered and divided by their largest per-dimension
    standard deviation times sqrt(D) before the transport is computed.

    Args:
        x: Particle sets (B, N, D).
        logw: Log-weights (B, N).
        eps: Entropic regularization, in the rescaled coordinates.
        scaling: Factor of the epsilon-scaling.
        threshold: Convergence threshold of the Sinkhorn potentials.
        max_iter: Maximum number of Sinkhorn iterations.
        n: Number of particles N.
        device: Device of the computation.

    Returns:
        Transport matrix (B, N, N).
    """
    eps = torch.tensor(eps, dtype=torch.float).to(device)
    float_n = torch.as_tensor(n, dtype=torch.float, device=device)
    uniform_log_weight = -torch.log(float_n).to(device) * torch.ones_like(
        logw
    ).to(device)
    dimension = torch.tensor(x.shape[-1]).to(device)
    centered_x = x - x.mean(dim=1, keepdim=True).detach().clone()
    scale = _diameter(x, x).reshape([-1, 1, 1]) * torch.sqrt(dimension)
    scaled_x = centered_x / scale.detach().clone()
    alpha, beta, _, _, _ = sinkhorn_potentials(
        logw,
        scaled_x,
        uniform_log_weight,
        scaled_x,
        eps,
        scaling,
        threshold,
        max_iter,
        device=device,
    )
    return transport_from_potentials(
        scaled_x, alpha, beta, eps, logw, float_n, device=device
    )


class _ClampedTransport(torch.autograd.Function):
    """Passes the transport matrix on, clamping its gradient to [-1, 1]."""

    @staticmethod
    def forward(ctx, x, logw, x_, logw_, transport_matrix_):
        ctx.save_for_backward(transport_matrix_, x_, logw_)
        return transport_matrix_.clone()

    @staticmethod
    def backward(ctx, d_transport):
        d_transport = torch.clamp(d_transport, -1.0, 1.0)
        transport_matrix_, x_, logw_ = ctx.saved_tensors
        dx, dlogw = torch.autograd.grad(
            transport_matrix_,
            [x_, logw_],
            grad_outputs=d_transport,
            retain_graph=True,
            allow_unused=True,
        )
        if dx is None:
            dx = torch.zeros_like(x_)
        if dlogw is None:
            dlogw = torch.zeros_like(logw_)
        return dx, dlogw, None, None, None


def resampler_ot(
    particles: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 0.5,
    scaling: float = 0.75,
    threshold: float = 1e-3,
    max_iter: int = 100,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EROT resampling of Corenflos et al. (2021).

    Args:
        particles: Particle sets (B, N, D).
        weights: Normalized particle weights (B, N).
        eps: Entropic regularization.
        scaling: Factor of the epsilon-scaling.
        threshold: Convergence threshold of the Sinkhorn potentials.
        max_iter: Maximum number of Sinkhorn iterations.
        device: Device of the computation; that of the particles if None.

    Returns:
        Resampled particles (B, N, D) and uniform weights (B, N), as float32.
    """
    if device is None:
        device = particles.device
    logw = weights.log()
    x_ = particles.detach().clone().requires_grad_()
    logw_ = logw.detach().clone().requires_grad_()
    transport_matrix_ = transport_function(
        x_, logw_, eps, scaling, threshold, max_iter, particles.shape[1], device
    )
    transport_matrix = _ClampedTransport.apply(
        particles, logw, x_, logw_, transport_matrix_
    )
    num_particles = torch.tensor(particles.shape[1]).float()
    resampled_particles = torch.matmul(
        transport_matrix.float(), particles.float()
    ).float()
    uniform_weights = (torch.ones_like(logw.exp()) / num_particles).float()
    return resampled_particles, uniform_weights


# -----------------------------------------------------------------------------
# Resampling inside the particle filter.
# -----------------------------------------------------------------------------


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Effective sample size 1 / sum_i w_i^2 of each particle set.

    Args:
        weights: Normalized particle weights (B, N).

    Returns:
        Effective sample sizes (B,).
    """
    return 1.0 / torch.sum(weights**2, dim=1)


def conditional_resample(
    weights: torch.Tensor,
    particles: torch.Tensor,
    resampling_method: ResamplingMethod = ResamplingMethod.OT,
    hard_systematic_resampling: bool = False,
    ess_threshold: float = 0.5,
    sys_tau: float = 0.5,
    morton_bits: int = 20,
    ot_eps: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resamples the particle sets whose ESS is below ess_threshold * N.

    Args:
        weights: Normalized particle weights (B, N).
        particles: Particle sets (B, N, D).
        resampling_method: Resampling method.
        hard_systematic_resampling: Use hard systematic resampling regardless
            of resampling_method, as at evaluation.
        ess_threshold: Resampling threshold as a fraction of N.
        sys_tau: Temperature of DSR.
        morton_bits: Grid resolution per dimension of Morton sorting.
        ot_eps: Entropic regularization of EROT.

    Returns:
        Weights (B, N) and particles (B, N, D), resampled where needed.
    """
    _, num_particles, _ = particles.shape
    resample_mask = (
        effective_sample_size(weights) < ess_threshold * num_particles
    )
    if not resample_mask.any():
        return weights, particles

    resample_indices = torch.where(resample_mask)[0]
    x = particles[resample_indices]
    w = weights[resample_indices]
    if hard_systematic_resampling:
        new_particles, new_weights = hard_resampling_ignore_grad(x, w)
    elif resampling_method == ResamplingMethod.SOFT:
        new_particles, new_weights = soft_resampling(
            x, w, alpha=0.5, num_particles=num_particles
        )
    elif resampling_method == ResamplingMethod.OT:
        new_particles, new_weights = resampler_ot(x, w, eps=ot_eps)
    elif resampling_method == ResamplingMethod.SYSTEMATIC:
        new_particles, new_weights = resampler_systematic(x, w, tau=sys_tau)
    elif resampling_method == ResamplingMethod.SYSTEMATIC_MORTON:
        new_particles, new_weights = resampler_systematic_morton(
            x, w, tau=sys_tau, bits=morton_bits
        )
    elif resampling_method == ResamplingMethod.SYSTEMATIC_HARD_IGNORE_GRAD:
        new_particles, new_weights = hard_resampling_ignore_grad(x, w)
    else:
        raise ValueError(f'Unknown resampling method: {resampling_method}')

    resampled_weights = weights.clone()
    resampled_particles = particles.clone()
    resampled_weights[resample_indices] = new_weights
    resampled_particles[resample_indices] = new_particles
    return resampled_weights, resampled_particles
