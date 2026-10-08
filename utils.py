"""Utility functions for particle filter experiments and analysis."""

import matplotlib.pyplot as plt
import numpy as np
import torch


class GradientTracker:
    """Track gradient statistics aggregated by epoch"""
    def __init__(self):
        self.epoch_grad_norms = []
        self.epochs = []
        self.current_epoch_grads = []
    
    def update(self, model):
        """Called once per batch"""
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += (p.grad.data ** 2).sum().item()
        grad_norm = grad_norm ** 0.5
        self.current_epoch_grads.append(grad_norm)
    
    def end_epoch(self, epoch):
        """Called at end of each epoch"""
        if len(self.current_epoch_grads) > 0:
            # Store mean and std for this epoch
            mean_grad = np.mean(self.current_epoch_grads)
            std_grad = np.std(self.current_epoch_grads)
            self.epoch_grad_norms.append((mean_grad, std_grad))
            self.epochs.append(epoch)
            self.current_epoch_grads = []  # Reset for next epoch
    
    def plot(self, save_path):
        """Plot epoch-level gradient statistics"""
        if len(self.epoch_grad_norms) == 0:
            return
        
        means = [x[0] for x in self.epoch_grad_norms]
        stds = [x[1] for x in self.epoch_grad_norms]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.epochs, means, linewidth=2, label='Mean Gradient Norm')
        means_array = np.array(means)
        stds_array = np.array(stds)
        ax.fill_between(self.epochs, 
                        means_array - stds_array, 
                        means_array + stds_array, 
                        alpha=0.3, label='±1 std')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norm Over Training', fontsize=14)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class NMSELossAccumulator:
    def __init__(self):
        self._total_nmse = 0.0
        self._num_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with a batch of predictions and targets"""
        assert predictions.shape == targets.shape
        
        batch_size = predictions.shape[0]
        mse = torch.sum((predictions - targets) ** 2, dim=(-1, -2))
        target_power = torch.sum(targets ** 2, dim=(-1, -2))
        nmse_db = 10 * torch.log10(mse / target_power)
        self._total_nmse += torch.sum(nmse_db).item()
        self._num_samples += batch_size
    
    def get_total_nmse_loss(self) -> float:
        """Compute final NMSE loss in dB"""
        if self._num_samples == 0:
            return 0.0
        return self._total_nmse / self._num_samples
    
    def reset(self):
        """Reset accumulator"""
        self._total_nmse = 0.0
        self._num_samples = 0


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'min'
    ):
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._best_score = float('inf') if mode == 'min' else float('-inf')
        self._mode = mode

    @property
    def patience(self):
        return self._patience

    def _is_better(self, new_score):
        if self._mode == 'min':
            return new_score < self._best_score - self._min_delta
        return new_score > self._best_score + self._min_delta
    
    def __call__(self, new_score, epoch):
        if self._is_better(new_score):
            self._best_score = new_score
            self._counter = 0
            return False
        else:
            self._counter += 1
            if self._counter >= self._patience:
                return True
            return False


def mvn_logprob(x, m, S, eps=1e-6):
    """Log-density of N(m, S) at x via Cholesky, with eps * I added to S."""
    D = x.shape[-1]
    S = S + eps * torch.eye(D, device=S.device, dtype=S.dtype)
    L = torch.linalg.cholesky(S)
    diff = (x - m).unsqueeze(-1)
    y = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
    maha = (y * y).sum(-1)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return -0.5 * (D * torch.log(torch.tensor(2 * torch.pi, device=x.device)) + logdet + maha)


def empirical_gaussian_kl(
    particles: torch.Tensor,
    kf_mean: torch.Tensor,
    kf_cov: torch.Tensor,
) -> torch.Tensor:
    """KL(empirical particle Gaussian || Kalman posterior Gaussian), averaged over batch.

    Fits a Gaussian to the (equally-weighted) resampled particles and computes
    KL divergence against the Kalman filter posterior.  This penalises both a
    wrong mean and a wrong covariance, making it a fairer quality metric than
    plain log-likelihood (which would reward collapsing to the mean).

    Args:
        particles: (B, N, D) — resampled particles (uniform weights assumed).
        kf_mean:   (B, D)    — Kalman posterior mean at this time step.
        kf_cov:    (B, D, D) — Kalman posterior covariance at this time step.
    Returns:
        Scalar mean KL(particle_gaussian || kalman_posterior) over the batch.
    """
    B, N, D = particles.shape
    mean_p = particles.mean(dim=1)                               # (B, D)
    diff = particles - mean_p.unsqueeze(1)                       # (B, N, D)
    cov_p = torch.einsum('bni,bnj->bij', diff, diff) / (N - 1)  # (B, D, D)

    # Symmetrize (eliminates floating-point asymmetry) then add relative jitter.
    cov_p = (cov_p + cov_p.transpose(-1, -2)) / 2
    trace = cov_p.diagonal(dim1=-2, dim2=-1).sum(-1)            # (B,)
    jitter = (trace / D).clamp(min=1e-6).unsqueeze(-1).unsqueeze(-1)
    eye = torch.eye(D, device=particles.device, dtype=particles.dtype)
    cov_p = cov_p + 1e-4 * jitter * eye

    try:
        p = torch.distributions.MultivariateNormal(mean_p, cov_p)
        q = torch.distributions.MultivariateNormal(kf_mean, kf_cov)
        return torch.distributions.kl_divergence(p, q).mean()
    except ValueError:
        return None


def compute_wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2):
   """Compute Wasserstein-2 distance between two multivariate Gaussians"""
   mean_diff = mu1 - mu2
   mean_term = torch.sum(mean_diff * mean_diff, dim=-1)
   sigma1_sqrt = torch.linalg.cholesky(sigma1)
   temp = torch.bmm(sigma1_sqrt, torch.bmm(sigma2, sigma1_sqrt))
   
   eigenvals = torch.linalg.eigvals(temp).real
   eigenvals = torch.clamp(eigenvals, min=1e-8)  # Numerical stability
   sqrt_trace = torch.sum(torch.sqrt(eigenvals), dim=-1)
   
   trace_sigma1 = torch.diagonal(sigma1, dim1=-2, dim2=-1).sum(-1)
   trace_sigma2 = torch.diagonal(sigma2, dim1=-2, dim2=-1).sum(-1)
   
   cov_term = trace_sigma1 + trace_sigma2 - 2 * sqrt_trace
   
   return torch.sqrt(mean_term + cov_term)