"""Transition and proposal models of the particle filter.

Each model maps particles to a Gaussian with diagonal covariance.
"""

from __future__ import annotations

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable
class TransitionModel(nn.Module):
    """MLP transition model p(x_t | x_(t-1)) with diagonal Gaussian output."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """Initializes the instance.

        Args:
            input_dim: Dimension of the state.
            hidden_dim: Width of the two hidden layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, input_dim)
        self.cov_head = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, x: torch.Tensor, *args
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts the distribution of the next state.

        Args:
            x: States (..., D).
            *args: Ignored, so the model can stand in for a proposal model.

        Returns:
            Mean (..., D) and diagonal covariance (..., D, D).
        """
        del args
        features = self.shared(x)
        mean = self.mean_head(features)
        cov = torch.diag_embed(F.softplus(self.cov_head(features)))
        return mean, cov

    def forward_k_steps(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Mean prediction k steps ahead, feeding each mean back as input.

        Args:
            x: States (..., D).
            k: Number of steps.

        Returns:
            Predicted means (..., D).
        """
        mean = x
        for _ in range(k):
            mean, _ = self.forward(mean)
        return mean


@gin.configurable
class LinearTransitionModel(nn.Module):
    """Linear Gaussian transition model x_t = A x_(t-1) + e_t.

    The noise covariance is diag(softplus(log_diag)).
    """

    def __init__(self, state_dim: int = 4):
        """Initializes the instance.

        Args:
            state_dim: Dimension of the state.
        """
        super().__init__()
        self.A = nn.Linear(state_dim, state_dim, bias=False)
        self.log_diag = nn.Parameter(torch.zeros(state_dim))

    def forward(
        self, x: torch.Tensor, *args
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts the distribution of the next state.

        Args:
            x: States (..., D).
            *args: Ignored, so the model can stand in for a proposal model.

        Returns:
            Mean (..., D) and diagonal covariance (..., D, D).
        """
        mean = self.A(x)
        cov = torch.diag_embed(
            F.softplus(self.log_diag).expand(*x.shape[:-1], -1)
        )
        return mean, cov

    def forward_k_steps(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Mean prediction k steps ahead, feeding each mean back as input.

        Args:
            x: States (..., D).
            k: Number of steps.

        Returns:
            Predicted means (..., D).
        """
        mean = x
        for _ in range(k):
            mean, _ = self.forward(mean)
        return mean


@gin.configurable
class ProposalModel(nn.Module):
    """MLP proposal q(x_t | x_(t-1), y_t) with diagonal Gaussian output."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 128, y_dim: int | None = None
    ):
        """Initializes the instance.

        Args:
            input_dim: Dimension of the state.
            hidden_dim: Width of the two hidden layers.
            y_dim: Dimension of the observation; input_dim if None.
        """
        super().__init__()
        self.input_dim = input_dim
        y_dim = input_dim if y_dim is None else y_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, input_dim)
        self.cov_head = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Proposes the next state of every particle given the observation.

        Args:
            x: Particles (B, N, D).
            y: Current observations (B, D_y), shared by the N particles.

        Returns:
            Mean (B, N, D) and diagonal covariance (B, N, D, D).
        """
        num_particles = x.shape[1]
        y = y.unsqueeze(1).repeat(1, num_particles, 1)
        features = self.shared(torch.cat([x, y], dim=-1))
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        mean = self.mean_head(features)
        cov = torch.diag_embed(F.softplus(self.cov_head(features)))
        return mean, cov


if __name__ == '__main__':
    transition = TransitionModel(input_dim=3)
    proposal = ProposalModel(input_dim=3, y_dim=64)
    particles = torch.randn(4, 25, 3)
    mean, cov = transition(particles)
    print(f'Transition: mean {tuple(mean.shape)}, cov {tuple(cov.shape)}')
    mean, cov = proposal(particles, torch.randn(4, 64))
    print(f'Proposal: mean {tuple(mean.shape)}, cov {tuple(cov.shape)}')
