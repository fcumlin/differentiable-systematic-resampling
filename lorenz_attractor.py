"""Stochastic Lorenz-63 and its camera observation map.

The discretized dynamics are x_(t+1) = F(x_t) x_t + e_t, with
F(x) = exp(A(x_1) delta) approximated by a Taylor series and e_t ~ N(0, Ce).
The observations are y_t = h(x_t) + w_t with w_t ~ N(0, Cw).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import gin
import numpy as np
import torch


def _db_to_linear(x: float) -> float:
    return 10 ** (x / 10)


def camera_observation(x):
    """Camera observation of Lorenz-63: an 8 x 8 Gaussian point spread image.

    Pixel i at grid point g_i of [-30, 30] x [-40, 40] has intensity
    10 exp(-||g_i - x_(1:2)||^2 / (2 (x_3 + 7))). The third coordinate acts as
    depth and is clamped at -6.9 to keep the spread positive.

    Args:
        x: States (..., 3), a numpy array or a torch tensor.

    Returns:
        Pixel intensities (..., 64), of the same type as x.
    """
    if isinstance(x, np.ndarray):
        grid = np.stack(
            np.meshgrid(
                np.linspace(-30, 30, 8), np.linspace(-40, 40, 8), indexing='ij'
            ),
            axis=-1,
        ).reshape(-1, 2)
        sq_dist = np.linalg.norm(grid - x[..., None, :2], axis=-1) ** 2
        depth = np.clip(x[..., None, 2], -6.9, np.inf)
        return 10.0 * np.exp(-0.5 * sq_dist / (1e-12 + depth + 7))
    grid = torch.cartesian_prod(
        torch.linspace(-30, 30, 8, dtype=torch.float32),
        torch.linspace(-40, 40, 8, dtype=torch.float32),
    ).to(x.device)
    spread = 1e-12 + (torch.clamp(x[..., 2], min=-6.9) + 7) ** 0.5
    sq_dist = (
        torch.norm((grid - x[..., None, :2]) / spread[..., None, None], dim=-1)
        ** 2
    )
    return 10.0 * torch.exp(-0.5 * sq_dist)


@gin.configurable
class LorenzSSM:
    """Stochastic Lorenz-63 state space model.

    The continuous-time system matrix is
        A(x_1) = [[-10, 10, 0], [28, -1, -x_1], [0, x_1, -8/3]].

    Attributes:
        n_states: State dimension.
        n_obs: Observation dimension.
        Ce: Process noise covariance; set by generate_state_sequence.
        Cw: Observation noise covariance; set by generate_single_sequence.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_obs: int = 3,
        taylor_order: int = 5,
        delta: float = 0.02,
        observation_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Initializes the instance.

        Args:
            n_states: State dimension.
            n_obs: Observation dimension.
            taylor_order: Order of the Taylor series of the matrix exponential.
            delta: Time step of the discretization.
            observation_fn: Observation map h; the identity if None.
        """
        self.n_states = n_states
        self.n_obs = n_obs
        self._taylor_order = taylor_order
        self._delta = delta
        self._observation_fn = (
            observation_fn if observation_fn is not None else lambda x: x
        )

    @property
    def observation_cov(self) -> np.ndarray:
        return self.Cw

    def f_linearize(self, x: np.ndarray) -> np.ndarray:
        """Mean of the next state, F(x) x.

        Args:
            x: State (n_states,).

        Returns:
            Mean of the next state (n_states,).
        """
        z = x[0]
        system_matrix = np.array(
            [[-10.0, 10.0, 0.0], [28.0, -1.0, -z], [0.0, z, -8.0 / 3]]
        )
        F = np.eye(self.n_states)
        for j in range(1, self._taylor_order + 1):
            F += np.linalg.matrix_power(
                system_matrix * self._delta, j
            ) / math.factorial(j)
        return F @ x

    def generate_state_sequence(self, T: int, sigma_e2_dB: float) -> np.ndarray:
        """Simulates states from x_0 = 0, and sets Ce.

        Args:
            T: Number of time steps.
            sigma_e2_dB: Process noise variance per dimension, in dB.

        Returns:
            States (T, n_states).
        """
        self.Ce = _db_to_linear(sigma_e2_dB) * np.eye(self.n_states)
        states = np.zeros((T, self.n_states))
        noise = np.random.multivariate_normal(
            np.zeros(self.n_states), self.Ce, size=(T + 1,)
        )
        for t in range(T - 1):
            states[t + 1] = self.f_linearize(states[t]) + noise[t]
        return states

    def generate_single_sequence(
        self, T: int, sigma_e2_dB: float, smnr_dB: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulates states and noisy observations, and sets Ce and Cw.

        The observation noise is isotropic, with its variance set by the SMNR
        and the variance of the noise-free observations over all entries.

        Args:
            T: Number of time steps.
            sigma_e2_dB: Process noise variance per dimension, in dB.
            smnr_dB: Signal-to-measurement noise ratio of the observations in
                dB.

        Returns:
            States (T, n_states) and noisy observations (T, n_obs).
        """
        states = self.generate_state_sequence(T, sigma_e2_dB)
        signal_power = np.var(self._observation_fn(states))
        self.Cw = signal_power / _db_to_linear(smnr_dB) * np.eye(self.n_obs)
        noise = np.random.multivariate_normal(
            np.zeros(self.n_obs), self.Cw, size=(T,)
        )
        observations = np.zeros((T, self.n_obs))
        for t in range(T):
            observations[t] = self._observation_fn(states[t]) + noise[t]
        return states, observations


def main():
    """Compares the numpy and torch branches of camera_observation.

    The torch branch places its grid in float32, so it differs from numpy
    by ~1e-7 relative to the peak intensity even for float64 states.
    """
    np.random.seed(0)
    model = LorenzSSM(n_obs=64, observation_fn=camera_observation)
    states, _ = model.generate_single_sequence(
        T=500, sigma_e2_dB=-10, smnr_dB=10
    )
    reference = camera_observation(states)
    for dtype in (torch.float64, torch.float32):
        observations = camera_observation(torch.from_numpy(states).to(dtype))
        diff = np.abs(observations.double().numpy() - reference)
        print(
            f'{dtype} states, torch vs numpy: max abs diff {diff.max():.2e}, '
            f'relative to the peak {diff.max() / np.abs(reference).max():.2e}'
        )


if __name__ == '__main__':
    main()
