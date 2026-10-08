"""Linear Gaussian state space model of the paper's linear experiments.

    x_t = A x_(t-1) + e_t,  e_t ~ N(0, Ce),
    y_t = x_t + w_t,        w_t ~ N(0, Cw),

with A = 0.9 blockdiag(R(0.3), R(0.5)), R(theta) a 2 x 2 rotation, Ce = 0.1 I,
and Cw set by the SMNR of each simulated sequence.
"""

import numpy as np


def _rotation(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


class LinearSSMRotational:
    """The 4-dimensional damped rotation model, with its Kalman filter.

    Attributes:
        d: State dimension.
        A: Transition matrix (d, d).
        H: Observation matrix (d, d), the identity.
        Ce: Process noise covariance (d, d).
        Cw: Observation noise covariance (d, d); set by
            generate_single_sequence.
        mu0: Initial mean of the Kalman filter (d,).
        P0: Initial covariance of the Kalman filter (d, d).
    """

    def __init__(self, d: int = 4):
        """Initializes the instance.

        Args:
            d: State dimension; must be 4, the size of A.
        """
        self.d = d
        self.A = 0.9 * np.block(
            [
                [_rotation(0.3), np.zeros((2, 2))],
                [np.zeros((2, 2)), _rotation(0.5)],
            ]
        )
        self.H = np.eye(d)
        self.Ce = 0.1 * np.eye(d)
        self.mu0 = np.zeros(d)
        self.P0 = 0.1 * np.eye(d)

    def generate_state_sequence(self, T: int) -> np.ndarray:
        """Simulates states, starting from x_0 ~ N(0, 0.01 I).

        Args:
            T: Number of time steps.

        Returns:
            States (T, d).
        """
        states = np.zeros((T, self.d))
        states[0] = 0.1 * np.random.randn(self.d)
        for t in range(1, T):
            states[t] = self.A @ states[t - 1] + np.random.multivariate_normal(
                np.zeros(self.d), self.Ce
            )
        return states

    def generate_single_sequence(
        self, T: int, smnr_dB: float = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulates a sequence, its observations and its Kalman posteriors.

        Sets Cw so that the observations have the given SMNR, with the signal
        power taken as the mean per-dimension variance of the states.

        Args:
            T: Number of time steps.
            smnr_dB: Signal-to-measurement noise ratio of the observations in
                dB.

        Returns:
            States (T, d), noisy observations (T, d), and the Kalman filter's
            posterior means (T, d) and covariances (T, d, d).
        """
        states = self.generate_state_sequence(T)
        signal_power = np.mean(np.var(states, axis=0))
        self.Cw = signal_power / (10 ** (smnr_dB / 10)) * np.eye(self.d)
        observations = states + np.random.multivariate_normal(
            np.zeros(self.d), self.Cw, size=T
        )
        posterior_mean, posterior_cov = self.kalman_filter(observations)
        return states, observations, posterior_mean, posterior_cov

    def kalman_filter(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman filter from the prior N(mu0, P0), predicting before y_0.

        Args:
            observations: Noisy observations (T, d).

        Returns:
            Posterior means (T, d) and covariances (T, d, d).
        """
        T, d = observations.shape
        A, H, Ce, Cw = self.A, self.H, self.Ce, self.Cw
        posterior_mean = np.zeros((T, d))
        posterior_cov = np.zeros((T, d, d))
        m = self.mu0.copy()
        P = self.P0.copy()
        for t in range(T):
            m_pred = A @ m
            P_pred = A @ P @ A.T + Ce
            S = H @ P_pred @ H.T + Cw
            K = P_pred @ H.T @ np.linalg.inv(S)
            m = m_pred + K @ (observations[t] - H @ m_pred)
            P = (np.eye(d) - K @ H) @ P_pred
            posterior_mean[t] = m
            posterior_cov[t] = P
        return posterior_mean, posterior_cov
