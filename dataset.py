"""Dataset definition and loader.

To make it easier to implement datasets into the framework, there is a baseclass
for datasets, called `BaseDynamicalDataset`. This class implements the
`torch.utils.data.Dataset` interface. Datasets suitable for the framework have
the following:
* A temporal component, and the states and observations are 1D tensors.
    The latter means that images, videos, etc. are not suitable. A trajectory
    (state/observation) is of shape (T, D), where T is the number of time steps
    and D is the dimension.
* A state to observation mapping, `observation_fn`. It is the identity by
    default and may be nonlinear, e.g. the camera map of Lorenz-63.
* Additive Gaussian observation noise, whose covariance is part of every item.

Test datasets subclass `BaseDynamicalTestDataset` and hold one-step
transitions x_(t-1) -> x_t instead of sequences.

Minimal example of implementing a dataset:

```
class NewDataset(BaseDynamicalDataset):

    def __init__(self):
        super().__init__(
            num_samples=100,
            signal_length=10,
            smnr_db=10,
        )

    @property
    def observation_noise_covariance(self):
        return torch.eye(2) * 0.1

    def _generate_data(self):
        states = torch.ones(self._num_samples, self._signal_length, 2)
        observations = self.state_to_observation(states)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(2),
            self.observation_noise_covariance,
        )
        noisy_observations = observations + mvn.sample(
            (self._num_samples, self._signal_length)
        )
        Cws = self.observation_noise_covariance.expand(
            self._num_samples, 1, 2, 2
        )
        return states, observations, noisy_observations, Cws
```
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
import dataclasses
import os

import gin
import numpy as np
import scipy.io
import torch
import torch.utils.data
import torch.utils.data.dataset

import linear_ssm as linear_ssm_lib
import lorenz_attractor as lorenz_attractor_lib

# Downloaded by scripts/download_characters.sh.
CHARACTERS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'mixoutALL_shifted.mat'
)


def _check_expected_dim(
    *arrays: np.ndarray | torch.Tensor, expected_dim: int
) -> None:
    """Checks that all arrays have `expected_dim` dimensions.

    Args:
        *arrays: Arrays to check.
        expected_dim: Expected number of dimensions of every array.

    Raises:
        ValueError: If an array has another number of dimensions.
    """
    for array in arrays:
        if len(array.shape) != expected_dim:
            raise ValueError(
                f'Array has {len(array.shape)} dimensions, expected'
                f' {expected_dim}'
            )


def _convert_arrays_to_tensors(*arrays: np.ndarray) -> list[torch.Tensor]:
    """Converts numpy arrays to float32 torch tensors.

    Args:
        *arrays: Arrays to convert.

    Returns:
        The arrays as tensors with dtype torch.float32.
    """
    return [torch.from_numpy(array).to(dtype=torch.float32) for array in arrays]


# -----------------------------------------------------------------------------
# Baseclass for training dynamical datasets.
# -----------------------------------------------------------------------------


class BaseDynamicalDataset(abc.ABC, torch.utils.data.dataset.Dataset):
    """Abstract base class for dynamical system datasets.

    Every item is one sequence: states (T, D), noise-free observations,
    noisy observations (T, D_y) and the observation noise covariance
    (1, D_y, D_y). Training uses the states, the noisy observations and the
    covariance; the datasets of this module fill the noise-free observations
    with a copy of the states.
    """

    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        smnr_db: float,
        observation_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initializes the instance and generates the data.

        Args:
            num_samples: Number of sequences to generate.
            signal_length: Number of time steps T of each sequence.
            smnr_db: Signal-to-measurement noise ratio of the observations in
                dB.
            observation_fn: Map from states to noise-free observations;
                identity if None.
        """
        self._num_samples = num_samples
        self._signal_length = signal_length
        self._smnr_db = smnr_db
        self._observation_fn = (
            observation_fn if observation_fn is not None else lambda x: x
        )
        (
            self._states,
            self._observations,
            self._noisy_observations,
            self._Cws,
        ) = self._generate_data()
        _check_expected_dim(
            self._states,
            self._observations,
            self._noisy_observations,
            expected_dim=3,
        )

    @abc.abstractmethod
    def _generate_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the dataset.

        Returns:
            States (num_samples, T, D), noise-free observations (unused by
            training), noisy observations (num_samples, T, D_y) and
            observation noise covariances (num_samples, 1, D_y, D_y).
        """

    def state_to_observation(self, states: torch.Tensor) -> torch.Tensor:
        """Maps states to noise-free observations with `observation_fn`.

        Args:
            states: States (..., D).

        Returns:
            Noise-free observations (..., D_y).
        """
        return self._observation_fn(states)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._states[index],
            self._observations[index],
            self._noisy_observations[index],
            self._Cws[index],
        )

    def collate_fn(
        self, batch: Sequence[tuple[torch.Tensor, ...]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stacks a list of items into batch tensors.

        Args:
            batch: Items as returned by __getitem__.

        Returns:
            States, observations, noisy observations and noise covariances,
            each with a leading batch dimension.
        """
        states, observations, noisy_observations, Cws = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(observations)),
            torch.FloatTensor(np.array(noisy_observations)),
            torch.FloatTensor(np.array(Cws)),
        )


# -----------------------------------------------------------------------------
# Dynamical datasets for training.
# -----------------------------------------------------------------------------


@gin.configurable
class LorenzAttractor(BaseDynamicalDataset):
    """Lorenz-63 observed through the 8 x 8 camera map.

    See lorenz_attractor.camera_observation for the observation map.
    """

    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        smnr_db: float,
        state_dim: int = 3,
        observation_dim: int = 64,
    ) -> None:
        """Initializes the instance and generates the data.

        Args:
            num_samples: Number of sequences to generate.
            signal_length: Number of time steps T of each sequence.
            smnr_db: Signal-to-measurement noise ratio of the observations in
                dB.
            state_dim: Dimension of the state.
            observation_dim: Number of camera pixels.
        """
        observation_fn = lorenz_attractor_lib.camera_observation
        self._lorenz_attractor_model = lorenz_attractor_lib.LorenzSSM(
            n_states=state_dim,
            n_obs=observation_dim,
            observation_fn=observation_fn,
        )
        super().__init__(
            num_samples=num_samples,
            signal_length=signal_length,
            smnr_db=smnr_db,
            observation_fn=observation_fn,
        )

    def _generate_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_states = []
        noisy_observations = []
        Cws = []
        for _ in range(self._num_samples):
            states, observations = (
                self._lorenz_attractor_model.generate_single_sequence(
                    self._signal_length,
                    sigma_e2_dB=-10,
                    smnr_dB=self._smnr_db,
                )
            )
            Cws.append(
                np.expand_dims(
                    self._lorenz_attractor_model.observation_cov, axis=0
                )
            )
            all_states.append(states)
            noisy_observations.append(observations)
        states = np.stack(all_states, axis=0)
        return _convert_arrays_to_tensors(
            states,
            states.copy(),
            np.stack(noisy_observations, axis=0),
            np.stack(Cws, axis=0),
        )


@gin.configurable
class CharacterTrajectoriesDataset(BaseDynamicalDataset):
    """CharacterTrajectories pen-tip sequences with added Gaussian noise.

    The sequences are 3-dimensional pen-tip positions, and the noise variance
    of each dimension is set by the SMNR. The learned state can be larger;
    `state_to_observation` observes its last 3 dimensions.
    """

    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        smnr_db: float,
        valid: str = 'train',
        data_path: str = CHARACTERS_PATH,
    ):
        """Initializes the instance and loads the data.

        Args:
            num_samples: Maximum number of sequences.
            signal_length: Number of time steps T of each sequence.
            smnr_db: Signal-to-measurement noise ratio of the observations in
                dB, per dimension.
            valid: 'train' (trajectories 0-1999) or 'valid' (2000-2399).
            data_path: Path of mixoutALL_shifted.mat.
        """
        self.data_path = data_path
        self._valid = valid
        super().__init__(
            num_samples,
            signal_length,
            smnr_db,
            observation_fn=lambda x: x[..., -3:],
        )

    def _generate_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectories = scipy.io.loadmat(self.data_path)['mixout'][0]
        if self._valid == 'train':
            trajectories = trajectories[:2000]
        elif self._valid == 'valid':
            trajectories = trajectories[2000:2400]
        else:
            raise ValueError(f"valid must be 'train' or 'valid': {self._valid}")

        sequences = []
        for trajectory in trajectories:
            trajectory = trajectory.T
            for i in range(
                0,
                len(trajectory) - self._signal_length + 1,
                self._signal_length,
            ):
                sequence = trajectory[i : i + self._signal_length]
                if len(sequence) == self._signal_length:
                    sequences.append(sequence)
                    if len(sequences) >= self._num_samples:
                        break
            if len(sequences) >= self._num_samples:
                break
        self._num_samples = min(self._num_samples, len(sequences))

        states = torch.FloatTensor(np.array(sequences[: self._num_samples]))
        observations = states.clone()
        dim = states.shape[-1]
        Cws = torch.zeros(self._num_samples, dim, dim)
        noisy_observations = observations.clone()
        smnr_linear = 10 ** (self._smnr_db / 10)
        for i in range(self._num_samples):
            noise_power = torch.var(states[i], dim=0) / smnr_linear
            Cws[i] = torch.diag(torch.clamp(noise_power, min=1e-3))
            noise = torch.distributions.MultivariateNormal(
                torch.zeros(dim), Cws[i]
            ).sample((self._signal_length,))
            noisy_observations[i] += noise
        return states, observations, noisy_observations, Cws.unsqueeze(1)


@gin.configurable
class LinearSSMRotationalDataset(BaseDynamicalDataset):
    """4-dimensional linear Gaussian SSM with Kalman posteriors.

    Items additionally hold the Kalman filter posterior means and covariances.
    """

    def __init__(self, num_samples: int, signal_length: int, smnr_db: float):
        """Initializes the instance and generates the data.

        Args:
            num_samples: Number of sequences to generate.
            signal_length: Number of time steps T of each sequence.
            smnr_db: Signal-to-measurement noise ratio of the observations in
                dB.
        """
        self._linear_model = linear_ssm_lib.LinearSSMRotational(d=4)
        super().__init__(num_samples, signal_length, smnr_db)

    def _generate_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_states, noisy_observations, Cws = [], [], []
        posterior_means, posterior_covs = [], []
        for _ in range(self._num_samples):
            states, observations, posterior_mean, posterior_cov = (
                self._linear_model.generate_single_sequence(
                    self._signal_length, smnr_dB=self._smnr_db
                )
            )
            all_states.append(states)
            noisy_observations.append(observations)
            Cws.append(np.expand_dims(self._linear_model.Cw, 0))
            posterior_means.append(posterior_mean)
            posterior_covs.append(posterior_cov)
        states = np.stack(all_states)
        self._posterior_mean, self._posterior_cov = _convert_arrays_to_tensors(
            np.stack(posterior_means), np.stack(posterior_covs)
        )
        return _convert_arrays_to_tensors(
            states, states.copy(), np.stack(noisy_observations), np.stack(Cws)
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self._states[index],
            self._observations[index],
            self._noisy_observations[index],
            self._Cws[index],
            self._posterior_mean[index],
            self._posterior_cov[index],
        )

    def collate_fn(
        self, batch: Sequence[tuple[torch.Tensor, ...]]
    ) -> tuple[torch.Tensor, ...]:
        """Stacks a list of items into batch tensors.

        Args:
            batch: Items as returned by __getitem__.

        Returns:
            States, observations, noisy observations, noise covariances,
            posterior means and posterior covariances, each with a leading
            batch dimension.
        """
        return tuple(
            torch.FloatTensor(np.array(field)) for field in zip(*batch)
        )


# -----------------------------------------------------------------------------
# Baseclass for test dynamical datasets.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class MarkovianStateDistribution:
    """One test transition with its Gaussian p(x_t | x_(t-1)).

    Attributes:
        prev_state: The previous state x_(t-1).
        next_state: The sampled next state x_t.
        mean_next_state: Mean of p(x_t | x_(t-1)); None if unknown.
        cov_next_state: Covariance of p(x_t | x_(t-1)); None if unknown.
        state_five_steps: The state x_(t+4), five steps after x_(t-1); None
            if unknown.
    """

    prev_state: torch.Tensor
    next_state: torch.Tensor
    mean_next_state: torch.Tensor | None
    cov_next_state: torch.Tensor | None
    state_five_steps: torch.Tensor | None = None


class BaseDynamicalTestDataset(abc.ABC, torch.utils.data.dataset.Dataset):
    """Abstract base class for test sets of one-step transitions."""

    def __init__(self, num_samples: int, signal_length: int) -> None:
        """Initializes the instance and generates the transitions.

        Args:
            num_samples: Number of sequences to draw transitions from.
            signal_length: Number of time steps of each sequence.
        """
        self._num_samples = num_samples
        self._signal_length = signal_length
        self._markovian_state_distributions = (
            self._generate_markovian_state_distributions()
        )

    @abc.abstractmethod
    def _generate_markovian_state_distributions(
        self,
    ) -> list[MarkovianStateDistribution]:
        """Generates the test transitions."""

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor | None, ...]:
        distribution = self._markovian_state_distributions[index]
        return (
            distribution.prev_state,
            distribution.next_state,
            distribution.mean_next_state,
            distribution.cov_next_state,
            distribution.state_five_steps,
        )

    def collate_fn(
        self, batch: Sequence[tuple[torch.Tensor | None, ...]]
    ) -> tuple[torch.Tensor | None, ...]:
        """Stacks a list of items into batch tensors.

        Args:
            batch: Items as returned by __getitem__.

        Returns:
            Previous states, next states, means and covariances of the next
            state, and states five steps ahead, each with a leading batch
            dimension. A field is None if any item lacks it; the mean and the
            covariance are both None if any mean is missing.
        """
        (
            prev_state,
            next_state,
            mean_next_state,
            cov_next_state,
            state_five_steps,
        ) = zip(*batch)
        prev_state = torch.FloatTensor(np.array(prev_state))
        next_state = torch.FloatTensor(np.array(next_state))
        if any(s is None for s in state_five_steps):
            state_five_steps = None
        else:
            state_five_steps = torch.FloatTensor(np.array(state_five_steps))
        if any(m is None for m in mean_next_state):
            return prev_state, next_state, None, None, state_five_steps
        return (
            prev_state,
            next_state,
            torch.FloatTensor(np.array(mean_next_state)),
            torch.FloatTensor(np.array(cov_next_state)),
            state_five_steps,
        )


# -----------------------------------------------------------------------------
# Dynamical datasets for testing.
# -----------------------------------------------------------------------------


@gin.configurable
class LorenzAttractorTest(BaseDynamicalTestDataset):
    """Transitions of Lorenz-63 with the true transition density."""

    def __init__(self, num_samples: int, signal_length: int):
        """Initializes the instance and generates the transitions.

        Args:
            num_samples: Number of sequences to draw transitions from.
            signal_length: Number of time steps of each sequence.
        """
        self._lorenz_attractor_model = lorenz_attractor_lib.LorenzSSM()
        super().__init__(num_samples=num_samples, signal_length=signal_length)

    def _generate_markovian_state_distributions(
        self,
    ) -> list[MarkovianStateDistribution]:
        distributions = []
        model = self._lorenz_attractor_model
        for _ in range(self._num_samples):
            states = model.generate_state_sequence(
                self._signal_length, sigma_e2_dB=-10
            )
            for t in range(1, self._signal_length - 4):
                distributions.append(
                    MarkovianStateDistribution(
                        prev_state=states[t - 1],
                        next_state=states[t],
                        mean_next_state=model.f_linearize(states[t - 1]),
                        cov_next_state=model.Ce,
                        state_five_steps=states[t + 4],
                    )
                )
        return distributions


@gin.configurable
class CharacterTrajectoriesTestDataset(BaseDynamicalTestDataset):
    """Transitions of the CharacterTrajectories test split.

    The true transition density is unknown, and every previous state is the
    whole history up to that step.
    """

    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        data_path: str = CHARACTERS_PATH,
    ):
        """Initializes the instance and loads the transitions.

        Args:
            num_samples: Number of transitions.
            signal_length: Unused; the trajectories keep their lengths.
            data_path: Path of mixoutALL_shifted.mat.
        """
        del signal_length
        self.data_path = data_path
        super().__init__(num_samples, signal_length=num_samples)

    def _generate_markovian_state_distributions(
        self,
    ) -> list[MarkovianStateDistribution]:
        trajectories = scipy.io.loadmat(self.data_path)['mixout'][0][2400:]
        distributions = []
        for trajectory in trajectories:
            trajectory = trajectory.T
            for i in range(len(trajectory) - 4):
                distributions.append(
                    MarkovianStateDistribution(
                        prev_state=torch.FloatTensor(
                            np.array(trajectory[: i + 1])
                        ),
                        next_state=torch.FloatTensor(
                            np.array(trajectory[i + 1])
                        ),
                        mean_next_state=None,
                        cov_next_state=None,
                    )
                )
                if len(distributions) >= self._num_samples:
                    break
            if len(distributions) >= self._num_samples:
                break
        if len(distributions) < self._num_samples:
            print(
                f'Warning: Only found {len(distributions)} state pairs,'
                f' requested {self._num_samples}'
            )
        return distributions[: self._num_samples]


@gin.configurable
class LinearSSMRotationalTestDataset(BaseDynamicalTestDataset):
    """Transitions of the linear Gaussian SSM with the true density."""

    def __init__(self, num_samples: int, signal_length: int):
        """Initializes the instance and generates the transitions.

        Args:
            num_samples: Number of sequences to draw transitions from.
            signal_length: Number of time steps of each sequence.
        """
        self._linear_model = linear_ssm_lib.LinearSSMRotational(d=4)
        super().__init__(num_samples, signal_length)

    def _generate_markovian_state_distributions(
        self,
    ) -> list[MarkovianStateDistribution]:
        distributions = []
        for _ in range(self._num_samples):
            states = self._linear_model.generate_state_sequence(
                self._signal_length
            )
            for t in range(1, self._signal_length - 4):
                distributions.append(
                    MarkovianStateDistribution(
                        prev_state=states[t - 1],
                        next_state=states[t],
                        mean_next_state=self._linear_model.A @ states[t - 1],
                        cov_next_state=self._linear_model.Ce,
                        state_five_steps=states[t + 4],
                    )
                )
        return distributions


# -----------------------------------------------------------------------------
# Dataloader for datasets.
# -----------------------------------------------------------------------------


@gin.configurable
def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Returns a dataloader that batches with the dataset's collate_fn.

    Args:
        dataset: Dataset with a collate_fn method.
        batch_size: Number of items per batch.
        num_workers: Number of worker processes.
        shuffle: Whether to shuffle the items every epoch.

    Returns:
        The dataloader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
