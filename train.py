"""Trains transition and proposal models with VSMC and differentiable
resampling.

Usage (from the repository root):

    python train.py --gin_path configs/lorenz63/dsr_10db.gin \
        --save_path runs/lorenz63/dsr_10db_0
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import shutil
import time

import gin
import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.nn
import tqdm

import dataset as dataset_lib
import model as model_lib  # noqa: F401, registers the models with gin.
import resampling as resampling_lib
import utils as utils_lib


def _draw_samples(means: torch.Tensor, covs: torch.Tensor) -> torch.Tensor:
    """Reparameterized samples from Gaussians with diagonal covariances.

    Args:
        means: Means (B, N, D).
        covs: Diagonal covariance matrices (B, N, D, D).

    Returns:
        Samples (B, N, D).
    """
    eps = torch.randn_like(means)
    return means + torch.einsum('bnij,bnj->bni', torch.sqrt(covs), eps)


def _calculate_log_likelihood_stable(
    x: torch.Tensor,
    means: torch.Tensor,
    covs: torch.Tensor,
    min_constant: float = -100,
) -> torch.Tensor:
    """Gaussian log-likelihood, clamped to [min_constant, 10000] for stability.

    Args:
        x: Points (B, N, D).
        means: Means (B, N, D).
        covs: Covariance matrices (B, N, D, D), or broadcastable to it.
        min_constant: Lower clamp of the log-likelihood.

    Returns:
        Log-likelihoods (B, N).
    """
    batch_size, num_samples, _ = x.shape
    log_probs = utils_lib.mvn_logprob(x, means, covs)
    log_probs = torch.clamp(log_probs, min=min_constant, max=10000)
    return log_probs.view(batch_size, num_samples)


@gin.configurable
class TrainingLoop:
    """Trains the models by maximizing the VSMC surrogate ELBO."""

    def __init__(
        self,
        *,
        proposal_model: type[torch.nn.Module] | None,
        transition_model: type[torch.nn.Module],
        dataset_cls: type[dataset_lib.BaseDynamicalDataset],
        test_dataset_cls: type[dataset_lib.BaseDynamicalTestDataset],
        save_path: str,
        num_particles: int,
        batch_size: int,
        state_dim: int,
        num_epochs: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        resampling_method: (
            resampling_lib.ResamplingMethod | str
        ) = resampling_lib.ResamplingMethod.OT,
        real_dataset: bool = False,
        particle_init_mean: float = 0.0,
        smnr_db: float = 0.0,
        systematic_tau: float = 0.5,
        tau_start: float | None = None,
        tau_end: float | None = None,
        tau_annealing_epochs: int = 0,
        lr_start: float | None = None,
        lr_end: float | None = None,
        lr_warmdown_epochs: int | None = None,
        track_gradients: bool = True,
        morton_bits: int = 20,
        ot_eps: float = 0.5,
    ):
        """Initializes the instance.

        Args:
            proposal_model: Proposal model class, q(x_t | x_(t-1), y_t); if
                None, the transition model is used as a bootstrap proposal.
            transition_model: Transition model class, p(x_t | x_(t-1)).
            dataset_cls: Training dataset class; also builds the validation
                set.
            test_dataset_cls: Test dataset class of one-step transitions.
            save_path: Directory of the log, checkpoints and plots.
            num_particles: Number of particles during training.
            batch_size: Number of sequences per batch.
            state_dim: Dimension of the state.
            num_epochs: Number of training epochs.
            learning_rate: Adam learning rate, unless lr_start is set.
            weight_decay: Adam weight decay.
            resampling_method: Resampling method, or its string value.
            real_dataset: Whether the data is CharacterTrajectories; its
                states are the last 3 dimensions and it has no test set.
            particle_init_mean: Mean of the initial particles.
            smnr_db: SMNR of the data in dB; sets the log-likelihood clamp.
            systematic_tau: DSR temperature when tau is not annealed.
            tau_start: First DSR temperature of the annealing.
            tau_end: Last DSR temperature of the annealing.
            tau_annealing_epochs: Epochs over which tau is annealed linearly;
                0 disables annealing.
            lr_start: First learning rate of the linear schedule.
            lr_end: Last learning rate of the linear schedule.
            lr_warmdown_epochs: Epochs over which the learning rate goes from
                lr_start to lr_end; num_epochs if None.
            track_gradients: Whether to log gradient norms of the proposal.
            morton_bits: Grid resolution per dimension of Morton sorting.
            ot_eps: Entropic regularization of EROT.
        """
        self._save_path = save_path
        self._log_path = os.path.join(save_path, 'train.log')
        logging.basicConfig(filename=self._log_path, level=logging.INFO)
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logging.info(f'Device: {self._device}')
        self._num_particles = num_particles
        self._batch_size = batch_size
        self._state_dim = state_dim
        self._proposal_model = (
            proposal_model()
            if proposal_model is not None
            else transition_model()
        )
        self._transition_model = transition_model()
        self._proposal_model.to(device=self._device)
        self._transition_model.to(device=self._device)

        self._train_dataset = dataset_cls()
        self._train_dataloader = dataset_lib.get_dataloader(
            self._train_dataset, self._batch_size
        )
        if real_dataset:
            self._val_dataset = dataset_cls(
                num_samples=10, signal_length=178, valid='valid'
            )
        else:
            self._val_dataset = dataset_cls(num_samples=10, signal_length=1000)
        self._valid_dataloader = dataset_lib.get_dataloader(
            self._val_dataset, self._batch_size
        )
        self._test_dataset = test_dataset_cls(
            num_samples=178 if real_dataset else 10, signal_length=1000
        )
        self._test_dataloader = dataset_lib.get_dataloader(
            self._test_dataset, 1 if real_dataset else self._batch_size
        )

        self._lr_start = learning_rate if lr_start is None else lr_start
        self._lr_end = learning_rate if lr_end is None else lr_end
        self._lr_warmdown_epochs = (
            num_epochs if lr_warmdown_epochs is None else lr_warmdown_epochs
        )
        self._optimizer = torch.optim.Adam(
            self._parameters(),
            eps=1e-8,
            lr=self._lr_start,
            weight_decay=weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lr_lambda=self._lr_factor
        )
        self._epoch = 0
        self._num_epochs = num_epochs
        self._resampling_method = resampling_lib.ResamplingMethod(
            resampling_method
        )

        self._state_to_observation = self._train_dataset.state_to_observation
        self._nmse_loss_accumulator = utils_lib.NMSELossAccumulator()
        self._early_stopping = utils_lib.EarlyStopping(
            patience=1000, min_delta=1e-4, mode='min'
        )
        self._track_gradients = track_gradients
        self._gradient_tracker = (
            utils_lib.GradientTracker() if track_gradients else None
        )
        self._real_dataset = real_dataset
        self._particle_init_mean = particle_init_mean
        self._min_constant = -50 if smnr_db == 10.0 else -100
        self._systematic_tau = systematic_tau
        self._tau_start = systematic_tau if tau_start is None else tau_start
        self._tau_end = systematic_tau if tau_end is None else tau_end
        self._tau_annealing_epochs = tau_annealing_epochs
        self._morton_bits = morton_bits
        self._ot_eps = ot_eps
        self._val_kl_history: list[float] = []

    def _parameters(self):
        return itertools.chain(
            self._proposal_model.parameters(),
            self._transition_model.parameters(),
        )

    def _lr_factor(self, epoch: int) -> float:
        """Learning rate of the linear schedule, relative to lr_start."""
        t = min(epoch, self._lr_warmdown_epochs)
        if self._lr_warmdown_epochs <= 0:
            return 1.0
        alpha = t / self._lr_warmdown_epochs
        return (
            self._lr_start + alpha * (self._lr_end - self._lr_start)
        ) / self._lr_start

    def _get_current_tau(self) -> float:
        """Returns the current DSR temperature, linearly annealed if set."""
        if self._tau_annealing_epochs <= 0:
            return self._systematic_tau
        t = min(self._epoch, self._tau_annealing_epochs)
        return (
            self._tau_start
            + (self._tau_end - self._tau_start) * t / self._tau_annealing_epochs
        )

    def _state_estimate(
        self, particles: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Weighted particle mean on the CPU, restricted to the state dims."""
        estimate = torch.einsum('bni,bn->bi', particles, weights).detach().cpu()
        return estimate[..., -3:] if self._real_dataset else estimate

    def _train_once(
        self,
        observations: torch.Tensor,
        Cws: torch.Tensor,
        num_particles: int,
        hard_systematic_resampling: bool = False,
        posterior_mean: torch.Tensor | None = None,
        posterior_cov: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float | None]:
        """Runs the differentiable particle filter over one batch.

        Args:
            observations: Noisy observations (B, T, D_y).
            Cws: Observation noise covariances, broadcastable to
                (B, N, D_y, D_y).
            num_particles: Number of particles N.
            hard_systematic_resampling: Resample with hard systematic
                resampling instead of the configured method.
            posterior_mean: Kalman posterior means (B, T, D), to log the KL
                of the particles to the posterior; None skips it.
            posterior_cov: Kalman posterior covariances (B, T, D, D).

        Returns:
            Negative surrogate ELBO, state estimates (B, T, D) on the CPU,
            mean ESS before resampling, and mean KL to the Kalman posterior
            (None if not computed).
        """
        batch_size, seq_len, _ = observations.shape
        device = observations.device
        particles = (
            torch.randn(
                batch_size, num_particles, self._state_dim, device=device
            )
            + self._particle_init_mean
        )
        weights = (
            torch.ones(batch_size, num_particles, device=device) / num_particles
        )
        loss = torch.Tensor([0.0]).to(device)
        ess_sum = 0.0
        kl_sum = 0.0
        kl_steps = 0
        state_dim = 3 if self._real_dataset else self._state_dim
        predicted_states = torch.zeros((batch_size, seq_len, state_dim))

        for t in range(1, seq_len):
            current_observations = observations[:, t, :]
            ess_sum += (
                resampling_lib.effective_sample_size(weights).mean().item()
            )
            weights, particles = resampling_lib.conditional_resample(
                weights,
                particles,
                resampling_method=self._resampling_method,
                hard_systematic_resampling=hard_systematic_resampling,
                sys_tau=self._get_current_tau(),
                morton_bits=self._morton_bits,
                ot_eps=self._ot_eps,
            )
            if posterior_mean is not None and posterior_cov is not None:
                with torch.no_grad():
                    kl = utils_lib.empirical_gaussian_kl(
                        particles,
                        posterior_mean[:, t, :].to(device),
                        posterior_cov[:, t, :, :].to(device),
                    )
                if kl is not None:
                    kl_sum += kl.item()
                    kl_steps += 1
            predicted_states[:, t - 1, :] = self._state_estimate(
                particles, weights
            )

            proposal_means, proposal_covs = self._proposal_model(
                particles, current_observations
            )
            proposal_particles = _draw_samples(proposal_means, proposal_covs)
            transition_means, transition_covs = self._transition_model(
                particles
            )
            log_likelihood_proposal = _calculate_log_likelihood_stable(
                proposal_particles,
                proposal_means,
                proposal_covs,
                min_constant=self._min_constant,
            )
            log_likelihood_transition = _calculate_log_likelihood_stable(
                proposal_particles,
                transition_means,
                transition_covs,
                min_constant=self._min_constant,
            )
            log_likelihood_observations = _calculate_log_likelihood_stable(
                current_observations.unsqueeze(1).repeat(1, num_particles, 1),
                self._state_to_observation(proposal_particles),
                Cws,
                min_constant=self._min_constant,
            )
            log_weights = torch.log(weights) + (
                log_likelihood_observations
                + (log_likelihood_transition - log_likelihood_proposal)
            )
            weights = torch.exp(
                log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
            )
            log_marginal_likelihood = torch.logsumexp(
                log_weights, dim=1
            ) - torch.log(torch.tensor(num_particles))
            loss -= torch.mean(log_marginal_likelihood)
            particles = proposal_particles

        predicted_states[:, -1, :] = self._state_estimate(particles, weights)
        mean_ess = ess_sum / (seq_len - 1) if seq_len > 1 else 0.0
        mean_kl = kl_sum / kl_steps if kl_steps > 0 else None
        return loss, predicted_states, mean_ess, mean_kl

    def _has_bad_gradients(self) -> bool:
        return any(
            param.grad is not None
            and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            for param in self._parameters()
        )

    def train(self) -> None:
        """Trains for num_epochs, validating and testing after every epoch."""
        start_time = time.time()
        while self._epoch < self._num_epochs:
            square_error = 0.0
            num_examples = 0
            loss_epoch = 0.0
            ess_epoch = 0.0
            kl_epoch = 0.0
            kl_batches = 0
            self._proposal_model.train()
            self._transition_model.train()
            for batch in tqdm.tqdm(
                self._train_dataloader,
                total=len(self._train_dataloader),
                ncols=0,
                desc='Train',
            ):
                self._optimizer.zero_grad()
                states, _, noisy_observations, Cws, *kalman_data = batch
                posterior_mean = (
                    kalman_data[0] if len(kalman_data) > 0 else None
                )
                posterior_cov = kalman_data[1] if len(kalman_data) > 1 else None
                noisy_observations = noisy_observations.to(device=self._device)
                Cws = Cws.to(device=self._device)
                loss, predicted_states, batch_ess, batch_kl = self._train_once(
                    noisy_observations,
                    Cws,
                    num_particles=self._num_particles,
                    posterior_mean=posterior_mean,
                    posterior_cov=posterior_cov,
                )
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f'NaN/Inf loss detected: {loss}')
                    print('Skipping this batch')
                    continue
                if loss.item() > 1e5:
                    print(f'Extremely large loss: {loss.item()}')
                    print('Skipping this batch')
                    continue

                loss_epoch += loss.item()
                ess_epoch += batch_ess
                if batch_kl is not None:
                    kl_epoch += batch_kl
                    kl_batches += 1
                loss.backward()
                if self._has_bad_gradients():
                    print('Skipping batch due to bad gradients')
                    self._optimizer.zero_grad()
                    continue
                if self._track_gradients:
                    self._gradient_tracker.update(self._proposal_model)
                torch.nn.utils.clip_grad_norm_(self._parameters(), max_norm=5.0)
                self._optimizer.step()
                square_error += torch.sum(
                    10
                    * torch.log10(
                        torch.mean(
                            (states - predicted_states) ** 2, axis=(1, 2)
                        )
                        / torch.mean(states**2, axis=(1, 2))
                    )
                ).item()
                num_examples += noisy_observations.shape[0]
            if num_examples == 0:
                logging.info(
                    f'Training diverged at epoch {self._epoch + 1}: every'
                    ' batch had a NaN/Inf loss or gradient.'
                )
                break
            if self._track_gradients:
                self._gradient_tracker.end_epoch(self._epoch)
            self._epoch += 1
            self._scheduler.step()

            num_batches = len(self._train_dataloader)
            logging.info(
                f'Epoch={self._epoch}, Average loss train='
                f'{loss_epoch / num_batches}'
            )
            logging.info(f'Train NMSE (states)={square_error / num_examples}.')
            logging.info(
                f'Train mean ESS={ess_epoch / num_batches:.2f} /'
                f' {self._num_particles}'
            )
            if kl_batches > 0:
                logging.info(
                    'Train mean KL(particles||Kalman) after resample='
                    f'{kl_epoch / kl_batches:.4f}'
                )
            if (
                self._track_gradients
                and self._gradient_tracker.epoch_grad_norms
            ):
                mean_grad, std_grad = self._gradient_tracker.epoch_grad_norms[
                    -1
                ]
                logging.info(
                    f'Mean gradient norm={mean_grad:.6f} (std={std_grad:.6f})'
                )
            avg_valid_loss, _ = self.evaluate()
            if not self._real_dataset:
                self.test()
            if self._early_stopping(avg_valid_loss, self._epoch):
                logging.info(
                    f'Early stopping at epoch {self._epoch}; validation loss'
                    ' did not improve for'
                    f' {self._early_stopping.patience} epochs.'
                )
                break
            if self._epoch % 100 == 0:
                self.save_model(self._epoch)
                logging.info(f'Model saved at epoch {self._epoch}')
                self._plot_epoch(predicted_states, states, noisy_observations)
        logging.info(
            f'Training completed in {time.time() - start_time} seconds.'
        )
        if self._val_kl_history:
            plt.figure()
            plt.plot(self._val_kl_history)
            plt.xlabel('Epoch')
            plt.ylabel('KL(particles || Kalman posterior)')
            plt.title('Particle posterior quality (lower = better)')
            plt.tight_layout()
            plt.savefig(os.path.join(self._save_path, 'val_posterior_kl.png'))
            plt.close()

    def _plot_epoch(
        self,
        predicted_states: torch.Tensor,
        states: torch.Tensor,
        noisy_observations: torch.Tensor,
    ) -> None:
        """Saves the gradient norms and the first sequence of the last batch."""
        if self._track_gradients:
            self._gradient_tracker.plot(
                os.path.join(
                    self._save_path, f'gradients_epoch_{self._epoch}.png'
                )
            )
        num_plot_dims = min(
            predicted_states.shape[-1],
            states.shape[-1],
            noisy_observations.shape[-1],
        )
        for i in range(num_plot_dims):
            plt.plot(predicted_states[0, :, i].numpy(), label='Predicted')
            plt.plot(states[0, :, i].cpu().numpy(), label='States')
            plt.plot(
                noisy_observations[0, :, i].cpu().numpy(),
                label='Noisy observations',
            )
            plt.legend()
            plt.savefig(
                os.path.join(
                    self._save_path, f'Epoch={self._epoch}_dim={i}.png'
                )
            )
            plt.close()

    def evaluate(self) -> tuple[float, float]:
        """Filters the validation set with N = 1000 and hard SR.

        Returns:
            Mean negative surrogate ELBO per batch and the NMSE in dB.
        """
        self._proposal_model.eval()
        self._transition_model.eval()
        self._nmse_loss_accumulator.reset()
        total_loss = 0
        kl_epoch = 0.0
        kl_batches = 0
        for batch in self._valid_dataloader:
            states, _, noisy_observations, Cws, *kalman_data = batch
            posterior_mean = kalman_data[0] if len(kalman_data) > 0 else None
            posterior_cov = kalman_data[1] if len(kalman_data) > 1 else None
            noisy_observations = noisy_observations.to(device=self._device)
            Cws = Cws.to(device=self._device)
            with torch.no_grad():
                loss, predicted_states, _, batch_kl = self._train_once(
                    noisy_observations,
                    Cws,
                    num_particles=1000,
                    hard_systematic_resampling=True,
                    posterior_mean=posterior_mean,
                    posterior_cov=posterior_cov,
                )
            self._nmse_loss_accumulator.update(predicted_states, states)
            total_loss += loss.item()
            if batch_kl is not None:
                kl_epoch += batch_kl
                kl_batches += 1
        avg_loss = total_loss / len(self._valid_dataloader)
        nmse = self._nmse_loss_accumulator.get_total_nmse_loss()
        logging.info(f'Validation NMSE: {nmse}')
        logging.info(f'Validation ELBO: {avg_loss}')
        if kl_batches > 0:
            mean_kl = kl_epoch / kl_batches
            logging.info(
                'Validation mean KL(particles||Kalman) after resample='
                f'{mean_kl:.4f}'
            )
            self._val_kl_history.append(mean_kl)
        return avg_loss, nmse

    def test(self) -> tuple[float, float | None, float | None]:
        """Evaluates the transition model on one-step test transitions.

        Logs the one- and five-step MSE, the log-likelihood, and, if the true
        transition density is known, its KL divergence and Wasserstein
        distance to the learned one.

        Returns:
            Mean log-likelihood, mean KL divergence (None if the true density
            is unknown) and ||A_hat - A||_F (None if not linear).
        """
        total_log_likelihood = 0.0
        total_kl_divergence = 0.0
        total_wasserstein = 0.0
        mse = 0.0
        mse_t5 = 0.0
        total_samples = 0
        self._transition_model.eval()
        for batch in self._test_dataloader:
            (
                prev_state,
                next_state,
                mean_next_state,
                cov_next_state,
                state_five_steps,
            ) = batch
            prev_state = prev_state.to(device=self._device)
            with torch.no_grad():
                pred_mean, pred_cov = self._transition_model(prev_state)
                pred_mean_five_steps = self._transition_model.forward_k_steps(
                    prev_state, k=5
                ).cpu()
            pred_mean = pred_mean.cpu()
            pred_cov = pred_cov.cpu()
            pred_dist = torch.distributions.MultivariateNormal(
                pred_mean, pred_cov
            )
            log_likelihood = pred_dist.log_prob(next_state)

            if isinstance(mean_next_state, torch.Tensor):
                true_dist = torch.distributions.MultivariateNormal(
                    mean_next_state, cov_next_state
                )
                kl_div = torch.distributions.kl_divergence(pred_dist, true_dist)
                wasserstein = utils_lib.compute_wasserstein_distance_gaussian(
                    pred_mean, pred_cov, mean_next_state, cov_next_state
                )
                total_kl_divergence += kl_div.sum().item()
                total_wasserstein += wasserstein.sum().item()
            if isinstance(state_five_steps, torch.Tensor):
                mse_t5 += (
                    torch.sum(
                        (pred_mean_five_steps - state_five_steps) ** 2, dim=-1
                    )
                    .sum()
                    .item()
                )
            total_log_likelihood += log_likelihood.sum().item()
            mse += torch.sum((pred_mean - next_state) ** 2, dim=-1).sum().item()
            total_samples += prev_state.shape[0]

        a_distance = self._transition_matrix_distance()
        logging.info('----------------------------')
        logging.info(f'Test MSE: {mse / total_samples}')
        logging.info(f'Test 5-step MSE: {mse_t5 / total_samples}')
        logging.info(
            f'Test log likelihood: {total_log_likelihood / total_samples}'
        )
        if a_distance is not None:
            logging.info(f'Test A matrix distance (Frobenius): {a_distance}')
        if isinstance(mean_next_state, torch.Tensor):
            logging.info(
                f'Test KL divergence: {total_kl_divergence / total_samples}'
            )
            logging.info(
                'Test Wasserstein distance:'
                f' {total_wasserstein / total_samples}'
            )
            mean_kl_divergence = total_kl_divergence / total_samples
        else:
            logging.info(
                'KL divergence and Wasserstein distance not computed (no target'
                ' distribution)'
            )
            mean_kl_divergence = None
        logging.info('----------------------------')
        return (
            total_log_likelihood / total_samples,
            mean_kl_divergence,
            a_distance,
        )

    def _transition_matrix_distance(self) -> float | None:
        """||A_hat - A||_F for a linear transition model, else None."""
        model = self._transition_model
        true_model = getattr(self._test_dataset, '_linear_model', None)
        if not (
            isinstance(getattr(model, 'A', None), torch.nn.Linear)
            and hasattr(true_model, 'A')
        ):
            return None
        true_A = torch.tensor(
            true_model.A,
            dtype=model.A.weight.dtype,
            device=model.A.weight.device,
        )
        return torch.norm(model.A.weight - true_A, p='fro').item()

    def save_model(self, epoch: int | str | None = None) -> None:
        """Saves the transition and proposal model state dicts.

        Args:
            epoch: Suffix of the file names; the current epoch if None.
        """
        if epoch is None:
            epoch = self._epoch
        torch.save(
            self._transition_model.state_dict(),
            os.path.join(self._save_path, f'transition_model_epoch_{epoch}.pt'),
        )
        torch.save(
            self._proposal_model.state_dict(),
            os.path.join(self._save_path, f'proposal_model_epoch_{epoch}.pt'),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--gin_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    gin.parse_config_file(args.gin_path)
    gin.finalize()
    shutil.copy(args.gin_path, os.path.join(args.save_path, 'gin_config.gin'))
    training_loop = TrainingLoop(save_path=args.save_path)
    training_loop.train()
    training_loop.save_model(epoch='last')


if __name__ == '__main__':
    main()
