from typing import Literal, cast

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class Diffusion(nn.Module):
    """
    Diffusion model for time series data.

    The model is initialized with a denoising network, a noise sampler and a way to compute Mahalanobis distances.
    Ususally, the noise sampler and the Mahalanobis distances will be based the same Gaussian Process.

    The noise schedule and variance type are configured through diffusers DDPMScheduler.
    """

    def __init__(
        self,
        network,
        diffusion_time_steps,
        noise_sampler,
        mal_dist_computer,
        schedule: Literal[
            "linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"
        ] = "linear",
        variance_type: Literal[
            "fixed_small",
            "fixed_small_log",
            "fixed_large",
            "fixed_large_log",
            "learned",
            "learned_range",
        ] = "fixed_small",
        start_beta=1e-4,
        end_beta=0.02,
    ):
        super().__init__()
        assert network.signal_length == noise_sampler.signal_length
        assert network.signal_length == mal_dist_computer.signal_length

        self.device = "cpu"  # default
        self.network = network
        self.noise_sampler = noise_sampler
        self.mal_dist_computer = mal_dist_computer
        self.diffusion_time_steps = diffusion_time_steps

        self.scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_time_steps,
            beta_start=start_beta,
            beta_end=end_beta,
            beta_schedule=schedule,
            variance_type=variance_type,
            prediction_type="epsilon",
            clip_sample=False,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.noise_sampler.to(*args, **kwargs)
        self.mal_dist_computer.to(*args, **kwargs)
        return self

    def train_batch(self, batch, cond=None, mask=None):
        self.train()
        batch_size = batch.shape[0]
        time_index = torch.randint(
            0, self.diffusion_time_steps, (batch_size,), device=batch.device
        )
        time_index = time_index.to(batch.device, dtype=torch.long)
        scheduler_timesteps = cast(torch.IntTensor, time_index)
        noise = self.noise_sampler.sample(
            sample_shape=(
                batch_size,
                self.network.signal_channel,
                self.network.signal_length,
            )
        )
        assert noise.shape == batch.shape
        noisy_sig = self.scheduler.add_noise(batch, noise, scheduler_timesteps)
        res = self.network.forward(noisy_sig, time_index, cond=cond)
        diff = noise - res
        malhabonis = self.mal_dist_computer.sqrt_mal(diff)
        if mask is not None:
            malhabonis = malhabonis * mask
        return torch.einsum("icl,icl->i", malhabonis, malhabonis)

    def sample(
        self,
        num_samples,
        cond=None,
        sample_length=None,
        sampler=None,
    ):
        if sampler is None:
            sampler = self.noise_sampler
        if sample_length is None:
            sample_length = self.noise_sampler.signal_length

        if cond is not None:
            cond_batch, cond_channel, cond_length = cond.shape
            assert cond_batch == 1 or cond_batch == num_samples
            assert cond_length == sample_length
            if cond_batch == 1:
                cond = cond.repeat(num_samples, 1, 1)

        self.eval()
        self.scheduler.set_timesteps(self.diffusion_time_steps, device=self.device)
        with torch.no_grad():
            state = sampler.sample(
                sample_shape=(
                    num_samples,
                    self.network.signal_channel,
                    sample_length,
                )
            )

            for timestep in self.scheduler.timesteps:
                time_value = int(timestep.item())
                time_vector = torch.full(
                    (num_samples,), time_value, device=self.device, dtype=torch.long
                )

                predicted_noise = self.network.forward(
                    state,
                    time_vector,
                    cond=cond,
                )
                state = self.scheduler.step(
                    model_output=predicted_noise,
                    timestep=time_value,
                    sample=state,
                    return_dict=False,
                )[0]

            return state

    # Mask: 1 -> sample is present, 0 sample is not present
    def impute(self, signal, mask, cond=None):
        signal_batch, signal_channel, signal_length = signal.shape
        assert signal_channel == self.network.signal_channel
        assert signal.shape == mask.shape
        mask = mask.to(device=signal.device, dtype=signal.dtype)

        if cond is not None:
            cond_batch, _cond_channel, cond_length = cond.shape
            assert cond_batch == signal_batch
            assert signal_length == cond_length

        self.eval()
        self.scheduler.set_timesteps(self.diffusion_time_steps, device=self.device)
        with torch.no_grad():
            state = self.noise_sampler.sample(
                sample_shape=(
                    signal_batch,
                    self.network.signal_channel,
                    self.network.signal_length,
                )
            )
            for timestep in self.scheduler.timesteps:
                time_value = int(timestep.item())
                time_vector = torch.full(
                    (signal_batch,), time_value, device=self.device, dtype=torch.long
                )

                known_noise = self.noise_sampler.sample(
                    sample_shape=(
                        signal_batch,
                        self.network.signal_channel,
                        self.network.signal_length,
                    )
                )
                timestep_vector = torch.full(
                    (signal_batch,), time_value, device=signal.device, dtype=torch.long
                )
                scheduler_timesteps = cast(torch.IntTensor, timestep_vector)
                known_state = self.scheduler.add_noise(
                    signal, known_noise, scheduler_timesteps
                )
                state = mask * known_state + (1.0 - mask) * state

                predicted_noise = self.network.forward(
                    state,
                    time_vector,
                    cond=cond,
                )
                state = self.scheduler.step(
                    model_output=predicted_noise,
                    timestep=time_value,
                    sample=state,
                    return_dict=False,
                )[0]
            return state


class Trainer:
    """
    Training class for a diffusion model.

    Given a data loader and optimizer, it trains the model for one epoch.
    """

    def __init__(self, model, data_loader, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.data_loader = data_loader

    def train_epoch(self):
        batchwise_losses = []
        for batch in self.data_loader:
            sig_batch = batch["signal"]
            batch_size = sig_batch.shape[0]
            sig_batch = sig_batch.to(self.model.device)
            # If a Dataloader provides these, use them. If not, don't.
            try:
                cond_batch = batch["cond"]
                cond_batch = cond_batch.to(self.model.device)
            except KeyError:
                cond_batch = None
            try:
                mask_batch = batch["mask"]
                mask_batch = mask_batch.to(self.model.device)
            except KeyError:
                mask_batch = None

            batch_loss = self.model.train_batch(
                sig_batch, cond=cond_batch, mask=mask_batch
            )
            batch_loss = torch.mean(batch_loss)

            batchwise_losses.append((batch_size, batch_loss.item()))

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        return batchwise_losses
