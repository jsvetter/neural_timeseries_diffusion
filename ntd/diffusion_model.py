from typing import Any, Literal, cast

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
        network: nn.Module,
        diffusion_time_steps: int,
        noise_sampler: Any,  # TODO
        mal_dist_computer: Any,  # TODO
        schedule: Literal[
            "linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"
        ] = "linear",
        variance_type: Literal[
            "fixed_small",
            "fixed_small_log",
            "fixed_large",
            "fixed_large_log",
            "learned",  # TODO remove, will not be supported by tailored DDPM scheduler
            "learned_range",  # TODO see above
        ] = "fixed_small",
        start_beta: float = 1e-4,
        end_beta: float = 0.02,
    ) -> None:
        super().__init__()
        if network.signal_length != noise_sampler.signal_length:
            raise ValueError(
                "network.signal_length must match noise_sampler.signal_length"
            )
        if network.signal_length != mal_dist_computer.signal_length:
            raise ValueError(
                "network.signal_length must match mal_dist_computer.signal_length"
            )

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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to(self, *args, **kwargs) -> "Diffusion":
        super().to(*args, **kwargs)
        self.noise_sampler.to(*args, **kwargs)
        self.mal_dist_computer.to(*args, **kwargs)
        return self

    def train_batch(
        self,
        batch: torch.Tensor,
        cond: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = batch.shape[0]
        time_index = torch.randint(
            0, self.diffusion_time_steps, (batch_size,), device=batch.device
        )
        time_index = time_index.to(batch.device, dtype=torch.long)
        noise = self.noise_sampler.sample(
            sample_shape=(
                batch_size,
                self.network.signal_channel,
                self.network.signal_length,
            )
        )
        if noise.shape != batch.shape:
            raise ValueError(
                f"noise shape {noise.shape} must match batch shape {batch.shape}"
            )
        scheduler_timesteps = cast(torch.IntTensor, time_index)
        noisy_sig = self.scheduler.add_noise(batch, noise, scheduler_timesteps)
        res = self.network(noisy_sig, time_index, cond=cond)
        diff = noise - res
        mahalanobis = self.mal_dist_computer.sqrt_mal(diff)
        if mask is not None:
            mahalanobis = mahalanobis * mask
        return torch.einsum("icl,icl->i", mahalanobis, mahalanobis)

    def sample(
        self,
        num_samples: int,
        cond: torch.Tensor | None = None,
        sample_length: int | None = None,
    ) -> torch.Tensor:
        if sample_length is None:
            sample_length = self.noise_sampler.signal_length

        if cond is not None:
            cond_batch, _cond_channel, cond_length = cond.shape
            if cond_batch != 1 and cond_batch != num_samples:
                raise ValueError("cond batch size must be 1 or num_samples")
            if cond_length != sample_length:
                raise ValueError("cond length must match sample_length")
            if cond_batch == 1:
                cond = cond.repeat(num_samples, 1, 1)

        self.scheduler.set_timesteps(self.diffusion_time_steps, device=self.device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            state = self.noise_sampler.sample(
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

                predicted_noise = self.network(
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

        self.train(was_training)
        return state

    # Mask: 1 -> sample is present, 0 sample is not present
    def impute(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        signal_batch, signal_channel, signal_length = signal.shape
        if signal_channel != self.network.signal_channel:
            raise ValueError("signal channel must match network.signal_channel")
        if signal.shape != mask.shape:
            raise ValueError("signal shape must match mask shape")
        mask = mask.to(device=signal.device, dtype=signal.dtype)

        if cond is not None:
            cond_batch, _cond_channel, cond_length = cond.shape
            if cond_batch != signal_batch:
                raise ValueError("cond batch size must match signal batch size")
            if signal_length != cond_length:
                raise ValueError("cond length must match signal length")

        self.scheduler.set_timesteps(self.diffusion_time_steps, device=self.device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            state = self.noise_sampler.sample(
                sample_shape=(
                    signal_batch,
                    self.network.signal_channel,
                    signal_length,
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
                        signal_length,
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

                predicted_noise = self.network(
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
        self.train(was_training)
        return state


class Trainer:
    """
    Training class for a diffusion model.

    Given a data loader and optimizer, it trains the model for one epoch.
    """

    def __init__(self, model: Diffusion, data_loader: Any, optimizer: Any, device: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.data_loader = data_loader

    def train_epoch(self) -> list[tuple[int, float]]:
        batchwise_losses = []
        self.model.train()
        for batch in self.data_loader:
            sig_batch = batch["signal"]
            batch_size = sig_batch.shape[0]
            sig_batch = sig_batch.to(self.model.device)
            # If a Dataloader provides these, use them. If not, don't.
            cond_batch = batch.get("cond")
            if cond_batch is not None:
                cond_batch = cond_batch.to(self.model.device)
            mask_batch = batch.get("mask")
            if mask_batch is not None:
                mask_batch = mask_batch.to(self.model.device)

            batch_loss = self.model.train_batch(
                sig_batch, cond=cond_batch, mask=mask_batch
            )
            batch_loss = torch.mean(batch_loss)

            batchwise_losses.append((batch_size, batch_loss.item()))

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        return batchwise_losses
