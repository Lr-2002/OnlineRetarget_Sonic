"""Compact causal Diffusion Policy UNet wrapper."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class TinyDDPMScheduler(nn.Module):
    """Small local DDPM scheduler to avoid pulling a full robotics stack."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 32,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if beta_start <= 0.0 or beta_end <= 0.0 or beta_end <= beta_start:
            raise ValueError("expected 0 < beta_start < beta_end")
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.num_train_timesteps = num_train_timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].to(device=clean.device, dtype=clean.dtype)
        while alpha_bar.ndim < clean.ndim:
            alpha_bar = alpha_bar.unsqueeze(-1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def step(
        self,
        pred_noise: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        index = int(timestep.item())
        beta = self.betas[index].to(device=sample.device, dtype=sample.dtype)
        alpha = self.alphas[index].to(device=sample.device, dtype=sample.dtype)
        alpha_bar = self.alpha_bars[index].to(device=sample.device, dtype=sample.dtype)
        return (sample - beta * pred_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()


class DiffusionPolicyUNetSmall(nn.Module):
    """Strict-causal DDPM denoiser for future G1 action windows.

    The model conditions on past ``reference_history_tokens`` plus current robot
    state and previous action only. Future G1 joints enter through
    ``target_action`` in the loss and are never accepted by ``forward``.
    """

    def __init__(
        self,
        *,
        action_dim: int = 29,
        reference_body_token_dim: int = 15,
        reference_history_frames: int = 10,
        reference_body_count: int = 30,
        robot_state_dim: int = 94,
        down_dims: Sequence[int] = (128, 256),
        condition_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        kernel_size: int = 5,
        groups: int = 8,
        cond_predict_scale: bool = True,
        diffusion_steps: int = 32,
        inference_steps: int | None = None,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        max_action_horizon: int = 64,
        output_mode: str = "residual_prev_action",
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if reference_body_token_dim <= 0:
            raise ValueError("reference_body_token_dim must be positive")
        if reference_history_frames not in {5, 10}:
            raise ValueError(
                "reference_history_frames must be 5 or 10 for the strict causal preset"
            )
        if reference_body_count <= 0:
            raise ValueError("reference_body_count must be positive")
        if robot_state_dim < 0:
            raise ValueError("robot_state_dim must be non-negative")
        if output_mode not in {"absolute", "residual_prev_action"}:
            raise ValueError("output_mode must be absolute or residual_prev_action")
        dims = tuple(int(value) for value in down_dims)
        if not dims or any(value <= 0 for value in dims):
            raise ValueError("down_dims must contain positive channel widths")
        self.action_dim = action_dim
        self.reference_body_token_dim = reference_body_token_dim
        self.reference_history_frames = reference_history_frames
        self.reference_body_count = reference_body_count
        self.robot_state_dim = robot_state_dim
        self.condition_dim = condition_dim
        self.diffusion_steps = diffusion_steps
        self.inference_steps = inference_steps or diffusion_steps
        self.max_action_horizon = max_action_horizon
        self.output_mode = output_mode

        self.scheduler = TinyDDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, condition_dim),
            nn.Mish(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.reference_encoder = nn.Sequential(
            nn.Linear(reference_body_token_dim, condition_dim),
            nn.Mish(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.robot_encoder = (
            nn.Sequential(
                nn.Linear(robot_state_dim, condition_dim),
                nn.Mish(),
                nn.Linear(condition_dim, condition_dim),
            )
            if robot_state_dim > 0
            else None
        )
        self.prev_action_encoder = nn.Sequential(
            nn.Linear(action_dim, condition_dim),
            nn.Mish(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.position = nn.Parameter(torch.zeros(1, dims[0], max_action_horizon))
        self.input_proj = nn.Conv1d(action_dim, dims[0], kernel_size=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_dim = dims[0]
        for next_dim in dims:
            self.down_blocks.append(
                ConditionalResidualBlock1d(
                    current_dim,
                    next_dim,
                    condition_dim=condition_dim,
                    kernel_size=kernel_size,
                    groups=groups,
                    cond_predict_scale=cond_predict_scale,
                )
            )
            self.downsamples.append(
                Downsample1d(next_dim) if next_dim != dims[-1] else nn.Identity()
            )
            current_dim = next_dim

        self.mid_block1 = ConditionalResidualBlock1d(
            dims[-1],
            dims[-1],
            condition_dim=condition_dim,
            kernel_size=kernel_size,
            groups=groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.mid_block2 = ConditionalResidualBlock1d(
            dims[-1],
            dims[-1],
            condition_dim=condition_dim,
            kernel_size=kernel_size,
            groups=groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        reversed_dims = tuple(reversed(dims))
        current_dim = reversed_dims[0]
        for skip_dim in reversed_dims[1:]:
            self.upsamples.append(Upsample1d(current_dim))
            self.up_blocks.append(
                ConditionalResidualBlock1d(
                    current_dim + skip_dim,
                    skip_dim,
                    condition_dim=condition_dim,
                    kernel_size=kernel_size,
                    groups=groups,
                    cond_predict_scale=cond_predict_scale,
                )
            )
            current_dim = skip_dim
        self.final_block = ConditionalResidualBlock1d(
            current_dim,
            current_dim,
            condition_dim=condition_dim,
            kernel_size=kernel_size,
            groups=groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.final_proj = nn.Conv1d(current_dim, action_dim, kernel_size=1)

    def forward(
        self,
        reference_history_tokens: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        robot_state: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_reference(reference_history_tokens)
        if noisy_action.ndim != 3:
            raise ValueError("noisy_action must have shape [B,T,J]")
        if noisy_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"noisy_action width must match action_dim={self.action_dim}; "
                f"got {noisy_action.shape[-1]}"
            )
        if noisy_action.shape[1] > self.max_action_horizon:
            raise ValueError("action horizon exceeds max_action_horizon")
        condition = self._condition_embedding(
            reference_history_tokens,
            timesteps,
            robot_state=robot_state,
            prev_action=prev_action,
        )
        x = self.input_proj(noisy_action.transpose(1, 2))
        x = x + self.position[:, :, : x.shape[-1]].to(dtype=x.dtype)
        skips = []
        for block, downsample in zip(self.down_blocks, self.downsamples):
            x = block(x, condition)
            skips.append(x)
            x = downsample(x)
        x = self.mid_block1(x, condition)
        x = self.mid_block2(x, condition)
        for upsample, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips[:-1])):
            x = upsample(x, target_length=skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = block(x, condition)
        x = self.final_block(x, condition)
        return self.final_proj(x).transpose(1, 2)

    def diffusion_loss(
        self,
        reference_history_tokens: torch.Tensor,
        target_action: torch.Tensor,
        *,
        robot_state: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        loss_config: dict[str, float] | None = None,
        fps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del fps
        model_target = self._to_model_action(target_action, prev_action)
        if noise is None:
            noise = torch.randn_like(model_target)
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.diffusion_steps,
                (model_target.shape[0],),
                device=model_target.device,
            )
        noisy_action = self.scheduler.add_noise(model_target, noise, timesteps)
        pred_noise = self.forward(
            reference_history_tokens,
            noisy_action,
            timesteps,
            robot_state=robot_state,
            prev_action=prev_action,
        )
        loss_cfg = loss_config or {}
        weight = float(loss_cfg.get("denoise", loss_cfg.get("noise_mse", 1.0)))
        return weight * torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        reference_history_tokens: torch.Tensor,
        *,
        robot_state: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
        action_horizon: int | None = None,
        steps: int | None = None,
        start: str = "noise",
    ) -> torch.Tensor:
        self._validate_reference(reference_history_tokens)
        horizon = int(action_horizon or self.max_action_horizon)
        if horizon <= 0 or horizon > self.max_action_horizon:
            raise ValueError("action_horizon must be in [1, max_action_horizon]")
        shape = (reference_history_tokens.shape[0], horizon, self.action_dim)
        if start == "noise":
            state = torch.randn(
                *shape,
                device=reference_history_tokens.device,
                dtype=reference_history_tokens.dtype,
            )
        elif start == "zeros":
            state = torch.zeros(
                *shape,
                device=reference_history_tokens.device,
                dtype=reference_history_tokens.dtype,
            )
        else:
            raise ValueError(f"unsupported diffusion start: {start}")
        solve_steps = int(steps or self.inference_steps)
        if solve_steps != self.diffusion_steps:
            raise ValueError(
                "TinyDDPMScheduler sampling requires inference_steps == diffusion_steps; "
                "skipped timestep updates are not implemented"
            )
        indices = torch.linspace(
            self.diffusion_steps - 1,
            0,
            solve_steps,
            device=reference_history_tokens.device,
        ).round().to(torch.long)
        for index in indices:
            timestep = torch.full(
                (reference_history_tokens.shape[0],),
                int(index.item()),
                device=reference_history_tokens.device,
                dtype=torch.long,
            )
            pred_noise = self.forward(
                reference_history_tokens,
                state,
                timestep,
                robot_state=robot_state,
                prev_action=prev_action,
            )
            state = self.scheduler.step(pred_noise, index, state)
        return self._from_model_action(state, prev_action)

    def _condition_embedding(
        self,
        reference_history_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        robot_state: torch.Tensor | None,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        if timesteps.ndim == 2 and timesteps.shape[-1] == 1:
            timesteps = timesteps.squeeze(-1)
        time = timesteps.to(
            device=reference_history_tokens.device,
            dtype=reference_history_tokens.dtype,
        ).unsqueeze(-1)
        time = time / max(1, self.diffusion_steps - 1)
        condition = self.time_encoder(time)
        reference = self.reference_encoder(reference_history_tokens).mean(dim=(1, 2))
        condition = condition + reference
        if self.robot_encoder is not None:
            condition = condition + self.robot_encoder(
                self._robot_state_reference(reference_history_tokens, robot_state)
            )
        condition = condition + self.prev_action_encoder(
            self._prev_action_vector(reference_history_tokens, prev_action)
        )
        return condition

    def _validate_reference(self, reference_history_tokens: torch.Tensor) -> None:
        if reference_history_tokens.ndim != 4:
            raise ValueError("reference_history_tokens must have shape [B,H,N,D]")
        if reference_history_tokens.shape[1] != self.reference_history_frames:
            raise ValueError(
                "reference_history_tokens history length must match "
                f"reference_history_frames={self.reference_history_frames}; "
                f"got {reference_history_tokens.shape[1]}"
            )
        if reference_history_tokens.shape[2] != self.reference_body_count:
            raise ValueError(
                "reference_history_tokens body count must match "
                f"reference_body_count={self.reference_body_count}; "
                f"got {reference_history_tokens.shape[2]}"
            )
        if reference_history_tokens.shape[3] != self.reference_body_token_dim:
            raise ValueError(
                "reference_history_tokens token dim must match "
                f"reference_body_token_dim={self.reference_body_token_dim}; "
                f"got {reference_history_tokens.shape[3]}"
            )

    def _to_model_action(
        self,
        target_action: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.output_mode == "absolute":
            return target_action
        return target_action - self._prev_action_reference(target_action, prev_action)

    def _from_model_action(
        self,
        model_action: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.output_mode == "absolute":
            return model_action
        return self._prev_action_reference(model_action, prev_action) + model_action

    def _prev_action_reference(
        self,
        action: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._prev_action_vector(action, prev_action).unsqueeze(1)

    def _prev_action_vector(
        self,
        reference: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        if prev_action is None:
            if self.output_mode == "residual_prev_action":
                raise ValueError("residual_prev_action output_mode requires prev_action")
            return torch.zeros(
                reference.shape[0],
                self.action_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
        vector = prev_action.to(device=reference.device, dtype=reference.dtype).reshape(
            reference.shape[0],
            -1,
        )
        if vector.shape[-1] != self.action_dim:
            raise ValueError(
                f"prev_action width must match action_dim={self.action_dim}; "
                f"got {vector.shape[-1]}"
            )
        return vector

    def _robot_state_reference(
        self,
        reference: torch.Tensor,
        robot_state: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.robot_state_dim <= 0:
            return reference.new_zeros((reference.shape[0], 0))
        if robot_state is None:
            raise ValueError("robot_state is required when robot_state_dim > 0")
        vector = robot_state.to(device=reference.device, dtype=reference.dtype).reshape(
            reference.shape[0],
            -1,
        )
        if vector.shape[-1] != self.robot_state_dim:
            raise ValueError(
                f"robot_state width must match robot_state_dim={self.robot_state_dim}; "
                f"got {vector.shape[-1]}"
            )
        return vector


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 0:
            return time
        freqs = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * -(math.log(10000.0) / max(1, half - 1))
        )
        args = time * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = torch.cat([embedding, time], dim=-1)
        return embedding[:, : self.dim]


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        condition_dim: int,
        kernel_size: int,
        groups: int,
        cond_predict_scale: bool,
    ) -> None:
        super().__init__()
        self.block1 = Conv1dBlock(in_channels, out_channels, kernel_size=kernel_size, groups=groups)
        self.block2 = Conv1dBlock(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.cond_predict_scale = cond_predict_scale
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.condition = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, cond_channels))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        cond = self.condition(condition).unsqueeze(-1)
        if self.cond_predict_scale:
            scale, shift = cond.chunk(2, dim=1)
            out = out * (1.0 + scale) + shift
        else:
            out = out + cond
        out = self.block2(out)
        return out + self.residual(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(_valid_group_count(out_channels, groups), out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, *, target_length: int) -> torch.Tensor:
        out = self.conv(x)
        if out.shape[-1] > target_length:
            return out[..., :target_length]
        if out.shape[-1] < target_length:
            padding = target_length - out.shape[-1]
            return torch.nn.functional.pad(out, (0, padding))
        return out


def _valid_group_count(channels: int, requested: int) -> int:
    groups = max(1, min(channels, requested))
    while channels % groups != 0:
        groups -= 1
    return groups
