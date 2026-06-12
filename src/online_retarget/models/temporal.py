"""Temporal retargeting model families."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .mlp import OnlineRetargetMLP


class TemporalTransformerRetargeter(nn.Module):
    """Bidirectional temporal Transformer over flattened window observations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        history_frames: int,
        source_feature_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if source_feature_dim <= 0 or source_feature_dim % history_frames != 0:
            raise ValueError("source_feature_dim must be divisible by history_frames")
        if input_dim < source_feature_dim:
            raise ValueError("input_dim must be >= source_feature_dim")
        self.history_frames = history_frames
        self.source_feature_dim = source_feature_dim
        self.per_frame_dim = source_feature_dim // history_frames
        self.side_dim = input_dim - source_feature_dim
        self.pooling = pooling

        self.source_proj = nn.Linear(self.per_frame_dim, d_model)
        self.side_proj = nn.Linear(self.side_dim, d_model) if self.side_dim > 0 else None
        token_count = history_frames + (1 if self.side_dim > 0 else 0)
        self.position = nn.Parameter(torch.zeros(1, token_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        source = observation[:, : self.source_feature_dim]
        tokens = self.source_proj(
            source.reshape(source.shape[0], self.history_frames, self.per_frame_dim)
        )
        if self.side_proj is not None:
            side = observation[:, self.source_feature_dim :]
            side_token = self.side_proj(side).unsqueeze(1)
            tokens = torch.cat([side_token, tokens], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        elif self.pooling == "last":
            pooled = encoded[:, -1]
        elif self.pooling == "side":
            pooled = encoded[:, 0]
        else:
            raise ValueError(f"unsupported transformer pooling: {self.pooling}")
        return self.head(self.norm(pooled))


class TokenizedTransformerRetargeter(nn.Module):
    """Continuous-token cross-attention baseline for next-frame G1 prediction.

    The model keeps token semantics explicit without requiring a new dataset
    format: the current flattened observation is sliced into source motion,
    morphology/skeleton proxy, and robot-state side-channel blocks according to
    ``ObservationSpec`` dimensions. Auxiliary autoencoder heads make the 128D
    skeleton, motion, and previous-state tokens trainable/debuggable before
    richer proposal-file skeleton features are wired in.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        history_frames: int,
        source_feature_dim: int,
        morphology_dim: int,
        robot_state_dim: int,
        latent_dim: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_mode: str = "position",
        use_prev_state: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if input_dim < source_feature_dim + morphology_dim + robot_state_dim:
            raise ValueError("input_dim is smaller than observation slices")
        if latent_dim % nhead != 0:
            raise ValueError("latent_dim must be divisible by nhead")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_frames = history_frames
        self.source_feature_dim = source_feature_dim
        self.morphology_dim = morphology_dim
        self.robot_state_dim = robot_state_dim
        self.latent_dim = latent_dim
        self.output_mode = output_mode
        self.use_prev_state = use_prev_state

        self.motion_encoder = _mlp(source_feature_dim, latent_dim, hidden_dim=max(latent_dim, 256))
        self.motion_decoder = _mlp(latent_dim, source_feature_dim, hidden_dim=max(latent_dim, 256))
        self.skeleton_encoder = _mlp(
            max(1, morphology_dim),
            latent_dim,
            hidden_dim=max(latent_dim, 128),
        )
        self.skeleton_decoder = _mlp(
            latent_dim,
            max(1, morphology_dim),
            hidden_dim=max(latent_dim, 128),
        )
        self.state_encoder = _mlp(output_dim, latent_dim, hidden_dim=max(latent_dim, 128))
        self.state_decoder = _mlp(latent_dim, output_dim, hidden_dim=max(latent_dim, 128))

        self.memory_type = nn.Parameter(torch.zeros(1, 2, latent_dim))
        self.query_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.query_type = nn.Parameter(torch.zeros(1, 1, latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, observation: torch.Tensor, prev_state: torch.Tensor | None = None) -> torch.Tensor:
        output, _ = self.forward_with_aux(observation, prev_state=prev_state)
        return output

    def forward_with_aux(
        self,
        observation: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source, skeleton, side = self._split_observation(observation)
        if prev_state is None:
            prev_state = self._prev_state_from_side(side, observation)
        z_motion = self.motion_encoder(source)
        z_skeleton = self.skeleton_encoder(skeleton)
        z_state = self.state_encoder(prev_state)
        memory = torch.stack([z_skeleton, z_motion], dim=1) + self.memory_type
        memory = self.encoder(memory)
        query = self.query_token.expand(observation.shape[0], -1, -1) + self.query_type
        if self.use_prev_state:
            query = query + z_state.unsqueeze(1)
        decoded = self.decoder(query, memory).squeeze(1)
        predicted = self.head(decoded)
        if self.output_mode == "delta":
            predicted = prev_state + predicted
        elif self.output_mode != "position":
            raise ValueError(f"unsupported output_mode: {self.output_mode}")
        aux = {
            "source": source,
            "skeleton": skeleton,
            "prev_state": prev_state,
            "motion_reconstruction": self.motion_decoder(z_motion),
            "skeleton_reconstruction": self.skeleton_decoder(z_skeleton)[
                :, : self.morphology_dim
            ],
            "state_reconstruction": self.state_decoder(z_state),
            "z_motion": z_motion,
            "z_skeleton": z_skeleton,
            "z_state": z_state,
        }
        return predicted, aux

    def _split_observation(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = observation[:, : self.source_feature_dim]
        start = self.source_feature_dim
        end = start + self.morphology_dim
        if self.morphology_dim > 0:
            skeleton = observation[:, start:end]
        else:
            skeleton = observation.new_zeros((observation.shape[0], 1))
        side_end = end + self.robot_state_dim
        side = observation[:, end:side_end] if self.robot_state_dim > 0 else observation.new_zeros((observation.shape[0], 0))
        return source, skeleton, side

    def _prev_state_from_side(self, side: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        if side.shape[1] >= self.output_dim:
            return side[:, : self.output_dim]
        return observation.new_zeros((observation.shape[0], self.output_dim))


class FlowMatchingRetargeter(nn.Module):
    """Conditional flow-matching baseline for G1 joint targets.

    The model learns a vector field from a simple Gaussian source to the target
    joint vector conditioned on the flattened temporal observation. Inference
    uses a deterministic zero-start Euler solve by default so debug runs are
    reproducible.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        activation: str = "silu",
        dropout: float = 0.0,
        time_embed_dim: int = 32,
        inference_steps: int = 8,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.time_embed_dim = time_embed_dim
        self.inference_steps = inference_steps
        self.vector_field = OnlineRetargetMLP(
            input_dim=input_dim + output_dim + time_embed_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        return self.vector_field(torch.cat([observation, state, _time_embedding(time, self.time_embed_dim)], dim=-1))

    def flow_matching_loss(
        self,
        observation: torch.Tensor,
        target: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], 1, device=target.device, dtype=target.dtype)
        state = (1.0 - time) * noise + time * target
        target_velocity = target - noise
        pred_velocity = self.forward(observation, state, time)
        return torch.nn.functional.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample(
        self,
        observation: torch.Tensor,
        *,
        steps: int | None = None,
        start: str = "zeros",
    ) -> torch.Tensor:
        solve_steps = max(1, int(steps or self.inference_steps))
        if start == "noise":
            state = torch.randn(
                observation.shape[0],
                self.output_dim,
                device=observation.device,
                dtype=observation.dtype,
            )
        elif start == "zeros":
            state = torch.zeros(
                observation.shape[0],
                self.output_dim,
                device=observation.device,
                dtype=observation.dtype,
            )
        else:
            raise ValueError(f"unsupported flow start: {start}")
        dt = 1.0 / solve_steps
        for index in range(solve_steps):
            time = torch.full(
                (observation.shape[0], 1),
                index / solve_steps,
                device=observation.device,
                dtype=observation.dtype,
            )
            state = state + dt * self.forward(observation, state, time)
        return state


class DiffusionPolicyRetargeter(nn.Module):
    """Conditional DDPM-style denoiser for flattened future G1 action windows."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        activation: str = "silu",
        dropout: float = 0.0,
        time_embed_dim: int = 32,
        diffusion_steps: int = 32,
        inference_steps: int | None = None,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        if beta_start <= 0.0 or beta_end <= 0.0 or beta_end <= beta_start:
            raise ValueError("expected 0 < beta_start < beta_end")
        self.output_dim = output_dim
        self.time_embed_dim = time_embed_dim
        self.diffusion_steps = diffusion_steps
        self.inference_steps = inference_steps or diffusion_steps
        self.denoiser = OnlineRetargetMLP(
            input_dim=input_dim + output_dim + time_embed_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(
        self,
        observation: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.ndim == 2 and timesteps.shape[-1] == 1:
            timesteps = timesteps.squeeze(-1)
        time = timesteps.to(device=observation.device, dtype=observation.dtype).unsqueeze(-1)
        time = time / max(1, self.diffusion_steps - 1)
        conditioning = torch.cat(
            [observation, noisy_action, _time_embedding(time, self.time_embed_dim)],
            dim=-1,
        )
        return self.denoiser(conditioning)

    def diffusion_loss(
        self,
        observation: torch.Tensor,
        target_action: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(target_action)
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.diffusion_steps,
                (target_action.shape[0],),
                device=target_action.device,
            )
        alpha_bar = self.alpha_bars[timesteps].to(target_action.dtype).unsqueeze(-1)
        noisy_action = alpha_bar.sqrt() * target_action + (1.0 - alpha_bar).sqrt() * noise
        pred_noise = self.forward(observation, noisy_action, timesteps)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        observation: torch.Tensor,
        *,
        steps: int | None = None,
        start: str = "noise",
    ) -> torch.Tensor:
        solve_steps = max(1, min(self.diffusion_steps, int(steps or self.inference_steps)))
        if start == "noise":
            state = torch.randn(
                observation.shape[0],
                self.output_dim,
                device=observation.device,
                dtype=observation.dtype,
            )
        elif start == "zeros":
            state = torch.zeros(
                observation.shape[0],
                self.output_dim,
                device=observation.device,
                dtype=observation.dtype,
            )
        else:
            raise ValueError(f"unsupported diffusion start: {start}")

        indices = torch.linspace(
            self.diffusion_steps - 1,
            0,
            solve_steps,
            device=observation.device,
        ).round().to(torch.long)
        for index in indices:
            timestep = torch.full(
                (observation.shape[0],),
                int(index.item()),
                device=observation.device,
                dtype=torch.long,
            )
            beta = self.betas[index].to(dtype=observation.dtype)
            alpha = self.alphas[index].to(dtype=observation.dtype)
            alpha_bar = self.alpha_bars[index].to(dtype=observation.dtype)
            pred_noise = self.forward(observation, state, timestep)
            state = (state - beta * pred_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
        return state


class TemporalDiffusionPolicyRetargeter(nn.Module):
    """Temporal denoiser over structured source tokens and G1 action horizons."""

    def __init__(
        self,
        *,
        action_dim: int = 29,
        source_body_token_dim: int = 15,
        source_skeleton_dim: int = 120,
        morphology_dim: int = 13,
        robot_state_dim: int = 94,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        time_embed_dim: int = 32,
        diffusion_steps: int = 32,
        inference_steps: int | None = None,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        max_horizon: int = 64,
        output_mode: str = "absolute",
    ) -> None:
        super().__init__()
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if beta_start <= 0.0 or beta_end <= 0.0 or beta_end <= beta_start:
            raise ValueError("expected 0 < beta_start < beta_end")
        if output_mode not in {"absolute", "residual_prev_action"}:
            raise ValueError("output_mode must be absolute or residual_prev_action")
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.inference_steps = inference_steps or diffusion_steps
        self.time_embed_dim = time_embed_dim
        self.source_skeleton_dim = source_skeleton_dim
        self.morphology_dim = morphology_dim
        self.robot_state_dim = robot_state_dim
        self.output_mode = output_mode

        self.body_encoder = nn.Sequential(
            nn.Linear(source_body_token_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.action_encoder = nn.Linear(action_dim, d_model)
        self.time_encoder = nn.Linear(time_embed_dim, d_model) if time_embed_dim > 0 else None
        global_dim = source_skeleton_dim + morphology_dim + robot_state_dim + action_dim
        self.global_encoder = nn.Linear(global_dim, d_model) if global_dim > 0 else None
        self.position = nn.Parameter(torch.zeros(1, max_horizon, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, action_dim)

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(
        self,
        source_body_tokens: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        source_skeleton: torch.Tensor | None = None,
        morphology: torch.Tensor | None = None,
        robot_state: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_body_tokens.ndim != 4:
            raise ValueError("source_body_tokens must have shape [B,T,N,D]")
        if noisy_action.ndim != 3:
            raise ValueError("noisy_action must have shape [B,T,J]")
        if source_body_tokens.shape[1] != noisy_action.shape[1]:
            raise ValueError("source/action horizons must match")
        body_tokens = self.body_encoder(source_body_tokens).mean(dim=2)
        tokens = body_tokens + self.action_encoder(noisy_action)
        if timesteps.ndim == 2 and timesteps.shape[-1] == 1:
            timesteps = timesteps.squeeze(-1)
        time = timesteps.to(device=noisy_action.device, dtype=noisy_action.dtype).unsqueeze(-1)
        time = time / max(1, self.diffusion_steps - 1)
        if self.time_encoder is not None:
            tokens = tokens + self.time_encoder(_time_embedding(time, self.time_embed_dim)).unsqueeze(1)
        global_token = self._global_token(
            noisy_action,
            source_skeleton=source_skeleton,
            morphology=morphology,
            robot_state=robot_state,
            prev_action=prev_action,
        )
        if global_token is not None:
            tokens = tokens + global_token.unsqueeze(1)
        if noisy_action.shape[1] > self.position.shape[1]:
            raise ValueError("action horizon exceeds max_horizon")
        tokens = tokens + self.position[:, : noisy_action.shape[1]]
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded))

    def diffusion_loss(
        self,
        source_body_tokens: torch.Tensor,
        target_action: torch.Tensor,
        *,
        source_skeleton: torch.Tensor | None = None,
        morphology: torch.Tensor | None = None,
        robot_state: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        loss_config: dict[str, Any] | None = None,
        fps: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        alpha_bar = self.alpha_bars[timesteps].to(model_target.dtype).view(-1, 1, 1)
        noisy_action = alpha_bar.sqrt() * model_target + (1.0 - alpha_bar).sqrt() * noise
        pred_noise = self.forward(
            source_body_tokens,
            noisy_action,
            timesteps,
            source_skeleton=source_skeleton,
            morphology=morphology,
            robot_state=robot_state,
            prev_action=prev_action,
        )
        loss_cfg = loss_config or {}
        denoise_weight = float(loss_cfg.get("denoise", loss_cfg.get("noise_mse", 1.0)))
        total = denoise_weight * torch.nn.functional.mse_loss(
            pred_noise, noise
        )
        if self._has_auxiliary_loss(loss_cfg):
            pred_model_action = (
                noisy_action - (1.0 - alpha_bar).sqrt() * pred_noise
            ) / alpha_bar.sqrt()
            pred_action = self._from_model_action(pred_model_action, prev_action)
            total = total + self._stability_loss(
                pred_action,
                target_action,
                loss_cfg,
                prev_action=prev_action,
                fps=fps,
            )
        return total

    @torch.no_grad()
    def sample(
        self,
        source_body_tokens: torch.Tensor,
        *,
        source_skeleton: torch.Tensor | None = None,
        morphology: torch.Tensor | None = None,
        robot_state: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
        steps: int | None = None,
        start: str = "noise",
    ) -> torch.Tensor:
        solve_steps = max(1, min(self.diffusion_steps, int(steps or self.inference_steps)))
        shape = (source_body_tokens.shape[0], source_body_tokens.shape[1], self.action_dim)
        if start == "noise":
            state = torch.randn(*shape, device=source_body_tokens.device, dtype=source_body_tokens.dtype)
        elif start == "zeros":
            state = torch.zeros(*shape, device=source_body_tokens.device, dtype=source_body_tokens.dtype)
        else:
            raise ValueError(f"unsupported diffusion start: {start}")
        indices = torch.linspace(
            self.diffusion_steps - 1,
            0,
            solve_steps,
            device=source_body_tokens.device,
        ).round().to(torch.long)
        for index in indices:
            timestep = torch.full(
                (source_body_tokens.shape[0],),
                int(index.item()),
                device=source_body_tokens.device,
                dtype=torch.long,
            )
            beta = self.betas[index].to(dtype=source_body_tokens.dtype)
            alpha = self.alphas[index].to(dtype=source_body_tokens.dtype)
            alpha_bar = self.alpha_bars[index].to(dtype=source_body_tokens.dtype)
            pred_noise = self.forward(
                source_body_tokens,
                state,
                timestep,
                source_skeleton=source_skeleton,
                morphology=morphology,
                robot_state=robot_state,
                prev_action=prev_action,
            )
            state = (state - beta * pred_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
        return self._from_model_action(state, prev_action)

    def _to_model_action(
        self,
        target_action: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.output_mode == "absolute":
            return target_action
        reference = self._prev_action_reference(target_action, prev_action)
        return target_action - reference

    def _from_model_action(
        self,
        model_action: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.output_mode == "absolute":
            return model_action
        reference = self._prev_action_reference(model_action, prev_action)
        return reference + model_action

    def _prev_action_reference(
        self,
        action: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        if prev_action is None:
            raise ValueError("residual_prev_action output_mode requires prev_action")
        reference = prev_action.to(device=action.device, dtype=action.dtype)
        if reference.ndim != 2:
            reference = reference.reshape(action.shape[0], -1)
        if reference.shape[-1] != self.action_dim:
            raise ValueError(
                f"prev_action width must match action_dim={self.action_dim}; "
                f"got {reference.shape[-1]}"
            )
        return reference.unsqueeze(1)

    @staticmethod
    def _has_auxiliary_loss(loss_config: dict[str, Any]) -> bool:
        keys = (
            "x0_reconstruction",
            "q_reconstruction",
            "velocity",
            "acceleration",
            "jerk",
            "delta_smoothness",
            "joint_jump",
            "joint_limit",
        )
        return any(float(loss_config.get(key, 0.0)) != 0.0 for key in keys)

    def _stability_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        loss_config: dict[str, Any],
        *,
        prev_action: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = prediction.new_tensor(0.0)
        reconstruction_weight = float(
            loss_config.get("x0_reconstruction", loss_config.get("q_reconstruction", 0.0))
        )
        if reconstruction_weight:
            total = total + reconstruction_weight * torch.nn.functional.mse_loss(prediction, target)
        velocity_weight = float(loss_config.get("velocity", 0.0))
        if velocity_weight and prediction.shape[1] >= 2:
            pred_velocity = prediction[:, 1:] - prediction[:, :-1]
            target_velocity = target[:, 1:] - target[:, :-1]
            velocity_loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)
            total = total + velocity_weight * velocity_loss
        delta_smoothness_weight = float(loss_config.get("delta_smoothness", 0.0))
        if delta_smoothness_weight:
            total = total + delta_smoothness_weight * self._delta_smoothness_loss(
                prediction,
                prev_action,
            )
        acceleration_weight = float(loss_config.get("acceleration", 0.0))
        if acceleration_weight and prediction.shape[1] >= 3:
            pred_accel = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
            target_accel = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
            total = total + acceleration_weight * torch.nn.functional.mse_loss(
                pred_accel,
                target_accel,
            )
        jerk_weight = float(loss_config.get("jerk", 0.0))
        if jerk_weight and prediction.shape[1] >= 4:
            pred_jerk = (
                prediction[:, 3:]
                - 3.0 * prediction[:, 2:-1]
                + 3.0 * prediction[:, 1:-2]
                - prediction[:, :-3]
            )
            target_jerk = (
                target[:, 3:]
                - 3.0 * target[:, 2:-1]
                + 3.0 * target[:, 1:-2]
                - target[:, :-3]
            )
            total = total + jerk_weight * torch.nn.functional.mse_loss(pred_jerk, target_jerk)
        jump_weight = float(loss_config.get("joint_jump", 0.0))
        if jump_weight and prediction.shape[1] >= 2:
            threshold = self._joint_jump_delta_threshold(
                prediction,
                loss_config,
                fps=fps,
            )
            pred_jump = torch.relu((prediction[:, 1:] - prediction[:, :-1]).abs() - threshold)
            target_jump = torch.relu((target[:, 1:] - target[:, :-1]).abs() - threshold)
            total = total + jump_weight * torch.nn.functional.mse_loss(pred_jump, target_jump)
        limit_weight = float(loss_config.get("joint_limit", 0.0))
        if limit_weight:
            total = total + limit_weight * self._joint_limit_loss(prediction, loss_config)
        return total

    def _delta_smoothness_loss(
        self,
        prediction: torch.Tensor,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        deltas = []
        if prev_action is not None:
            reference = self._prev_action_reference(prediction, prev_action)
            deltas.append(prediction[:, :1] - reference)
        if prediction.shape[1] >= 2:
            deltas.append(prediction[:, 1:] - prediction[:, :-1])
        if not deltas:
            return prediction.new_tensor(0.0)
        return torch.mean(torch.cat(deltas, dim=1).square())

    def _joint_jump_delta_threshold(
        self,
        reference: torch.Tensor,
        loss_config: dict[str, Any],
        *,
        fps: torch.Tensor | None,
    ) -> torch.Tensor:
        explicit_delta = loss_config.get(
            "joint_jump_delta_threshold",
            loss_config.get("joint_jump_delta"),
        )
        if explicit_delta is not None:
            return self._threshold_tensor(reference, explicit_delta)
        velocity = float(loss_config.get("joint_jump_velocity", 20.0))
        if fps is None:
            fps = reference.new_full(
                (reference.shape[0],),
                float(loss_config.get("joint_jump_fps", loss_config.get("fps", 30.0))),
            )
        fps_tensor = torch.as_tensor(fps, device=reference.device, dtype=reference.dtype)
        fps_tensor = fps_tensor.reshape(reference.shape[0], -1)[:, :1]
        if torch.any(fps_tensor <= 0):
            raise ValueError("joint_jump fps must be positive")
        return (velocity / fps_tensor).reshape(reference.shape[0], 1, 1)

    def _threshold_tensor(
        self,
        reference: torch.Tensor,
        value,
    ) -> torch.Tensor:
        if isinstance(value, (int, float)):
            return reference.new_full((1, 1, 1), float(value))
        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        return tensor.reshape(1, 1, -1)

    def _joint_limit_loss(
        self,
        prediction: torch.Tensor,
        loss_config: dict[str, Any],
    ) -> torch.Tensor:
        lower = self._optional_limit_tensor(
            prediction,
            loss_config.get("joint_limit_min"),
        )
        upper = self._optional_limit_tensor(
            prediction,
            loss_config.get("joint_limit_max"),
        )
        penalty = prediction.new_zeros(prediction.shape)
        if lower is not None:
            penalty = penalty + torch.relu(lower - prediction)
        if upper is not None:
            penalty = penalty + torch.relu(prediction - upper)
        return torch.mean(penalty.square())

    def _optional_limit_tensor(
        self,
        reference: torch.Tensor,
        value,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return reference.new_full((1, 1, self.action_dim), float(value))
        tensor = torch.as_tensor(
            value,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, -1)
        if tensor.shape[-1] == 1:
            return tensor.expand(1, 1, self.action_dim)
        if tensor.shape[-1] != self.action_dim:
            raise ValueError(
                f"joint limit width must be 1 or action_dim={self.action_dim}; "
                f"got {tensor.shape[-1]}"
            )
        return tensor

    def _global_token(
        self,
        noisy_action: torch.Tensor,
        *,
        source_skeleton: torch.Tensor | None,
        morphology: torch.Tensor | None,
        robot_state: torch.Tensor | None,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.global_encoder is None:
            return None
        batch = noisy_action.shape[0]
        dtype = noisy_action.dtype
        device = noisy_action.device
        pieces = []
        for value, width in (
            (source_skeleton, self.source_skeleton_dim),
            (morphology, self.morphology_dim),
            (robot_state, self.robot_state_dim),
            (prev_action, self.action_dim),
        ):
            if width <= 0:
                continue
            if value is None:
                pieces.append(torch.zeros(batch, width, device=device, dtype=dtype))
                continue
            piece = value.to(device=device, dtype=dtype).reshape(batch, -1)
            if piece.shape[-1] < width:
                padding = torch.zeros(batch, width - piece.shape[-1], device=device, dtype=dtype)
                piece = torch.cat([piece, padding], dim=-1)
            elif piece.shape[-1] > width:
                piece = piece[:, :width]
            pieces.append(piece)
        if not pieces:
            return None
        return self.global_encoder(torch.cat(pieces, dim=-1))


def _time_embedding(time: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        return time.new_zeros((time.shape[0], 0))
    half = dim // 2
    if half == 0:
        return time
    freqs = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * -(math.log(10000.0) / max(1, half - 1))
    )
    args = time * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = torch.cat([embedding, time], dim=-1)
    return embedding[:, :dim]


def _mlp(input_dim: int, output_dim: int, *, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )
