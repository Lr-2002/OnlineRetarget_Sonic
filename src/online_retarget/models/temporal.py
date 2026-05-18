"""Temporal retargeting model families."""

from __future__ import annotations

import math

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
