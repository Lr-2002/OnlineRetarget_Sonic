"""Compact direct-output baseline for online G1 retargeting."""

from __future__ import annotations

import torch
from torch import nn


class OnlineRetargetMLP(nn.Module):
    """Temporal MLP that maps flattened observations to G1 joint commands."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 29,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def _activation(name: str) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key == "silu":
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported MLP activation: {name}")
