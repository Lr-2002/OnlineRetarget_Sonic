"""Small MLP VAE modules for continuous retargeting tokens."""

from __future__ import annotations

import torch
from torch import nn


class MLPTokenVAE(nn.Module):
    """Encode one continuous token family and reconstruct its numeric input."""

    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.dropout = dropout

        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(_activation(activation))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(prev_dim, latent_dim)
        self.logvar = nn.Linear(prev_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(_activation(activation))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar) if sample else mu
        reconstruction = self.decode(latent)
        return reconstruction, mu, logvar, latent


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction MSE, and KL losses."""

    reconstruction_mse = torch.nn.functional.mse_loss(reconstruction, target)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_mse + beta * kl, reconstruction_mse, kl


def _activation(name: str) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported VAE activation: {name}")
