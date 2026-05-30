"""Frozen Skeleton Geometry AE model and artifact loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from online_retarget.data.skeleton_ae_registry import SKELETON_GEOMETRY_DIM


SKELETON_GEOMETRY_AE_ARCHITECTURE = [104, 256, 128, 64, 128, 256, 104]
SKELETON_GEOMETRY_LATENT_DIM = 64


class SkeletonGeometryAE(nn.Module):
    """MLP AE used by the all-skeleton geometry gate."""

    def __init__(self, *, input_dim: int = SKELETON_GEOMETRY_DIM, latent_dim: int = 64) -> None:
        super().__init__()
        if input_dim != SKELETON_GEOMETRY_DIM:
            raise ValueError(f"input_dim must be {SKELETON_GEOMETRY_DIM}")
        if latent_dim != SKELETON_GEOMETRY_LATENT_DIM:
            raise ValueError(f"latent_dim must be {SKELETON_GEOMETRY_LATENT_DIM}")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def validate_skeleton_ae_architecture(payload: Mapping[str, Any]) -> None:
    architecture = payload.get("architecture")
    if list(architecture or []) != SKELETON_GEOMETRY_AE_ARCHITECTURE:
        raise ValueError(
            "Skeleton AE checkpoint architecture must be "
            f"{SKELETON_GEOMETRY_AE_ARCHITECTURE}, got {architecture!r}"
        )


def load_skeleton_geometry_ae_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    freeze_encoder: bool = True,
) -> tuple[SkeletonGeometryAE, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"skeleton_ae.checkpoint is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Skeleton AE checkpoint must contain a mapping: {checkpoint_path}")
    validate_skeleton_ae_architecture(payload)
    model = SkeletonGeometryAE().to(device)
    model.load_state_dict(payload["model"], strict=True)
    if freeze_encoder:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model, dict(payload)


def load_skeleton_geometry_ae_stats(stats_path: Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    if not stats_path.exists():
        raise FileNotFoundError(f"skeleton_ae.normalization is missing: {stats_path}")
    payload = torch.load(stats_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Skeleton AE stats must contain a mapping: {stats_path}")
    stats = {key: value.to(device) for key, value in payload.items() if torch.is_tensor(value)}
    for key in ("skeleton_mean", "skeleton_std"):
        if key not in stats:
            raise ValueError(f"Skeleton AE stats missing {key}: {stats_path}")
        if tuple(stats[key].shape) != (SKELETON_GEOMETRY_DIM,):
            raise ValueError(f"Skeleton AE stats {key} must have shape [104], got {tuple(stats[key].shape)}")
    return stats
