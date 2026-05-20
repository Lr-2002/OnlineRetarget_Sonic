"""Hydra-compatible encoder modules for SONIC-native retarget variants."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "SiLU":
        return nn.SiLU()
    if name == "ReLU":
        return nn.ReLU()
    if name == "GELU":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name}")


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    activation: str = "SiLU",
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous, int(hidden_dim)))
        layers.append(_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        previous = int(hidden_dim)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def _deterministic_cluster_routes(
    conditioning: torch.Tensor,
    *,
    num_routes: int,
    route_feature_index: int = -1,
) -> torch.Tensor:
    """Map the normalized skeleton-cluster scalar to a deterministic route id."""

    if num_routes <= 0:
        raise ValueError("num_routes must be positive")
    cluster_scalar = conditioning[..., route_feature_index]
    routes = torch.round(cluster_scalar.clamp(0.0, 1.0) * float(num_routes - 1)).long()
    return routes.clamp_(0, num_routes - 1)


class _SonicEncoderBase(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_input_temporal_dims: int | None = None,
        num_output_temporal_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.input_feature_dim = int(input_dim)
        self.output_feature_dim = int(output_dim)
        self.num_input_temporal_dims = num_input_temporal_dims
        self.num_output_temporal_dims = num_output_temporal_dims
        self.input_dim = self.input_feature_dim
        self.output_dim = self.output_feature_dim
        if self.num_input_temporal_dims is not None:
            self.input_dim *= int(self.num_input_temporal_dims)
        if self.num_output_temporal_dims is not None:
            self.output_dim *= int(self.num_output_temporal_dims)

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_input_temporal_dims is not None:
            return x.reshape(*x.shape[:-2], self.input_dim)
        return x

    def _reshape_output(self, y: torch.Tensor) -> torch.Tensor:
        if self.num_output_temporal_dims is None:
            return y
        return y.reshape(
            *y.shape[:-1],
            int(self.num_output_temporal_dims),
            self.output_feature_dim,
        )

    def _effective_conditioning_dim(self, conditioning_dim: int) -> int:
        conditioning_dim = int(conditioning_dim)
        if self.num_input_temporal_dims is not None:
            conditioning_dim *= int(self.num_input_temporal_dims)
        return conditioning_dim

    def _split_motion_conditioning(
        self,
        x: torch.Tensor,
        conditioning_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioning_dim = int(conditioning_dim)
        motion_feature_dim = self.input_feature_dim - conditioning_dim
        if self.num_input_temporal_dims is None:
            return x.split([motion_feature_dim, conditioning_dim], dim=-1)

        motion = x[..., :motion_feature_dim].reshape(*x.shape[:-2], -1)
        conditioning = x[..., motion_feature_dim:].reshape(*x.shape[:-2], -1)
        return motion, conditioning


class ConcatSomaEncoderModule(_SonicEncoderBase):
    """A1: compact MLP over concatenated source motion and skeleton features."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        activation: str = "SiLU",
        dropout: float = 0.0,
        num_input_temporal_dims: int | None = None,
        num_output_temporal_dims: int | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_input_temporal_dims=num_input_temporal_dims,
            num_output_temporal_dims=num_output_temporal_dims,
        )
        self.net = _mlp(
            self.input_dim,
            hidden_dims,
            self.output_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self._reshape_output(self.net(self._flatten_input(x)))


class FilmSomaEncoderModule(_SonicEncoderBase):
    """A2: FiLM-conditioned MLP using appended skeleton/contact features."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        conditioning_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        activation: str = "SiLU",
        num_input_temporal_dims: int | None = None,
        num_output_temporal_dims: int | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_input_temporal_dims=num_input_temporal_dims,
            num_output_temporal_dims=num_output_temporal_dims,
        )
        if conditioning_dim <= 0 or conditioning_dim >= input_dim:
            raise ValueError("conditioning_dim must split motion and conditioning features")
        self.conditioning_feature_dim = int(conditioning_dim)
        self.conditioning_dim = self._effective_conditioning_dim(conditioning_dim)
        self.motion_dim = self.input_dim - self.conditioning_dim
        self.input = nn.Linear(self.motion_dim, hidden_dim)
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, num_layers - 1))
        )
        self.film = nn.Linear(self.conditioning_dim, 2 * hidden_dim * max(1, num_layers))
        self.output = nn.Linear(hidden_dim, self.output_dim)
        self.act = _activation(activation)
        self.num_layers = max(1, int(num_layers))
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        motion, conditioning = self._split_motion_conditioning(x, self.conditioning_feature_dim)
        film = self.film(conditioning).reshape(
            *conditioning.shape[:-1],
            self.num_layers,
            2,
            self.hidden_dim,
        )
        h = self.act(self.input(motion))
        gamma, beta = film[..., 0, 0, :], film[..., 0, 1, :]
        h = h * (1.0 + gamma) + beta
        for index, layer in enumerate(self.layers, start=1):
            h = self.act(layer(h))
            gamma, beta = film[..., index, 0, :], film[..., index, 1, :]
            h = h * (1.0 + gamma) + beta
        return self._reshape_output(self.output(h))


class AdapterSomaEncoderModule(_SonicEncoderBase):
    """B1: shared trunk plus deterministic route-specific residual adapters."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        conditioning_dim: int,
        hidden_dim: int = 512,
        adapter_dim: int = 128,
        num_adapters: int = 4,
        routing: str = "deterministic_cluster",
        route_feature_index: int = -1,
        num_input_temporal_dims: int | None = None,
        num_output_temporal_dims: int | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_input_temporal_dims=num_input_temporal_dims,
            num_output_temporal_dims=num_output_temporal_dims,
        )
        if conditioning_dim <= 0 or conditioning_dim >= input_dim:
            raise ValueError("conditioning_dim must split motion and conditioning features")
        self.conditioning_feature_dim = int(conditioning_dim)
        self.conditioning_dim = self._effective_conditioning_dim(conditioning_dim)
        self.motion_dim = self.input_dim - self.conditioning_dim
        self.num_adapters = int(num_adapters)
        self.routing = routing
        self.route_feature_index = int(route_feature_index)
        self.trunk = _mlp(self.motion_dim, (hidden_dim, hidden_dim), hidden_dim)
        self.router = (
            nn.Linear(self.conditioning_dim, self.num_adapters)
            if routing == "learned"
            else None
        )
        self.adapters = nn.ModuleList(
            _mlp(hidden_dim, (adapter_dim,), hidden_dim) for _ in range(self.num_adapters)
        )
        self.output = nn.Linear(hidden_dim, self.output_dim)
        self.last_routes: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        motion, conditioning = self._split_motion_conditioning(x, self.conditioning_feature_dim)
        h = self.trunk(motion)
        routes = self._routes(conditioning)
        self.last_routes = routes.detach()
        residual = torch.zeros_like(h)
        for index, adapter in enumerate(self.adapters):
            mask = routes == index
            if mask.any():
                residual[mask] = adapter(h[mask])
        return self._reshape_output(self.output(h + residual))

    def _routes(self, conditioning: torch.Tensor) -> torch.Tensor:
        if self.routing in {"deterministic_cluster", "cluster"}:
            return _deterministic_cluster_routes(
                conditioning,
                num_routes=self.num_adapters,
                route_feature_index=self.route_feature_index,
            )
        if self.routing == "learned" and self.router is not None:
            return torch.argmax(self.router(conditioning), dim=-1)
        raise ValueError(f"unsupported routing mode: {self.routing}")

    def route_summary(self) -> dict[str, object]:
        return _route_summary(self.last_routes, self.num_adapters)


class ExpertSomaEncoderModule(_SonicEncoderBase):
    """B2: deterministic skeleton-routed lightweight expert branches."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        conditioning_dim: int,
        hidden_dim: int = 512,
        num_experts: int = 4,
        routing: str = "deterministic_cluster",
        route_feature_index: int = -1,
        num_input_temporal_dims: int | None = None,
        num_output_temporal_dims: int | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_input_temporal_dims=num_input_temporal_dims,
            num_output_temporal_dims=num_output_temporal_dims,
        )
        if conditioning_dim <= 0 or conditioning_dim >= input_dim:
            raise ValueError("conditioning_dim must split motion and conditioning features")
        self.conditioning_feature_dim = int(conditioning_dim)
        self.conditioning_dim = self._effective_conditioning_dim(conditioning_dim)
        self.motion_dim = self.input_dim - self.conditioning_dim
        self.num_experts = int(num_experts)
        self.routing = routing
        self.route_feature_index = int(route_feature_index)
        self.router = (
            nn.Linear(self.conditioning_dim, self.num_experts)
            if routing == "learned"
            else None
        )
        self.experts = nn.ModuleList(
            _mlp(self.input_dim, (hidden_dim, hidden_dim), self.output_dim)
            for _ in range(self.num_experts)
        )
        self.last_routes: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        flat = self._flatten_input(x)
        _, conditioning = self._split_motion_conditioning(x, self.conditioning_feature_dim)
        routes = self._routes(conditioning)
        self.last_routes = routes.detach()
        output = flat.new_zeros(*flat.shape[:-1], self.output_dim)
        for index, expert in enumerate(self.experts):
            mask = routes == index
            if mask.any():
                output[mask] = expert(flat[mask])
        return self._reshape_output(output)

    def _routes(self, conditioning: torch.Tensor) -> torch.Tensor:
        if self.routing in {"deterministic_cluster", "cluster"}:
            return _deterministic_cluster_routes(
                conditioning,
                num_routes=self.num_experts,
                route_feature_index=self.route_feature_index,
            )
        if self.routing == "learned" and self.router is not None:
            return torch.argmax(self.router(conditioning), dim=-1)
        raise ValueError(f"unsupported routing mode: {self.routing}")

    def route_summary(self) -> dict[str, object]:
        return _route_summary(self.last_routes, self.num_experts)


def _route_summary(routes: torch.Tensor | None, num_routes: int) -> dict[str, object]:
    if routes is None:
        return {"available": False, "counts": [0] * int(num_routes)}
    flat = routes.detach().reshape(-1).cpu()
    counts = torch.bincount(flat, minlength=int(num_routes))[: int(num_routes)]
    return {
        "available": True,
        "counts": [int(value) for value in counts.tolist()],
        "total": int(flat.numel()),
    }
