from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from .loaders import LoadedVisPacket


@dataclass(frozen=True)
class RenderRequest:
    loaded_packet: LoadedVisPacket
    output_dir: Path | None = None


@dataclass(frozen=True)
class RenderArtifact:
    path: Path | None = None
    bytes: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class StaticRenderer(Protocol):
    """Optional renderer interface; adapters live outside vis_core."""

    def render(self, request: RenderRequest) -> RenderArtifact:
        ...


class PhysicsAdapter(Protocol):
    """Optional physics interface for dynamic paths outside static validation."""

    def reset(self, loaded_packet: LoadedVisPacket) -> Mapping[str, Any]:
        ...

    def step(self, action: Mapping[str, Any], *, dt: float) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...


class DiagnoseAdapter(Protocol):
    """Optional diagnostics interface for adapter-specific collection."""

    def collect(self, loaded_packet: LoadedVisPacket) -> Mapping[str, Any]:
        ...
