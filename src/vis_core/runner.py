from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from .interfaces import RenderArtifact, RenderRequest, StaticRenderer
from .loaders import LoadedVisPacket, load_vis_packet, validate_loaded_packet


RendererLike = StaticRenderer | Callable[[RenderRequest], RenderArtifact]


@dataclass(frozen=True)
class StaticRunnerDiagnostics:
    status: str
    schema_version: str
    frame_count: int
    duration_sec: float
    tracks: tuple[str, ...]
    load_sec: float
    validate_sec: float
    render_sec: float
    wall_sec: float
    physics_requested: bool = False
    physics_executed: bool = False
    renderer: str | None = None
    output_path: str | None = None
    output_bytes: int | None = None
    adapter_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "schema_version": self.schema_version,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "tracks": list(self.tracks),
            "load_sec": self.load_sec,
            "validate_sec": self.validate_sec,
            "render_sec": self.render_sec,
            "wall_sec": self.wall_sec,
            "physics_requested": self.physics_requested,
            "physics_executed": self.physics_executed,
            "renderer": self.renderer,
            "output_path": self.output_path,
            "output_bytes": self.output_bytes,
            "adapter_diagnostics": dict(self.adapter_diagnostics),
        }


@dataclass(frozen=True)
class StaticRunnerResult:
    loaded_packet: LoadedVisPacket
    diagnostics: StaticRunnerDiagnostics
    artifact: RenderArtifact | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "packet": self.loaded_packet.packet.as_dict(),
            "diagnostics": self.diagnostics.as_dict(),
        }


def run_static_packet(
    manifest_path: str | Path,
    *,
    renderer: RendererLike | None = None,
    output_dir: str | Path | None = None,
) -> StaticRunnerResult:
    wall_start = perf_counter()

    load_start = perf_counter()
    loaded = load_vis_packet(manifest_path, validate=False)
    load_sec = perf_counter() - load_start

    validate_start = perf_counter()
    validate_loaded_packet(loaded)
    validate_sec = perf_counter() - validate_start

    artifact: RenderArtifact | None = None
    render_sec = 0.0
    renderer_name: str | None = None
    if renderer is not None:
        render_start = perf_counter()
        request = RenderRequest(
            loaded_packet=loaded,
            output_dir=Path(output_dir) if output_dir is not None else None,
        )
        artifact = _render(renderer, request)
        render_sec = perf_counter() - render_start
        renderer_name = _renderer_name(renderer)

    output_path, output_bytes = _artifact_output(artifact)
    diagnostics = StaticRunnerDiagnostics(
        status="ok",
        schema_version=loaded.packet.schema_version,
        frame_count=loaded.packet.timeline.frame_count,
        duration_sec=loaded.packet.timeline.duration_sec,
        tracks=tuple(loaded.tracks),
        load_sec=load_sec,
        validate_sec=validate_sec,
        render_sec=render_sec,
        wall_sec=perf_counter() - wall_start,
        physics_requested=loaded.packet.physics.enabled,
        physics_executed=False,
        renderer=renderer_name,
        output_path=str(output_path) if output_path is not None else None,
        output_bytes=output_bytes,
        adapter_diagnostics=artifact.diagnostics if artifact is not None else {},
    )
    return StaticRunnerResult(loaded_packet=loaded, diagnostics=diagnostics, artifact=artifact)


def _render(renderer: RendererLike, request: RenderRequest) -> RenderArtifact:
    if hasattr(renderer, "render"):
        artifact = renderer.render(request)  # type: ignore[union-attr]
    else:
        artifact = renderer(request)
    if not isinstance(artifact, RenderArtifact):
        raise TypeError("static renderer must return RenderArtifact")
    return artifact


def _renderer_name(renderer: RendererLike) -> str:
    if hasattr(renderer, "__name__"):
        return str(renderer.__name__)  # type: ignore[attr-defined]
    return renderer.__class__.__name__


def _artifact_output(artifact: RenderArtifact | None) -> tuple[Path | None, int | None]:
    if artifact is None or artifact.path is None:
        return None, None
    if artifact.bytes is not None:
        return artifact.path, artifact.bytes
    try:
        return artifact.path, artifact.path.stat().st_size
    except OSError:
        return artifact.path, None
