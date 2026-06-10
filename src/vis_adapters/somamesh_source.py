from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping

from vis_core import RenderArtifact, RenderRequest

from .common import (
    AdapterPreflight,
    CommandSpec,
    output_path,
    resolve_packet_path,
    run_command_spec,
    script_path,
)


@dataclass(frozen=True)
class SomaMeshSourceRenderAdapter:
    """Command-backed adapter for the accepted SomaMesh source renderer."""

    python: str = sys.executable
    script: Path = field(default_factory=lambda: script_path("scripts", "render_somamesh_source.py"))
    output_name: str = "somamesh_source.mp4"
    timeout_sec: float | None = None
    extra_args: tuple[str, ...] = ()

    adapter_name: str = "somamesh_source_render"

    def preflight(self, request: RenderRequest) -> AdapterPreflight:
        reasons: list[str] = []
        if not self.script.exists():
            reasons.append("somamesh_source_script_missing")
        source_bvh = _source_bvh(request.loaded_packet.packet.render.config, request)
        if source_bvh is None:
            reasons.append("source_bvh_uri_missing")
        return AdapterPreflight(
            adapter=self.adapter_name,
            status="blocked" if reasons else "ok",
            connected=True,
            executable=not reasons,
            reasons=tuple(reasons),
            details={
                "script": str(self.script),
                "source_bvh": str(source_bvh) if source_bvh is not None else None,
                "optional_backend": "SomaMesh LBS",
            },
        )

    def build_command(self, request: RenderRequest) -> CommandSpec:
        preflight = self.preflight(request)
        if not preflight.ok:
            raise ValueError(f"{self.adapter_name} preflight blocked: {preflight.reasons}")
        packet = request.loaded_packet.packet
        config = dict(packet.render.config)
        out_path = output_path(request.output_dir, self.output_name)
        argv = [
            self.python,
            str(self.script),
            "--bvh",
            str(preflight.details["source_bvh"]),
            "--output",
            str(out_path),
            "--fps",
            str(packet.timeline.fps),
            "--frame-count",
            str(packet.timeline.frame_count),
        ]
        _extend_optional_path_arg(argv, "--retargeter-root", config.get("somamesh_retargeter_root"))
        _extend_optional_path_arg(argv, "--soma-usd", config.get("somamesh_usd"))
        _extend_optional_scalar_arg(argv, "--width", config.get("width"))
        _extend_optional_scalar_arg(argv, "--height", config.get("height"))
        _extend_optional_scalar_arg(argv, "--stride-triangles", config.get("stride_triangles"))
        argv.extend(str(arg) for arg in self.extra_args)
        return CommandSpec(
            adapter=self.adapter_name,
            argv=tuple(argv),
            cwd=self.script.parents[1],
            output_path=out_path,
            report_path=out_path.with_suffix(".json"),
            connected_script=self.script,
        )

    def render(self, request: RenderRequest) -> RenderArtifact:
        return run_command_spec(self.build_command(request), timeout_sec=self.timeout_sec)


def _source_bvh(config: Mapping[str, Any], request: RenderRequest) -> Path | None:
    raw = config.get("source_bvh_uri") or config.get("source_bvh")
    if isinstance(raw, str) and raw.strip():
        return resolve_packet_path(raw, base_dir=request.loaded_packet.packet.base_dir)
    return None


def _extend_optional_path_arg(argv: list[str], flag: str, raw: Any) -> None:
    if isinstance(raw, str) and raw.strip():
        argv.extend([flag, raw])


def _extend_optional_scalar_arg(argv: list[str], flag: str, raw: Any) -> None:
    if raw is not None:
        argv.extend([flag, str(raw)])
