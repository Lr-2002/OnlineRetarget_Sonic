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
class IsaacRenderAdapter:
    """Command-backed adapter for the existing Isaac G1 render script."""

    python: str = sys.executable
    script: Path = field(default_factory=lambda: script_path("scripts", "render_g1_isaac_pair.py"))
    output_name: str = "isaac_render.mp4"
    timeout_sec: float | None = None
    extra_args: tuple[str, ...] = ()

    adapter_name: str = "isaac_render"

    def preflight(self, request: RenderRequest) -> AdapterPreflight:
        reasons: list[str] = []
        if not self.script.exists():
            reasons.append("isaac_render_script_missing")
        motion_track = _g1_motion_track(request)
        if motion_track is None:
            reasons.append("no_csv_g1_motion_track")
        return AdapterPreflight(
            adapter=self.adapter_name,
            status="blocked" if reasons else "ok",
            connected=True,
            executable=not reasons,
            reasons=tuple(reasons),
            details={
                "script": str(self.script),
                "motion_role": motion_track,
                "optional_backend": "Isaac Lab",
            },
        )

    def build_command(self, request: RenderRequest) -> CommandSpec:
        preflight = self.preflight(request)
        if not preflight.ok:
            raise ValueError(f"{self.adapter_name} preflight blocked: {preflight.reasons}")
        packet = request.loaded_packet.packet
        motion_role = str(preflight.details["motion_role"])
        motion_path = request.loaded_packet.tracks[motion_role].source_path
        out_path = output_path(request.output_dir, self.output_name)
        config = dict(packet.render.config)

        argv = [
            self.python,
            str(self.script),
            "--g1-motion",
            str(motion_path),
            "--format",
            "csv",
            "--output",
            str(out_path),
            "--target-fps",
            str(packet.timeline.fps),
            "--duration-sec",
            str(packet.timeline.duration_sec),
            "--max-frames",
            str(packet.timeline.frame_count),
            "--fast-exit-after-report",
        ]
        source_bvh = _optional_source_bvh(config, base_dir=packet.base_dir)
        if source_bvh is not None:
            argv.extend(["--bvh", str(source_bvh)])
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


def _g1_motion_track(request: RenderRequest) -> str | None:
    for role in ("play_g1", "target_g1"):
        track = request.loaded_packet.tracks.get(role)
        if track is not None and track.spec.format == "csv":
            return role
    return None


def _optional_source_bvh(config: Mapping[str, Any], *, base_dir: Path | None) -> Path | None:
    raw = config.get("source_bvh_uri") or config.get("source_bvh")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return resolve_packet_path(raw, base_dir=base_dir)
