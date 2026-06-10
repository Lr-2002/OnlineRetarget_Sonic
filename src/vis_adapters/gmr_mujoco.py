from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

from vis_core import RenderRequest

from .common import AdapterPreflight, CommandSpec, output_path, sibling_repo_path


@dataclass(frozen=True)
class GMRMujocoKinematicPlaybackAdapter:
    """Interface-only migration point for GMR RobotMotionViewer playback.

    The current upstream GMR script runs an unbounded viewer loop, so Phase 2
    records the adapter boundary and command plan without making it executable.
    """

    python: str = sys.executable
    gmr_repo: Path = field(default_factory=lambda: sibling_repo_path("GMR_test"))
    output_name: str = "gmr_mujoco_playback.mp4"

    adapter_name: str = "gmr_mujoco_kinematic_playback"

    @property
    def script(self) -> Path:
        return self.gmr_repo / "scripts" / "vis_robot_motion.py"

    def preflight(self, request: RenderRequest | None = None) -> AdapterPreflight:
        reasons = ["interface_only_unbounded_viewer_loop"]
        if not self.script.exists():
            reasons.append("gmr_vis_robot_motion_script_missing")
        return AdapterPreflight(
            adapter=self.adapter_name,
            status="interface_only",
            connected=False,
            executable=False,
            reasons=tuple(reasons),
            details={
                "script": str(self.script),
                "optional_backend": "GMR RobotMotionViewer + MuJoCo",
                "migration_target": "bounded render/playback entry point around RobotMotionViewer.step",
            },
        )

    def command_plan(self, request: RenderRequest, *, robot_motion_path: Path | None = None) -> CommandSpec:
        out_path = output_path(request.output_dir, self.output_name)
        motion_path = robot_motion_path or request.loaded_packet.tracks["play_g1"].source_path
        return CommandSpec(
            adapter=self.adapter_name,
            argv=(
                self.python,
                str(self.script),
                "--robot_motion_path",
                str(motion_path),
                "--record_video",
                "--video_path",
                str(out_path),
            ),
            cwd=self.gmr_repo,
            output_path=out_path,
            connected_script=self.script,
            interface_only=True,
        )
