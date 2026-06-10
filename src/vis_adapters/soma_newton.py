from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping

from vis_core import LoadedVisPacket

from .common import AdapterPreflight, CommandSpec, sibling_repo_path


@dataclass(frozen=True)
class SOMANewtonDynamicSessionAdapter:
    """Interface-only migration point for SOMA/Newton dynamic sessions."""

    python: str = sys.executable
    soma_repo: Path = field(default_factory=lambda: sibling_repo_path("SOMA-online"))
    protocol: str = "tcp"
    listen_host: str = "0.0.0.0"
    port: int = 7003

    adapter_name: str = "soma_newton_dynamic_session"

    @property
    def threaded_module(self) -> str:
        return "app.threaded_online_bvh_retarget"

    @property
    def recorder_script(self) -> Path:
        return self.soma_repo / "app" / "record_online_retarget_output.py"

    def preflight(self, loaded_packet: LoadedVisPacket | None = None) -> AdapterPreflight:
        reasons = ["interface_only_external_runtime_session"]
        if not self.soma_repo.exists():
            reasons.append("soma_online_repo_missing")
        if not self.recorder_script.exists():
            reasons.append("soma_newton_recorder_script_missing")
        return AdapterPreflight(
            adapter=self.adapter_name,
            status="interface_only",
            connected=False,
            executable=False,
            reasons=tuple(reasons),
            details={
                "soma_repo": str(self.soma_repo),
                "threaded_module": self.threaded_module,
                "recorder_script": str(self.recorder_script),
                "optional_backend": "SOMA/Newton",
                "migration_target": "bounded reset/step/close wrapper around NewtonOnlinePipeline",
            },
        )

    def reset(self, loaded_packet: LoadedVisPacket) -> Mapping[str, Any]:
        return self.preflight(loaded_packet).as_dict()

    def step(self, action: Mapping[str, Any], *, dt: float) -> Mapping[str, Any]:
        raise NotImplementedError(
            "SOMA/Newton dynamic step is interface-only in Phase 2; "
            "wire a bounded NewtonOnlinePipeline session before executing policy feedback loops."
        )

    def close(self) -> None:
        return None

    def recorder_command_plan(self, *, output_csv: Path) -> CommandSpec:
        return CommandSpec(
            adapter=self.adapter_name,
            argv=(
                self.python,
                str(self.recorder_script),
                "--protocol",
                self.protocol,
                "--listen_host",
                self.listen_host,
                "--port",
                str(self.port),
                "--out",
                str(output_csv),
            ),
            cwd=self.soma_repo,
            output_path=output_csv,
            connected_script=self.recorder_script,
            interface_only=True,
        )
