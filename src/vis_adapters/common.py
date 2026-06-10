from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess
from time import perf_counter
from typing import Any, Mapping

from vis_core import RenderArtifact


@dataclass(frozen=True)
class AdapterPreflight:
    adapter: str
    status: str
    connected: bool
    executable: bool
    reasons: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "status": self.status,
            "connected": self.connected,
            "executable": self.executable,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CommandSpec:
    adapter: str
    argv: tuple[str, ...]
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    output_path: Path | None = None
    report_path: Path | None = None
    connected_script: Path | None = None
    interface_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "argv": list(self.argv),
            "cwd": str(self.cwd) if self.cwd is not None else None,
            "env": dict(self.env),
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "report_path": str(self.report_path) if self.report_path is not None else None,
            "connected_script": (
                str(self.connected_script) if self.connected_script is not None else None
            ),
            "interface_only": self.interface_only,
        }


class AdapterExecutionError(RuntimeError):
    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        self.diagnostics = dict(diagnostics)
        super().__init__(message)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return repo_root().parent


def script_path(*parts: str) -> Path:
    return repo_root().joinpath(*parts)


def sibling_repo_path(name: str) -> Path:
    return workspace_root() / name


def resolve_packet_path(raw: str | Path, *, base_dir: Path | None) -> Path:
    path = Path(raw)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def output_path(output_dir: Path | None, filename: str) -> Path:
    root = output_dir if output_dir is not None else Path.cwd()
    return root / filename


def run_command_spec(spec: CommandSpec, *, timeout_sec: float | None = None) -> RenderArtifact:
    started = perf_counter()
    env = os.environ.copy()
    env.update(spec.env)
    result = subprocess.run(
        spec.argv,
        cwd=str(spec.cwd) if spec.cwd is not None else None,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    elapsed_sec = perf_counter() - started
    diagnostics: dict[str, Any] = {
        "adapter": spec.adapter,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_sec": elapsed_sec,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": spec.as_dict(),
    }
    if result.returncode != 0:
        raise AdapterExecutionError(
            f"{spec.adapter} command failed with return code {result.returncode}",
            diagnostics,
        )
    output_bytes = None
    if spec.output_path is not None and spec.output_path.exists():
        output_bytes = spec.output_path.stat().st_size
    return RenderArtifact(
        path=spec.output_path,
        bytes=output_bytes,
        diagnostics=diagnostics,
    )
