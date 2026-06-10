from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .gmr_mujoco import GMRMujocoKinematicPlaybackAdapter
from .isaac_render import IsaacRenderAdapter
from .soma_newton import SOMANewtonDynamicSessionAdapter
from .somamesh_source import SomaMeshSourceRenderAdapter


AdapterStatus = Literal["connected_script", "interface_only"]


@dataclass(frozen=True)
class AdapterDescriptor:
    name: str
    status: AdapterStatus
    kind: str
    description: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "kind": self.kind,
            "description": self.description,
        }


def adapter_descriptors() -> tuple[AdapterDescriptor, ...]:
    return (
        AdapterDescriptor(
            name=IsaacRenderAdapter.adapter_name,
            status="connected_script",
            kind="static_render",
            description="Command-backed wrapper for scripts/render_g1_isaac_pair.py",
        ),
        AdapterDescriptor(
            name=SomaMeshSourceRenderAdapter.adapter_name,
            status="connected_script",
            kind="source_render",
            description="Command-backed wrapper for scripts/render_somamesh_source.py",
        ),
        AdapterDescriptor(
            name=GMRMujocoKinematicPlaybackAdapter.adapter_name,
            status="interface_only",
            kind="kinematic_playback",
            description="Migration point for GMR RobotMotionViewer/MuJoCo bounded playback",
        ),
        AdapterDescriptor(
            name=SOMANewtonDynamicSessionAdapter.adapter_name,
            status="interface_only",
            kind="dynamic_session",
            description="Migration point for SOMA/Newton reset-step-close dynamic sessions",
        ),
    )
