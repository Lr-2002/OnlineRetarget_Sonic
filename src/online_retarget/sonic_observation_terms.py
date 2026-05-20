"""Observation terms injected into SONIC for OnlineRetarget runs.

These functions are Hydra-compatible ``ObservationTermCfg.func`` targets. They
avoid importing Isaac Lab at module import time so the OnlineRetarget test suite
can still import them outside the simulator environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
import os
import re
from typing import Any

import torch

from .sonic_morphology import MORPHOLOGY_VECTOR_DIM, load_morphology_table


CONTACT_PHASE_DIM = 3
_ACTOR_PATTERN = re.compile(r"A\d{3,}")


def soma_morphology(
    env: Any,
    command_name: str,
    registry_csv: str | None = None,
    num_clusters: int = 4,
    non_flatten: bool = True,
    strict: bool = True,
) -> torch.Tensor:
    """Return per-env actor morphology features aligned to SONIC future frames."""

    command = env.command_manager.get_term(command_name)
    registry = _resolve_registry_path(registry_csv)
    table = _cached_table(registry) if registry else {}
    vectors = []
    for motion_key in _motion_keys_for_envs(command):
        actor_uid = _actor_uid_from_motion_key(motion_key)
        morphology = table.get(actor_uid) if actor_uid else None
        if morphology is None:
            if strict:
                raise KeyError(
                    f"no morphology row for motion_key={motion_key!r} actor_uid={actor_uid!r}"
                )
            vectors.append([0.0] * MORPHOLOGY_VECTOR_DIM)
        else:
            vectors.append(list(morphology.as_vector(num_clusters=num_clusters)))

    result = torch.tensor(vectors, dtype=torch.float32, device=command.device)
    return _repeat_future(result, command) if non_flatten else result


def soma_contact_phase(
    env: Any,
    command_name: str,
    non_flatten: bool = True,
) -> torch.Tensor:
    """Return optional source contact/phase features: contact flag, sin phase, cos phase."""

    command = env.command_manager.get_term(command_name)
    values = torch.zeros(command.num_envs, CONTACT_PHASE_DIM, device=command.device)

    denom = torch.clamp(command.motion_num_steps.float(), min=1.0)
    current_steps = (command.motion_start_time_steps + command.time_steps).float()
    phase = torch.clamp(current_steps / denom, 0.0, 1.0)
    values[:, 1] = torch.sin(phase * (2.0 * math.pi))
    values[:, 2] = torch.cos(phase * (2.0 * math.pi))

    contact_flags = getattr(command, "_motion_contact_flags", None)
    if contact_flags:
        motion_ids = command.motion_ids.detach().cpu().tolist()
        time_steps = (command.motion_start_time_steps + command.time_steps).detach().cpu().tolist()
        contacts = []
        for motion_id, time_step in zip(motion_ids, time_steps, strict=False):
            motion_contact = contact_flags.get(int(motion_id))
            if motion_contact is None or len(motion_contact) == 0:
                contacts.append(0.0)
                continue
            index = min(max(int(time_step), 0), len(motion_contact) - 1)
            contacts.append(float(motion_contact[index].item()))
        values[:, 0] = torch.tensor(contacts, dtype=torch.float32, device=command.device)

    return _repeat_future(values, command) if non_flatten else values


def g1_target_action(env: Any, command_name: str) -> torch.Tensor:
    """Return the current G1 target joint action in Sonic action-normalized space."""

    command = env.command_manager.get_term(command_name)
    action_manager = env.action_manager.get_term("joint_pos")
    action_offset = action_manager._offset  # noqa: SLF001
    action_scale = action_manager._scale  # noqa: SLF001
    action_joint_pos = command.joint_pos[:, action_manager._joint_ids]  # noqa: SLF001
    return (action_joint_pos - action_offset) / action_scale


def _repeat_future(value: torch.Tensor, command: Any) -> torch.Tensor:
    num_frames = int(getattr(command, "smpl_num_future_frames", command.num_future_frames))
    return value.unsqueeze(1).expand(-1, num_frames, -1)


def _resolve_registry_path(registry_csv: str | None) -> str | None:
    path = registry_csv or os.environ.get("ONLINE_RETARGET_SKELETON_REGISTRY")
    if not path:
        return None
    return str(Path(path).expanduser())


@lru_cache(maxsize=8)
def _cached_table(registry_csv: str):
    return load_morphology_table(registry_csv)


def _motion_keys_for_envs(command: Any) -> list[str]:
    keys = getattr(command.motion_lib, "curr_motion_keys", None)
    if keys is None:
        keys = getattr(command.motion_lib, "_motion_data_keys", None)
    if keys is None:
        return [""] * command.num_envs
    motion_ids = command.motion_ids.detach().cpu().tolist()
    return [str(keys[int(motion_id)]) for motion_id in motion_ids]


def _actor_uid_from_motion_key(motion_key: str) -> str | None:
    match = _ACTOR_PATTERN.search(Path(motion_key).stem)
    return match.group(0) if match else None
