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
    """Return per-env actor morphology features aligned to SONIC future frames.

    ``num_clusters`` controls the source skeleton/morphology bucket count, not actuator grouping.
    """

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


def root_pos_w_mf(
    env: Any,
    command_name: str,
    representation: str = "delta_xy_w_plus_z_w",
    root_body_index: int = 0,
) -> torch.Tensor:
    """Return target G1 root position in the formal root-pose representation.

    The train target is world-frame delta XY from the current root and absolute
    world Z height for each future frame.  If the Sonic command exposes only the
    current ``body_pos_w`` tensor, the function still returns a valid multi-
    future tensor with zero delta XY and the current Z repeated.
    """

    if representation != "delta_xy_w_plus_z_w":
        raise ValueError(f"unsupported root position representation: {representation}")
    command = env.command_manager.get_term(command_name)
    root_pos = _root_body_tensor(
        command,
        ("root_pos_w_mf", "root_pos_w_multi_future", "body_pos_w_multi_future", "body_pos_w"),
        root_body_index=root_body_index,
        width=3,
    )
    if root_pos.shape[-1] != 3:
        raise ValueError(f"root position tensor must end in 3 values, got {tuple(root_pos.shape)}")
    base_xy = root_pos[:, :1, :2]
    delta_xy = root_pos[..., :2] - base_xy
    return torch.cat((delta_xy, root_pos[..., 2:3]), dim=-1)


def root_rot_w_mf(
    env: Any,
    command_name: str,
    rotation_format: str = "rot6d_w",
    root_body_index: int = 0,
) -> torch.Tensor:
    """Return target G1 root rotation as world-frame rot6d future labels."""

    if rotation_format not in {"rot6d_w", "rot6d_w_from_quat_wxyz"}:
        raise ValueError(f"unsupported root rotation representation: {rotation_format}")
    command = env.command_manager.get_term(command_name)
    direct = _command_tensor(command, ("root_rot_w_mf", "root_rot6d_w_mf"))
    if direct is not None:
        return _ensure_future_tensor(direct, command, width=6)

    root_quat = _root_body_tensor(
        command,
        ("root_quat_w_mf", "root_quat_w_multi_future", "body_quat_w_multi_future", "body_quat_w"),
        root_body_index=root_body_index,
        width=4,
    )
    return _quat_wxyz_to_rot6d(root_quat)


def _repeat_future(value: torch.Tensor, command: Any) -> torch.Tensor:
    num_frames = int(getattr(command, "smpl_num_future_frames", command.num_future_frames))
    return value.unsqueeze(1).expand(-1, num_frames, -1)


def _future_frame_count(command: Any) -> int:
    return int(getattr(command, "smpl_num_future_frames", getattr(command, "num_future_frames", 1)))


def _command_tensor(command: Any, names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        value = getattr(command, name, None)
        if value is not None:
            return value
    return None


def _root_body_tensor(
    command: Any,
    names: tuple[str, ...],
    *,
    root_body_index: int,
    width: int,
) -> torch.Tensor:
    value = _command_tensor(command, names)
    if value is None:
        raise AttributeError(f"motion command exposes none of: {', '.join(names)}")
    flat_root = _flat_root_body_tensor(
        value,
        command,
        root_body_index=root_body_index,
        width=width,
    )
    if flat_root is not None:
        return flat_root
    tensor = _ensure_future_tensor(value, command, width=width)
    if tensor.shape[-1] == width and tensor.ndim == 3:
        return tensor
    if tensor.ndim >= 4 and tensor.shape[-1] == width:
        return tensor[..., int(root_body_index), :]
    raise ValueError(f"could not interpret root body tensor shape {tuple(tensor.shape)}")


def _ensure_future_tensor(value: torch.Tensor, command: Any, *, width: int) -> torch.Tensor:
    tensor = value
    if tensor.ndim == 2 and tensor.shape[-1] == width:
        return _repeat_future(tensor, command)
    if tensor.ndim == 2 and tensor.shape[-1] % width == 0:
        num_frames = _future_frame_count(command)
        if num_frames > 0 and tensor.shape[-1] == num_frames * width:
            return tensor.reshape(tensor.shape[0], num_frames, width)
    if tensor.ndim == 3 and tensor.shape[-1] == width:
        # Shape [B, N, C] is ambiguous.  If N looks like a body dimension, callers
        # that need a root body will select it before this function is returned.
        if tensor.shape[1] == _future_frame_count(command):
            return tensor
        return _repeat_future(tensor[:, 0], command)
    if tensor.ndim >= 4 and tensor.shape[-1] == width:
        return tensor
    raise ValueError(f"expected tensor with trailing dim {width}, got {tuple(tensor.shape)}")


def _flat_root_body_tensor(
    value: torch.Tensor,
    command: Any,
    *,
    root_body_index: int,
    width: int,
) -> torch.Tensor | None:
    """Select the root body from flattened SONIC body future tensors."""

    tensor = value
    num_frames = _future_frame_count(command)
    if tensor.ndim == 2 and tensor.shape[-1] != width:
        total = tensor.shape[-1]
        if num_frames > 0 and total % (num_frames * width) == 0:
            num_bodies = total // (num_frames * width)
            return tensor.reshape(tensor.shape[0], num_frames, num_bodies, width)[
                :, :, int(root_body_index), :
            ]
        if total % width == 0:
            num_bodies = total // width
            root_current = tensor.reshape(tensor.shape[0], num_bodies, width)[
                :, int(root_body_index), :
            ]
            return _repeat_future(root_current, command)
    if tensor.ndim == 3 and tensor.shape[-1] != width and tensor.shape[1] == num_frames:
        total = tensor.shape[-1]
        if total % width == 0:
            num_bodies = total // width
            return tensor.reshape(tensor.shape[0], num_frames, num_bodies, width)[
                :, :, int(root_body_index), :
            ]
    return None


def _quat_wxyz_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    row0 = torch.stack(
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        dim=-1,
    )
    row1 = torch.stack(
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        dim=-1,
    )
    row2 = torch.stack(
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        dim=-1,
    )
    matrix = torch.stack((row0, row1, row2), dim=-2)
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


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
