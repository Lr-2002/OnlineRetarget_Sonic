#!/usr/bin/env python3
"""Train skeleton-conditioned kin-only retargeting MLPs.

This is a supervised, simulator-free training lane.  It predicts the same
kinematic target fields used by SONIC's ``g1_kin`` decoder, with an opt-in root
pose target for OnlineRetarget diagnostics:

* ``command_multi_future_nonflat``: 29 joint positions + 29 joint velocities
* ``root_pos_w_mf``: G1 root position per future frame when enabled
* ``root_rot_w_mf`` or legacy ``motion_anchor_ori_b_mf_nonflat``: 6D root orientation
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online_retarget.data.skeleton_ae_registry import SKELETON_GEOMETRY_DIM  # noqa: E402
from online_retarget.models.skeleton_geometry_ae import (  # noqa: E402
    SKELETON_GEOMETRY_AE_ARCHITECTURE,
    SKELETON_GEOMETRY_LATENT_DIM,
    load_skeleton_geometry_ae_checkpoint,
    load_skeleton_geometry_ae_stats,
)


REQUIRED_NPZ_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
REQUIRED_ROBOT_MOTIONLIB_KEYS = ("dof", "root_rot", "fps")
REQUIRED_SOMA_MOTIONLIB_KEYS = ("soma_joints", "soma_root_quat", "fps")
SOMA_JOINT_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandMiddle1",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandMiddle1",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)
SOMA_CAPSULE_EDGES = (
    ("Hips", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Chest"),
    ("Chest", "Neck1"),
    ("Neck1", "Head"),
    ("Chest", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("LeftHand", "LeftHandThumb1"),
    ("LeftHand", "LeftHandMiddle1"),
    ("Chest", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("RightHand", "RightHandThumb1"),
    ("RightHand", "RightHandMiddle1"),
    ("Hips", "LeftLeg"),
    ("LeftLeg", "LeftShin"),
    ("LeftShin", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightLeg"),
    ("RightLeg", "RightShin"),
    ("RightShin", "RightFoot"),
    ("RightFoot", "RightToeBase"),
)
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "owner",
    "purpose",
    "source_repo",
    "source_rev",
    "input_data",
    "output_dir",
    "validation_command",
    "expected_artifacts",
    "variant",
    "features",
    "model",
    "training",
    "runtime",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp_compact() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stats(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def read_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS - set(data))
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")
    return data


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_status_short(root: Path) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ["not-a-git-worktree"]
    return out.splitlines()[:200]


def git_has_tracked_changes(root: Path) -> bool:
    checks = (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    )
    for cmd in checks:
        result = subprocess.run(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return True
    return False


def require_latest_git(root: Path, label: str) -> None:
    fetch_timeout = float(os.environ.get("GIT_FETCH_TIMEOUT_SECONDS", "60"))
    try:
        upstream = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        raise RuntimeError(f"{label} has no upstream tracking branch: {root}") from exc
    if "/" not in upstream:
        raise RuntimeError(f"{label} has unsupported upstream {upstream!r}: {root}")
    remote, branch = upstream.split("/", 1)
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", remote, branch],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=fetch_timeout,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{label} could not fetch {upstream}; refusing to train without a latest-code check"
        ) from exc

    head = git_revision(root)
    upstream_head = subprocess.check_output(
        ["git", "rev-parse", "FETCH_HEAD"],
        cwd=root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if head != upstream_head:
        raise RuntimeError(f"{label} is not latest: HEAD={head}, {upstream}={upstream_head}")


def resolve_text(value: str, config: dict[str, Any], run_group: str) -> str:
    return (
        value.replace("{run_group}", run_group)
        .replace("{variant}", str(config["variant"]["name"]))
        .replace("{timestamp}", timestamp_compact())
    )


def resolve_output_dir(config: dict[str, Any], run_group: str) -> Path:
    return Path(resolve_text(str(config["output_dir"]), config, run_group))


class StageTracer:
    def __init__(self, *, enabled: bool, env: Mapping[str, int], reason: str) -> None:
        self.enabled = enabled
        self.rank = int(env.get("rank", 0))
        self.world_size = int(env.get("world_size", 1))
        self.local_rank = int(env.get("local_rank", self.rank))
        self.reason = reason
        self.started = time.perf_counter()
        self.log_path: Path | None = None
        self._pending_lines: list[str] = []

    def update_runtime(self, runtime: Mapping[str, Any]) -> None:
        self.rank = int(runtime.get("rank", self.rank))
        self.world_size = int(runtime.get("world_size", self.world_size))
        self.local_rank = int(runtime.get("local_rank", self.local_rank))

    def attach(self, output_dir: Path) -> None:
        if not self.enabled:
            return
        log_dir = output_dir / "logs" / "a0_stage_trace"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"rank_{self.rank:04d}_local_{self.local_rank:04d}.jsonl"
        for line in self._pending_lines:
            self._write_line(line)
        self._pending_lines.clear()
        self.log("stage_trace_file", "attached", path=str(self.log_path))

    def log(self, stage: str, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "ts": utc_now(),
            "elapsed_sec": round(time.perf_counter() - self.started, 6),
            "pid": os.getpid(),
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "reason": self.reason,
            "stage": stage,
            "event": event,
            **fields,
        }
        line = json.dumps(payload, sort_keys=True, default=str)
        print(f"A0_STAGE_TRACE {line}", flush=True)
        if self.log_path is None:
            self._pending_lines.append(line)
        else:
            self._write_line(line)

    @contextmanager
    def span(self, stage: str, **fields: Any) -> Any:
        self.log(stage, "before", **fields)
        try:
            yield
        except BaseException as exc:
            self.log(stage, "error", error_type=type(exc).__name__, error=repr(exc))
            raise
        else:
            self.log(stage, "after")

    def _write_line(self, line: str) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def env_flag_enabled(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean env flag value, got {value!r}")


def should_enable_a0_stage_trace(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    env_enabled = env_flag_enabled(os.environ.get("A0_STAGE_TRACE"))
    if env_enabled is not None:
        return env_enabled, "A0_STAGE_TRACE"
    diagnostics = config.get("diagnostics", {})
    if isinstance(diagnostics, Mapping) and "a0_stage_trace" in diagnostics:
        return bool(diagnostics["a0_stage_trace"]), "config.diagnostics.a0_stage_trace"
    if bool(getattr(args, "stage_trace", False)):
        return True, "--stage-trace"
    if bool(getattr(args, "index_only", False)):
        return True, "--index-only"
    return bool(args.dry_run and is_skeleton_ae_enabled(config)), "a0_dry_run_default"


def should_run_a0_ddp_probe(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    stage_trace: StageTracer,
) -> tuple[bool, str]:
    env_enabled = env_flag_enabled(os.environ.get("A0_DDP_PROBE"))
    if env_enabled is not None:
        return env_enabled, "A0_DDP_PROBE"
    diagnostics = config.get("diagnostics", {})
    if isinstance(diagnostics, Mapping) and "a0_ddp_probe" in diagnostics:
        return bool(diagnostics["a0_ddp_probe"]), "config.diagnostics.a0_ddp_probe"
    return bool(args.dry_run and stage_trace.enabled and is_skeleton_ae_enabled(config)), "a0_stage_trace_default"


def should_run_a0_ddp_probe_only(config: Mapping[str, Any]) -> tuple[bool, str]:
    env_enabled = env_flag_enabled(os.environ.get("A0_DDP_PROBE_ONLY"))
    if env_enabled is not None:
        return env_enabled, "A0_DDP_PROBE_ONLY"
    diagnostics = config.get("diagnostics", {})
    if isinstance(diagnostics, Mapping) and "a0_ddp_probe_only" in diagnostics:
        return bool(diagnostics["a0_ddp_probe_only"]), "config.diagnostics.a0_ddp_probe_only"
    return False, "default"


def a0_ddp_broadcast_buffers(config: Mapping[str, Any]) -> tuple[bool, str]:
    env_enabled = env_flag_enabled(os.environ.get("A0_DDP_BROADCAST_BUFFERS"))
    if env_enabled is not None:
        return env_enabled, "A0_DDP_BROADCAST_BUFFERS"
    diagnostics = config.get("diagnostics", {})
    if isinstance(diagnostics, Mapping) and "a0_ddp_broadcast_buffers" in diagnostics:
        return bool(diagnostics["a0_ddp_broadcast_buffers"]), "config.diagnostics.a0_ddp_broadcast_buffers"
    if is_skeleton_ae_enabled(config):
        return False, "a0_default"
    return True, "torch_default"


def _ddp_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    ddp = config.get("ddp", {})
    return ddp if isinstance(ddp, Mapping) else {}


def a0_ddp_init_sync(config: Mapping[str, Any]) -> tuple[bool | None, str]:
    env_enabled = env_flag_enabled(os.environ.get("A0_DDP_INIT_SYNC"))
    if env_enabled is not None:
        return env_enabled, "A0_DDP_INIT_SYNC"
    ddp = _ddp_mapping(config)
    if "init_sync" in ddp:
        return bool(ddp["init_sync"]), "config.ddp.init_sync"
    return None, "unset"


def should_run_a0_ddp_probe_backward(config: Mapping[str, Any], *, probe_only: bool) -> tuple[bool, str]:
    env_enabled = env_flag_enabled(os.environ.get("A0_DDP_PROBE_BACKWARD"))
    if env_enabled is not None:
        return env_enabled, "A0_DDP_PROBE_BACKWARD"
    diagnostics = _diagnostics_mapping(config)
    if "a0_ddp_probe_backward" in diagnostics:
        return bool(diagnostics["a0_ddp_probe_backward"]), "config.diagnostics.a0_ddp_probe_backward"
    if probe_only:
        return True, "a0_ddp_probe_only_default"
    return False, "default"


def module_parameter_report(model: nn.Module) -> dict[str, Any]:
    hasher = hashlib.sha256()
    parameter_numel = 0
    trainable_parameter_numel = 0
    parameter_abs_sum = 0.0
    parameter_l2_sq = 0.0
    parameters = []
    for name, parameter in model.named_parameters():
        detached = parameter.detach().cpu().float().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(detached.shape)).encode("utf-8"))
        hasher.update(str(detached.dtype).encode("utf-8"))
        hasher.update(detached.numpy().tobytes())
        numel = int(detached.numel())
        parameter_numel += numel
        if parameter.requires_grad:
            trainable_parameter_numel += numel
        parameter_abs_sum += float(detached.abs().sum().item())
        parameter_l2_sq += float(detached.pow(2).sum().item())
        parameters.append(
            {
                "name": name,
                "shape": list(detached.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "requires_grad": bool(parameter.requires_grad),
                "numel": numel,
            }
        )
    return {
        "parameter_sha256": hasher.hexdigest(),
        "parameter_count": len(parameters),
        "parameter_numel": parameter_numel,
        "trainable_parameter_numel": trainable_parameter_numel,
        "parameter_abs_sum": parameter_abs_sum,
        "parameter_l2": math.sqrt(parameter_l2_sq),
        "parameters": parameters,
    }


def verify_rank_parameter_report(
    *,
    output_dir: Path,
    runtime: Mapping[str, Any],
    stage_trace: StageTracer,
    stage_name: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    rank = int(runtime.get("rank", 0))
    world_size = int(runtime.get("world_size", 1))
    checksum_dir = output_dir / "logs" / "a0_parameter_checksums"
    checksum_dir.mkdir(parents=True, exist_ok=True)
    payload = {"rank": rank, "world_size": world_size, **dict(report)}
    path = checksum_dir / f"{stage_name}_rank_{rank:04d}.json"
    tmp_path = checksum_dir / f".{stage_name}_rank_{rank:04d}.{os.getpid()}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)

    distributed_barrier(runtime)
    reports = []
    missing = []
    for expected_rank in range(world_size):
        expected_path = checksum_dir / f"{stage_name}_rank_{expected_rank:04d}.json"
        if not expected_path.exists():
            missing.append(str(expected_path))
            continue
        reports.append(json.loads(expected_path.read_text(encoding="utf-8")))
    checksums = sorted({str(item.get("parameter_sha256")) for item in reports})
    summary = {
        "checksum_dir": str(checksum_dir),
        "rank_reports": reports,
        "missing_rank_report_paths": missing,
        "all_rank_parameter_checksums": checksums,
        "all_rank_parameter_checksums_equal": not missing and len(checksums) == 1,
    }
    stage_trace.log(f"{stage_name}_all_rank_parameter_checksums", "details", **summary)
    if missing:
        raise RuntimeError(f"missing parameter checksum report(s): {missing}")
    if len(checksums) != 1:
        raise RuntimeError(f"rank parameter checksums differ: {checksums}")
    distributed_barrier(runtime)
    return summary


def module_gradient_report(model: nn.Module) -> dict[str, Any]:
    hasher = hashlib.sha256()
    gradient_numel = 0
    gradient_abs_sum = 0.0
    gradient_l2_sq = 0.0
    missing_gradients = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_gradients.append(name)
            continue
        detached = parameter.grad.detach().cpu().float().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(detached.shape)).encode("utf-8"))
        hasher.update(str(detached.dtype).encode("utf-8"))
        hasher.update(detached.numpy().tobytes())
        gradient_numel += int(detached.numel())
        gradient_abs_sum += float(detached.abs().sum().item())
        gradient_l2_sq += float(detached.pow(2).sum().item())
    return {
        "gradient_sha256": hasher.hexdigest(),
        "gradient_numel": gradient_numel,
        "gradient_abs_sum": gradient_abs_sum,
        "gradient_l2": math.sqrt(gradient_l2_sq),
        "missing_gradient_names": missing_gradients,
    }


def tensor_stage_snapshot(kind: str, name: str, tensor: torch.Tensor, requires_grad: bool | None) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "requires_grad": requires_grad,
        "numel": int(tensor.numel()),
    }


def module_ddp_preflight_snapshot(model: nn.Module) -> dict[str, Any]:
    named_parameters = [
        tensor_stage_snapshot("parameter", name, parameter, bool(parameter.requires_grad))
        for name, parameter in model.named_parameters()
    ]
    named_buffers = [
        tensor_stage_snapshot("buffer", name, buffer, None)
        for name, buffer in model.named_buffers()
    ]
    named_tensors = named_parameters + named_buffers
    encoder_name_markers = ("skeleton_ae", "skeleton_encoder", "encoder")
    contains_encoder_named_tensor = any(
        any(marker in item["name"] for marker in encoder_name_markers)
        for item in named_tensors
    )
    contains_skeleton_encoder_params = any(
        any(marker in item["name"] for marker in encoder_name_markers)
        for item in named_parameters
    )
    contains_frozen_encoder_parameter = any(
        any(marker in item["name"] for marker in encoder_name_markers) and not bool(item["requires_grad"])
        for item in named_parameters
    )
    return {
        "module_type": type(model).__name__,
        "parameter_count": len(named_parameters),
        "buffer_count": len(named_buffers),
        "parameter_numel": int(sum(item["numel"] for item in named_parameters)),
        "buffer_numel": int(sum(item["numel"] for item in named_buffers)),
        "trainable_parameter_numel": int(
            sum(item["numel"] for item in named_parameters if item["requires_grad"])
        ),
        "frozen_parameter_numel": int(sum(item["numel"] for item in named_parameters if not item["requires_grad"])),
        "contains_encoder_named_tensor": contains_encoder_named_tensor,
        "contains_skeleton_encoder_params": contains_skeleton_encoder_params,
        "contains_frozen_encoder_parameter": contains_frozen_encoder_parameter,
        "named_parameters": named_parameters,
        "named_buffers": named_buffers,
    }


def ddp_constructor_kwargs(
    runtime: Mapping[str, Any],
    device: torch.device,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    broadcast_buffers, broadcast_source = a0_ddp_broadcast_buffers(config)
    init_sync, init_sync_source = a0_ddp_init_sync(config)
    if device.type == "cuda":
        local_rank = int(runtime["local_rank"])
        kwargs: dict[str, Any] = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "broadcast_buffers": broadcast_buffers,
        }
    else:
        kwargs = {"broadcast_buffers": broadcast_buffers}
    if init_sync is not None:
        kwargs["init_sync"] = init_sync
    serializable = {
        "device_ids": kwargs.get("device_ids"),
        "output_device": kwargs.get("output_device"),
        "broadcast_buffers": bool(kwargs["broadcast_buffers"]),
        "broadcast_buffers_source": broadcast_source,
    }
    if init_sync is not None:
        serializable["init_sync"] = bool(init_sync)
        serializable["init_sync_source"] = init_sync_source
    supported = _supported_ddp_kwargs()
    if supported and "init_sync" in kwargs and "init_sync" not in supported:
        if int(runtime.get("world_size", 1)) > 1:
            raise RuntimeError("config requested DDP init_sync, but this torch DistributedDataParallel lacks init_sync")
        kwargs.pop("init_sync", None)
        serializable["unsupported_kwargs"] = ["init_sync"]
    return kwargs, serializable


def _diagnostics_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = config.get("diagnostics", {})
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _optional_bool_control(
    config: Mapping[str, Any],
    *,
    env_name: str,
    diagnostics_key: str,
) -> tuple[bool | None, str]:
    env_enabled = env_flag_enabled(os.environ.get(env_name))
    if env_enabled is not None:
        return env_enabled, env_name
    diagnostics = _diagnostics_mapping(config)
    if diagnostics_key in diagnostics:
        return bool(diagnostics[diagnostics_key]), f"config.diagnostics.{diagnostics_key}"
    return None, "unset"


def _optional_float_control(
    config: Mapping[str, Any],
    *,
    env_name: str,
    diagnostics_key: str,
) -> tuple[float | None, str]:
    raw = os.environ.get(env_name)
    if raw not in (None, ""):
        return float(raw), env_name
    diagnostics = _diagnostics_mapping(config)
    value = diagnostics.get(diagnostics_key)
    if value not in (None, ""):
        return float(value), f"config.diagnostics.{diagnostics_key}"
    return None, "unset"


def _supported_ddp_kwargs() -> set[str]:
    try:
        return set(inspect.signature(nn.parallel.DistributedDataParallel).parameters)
    except (TypeError, ValueError):
        return set()


def ddp_probe_constructor_kwargs(
    runtime: Mapping[str, Any],
    device: torch.device,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs, serializable = ddp_constructor_kwargs(runtime, device, config)
    optional_sources: dict[str, str] = {}

    bucket_cap_mb, bucket_source = _optional_float_control(
        config,
        env_name="A0_DDP_PROBE_BUCKET_CAP_MB",
        diagnostics_key="a0_ddp_probe_bucket_cap_mb",
    )
    if bucket_cap_mb is not None:
        bucket_value = int(bucket_cap_mb) if bucket_cap_mb.is_integer() else bucket_cap_mb
        kwargs["bucket_cap_mb"] = bucket_value
        serializable["bucket_cap_mb"] = bucket_value
        optional_sources["bucket_cap_mb"] = bucket_source

    for kwarg, env_name, diagnostics_key in (
        ("init_sync", "A0_DDP_PROBE_INIT_SYNC", "a0_ddp_probe_init_sync"),
        ("static_graph", "A0_DDP_PROBE_STATIC_GRAPH", "a0_ddp_probe_static_graph"),
        (
            "find_unused_parameters",
            "A0_DDP_PROBE_FIND_UNUSED_PARAMETERS",
            "a0_ddp_probe_find_unused_parameters",
        ),
    ):
        value, source = _optional_bool_control(config, env_name=env_name, diagnostics_key=diagnostics_key)
        if value is not None:
            kwargs[kwarg] = value
            serializable[kwarg] = value
            serializable[f"{kwarg}_source"] = source
            optional_sources[kwarg] = source

    supported = _supported_ddp_kwargs()
    unsupported = []
    if supported:
        for key in list(kwargs):
            if key not in supported:
                unsupported.append(key)
                kwargs.pop(key)
                serializable.pop(key, None)
                serializable.pop(f"{key}_source", None)
                optional_sources.pop(key, None)
    if optional_sources:
        serializable["optional_sources"] = optional_sources
    if unsupported:
        serializable["unsupported_probe_kwargs"] = unsupported
    return kwargs, serializable


def a0_probe_expected_dims(config: Mapping[str, Any]) -> tuple[int, int, int]:
    expected_dims = config.get("features", {}).get("expected_dims", {})
    if not isinstance(expected_dims, Mapping):
        raise ValueError("features.expected_dims is required for A0 DDP probe-only mode")
    motion_dim = int(expected_dims["motion_token"])
    skeleton_dim = int(expected_dims["z_skel"])
    target_dim = int(expected_dims["target"])
    return motion_dim, skeleton_dim, target_dim


def run_ddp_probe_model(
    *,
    stage_name: str,
    model: nn.Module,
    input_factory: Callable[[torch.device], tuple[torch.Tensor, ...]],
    stage_trace: StageTracer,
    runtime: Mapping[str, Any],
    device: torch.device,
    config: Mapping[str, Any],
    output_dir: Path | None = None,
    run_backward: bool = False,
) -> None:
    parameter_report = module_parameter_report(model)
    stage_trace.log(
        f"{stage_name}_parameter_checksum",
        "details",
        **parameter_report,
    )
    if output_dir is not None and runtime.get("distributed"):
        parameter_consistency = verify_rank_parameter_report(
            output_dir=output_dir,
            runtime=runtime,
            stage_trace=stage_trace,
            stage_name=f"{stage_name}_parameter_checksum",
            report=parameter_report,
        )
    else:
        parameter_consistency = {
            "all_rank_parameter_checksums": [parameter_report["parameter_sha256"]],
            "all_rank_parameter_checksums_equal": True,
        }
    with stage_trace.span(f"{stage_name}_to_device", module_type=type(model).__name__, device=str(device)):
        model = model.to(device)
    if device.type == "cuda":
        with stage_trace.span(f"{stage_name}_cuda_synchronize_pre_ddp", device=str(device)):
            torch.cuda.synchronize(device)
    inputs = input_factory(device)
    ddp_kwargs, ddp_kwargs_log = ddp_probe_constructor_kwargs(runtime, device, config)
    stage_trace.log(
        f"{stage_name}_preflight",
        "details",
        kwargs=ddp_kwargs_log,
        input_shapes=[list(tensor.shape) for tensor in inputs],
        input_devices=[str(tensor.device) for tensor in inputs],
        parameter_checksum=parameter_report,
        parameter_checksum_consistency=parameter_consistency,
        **module_ddp_preflight_snapshot(model),
    )
    with stage_trace.span(f"{stage_name}_ddp_ctor", **ddp_kwargs_log):
        wrapped_probe = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    with stage_trace.span(
        f"{stage_name}_forward",
        input_shapes=[list(tensor.shape) for tensor in inputs],
        input_devices=[str(tensor.device) for tensor in inputs],
    ):
        output = wrapped_probe(*inputs)
    stage_trace.log(
        f"{stage_name}_forward",
        "details",
        output_shape=list(output.shape),
        output_device=str(output.device),
    )
    if device.type == "cuda":
        with stage_trace.span(f"{stage_name}_cuda_synchronize_post_forward", device=str(device)):
            torch.cuda.synchronize(device)
    if run_backward:
        with stage_trace.span(f"{stage_name}_backward", output_shape=list(output.shape)):
            dummy_loss = output.float().pow(2).mean()
            dummy_loss.backward()
        if device.type == "cuda":
            with stage_trace.span(f"{stage_name}_cuda_synchronize_post_backward", device=str(device)):
                torch.cuda.synchronize(device)
        stage_trace.log(
            f"{stage_name}_backward",
            "details",
            dummy_loss=float(dummy_loss.detach().cpu().item()),
            **module_gradient_report(model),
        )


def run_a0_minimal_ddp_wrap_probe(
    *,
    stage_trace: StageTracer,
    runtime: Mapping[str, Any],
    device: torch.device,
    config: Mapping[str, Any],
    output_dir: Path | None = None,
    run_backward: bool = False,
) -> None:
    if not runtime.get("distributed"):
        stage_trace.log("ddp_wrap_probe_minimal_mlp", "skipped", reason="not_distributed")
        return
    set_model_init_seed(config, stage_trace, "ddp_wrap_probe_minimal_mlp_model_init_seed")
    run_ddp_probe_model(
        stage_name="ddp_wrap_probe_minimal_mlp",
        model=nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2)),
        input_factory=lambda probe_device: (torch.ones(2, 4, device=probe_device),),
        stage_trace=stage_trace,
        runtime=runtime,
        device=device,
        config=config,
        output_dir=output_dir,
        run_backward=run_backward,
    )


def run_a0_ddp_probe_suite(
    *,
    stage_trace: StageTracer,
    runtime: Mapping[str, Any],
    device: torch.device,
    config: Mapping[str, Any],
    motion_dim: int,
    skeleton_dim: int,
    target_dim: int,
    output_dir: Path | None = None,
    run_backward: bool = False,
) -> None:
    if not runtime.get("distributed"):
        stage_trace.log("ddp_probe_suite", "skipped", reason="not_distributed")
        return
    hidden_dim = int(config["model"]["hidden_dim"])
    num_layers = int(config["model"]["num_layers"])
    dropout = float(config["model"].get("dropout", 0.0))
    concat_input_dim = int(motion_dim) + int(skeleton_dim)
    with stage_trace.span(
        "ddp_probe_suite",
        motion_dim=int(motion_dim),
        skeleton_dim=int(skeleton_dim),
        target_dim=int(target_dim),
        concat_input_dim=concat_input_dim,
        run_backward=bool(run_backward),
    ):
        run_a0_minimal_ddp_wrap_probe(
            stage_trace=stage_trace,
            runtime=runtime,
            device=device,
            config=config,
            output_dir=output_dir,
            run_backward=run_backward,
        )
        set_model_init_seed(config, stage_trace, "ddp_probe_same_shape_sequential_model_init_seed")
        run_ddp_probe_model(
            stage_name="ddp_probe_same_shape_sequential",
            model=build_mlp(concat_input_dim, hidden_dim, int(target_dim), num_layers, dropout),
            input_factory=lambda probe_device: (
                torch.ones(2, concat_input_dim, device=probe_device),
            ),
            stage_trace=stage_trace,
            runtime=runtime,
            device=device,
            config=config,
            output_dir=output_dir,
            run_backward=run_backward,
        )
        set_model_init_seed(config, stage_trace, "ddp_probe_fresh_concat_retargeter_model_init_seed")
        run_ddp_probe_model(
            stage_name="ddp_probe_fresh_concat_retargeter",
            model=ConcatRetargeter(int(motion_dim), int(skeleton_dim), int(target_dim), dict(config["model"])),
            input_factory=lambda probe_device: (
                torch.ones(2, int(motion_dim), device=probe_device),
                torch.ones(2, int(skeleton_dim), device=probe_device),
            ),
            stage_trace=stage_trace,
            runtime=runtime,
            device=device,
            config=config,
            output_dir=output_dir,
            run_backward=run_backward,
        )
        for stage_name, features in (
            ("ddp_probe_single_linear_512", 512),
            ("ddp_probe_single_linear_1024", 1024),
            ("ddp_probe_single_linear_1154", 1154),
        ):
            set_model_init_seed(config, stage_trace, f"{stage_name}_model_init_seed")
            run_ddp_probe_model(
                stage_name=stage_name,
                model=nn.Linear(features, features),
                input_factory=lambda probe_device, input_features=features: (
                    torch.ones(2, input_features, device=probe_device),
                ),
                stage_trace=stage_trace,
                runtime=runtime,
                device=device,
                config=config,
                output_dir=output_dir,
                run_backward=run_backward,
            )


class SkeletonAEFeatureLookup:
    def __init__(
        self,
        *,
        embeddings_by_id: dict[str, np.ndarray],
        registry_rows_by_id: dict[str, dict[str, Any]],
        lookup_index: dict[tuple[str, str], set[str]],
        artifact_info: dict[str, Any],
    ) -> None:
        self.embeddings_by_id = embeddings_by_id
        self.registry_rows_by_id = registry_rows_by_id
        self.lookup_index = lookup_index
        self.artifact_info = artifact_info
        self.mapping_report: dict[str, Any] = {
            "missing_skeleton_geometry_count": 0,
            "ambiguous_skeleton_geometry_count": 0,
            "resolved_row_count": 0,
            "resolved_samples": [],
        }
        self.cache_path = ""

    @property
    def embedding_dim(self) -> int:
        return SKELETON_GEOMETRY_LATENT_DIM

    def resolve_row_ids(self, row: Mapping[str, Any]) -> set[str]:
        resolved: set[str] = set()
        for field, value in _row_skeleton_lookup_values(row):
            if field == "source_path":
                for alias in _path_aliases(value):
                    resolved.update(self.lookup_index.get((field, alias), set()))
            else:
                text = _clean_text(value)
                if text:
                    resolved.update(self.lookup_index.get((field, text), set()))
        return resolved

    def validate_and_annotate_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        missing: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []
        resolved_samples: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            resolved = self.resolve_row_ids(row)
            sample = {
                "row_index": row_index,
                "relative_path": row.get("relative_path", ""),
                "actor_uid": row.get("actor_uid", ""),
                "source_soma_proportional_path": row.get("source_soma_proportional_path", ""),
                "source_bvh": row.get("source_bvh", ""),
                "candidate_encoder_ids": sorted(resolved),
            }
            if not resolved:
                missing.append(sample)
                continue
            if len(resolved) > 1:
                ambiguous.append(sample)
                continue
            encoder_id = next(iter(resolved))
            registry_row = self.registry_rows_by_id[encoder_id]
            row["skeleton_ae_encoder_id"] = encoder_id
            row["skeleton_ae_source_soma_proportional_path"] = registry_row.get(
                "source_soma_proportional_path",
                "",
            )
            if len(resolved_samples) < 10:
                resolved_samples.append(
                    {
                        "row_index": row_index,
                        "relative_path": row.get("relative_path", ""),
                        "encoder_id": encoder_id,
                        "source_soma_proportional_path": registry_row.get(
                            "source_soma_proportional_path",
                            "",
                        ),
                    }
                )
        self.mapping_report = {
            "missing_skeleton_geometry_count": len(missing),
            "ambiguous_skeleton_geometry_count": len(ambiguous),
            "resolved_row_count": len(rows) - len(missing) - len(ambiguous),
            "row_count": len(rows),
            "missing_examples": missing[:10],
            "ambiguous_examples": ambiguous[:10],
            "resolved_samples": resolved_samples,
        }
        if missing or ambiguous:
            raise ValueError(
                "Skeleton AE registry mapping failed: "
                f"missing_skeleton_geometry_count={len(missing)} "
                f"ambiguous_skeleton_geometry_count={len(ambiguous)} "
                f"missing_examples={missing[:3]} ambiguous_examples={ambiguous[:3]}"
            )
        return self.mapping_report

    def embedding_for_row(self, row: Mapping[str, Any]) -> np.ndarray:
        encoder_id = str(row.get("skeleton_ae_encoder_id", ""))
        if not encoder_id:
            resolved = self.resolve_row_ids(row)
            if len(resolved) != 1:
                raise ValueError(f"row does not resolve to one Skeleton AE embedding: {row}")
            encoder_id = next(iter(resolved))
        return self.embeddings_by_id[encoder_id]

    def write_cache(self, output_dir: Path) -> str:
        cache_path = output_dir / "cache" / "skeleton_embedding_cache.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings_by_id": {
                    key: torch.from_numpy(value.astype(np.float32, copy=False))
                    for key, value in self.embeddings_by_id.items()
                },
                "artifact_info": self.artifact_info,
                "mapping_report": self.mapping_report,
            },
            cache_path,
        )
        self.cache_path = str(cache_path)
        return self.cache_path


def skeleton_ae_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    cfg = config.get("skeleton_ae")
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        return None
    return cfg


def is_skeleton_ae_enabled(config: Mapping[str, Any]) -> bool:
    return skeleton_ae_config(config) is not None


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _path_aliases(value: Any) -> set[str]:
    text = _clean_text(value)
    if not text:
        return set()
    path = Path(text)
    normalized = text.replace("\\", "/")
    return {
        text,
        normalized,
        normalized.lstrip("./"),
        path.name,
        path.stem,
    } - {""}


def _actor_ids_from_text(value: Any) -> set[str]:
    text = _clean_text(value)
    if not text:
        return set()
    return {match.group(1) for match in re.finditer(r"(A\d{3,})", text)}


def _row_skeleton_lookup_values(row: Mapping[str, Any]) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    for field in ("encoder_id", "skeleton_id", "actor_uid"):
        text = _clean_text(row.get(field, ""))
        if text:
            values.append((field, text))
    for field in ("source_soma_proportional_path", "source_bvh", "relative_path", "path"):
        text = _clean_text(row.get(field, ""))
        if text:
            values.append(("source_path", text))
            for actor_id in _actor_ids_from_text(text):
                values.append(("actor_uid", actor_id))
                values.append(("encoder_id", actor_id))
    return values


def _add_lookup_key(
    lookup_index: dict[tuple[str, str], set[str]],
    field: str,
    value: Any,
    encoder_id: str,
) -> None:
    text = _clean_text(value)
    if not text:
        return
    lookup_index.setdefault((field, text), set()).add(encoder_id)


def _load_skeleton_ae_registry(
    registry_csv: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], set[str]]]:
    if not registry_csv.exists():
        raise FileNotFoundError(f"skeleton_ae.registry_csv is missing: {registry_csv}")
    rows_by_id: dict[str, dict[str, Any]] = {}
    lookup_index: dict[tuple[str, str], set[str]] = {}
    with registry_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            encoder_id = _clean_text(row.get("encoder_id") or row.get("actor_uid"))
            if not encoder_id:
                raise ValueError(f"registry row has no encoder_id/actor_uid: {row}")
            if encoder_id in rows_by_id:
                raise ValueError(f"duplicate encoder_id in Skeleton AE registry: {encoder_id}")
            geometry = json.loads(row["geometry_json"])
            if len(geometry) != SKELETON_GEOMETRY_DIM:
                raise ValueError(f"{encoder_id} has geometry dim {len(geometry)}")
            payload = dict(row)
            payload["geometry"] = np.asarray(geometry, dtype=np.float32)
            rows_by_id[encoder_id] = payload
            for field in ("encoder_id", "actor_uid"):
                _add_lookup_key(lookup_index, field, row.get(field, ""), encoder_id)
            for alias in _path_aliases(row.get("source_soma_proportional_path", "")):
                _add_lookup_key(lookup_index, "source_path", alias, encoder_id)
            for actor_id in _actor_ids_from_text(row.get("source_soma_proportional_path", "")):
                _add_lookup_key(lookup_index, "actor_uid", actor_id, encoder_id)
                _add_lookup_key(lookup_index, "encoder_id", actor_id, encoder_id)
    if not rows_by_id:
        raise ValueError(f"Skeleton AE registry has no rows: {registry_csv}")
    return rows_by_id, lookup_index


def skeleton_ae_registry_cache_device(cfg: Mapping[str, Any]) -> torch.device:
    requested = str(cfg.get("cache_device", "cpu") or "cpu").strip().lower()
    if requested != "cpu":
        raise ValueError(
            "A0 frozen Skeleton AE registry cache must be built on CPU; "
            f"got skeleton_ae.cache_device={requested!r}"
        )
    return torch.device("cpu")


def build_skeleton_ae_feature_lookup(
    config: Mapping[str, Any],
    device: torch.device,
    stage_trace: StageTracer | None = None,
) -> SkeletonAEFeatureLookup | None:
    cfg = skeleton_ae_config(config)
    if cfg is None:
        return None
    freeze_encoder = bool(cfg.get("freeze_encoder", True))
    if not freeze_encoder:
        raise ValueError("A0 requires skeleton_ae.freeze_encoder=true")
    checkpoint_path = Path(str(cfg["checkpoint"]))
    stats_path = Path(str(cfg["normalization"]))
    registry_csv = Path(str(cfg["registry_csv"]))
    with (stage_trace.span("skeleton_ae_registry_load", registry_csv=str(registry_csv)) if stage_trace else nullcontext()):
        rows_by_id, lookup_index = _load_skeleton_ae_registry(registry_csv)
    cache_device = skeleton_ae_registry_cache_device(cfg)
    if stage_trace is not None:
        stage_trace.log(
            "skeleton_ae_registry_load",
            "details",
            registry_rows=len(rows_by_id),
            lookup_keys=len(lookup_index),
        )
    with (
        stage_trace.span(
            "skeleton_ae_checkpoint_load",
            checkpoint=str(checkpoint_path),
            cache_device=str(cache_device),
        )
        if stage_trace
        else nullcontext()
    ):
        ae_model, checkpoint = load_skeleton_geometry_ae_checkpoint(
            checkpoint_path,
            device=cache_device,
            freeze_encoder=True,
        )
    with (
        stage_trace.span("skeleton_ae_stats_load", stats=str(stats_path), cache_device=str(cache_device))
        if stage_trace
        else nullcontext()
    ):
        ae_stats = load_skeleton_geometry_ae_stats(stats_path, device=cache_device)
    embeddings_by_id: dict[str, np.ndarray] = {}
    with (
        stage_trace.span(
            "skeleton_ae_cpu_z_cache_build",
            row_count=len(rows_by_id),
            cache_device=str(cache_device),
        )
        if stage_trace
        else nullcontext()
    ):
        with torch.inference_mode():
            for encoder_id, row in rows_by_id.items():
                x = torch.from_numpy(row["geometry"]).to(device=cache_device, dtype=torch.float32).unsqueeze(0)
                x_norm = (x - ae_stats["skeleton_mean"]) / ae_stats["skeleton_std"]
                z = ae_model.encode(x_norm).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                if z.shape != (SKELETON_GEOMETRY_LATENT_DIM,):
                    raise ValueError(f"{encoder_id} encoded to unexpected shape {z.shape}")
                embeddings_by_id[encoder_id] = z
    artifact_info = {
        "checkpoint": file_stats(checkpoint_path),
        "normalization": file_stats(stats_path),
        "registry_csv": file_stats(registry_csv),
        "checkpoint_architecture": list(checkpoint.get("architecture") or []),
        "expected_architecture": SKELETON_GEOMETRY_AE_ARCHITECTURE,
        "x_skel_dim": SKELETON_GEOMETRY_DIM,
        "z_skel_dim": SKELETON_GEOMETRY_LATENT_DIM,
        "skeleton_encoder_frozen": True,
        "embedding_cache_device": str(cache_device),
        "training_device": str(device),
    }
    return SkeletonAEFeatureLookup(
        embeddings_by_id=embeddings_by_id,
        registry_rows_by_id=rows_by_id,
        lookup_index=lookup_index,
        artifact_info=artifact_info,
    )


def distributed_env(env: Mapping[str, str] | None = None) -> dict[str, int]:
    source = os.environ if env is None else env
    rank = int(source.get("RANK", "0"))
    world_size = int(source.get("WORLD_SIZE", "1"))
    local_rank = int(source.get("LOCAL_RANK", str(rank)))
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def setup_distributed_runtime(stage_trace: StageTracer | None = None) -> dict[str, Any]:
    env = distributed_env()
    distributed = env["world_size"] > 1
    if stage_trace is not None:
        stage_trace.log("distributed_env", "after", **env, distributed=distributed)
        stage_trace.log("cuda_available", "before")
    cuda_available = torch.cuda.is_available()
    if stage_trace is not None:
        stage_trace.log("cuda_available", "after", cuda_available=cuda_available)
    if cuda_available:
        device_index = env["local_rank"] if distributed else 0
        if stage_trace is not None:
            stage_trace.log("cuda_set_device", "before", device_index=device_index)
        torch.cuda.set_device(device_index)
        if stage_trace is not None:
            stage_trace.log("cuda_set_device", "after", device_index=device_index)
        device = torch.device("cuda", device_index)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if distributed:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is required for WORLD_SIZE > 1")
        if not torch.distributed.is_initialized():
            if stage_trace is not None:
                stage_trace.log("distributed_init_process_group", "before", backend=backend)
            torch.distributed.init_process_group(backend=backend, init_method="env://")
            if stage_trace is not None:
                stage_trace.log("distributed_init_process_group", "after", backend=backend)
    return {**env, "distributed": distributed, "device": device, "backend": backend}


def cleanup_distributed_runtime(runtime: Mapping[str, Any]) -> None:
    if runtime.get("distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def distributed_barrier(runtime: Mapping[str, Any]) -> None:
    if runtime.get("distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def is_main_process(runtime: Mapping[str, Any]) -> bool:
    return int(runtime.get("rank", 0)) == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def average_metric_dict(
    metrics: dict[str, float],
    runtime: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    if not metrics or not runtime.get("distributed"):
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    values /= int(runtime["world_size"])
    return {key: float(value) for key, value in zip(keys, values.detach().cpu().tolist())}


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if args.max_steps is not None:
        updated["training"]["max_steps"] = int(args.max_steps)
    if args.disable_visual_validation:
        visual = dict(updated.get("visual_validation", {}))
        visual["enabled"] = False
        updated["visual_validation"] = visual
    if args.wandb_mode:
        wandb_cfg = dict(updated.get("wandb", {}))
        if args.wandb_mode == "disabled":
            wandb_cfg["enabled"] = False
        else:
            wandb_cfg["mode"] = args.wandb_mode
            os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        updated["wandb"] = wandb_cfg
    return updated


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(norm < 1e-8, 1.0, norm)


def robot_root_rot_to_wxyz(root_rot: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    fmt = str(config.get("input_data", {}).get("robot_root_rot_format", "xyzw")).lower()
    root_rot = np.asarray(root_rot, dtype=np.float32)
    if fmt == "xyzw":
        return quat_normalize(root_rot[..., [3, 0, 1, 2]])
    if fmt == "wxyz":
        return quat_normalize(root_rot)
    raise ValueError(f"unsupported input_data.robot_root_rot_format: {fmt}")


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )
    return quat_normalize(out)


def quat_mul_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                axis=-1,
            ),
            np.stack(
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                axis=-1,
            ),
            np.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def future_indices(frame_indices: np.ndarray, total_frames: int, window: int, step: int) -> np.ndarray:
    offsets = np.arange(window, dtype=np.int64) * int(step)
    return np.minimum(frame_indices[:, None] + offsets[None, :], total_frames - 1)


def relative_root_rot6d(root_quat: np.ndarray, frame_indices: np.ndarray, idx: np.ndarray) -> np.ndarray:
    base = quat_normalize(root_quat[frame_indices])
    refs = quat_normalize(root_quat[idx])
    rel = quat_mul(quat_conjugate(base[:, None, :]), refs)
    return quat_to_rot6d(rel).reshape(frame_indices.shape[0], idx.shape[1], 6)


def quat_to_rot6d(root_quat: np.ndarray) -> np.ndarray:
    mat = quat_to_matrix(root_quat)
    return mat[..., :, :2].reshape(*mat.shape[:-2], 6)


def include_root_pos_target(config: Mapping[str, Any]) -> bool:
    features = config.get("features", {})
    if not isinstance(features, Mapping):
        return False
    explicit = features.get("include_root_pos_target")
    if explicit is not None:
        return bool(explicit)
    target_text = " ".join(
        str(features.get(key, ""))
        for key in ("target_feature", "target_features", "target_pose_feature")
    )
    return "root_pos" in target_text


def root_pose_target_dim(config: Mapping[str, Any], window: int) -> int:
    per_frame = 9 if include_root_pos_target(config) else 6
    return int(window) * per_frame


def target_command_dim(target_dim: int, window: int, config: Mapping[str, Any]) -> int:
    return int(target_dim) - root_pose_target_dim(config, window)


def visual_validation_interval_seconds(config: Mapping[str, Any]) -> float | None:
    cfg = config.get("visual_validation", {})
    if not isinstance(cfg, Mapping):
        return None
    every_seconds = cfg.get("every_seconds")
    if every_seconds not in ("", None):
        return float(every_seconds)
    every_minutes = cfg.get("every_minutes")
    if every_minutes not in ("", None):
        return float(every_minutes) * 60.0
    return None


def build_features(
    arrays: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    window: int,
    step: int,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_pos = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    body_pos = arrays["body_pos_w"]
    body_quat = arrays["body_quat_w"]
    total_frames = min(joint_pos.shape[0], joint_vel.shape[0], body_pos.shape[0], body_quat.shape[0])
    idx = future_indices(frame_indices, total_frames, window, step)

    root_pos0 = body_pos[frame_indices, 0]
    root_quat0 = quat_normalize(body_quat[frame_indices, 0])
    root_rot0_t = np.swapaxes(quat_to_matrix(root_quat0), -1, -2)
    rel_body_pos = body_pos[idx] - root_pos0[:, None, None, :]
    local_body_pos = np.einsum("nij,nwbj->nwbi", root_rot0_t, rel_body_pos)
    root_quat = body_quat[:, 0]
    root_ori6d = relative_root_rot6d(root_quat, frame_indices, idx)
    root_z_rel = (body_pos[idx, 0, 2] - root_pos0[:, None, 2])[..., None]

    motion = np.concatenate(
        [
            local_body_pos.reshape(frame_indices.shape[0], -1),
            root_ori6d.reshape(frame_indices.shape[0], -1),
            root_z_rel.reshape(frame_indices.shape[0], -1),
        ],
        axis=-1,
    )

    skeleton_anchor = local_body_pos[:, 0]
    skeleton_lengths = np.linalg.norm(skeleton_anchor, axis=-1)
    skeleton = np.concatenate(
        [skeleton_anchor.reshape(frame_indices.shape[0], -1), skeleton_lengths],
        axis=-1,
    )

    command = np.concatenate([joint_pos[idx], joint_vel[idx]], axis=-1)
    if config is not None and include_root_pos_target(config):
        target_root_pos = body_pos[idx, 0]
        target_root_ori6d = quat_to_rot6d(root_quat[idx])
        target = np.concatenate(
            [
                command.reshape(frame_indices.shape[0], -1),
                target_root_pos.reshape(frame_indices.shape[0], -1),
                target_root_ori6d.reshape(frame_indices.shape[0], -1),
            ],
            axis=-1,
        )
        return (
            motion.astype(np.float32, copy=False),
            skeleton.astype(np.float32, copy=False),
            target.astype(np.float32, copy=False),
        )

    target = np.concatenate(
        [command.reshape(frame_indices.shape[0], -1), root_ori6d.reshape(frame_indices.shape[0], -1)],
        axis=-1,
    )
    return (
        motion.astype(np.float32, copy=False),
        skeleton.astype(np.float32, copy=False),
        target.astype(np.float32, copy=False),
    )


def finite_difference_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] <= 1:
        return vel
    vel[1:] = (values[1:] - values[:-1]) * float(fps)
    vel[0] = vel[1]
    return vel


def resample_soma_array(
    data: np.ndarray,
    fps_source: float,
    fps_target: float,
    *,
    target_len: int | None = None,
) -> np.ndarray:
    if data.shape[0] <= 1 or fps_source <= 0 or fps_target <= 0 or abs(fps_source - fps_target) < 1e-6:
        out = data.astype(np.float32, copy=False)
    else:
        duration = (data.shape[0] - 1) / float(fps_source)
        target_times = np.arange(0.0, duration, 1.0 / float(fps_target), dtype=np.float32)
        if target_times.shape[0] <= 1:
            out = data[:1].astype(np.float32, copy=False)
        else:
            phase = target_times / duration
            src_pos = phase * (data.shape[0] - 1)
            idx0 = np.floor(src_pos).astype(np.int64)
            idx1 = np.minimum(idx0 + 1, data.shape[0] - 1)
            blend = (src_pos - idx0).astype(np.float32)
            while blend.ndim < data.ndim:
                blend = blend[..., None]
            out = data[idx0] * (1.0 - blend) + data[idx1] * blend
            out = out.astype(np.float32, copy=False)

    if target_len is None or out.shape[0] == target_len:
        return out
    if out.shape[0] > target_len:
        return out[:target_len]
    pad = np.repeat(out[-1:], target_len - out.shape[0], axis=0)
    return np.concatenate([out, pad], axis=0).astype(np.float32, copy=False)


def build_soma_motionlib_features(
    arrays: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    window: int,
    step: int,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    soma_joints = arrays["soma_joints"]
    soma_root_quat = quat_normalize(arrays["soma_root_quat"])
    dof = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    root_rot = quat_normalize(arrays["root_rot"])
    root_pos = arrays.get("root_pos")
    if root_pos is None:
        root_pos = np.zeros((dof.shape[0], 3), dtype=np.float32)
    total_frames = min(
        soma_joints.shape[0],
        soma_root_quat.shape[0],
        dof.shape[0],
        joint_vel.shape[0],
        root_rot.shape[0],
        root_pos.shape[0],
    )
    idx = future_indices(frame_indices, total_frames, window, step)

    root_rot_t = np.swapaxes(quat_to_matrix(soma_root_quat[idx]), -1, -2)
    soma_local = np.einsum("nwij,nwbj->nwbi", root_rot_t, soma_joints[idx])
    soma_root_ori6d = relative_root_rot6d(soma_root_quat, frame_indices, idx)
    motion = np.concatenate(
        [
            soma_local.reshape(frame_indices.shape[0], -1),
            soma_root_ori6d.reshape(frame_indices.shape[0], -1),
        ],
        axis=-1,
    )

    skeleton_anchor = soma_local[:, 0]
    skeleton_lengths = np.linalg.norm(skeleton_anchor, axis=-1)
    skeleton = np.concatenate(
        [skeleton_anchor.reshape(frame_indices.shape[0], -1), skeleton_lengths],
        axis=-1,
    )

    command = np.concatenate([dof[idx], joint_vel[idx]], axis=-1)
    if config is not None and include_root_pos_target(config):
        target_root_pos = root_pos[idx].copy()
        target_root_pos[..., :2] -= root_pos[frame_indices, None, :2]
        target_root_ori6d = quat_to_rot6d(root_rot[idx])
        target = np.concatenate(
            [
                command.reshape(frame_indices.shape[0], -1),
                target_root_pos.reshape(frame_indices.shape[0], -1),
                target_root_ori6d.reshape(frame_indices.shape[0], -1),
            ],
            axis=-1,
        )
        return (
            motion.astype(np.float32, copy=False),
            skeleton.astype(np.float32, copy=False),
            target.astype(np.float32, copy=False),
        )

    target_root_ori6d = relative_root_rot6d(root_rot, frame_indices, idx)
    target = np.concatenate(
        [command.reshape(frame_indices.shape[0], -1), target_root_ori6d.reshape(frame_indices.shape[0], -1)],
        axis=-1,
    )
    return (
        motion.astype(np.float32, copy=False),
        skeleton.astype(np.float32, copy=False),
        target.astype(np.float32, copy=False),
    )


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key].astype(np.float32, copy=False) for key in REQUIRED_NPZ_KEYS}


def _import_joblib():
    import joblib

    return joblib


def _single_motionlib_entry(path: Path) -> tuple[str, Mapping[str, Any]]:
    loaded = _import_joblib().load(path)
    if not isinstance(loaded, Mapping) or not loaded:
        raise ValueError(f"motionlib file must contain a non-empty mapping: {path}")
    key = path.stem
    if key in loaded:
        return key, loaded[key]
    first_key = next(iter(loaded))
    return str(first_key), loaded[first_key]


def _require_motionlib_keys(entry: Mapping[str, Any], keys: Sequence[str], path: Path) -> None:
    missing = [key for key in keys if key not in entry]
    if missing:
        raise ValueError(f"{path} is missing required motionlib keys: {', '.join(missing)}")


def load_soma_motionlib_arrays(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    input_cfg = config["input_data"]
    robot_path = Path(input_cfg["robot_motion_dir"]) / str(row["robot_relative_path"])
    soma_path = Path(input_cfg["soma_motion_dir"]) / str(row["soma_relative_path"])
    _, robot = _single_motionlib_entry(robot_path)
    _, soma = _single_motionlib_entry(soma_path)
    _require_motionlib_keys(robot, REQUIRED_ROBOT_MOTIONLIB_KEYS, robot_path)
    _require_motionlib_keys(soma, REQUIRED_SOMA_MOTIONLIB_KEYS, soma_path)

    dof = np.asarray(robot["dof"], dtype=np.float32)
    root_rot = robot_root_rot_to_wxyz(np.asarray(robot["root_rot"], dtype=np.float32), config)
    root_pos = np.asarray(
        robot.get("root_trans_offset", np.zeros((dof.shape[0], 3), dtype=np.float32)),
        dtype=np.float32,
    )
    robot_fps = float(robot.get("fps") or input_cfg.get("target_fps") or 50.0)
    soma_fps = float(soma.get("fps") or input_cfg.get("source_fps") or robot_fps)
    target_len = min(dof.shape[0], root_rot.shape[0], root_pos.shape[0])
    soma_joints = resample_soma_array(
        np.asarray(soma["soma_joints"], dtype=np.float32),
        soma_fps,
        robot_fps,
        target_len=target_len,
    )
    soma_root_quat = quat_normalize(
        resample_soma_array(
            np.asarray(soma["soma_root_quat"], dtype=np.float32),
            soma_fps,
            robot_fps,
            target_len=target_len,
        )
    )
    dof = dof[:target_len]
    root_rot = root_rot[:target_len]
    root_pos = root_pos[:target_len]
    return {
        "soma_joints": soma_joints,
        "soma_root_quat": soma_root_quat,
        "joint_pos": dof,
        "joint_vel": finite_difference_velocity(dof, robot_fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "joint_names": list(soma.get("joint_names", SOMA_JOINT_NAMES)),
        "fps": np.asarray(robot_fps, dtype=np.float32),
    }


def index_path_from_config(config: dict[str, Any]) -> Path:
    if config["input_data"].get("format") == "soma_motionlib":
        return Path(config["input_data"]["robot_motion_dir"])
    indexing = config["input_data"].get("indexing", {})
    raw_path = indexing.get("index_csv") or indexing.get("prebuilt_index_jsonl")
    if not raw_path:
        raise ValueError("input_data.indexing must define index_csv or prebuilt_index_jsonl")
    return Path(raw_path)


def remap_source_path(source_path: str, indexing: dict[str, Any]) -> Path:
    source_prefix = indexing.get("source_path_prefix", "")
    target_prefix = indexing.get("target_path_prefix", "")
    if source_prefix and target_prefix and source_path.startswith(source_prefix):
        return Path(target_prefix + source_path[len(source_prefix) :])
    return Path(source_path)


def rows_from_jsonl_index(config: dict[str, Any], data_root: Path) -> tuple[list[dict[str, Any]], int]:
    indexing = config["input_data"].get("indexing", {})
    index_path = Path(indexing["prebuilt_index_jsonl"])
    max_clips = int(indexing.get("max_clips", 0))
    rows = []
    skipped = 0
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            source = json.loads(line)
            if not source.get("source_npz_exists", False):
                skipped += 1
                continue
            source_path = str(source["source_npz_path"])
            path = remap_source_path(source_path, indexing)
            try:
                relative_path = str(path.relative_to(data_root))
            except ValueError:
                skipped += 1
                continue
            frame_count = int(float(source.get("move_duration_frames") or -1))
            if frame_count <= 1:
                skipped += 1
                continue
            rows.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "frame_count": frame_count,
                    "source_move_name": source.get("move_name"),
                    "source_labels": source.get("labels", {}),
                    "source_soma_proportional_path": source.get("source_soma_proportional_path", ""),
                    "fps": source.get("fps", ""),
                    "filename": source.get("filename") or source.get("move_name") or Path(relative_path).stem,
                    "actor_uid": source.get("actor_uid", ""),
                    "category": source.get("category", ""),
                }
            )
            if max_clips > 0 and len(rows) >= max_clips:
                break
    return rows, skipped


def rows_from_csv_index(config: dict[str, Any], data_root: Path) -> tuple[list[dict[str, Any]], int]:
    indexing = config["input_data"].get("indexing", {})
    index_path = Path(indexing["index_csv"])
    max_clips = int(indexing.get("max_clips", 0))
    rows = []
    skipped = 0
    with index_path.open("r", encoding="utf-8", newline="") as f:
        for source in csv.DictReader(f):
            if source.get("schema_status") not in {"", "ok", None}:
                skipped += 1
                continue
            source_path = str(source.get("sonic_path") or source.get("source_npz_path") or "")
            if not source_path:
                skipped += 1
                continue
            path = remap_source_path(source_path, indexing)
            try:
                relative_path = str(path.relative_to(data_root))
            except ValueError:
                skipped += 1
                continue
            frame_count_text = source.get("frame_count") or source.get("move_duration_frames") or ""
            try:
                frame_count = int(float(frame_count_text))
            except ValueError:
                skipped += 1
                continue
            if frame_count <= 1:
                skipped += 1
                continue
            timing = source_target_timing_summary(source, frame_count=frame_count, indexing=indexing)
            if timing.get("status") == "invalid":
                skipped += 1
                continue
            rows.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "frame_count": frame_count,
                    "source_frame_count": timing.get("source_frame_count", ""),
                    "source_fps": timing.get("source_fps", ""),
                    "target_fps": timing.get("target_fps", ""),
                    "source_duration_sec": timing.get("source_duration_sec", ""),
                    "target_duration_sec": timing.get("target_duration_sec", ""),
                    "source_move_name": source.get("filename") or source.get("move_name"),
                    "source_labels": {
                        "package": source.get("package", ""),
                        "category": source.get("category", ""),
                        "actor_uid": source.get("actor_uid", ""),
                    },
                    "source_soma_proportional_path": source.get("source_soma_proportional_path", ""),
                    "fps": source.get("fps", ""),
                    "filename": source.get("filename") or Path(relative_path).stem,
                    "actor_uid": source.get("actor_uid", ""),
                    "category": source.get("category", ""),
                }
            )
            if max_clips > 0 and len(rows) >= max_clips:
                break
    return rows, skipped


ROWS_FROM_INDEX_CACHE_SUBDIR = "cache/rows_from_index"


def rows_from_index_cache_path(output_dir: Path) -> Path:
    return output_dir / ROWS_FROM_INDEX_CACHE_SUBDIR / "rows_from_index_cache.json"


def rows_from_soma_motionlib_pair(
    config: dict[str, Any],
    *,
    stage_trace: StageTracer | None = None,
) -> tuple[list[dict[str, Any]], int]:
    input_cfg = config["input_data"]
    robot_dir = Path(input_cfg["robot_motion_dir"])
    soma_dir = Path(input_cfg["soma_motion_dir"])
    max_clips = int(input_cfg.get("max_clips", 0))
    max_duration_delta = float(input_cfg.get("max_duration_delta_sec", 0.05))
    rows: list[dict[str, Any]] = []
    skipped = 0
    missing_soma = 0
    invalid_rows = 0
    duration_filtered = 0
    accepted_samples: list[dict[str, Any]] = []
    if stage_trace:
        with stage_trace.span(
            "rows_from_index_stat",
            robot_motion_dir=str(robot_dir),
            soma_motion_dir=str(soma_dir),
        ):
            robot_dir_exists = robot_dir.exists()
            soma_dir_exists = soma_dir.exists()
        stage_trace.log(
            "rows_from_index_stat",
            "details",
            robot_motion_dir=str(robot_dir),
            robot_motion_dir_exists=robot_dir_exists,
            soma_motion_dir=str(soma_dir),
            soma_motion_dir_exists=soma_dir_exists,
        )
    with (
        stage_trace.span(
            "rows_from_index_glob",
            robot_motion_dir=str(robot_dir),
            soma_motion_dir=str(soma_dir),
            max_clips=max_clips,
            max_duration_delta_sec=max_duration_delta,
        )
        if stage_trace
        else nullcontext()
    ):
        robot_paths = sorted(robot_dir.glob("*.pkl"))
    if stage_trace:
        stage_trace.log(
            "rows_from_index_glob",
            "details",
            robot_path_count=len(robot_paths),
            robot_motion_dir=str(robot_dir),
            soma_motion_dir=str(soma_dir),
            max_clips=max_clips,
            max_duration_delta_sec=max_duration_delta,
        )
    for robot_path_index, robot_path in enumerate(robot_paths, start=1):
        if robot_path.name == "metadata.pkl":
            continue
        soma_path = soma_dir / robot_path.name
        if not soma_path.exists():
            missing_soma += 1
            skipped += 1
            continue
        try:
            if stage_trace and (robot_path_index == 1 or robot_path_index % 100 == 0):
                stage_trace.log(
                    "rows_from_index_progress",
                    "details",
                    robot_path_index=robot_path_index,
                    robot_path_count=len(robot_paths),
                    robot_path=str(robot_path),
                    soma_path=str(soma_path),
                    accepted_count=len(rows),
                    skipped_count=skipped,
                    missing_soma_count=missing_soma,
                    invalid_rows_count=invalid_rows,
                    duration_filtered_count=duration_filtered,
                )
            with (
                stage_trace.span(
                    "rows_from_index_read",
                    robot_path=str(robot_path),
                    soma_path=str(soma_path),
                )
                if stage_trace and robot_path_index <= 3
                else nullcontext()
            ):
                _, robot = _single_motionlib_entry(robot_path)
                _, soma = _single_motionlib_entry(soma_path)
            _require_motionlib_keys(robot, REQUIRED_ROBOT_MOTIONLIB_KEYS, robot_path)
            _require_motionlib_keys(soma, REQUIRED_SOMA_MOTIONLIB_KEYS, soma_path)
            robot_frames = int(np.asarray(robot["dof"]).shape[0])
            soma_frames = int(np.asarray(soma["soma_joints"]).shape[0])
            target_fps = float(robot.get("fps") or input_cfg.get("target_fps") or 50.0)
            source_fps = float(soma.get("fps") or input_cfg.get("source_fps") or target_fps)
            if robot_frames <= 1 or soma_frames <= 1 or target_fps <= 0 or source_fps <= 0:
                skipped += 1
                continue
            source_duration = soma_frames / source_fps
            target_duration = robot_frames / target_fps
            if stage_trace and robot_path_index <= 3:
                stage_trace.log(
                    "rows_from_index_parse",
                    "details",
                    robot_path=str(robot_path),
                    soma_path=str(soma_path),
                    robot_frames=robot_frames,
                    soma_frames=soma_frames,
                    target_fps=target_fps,
                    source_fps=source_fps,
                    source_duration_sec=source_duration,
                    target_duration_sec=target_duration,
                )
            if target_fps >= source_fps or abs(source_duration - target_duration) > max_duration_delta:
                duration_filtered += 1
                skipped += 1
                continue
        except Exception:
            invalid_rows += 1
            skipped += 1
            continue

        rows.append(
            {
                "path": str(robot_path),
                "relative_path": robot_path.name,
                "robot_relative_path": robot_path.name,
                "soma_relative_path": soma_path.name,
                "frame_count": robot_frames,
                "source_frame_count": soma_frames,
                "source_fps": source_fps,
                "target_fps": target_fps,
                "source_duration_sec": source_duration,
                "target_duration_sec": target_duration,
                "source_bvh": str(soma.get("source_bvh", "")),
                "source_soma_proportional_path": str(soma.get("source_bvh", "")),
                "filename": robot_path.stem,
                "actor_uid": "",
                "category": "",
            }
        )
        if len(accepted_samples) < 5:
            accepted_samples.append(
                {
                    "robot_path": str(robot_path),
                    "soma_path": str(soma_path),
                    "frame_count": robot_frames,
                    "source_frame_count": soma_frames,
                    "source_fps": source_fps,
                    "target_fps": target_fps,
                }
            )
        if max_clips > 0 and len(rows) >= max_clips:
            break
    if stage_trace:
        stage_trace.log(
            "rows_from_index_row_count",
            "details",
            robot_path_count=len(robot_paths),
            row_count=len(rows),
            skipped_count=skipped,
            missing_soma_count=missing_soma,
            invalid_rows_count=invalid_rows,
            duration_filtered_count=duration_filtered,
            accepted_samples=accepted_samples,
        )
        stage_trace.log(
            "rows_from_index_filter",
            "details",
            skipped_count=skipped,
            missing_soma_count=missing_soma,
            invalid_rows_count=invalid_rows,
            duration_filtered_count=duration_filtered,
        )
        stage_trace.log("rows_from_index_sample", "details", accepted_samples=accepted_samples)
    return rows, skipped


def source_target_timing_summary(
    source: Mapping[str, Any],
    *,
    frame_count: int,
    indexing: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate metadata-level timing for a 120 Hz source BVH / 50 Hz SONIC target pair."""

    source_frames = _optional_float(source.get("move_duration_frames"))
    target_fps = _optional_float(source.get("fps"))
    source_fps = _optional_float(indexing.get("source_fps", 120.0)) or 120.0
    max_delta = _optional_float(indexing.get("max_duration_delta_sec", 0.02)) or 0.02
    summary: dict[str, Any] = {
        "status": "unknown",
        "flags": [],
        "source_fps": source_fps,
        "target_fps": target_fps if target_fps is not None else "",
        "source_frame_count": int(source_frames) if source_frames is not None else "",
    }
    if source_frames is None or target_fps is None or source_frames <= 1 or target_fps <= 0:
        return summary

    source_duration = source_frames / source_fps
    target_duration = frame_count / target_fps
    flags: list[str] = []
    if target_fps >= source_fps:
        flags.append("target_fps_not_below_source_fps")
    if frame_count > source_frames:
        flags.append("target_frame_count_exceeds_source_frame_count")
    if abs(target_duration - source_duration) > max_delta:
        flags.append("source_target_duration_mismatch")
    summary.update(
        {
            "status": "invalid" if flags else "ok",
            "flags": flags,
            "source_duration_sec": source_duration,
            "target_duration_sec": target_duration,
            "duration_delta_sec": target_duration - source_duration,
        }
    )
    return summary


def _optional_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def rows_from_index(
    config: dict[str, Any],
    data_root: Path,
    *,
    stage_trace: StageTracer | None = None,
    output_dir: Path | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    cache_path = rows_from_index_cache_path(output_dir) if output_dir is not None else None
    use_cache = cache_path is not None
    if config["input_data"].get("format") == "soma_motionlib":
        if use_cache and runtime is not None and runtime.get("distributed"):
            if is_main_process(runtime):
                rows, skipped = rows_from_soma_motionlib_pair(config, stage_trace=stage_trace)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "cache_version": 1,
                    "created_at": utc_now(),
                    "config": {
                        "format": config["input_data"].get("format"),
                        "robot_motion_dir": config["input_data"].get("robot_motion_dir"),
                        "soma_motion_dir": config["input_data"].get("soma_motion_dir"),
                        "data_root": str(data_root),
                        "max_clips": int(config["input_data"].get("max_clips", 0)),
                        "max_duration_delta_sec": float(config["input_data"].get("max_duration_delta_sec", 0.05)),
                    },
                    "rows": rows,
                    "row_count": len(rows),
                    "skipped_count": int(skipped),
                }
                cache_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
                if stage_trace:
                    stage_trace.log(
                        "rows_from_index_cache",
                        "written",
                        cache_path=str(cache_path),
                        row_count=len(rows),
                        skipped_count=int(skipped),
                    )
            distributed_barrier(runtime)
            if not is_main_process(runtime):
                if stage_trace:
                    stage_trace.log("rows_from_index_cache", "before_read", cache_path=str(cache_path))
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                rows = payload["rows"]
                skipped = int(payload["skipped_count"])
                if stage_trace:
                    stage_trace.log(
                        "rows_from_index_cache",
                        "read",
                        cache_path=str(cache_path),
                        row_count=len(rows),
                        skipped_count=skipped,
                    )
            distributed_barrier(runtime)
            return rows, skipped
        rows, skipped = rows_from_soma_motionlib_pair(config, stage_trace=stage_trace)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cache_version": 1,
                "created_at": utc_now(),
                "config": {
                    "format": config["input_data"].get("format"),
                    "robot_motion_dir": config["input_data"].get("robot_motion_dir"),
                    "soma_motion_dir": config["input_data"].get("soma_motion_dir"),
                    "data_root": str(data_root),
                    "max_clips": int(config["input_data"].get("max_clips", 0)),
                    "max_duration_delta_sec": float(config["input_data"].get("max_duration_delta_sec", 0.05)),
                },
                "rows": rows,
                "row_count": len(rows),
                "skipped_count": int(skipped),
            }
            cache_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            if stage_trace:
                stage_trace.log(
                    "rows_from_index_cache",
                    "written",
                    cache_path=str(cache_path),
                    row_count=len(rows),
                    skipped_count=int(skipped),
                )
        return rows, skipped
    indexing = config["input_data"].get("indexing", {})
    if indexing.get("index_csv"):
        rows, skipped = rows_from_csv_index(config, data_root)
    else:
        rows, skipped = rows_from_jsonl_index(config, data_root)
    if stage_trace:
        stage_trace.log(
            "rows_from_index_row_count",
            "details",
            row_count=len(rows),
            skipped_count=int(skipped),
            index_path=str(index_path_from_config(config)),
            data_root=str(data_root),
        )
    return rows, skipped


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def split_rows(rows: list[dict[str, Any]], validation_ratio: float, hash_salt: str) -> None:
    threshold = int(validation_ratio * 1_000_000)
    for row in rows:
        value = stable_hash_int(row["relative_path"] + hash_salt) % 1_000_000
        row["split"] = "validation" if value < threshold else "train"
    if not any(row["split"] == "validation" for row in rows):
        rows[-1]["split"] = "validation"
    if not any(row["split"] == "train" for row in rows):
        rows[0]["split"] = "train"


class KinWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        split: str,
        config: dict[str, Any],
        skeleton_feature_lookup: SkeletonAEFeatureLookup | None = None,
    ) -> None:
        self.rows = [row for row in rows if row["split"] == split]
        self.config = config
        self.skeleton_feature_lookup = skeleton_feature_lookup
        self.input_format = str(config["input_data"].get("format", "npz"))
        self.data_root = Path(config["input_data"].get("data_root", config["input_data"].get("robot_motion_dir", ".")))
        self.window = int(config["features"]["future_window_frames"])
        self.step = int(config["features"]["future_step"])
        self.frame_stride = int(config["training"]["frame_stride"])
        self.chunk_frames = int(config["training"]["loader_chunk_frames"])
        self.samples: list[tuple[int, int, int]] = []
        for row_idx, row in enumerate(self.rows):
            frame_count = int(row["frame_count"])
            max_start = max(0, frame_count - 1)
            start = 0
            while start <= max_start:
                stop = min(frame_count, start + self.chunk_frames * self.frame_stride)
                if stop > start:
                    self.samples.append((row_idx, start, stop))
                start = stop
        if not self.samples:
            raise ValueError(f"no samples for split {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_idx, start, stop = self.samples[index]
        row = self.rows[row_idx]
        if self.input_format == "soma_motionlib":
            arrays = load_soma_motionlib_arrays(row, self.config)
            total_frames = min(
                arrays["soma_joints"].shape[0],
                arrays["soma_root_quat"].shape[0],
                arrays["joint_pos"].shape[0],
                arrays["joint_vel"].shape[0],
                arrays["root_pos"].shape[0],
                arrays["root_rot"].shape[0],
            )
        else:
            arrays = load_arrays(self.data_root / row["relative_path"])
            total_frames = min(
                arrays["joint_pos"].shape[0],
                arrays["joint_vel"].shape[0],
                arrays["body_pos_w"].shape[0],
                arrays["body_quat_w"].shape[0],
            )
        stop = min(stop, total_frames)
        if start >= stop:
            start = 0
            stop = total_frames
        frame_indices = np.arange(start, stop, self.frame_stride, dtype=np.int64)
        if self.input_format == "soma_motionlib":
            motion, skeleton, target = build_soma_motionlib_features(
                arrays,
                frame_indices,
                self.window,
                self.step,
                self.config,
            )
        else:
            motion, skeleton, target = build_features(
                arrays,
                frame_indices,
                self.window,
                self.step,
                self.config,
            )
        if self.skeleton_feature_lookup is not None:
            embedding = self.skeleton_feature_lookup.embedding_for_row(row)
            skeleton = np.repeat(embedding[None, :], motion.shape[0], axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(motion), torch.from_numpy(skeleton), torch.from_numpy(target)


def collate_chunks(
    chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    motion = torch.cat([item[0] for item in chunks], dim=0)
    skeleton = torch.cat([item[1] for item in chunks], dim=0)
    target = torch.cat([item[2] for item in chunks], dim=0)
    return motion, skeleton, target


class RunningStats:
    def __init__(self, dim: int, device: torch.device) -> None:
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.sum = torch.zeros(dim, dtype=torch.float64, device=device)
        self.sumsq = torch.zeros(dim, dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor) -> None:
        data = x.to(torch.float64)
        self.count += data.shape[0]
        self.sum += data.sum(dim=0)
        self.sumsq += (data * data).sum(dim=0)

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.sum / self.count.clamp_min(1.0)
        var = self.sumsq / self.count.clamp_min(1.0) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return mean.to(torch.float32), std.to(torch.float32)


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(prev, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ConcatRetargeter(nn.Module):
    def __init__(self, motion_dim: int, skeleton_dim: int, output_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.net = build_mlp(
            motion_dim + skeleton_dim,
            int(cfg["hidden_dim"]),
            output_dim,
            int(cfg["num_layers"]),
            float(cfg.get("dropout", 0.0)),
        )

    def forward(self, motion: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([motion, skeleton], dim=-1))


class FilmRetargeter(nn.Module):
    def __init__(self, motion_dim: int, skeleton_dim: int, output_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        num_layers = int(cfg["num_layers"])
        self.input = nn.Linear(motion_dim, hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, num_layers - 1)))
        self.film = nn.Linear(skeleton_dim, 2 * hidden_dim * max(1, num_layers))
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.SiLU()
        self.num_layers = max(1, num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, motion: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        film = self.film(skeleton).view(skeleton.shape[0], self.num_layers, 2, self.hidden_dim)
        h = self.activation(self.input(motion))
        gamma, beta = film[:, 0, 0], film[:, 0, 1]
        h = h * (1.0 + gamma) + beta
        for idx, layer in enumerate(self.layers, start=1):
            h = self.activation(layer(h))
            gamma, beta = film[:, idx, 0], film[:, idx, 1]
            h = h * (1.0 + gamma) + beta
        return self.output(h)


class AdapterRetargeter(nn.Module):
    def __init__(self, motion_dim: int, skeleton_dim: int, output_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        adapter_dim = int(cfg.get("adapter_dim", max(32, hidden_dim // 4)))
        self.motion = build_mlp(motion_dim, hidden_dim, hidden_dim, int(cfg["num_layers"]), float(cfg.get("dropout", 0.0)))
        self.adapter = nn.Sequential(
            nn.Linear(skeleton_dim, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, motion: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        return self.output(self.motion(motion) + self.adapter(skeleton))


class ExpertRetargeter(nn.Module):
    def __init__(self, motion_dim: int, skeleton_dim: int, output_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        num_experts = int(cfg.get("num_experts", 4))
        self.experts = nn.ModuleList(
            build_mlp(motion_dim, hidden_dim, hidden_dim, int(cfg["num_layers"]), float(cfg.get("dropout", 0.0)))
            for _ in range(num_experts)
        )
        self.gate = nn.Linear(skeleton_dim, num_experts)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, motion: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(skeleton), dim=-1)
        expert_h = torch.stack([expert(motion) for expert in self.experts], dim=1)
        h = torch.sum(expert_h * weights.unsqueeze(-1), dim=1)
        return self.output(h)


def make_model(motion_dim: int, skeleton_dim: int, output_dim: int, config: dict[str, Any]) -> nn.Module:
    variant_type = config["variant"]["type"]
    cfg = config["model"]
    if variant_type == "concat":
        return ConcatRetargeter(motion_dim, skeleton_dim, output_dim, cfg)
    if variant_type == "film":
        return FilmRetargeter(motion_dim, skeleton_dim, output_dim, cfg)
    if variant_type == "adapter":
        return AdapterRetargeter(motion_dim, skeleton_dim, output_dim, cfg)
    if variant_type == "expert":
        return ExpertRetargeter(motion_dim, skeleton_dim, output_dim, cfg)
    raise ValueError(f"unknown variant type: {variant_type}")


def compute_or_load_stats(
    output_dir: Path,
    train_dataset: KinWindowDataset,
    config: dict[str, Any],
    device: torch.device,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    stats_path = output_dir / "stats" / "normalization.pt"
    skeleton_mean_key, skeleton_std_key = skeleton_normalization_keys(config)
    if stats_path.exists():
        payload = torch.load(stats_path, map_location=device, weights_only=False)
        stats = {key: value.to(device) for key, value in payload.items() if torch.is_tensor(value)}
        require_normalization_keys(stats, config)
        return stats
    if runtime is not None and runtime.get("distributed") and not is_main_process(runtime):
        distributed_barrier(runtime)
        payload = torch.load(stats_path, map_location=device, weights_only=False)
        stats = {key: value.to(device) for key, value in payload.items() if torch.is_tensor(value)}
        require_normalization_keys(stats, config)
        return stats

    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_chunks,
    )
    stats_device = torch.device("cpu") if is_skeleton_ae_enabled(config) else device
    motion_stats = None
    skeleton_stats = None
    target_stats = None
    seen = 0
    max_frames = int(config["normalization"]["max_frames"])
    frames_per_chunk = int(config["normalization"].get("frames_per_chunk", 1024))
    for motion, skeleton, target in loader:
        if motion.shape[0] > frames_per_chunk:
            select = torch.randperm(motion.shape[0])[:frames_per_chunk]
            motion, skeleton, target = motion[select], skeleton[select], target[select]
        motion = motion.to(stats_device)
        skeleton = skeleton.to(stats_device)
        target = target.to(stats_device)
        if motion_stats is None:
            motion_stats = RunningStats(motion.shape[-1], stats_device)
            skeleton_stats = RunningStats(skeleton.shape[-1], stats_device)
            target_stats = RunningStats(target.shape[-1], stats_device)
        motion_stats.update(motion)
        skeleton_stats.update(skeleton)
        target_stats.update(target)
        seen += int(motion.shape[0])
        if seen >= max_frames:
            break

    if motion_stats is None or skeleton_stats is None or target_stats is None:
        raise RuntimeError("could not compute normalization stats")
    motion_mean, motion_std = motion_stats.finalize()
    skeleton_mean, skeleton_std = skeleton_stats.finalize()
    target_mean, target_std = target_stats.finalize()
    payload = {
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        skeleton_mean_key: skeleton_mean,
        skeleton_std_key: skeleton_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in payload.items()}, stats_path)
    if runtime is not None and runtime.get("distributed"):
        distributed_barrier(runtime)
    return {key: value.to(device) for key, value in payload.items()}


def skeleton_normalization_keys(config: Mapping[str, Any]) -> tuple[str, str]:
    if is_skeleton_ae_enabled(config):
        return "skeleton_embedding_mean", "skeleton_embedding_std"
    return "skeleton_mean", "skeleton_std"


def require_normalization_keys(stats: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> None:
    required = ["motion_mean", "motion_std", "target_mean", "target_std", *skeleton_normalization_keys(config)]
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"normalization stats missing required keys: {', '.join(missing)}")


def skeleton_stats_pair(
    stats: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_key, std_key = skeleton_normalization_keys(config)
    return stats[mean_key], stats[std_key]


def skeleton_feature_dim(stats: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> int:
    mean, _std = skeleton_stats_pair(stats, config)
    return int(mean.numel())


def normalize_batch(
    motion: torch.Tensor,
    skeleton: torch.Tensor,
    target: torch.Tensor,
    stats: dict[str, torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    skeleton_mean, skeleton_std = skeleton_stats_pair(stats, config)
    return (
        (motion - stats["motion_mean"]) / stats["motion_std"],
        (skeleton - skeleton_mean) / skeleton_std,
        (target - stats["target_mean"]) / stats["target_std"],
    )


def loss_and_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    target_raw: torch.Tensor,
    stats: dict[str, torch.Tensor],
    command_dim: int,
    joint_dim: int,
    root_pose_dim: int,
    root_pos_enabled: bool,
    command_weight: float,
    root_pos_weight: float,
    root_rot_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_command = pred_norm[:, :command_dim]
    target_command = target_norm[:, :command_dim]
    pred_anchor = pred_norm[:, command_dim:]
    target_anchor = target_norm[:, command_dim:]
    command_loss = torch.mean((pred_command - target_command) ** 2)
    if root_pos_enabled:
        window = root_pose_dim // 9
        pred_pose = pred_anchor.reshape(pred_anchor.shape[0], window, 9)
        target_pose = target_anchor.reshape(target_anchor.shape[0], window, 9)
        root_pos_loss = torch.mean((pred_pose[..., :3] - target_pose[..., :3]) ** 2)
        root_rot_loss = torch.mean((pred_pose[..., 3:] - target_pose[..., 3:]) ** 2)
        anchor_loss = torch.mean((pred_anchor - target_anchor) ** 2)
        loss = (
            command_weight * command_loss
            + root_pos_weight * root_pos_loss
            + root_rot_weight * root_rot_loss
        )
    else:
        anchor_loss = torch.mean((pred_anchor - target_anchor) ** 2)
        root_pos_loss = pred_anchor.new_tensor(float("nan"))
        root_rot_loss = anchor_loss
        loss = command_weight * command_loss + root_rot_weight * anchor_loss

    with torch.no_grad():
        pred_raw = pred_norm * stats["target_std"] + stats["target_mean"]
        raw_error = pred_raw - target_raw
        command_raw = raw_error[:, :command_dim]
        anchor_raw = raw_error[:, command_dim:]
        command_frame_dim = joint_dim * 2
        joint_error = command_raw.reshape(command_raw.shape[0], -1, command_frame_dim)
        joint_pos_error = joint_error[..., :joint_dim]
        joint_vel_error = joint_error[..., joint_dim:]
        root_pos_rmse = float("nan")
        root_rot6d_error = anchor_raw.reshape(anchor_raw.shape[0], -1, 6)
        if root_pos_enabled:
            window = root_pose_dim // 9
            root_pose = anchor_raw.reshape(anchor_raw.shape[0], window, 9)
            root_pos_rmse = float(torch.sqrt(torch.mean(root_pose[..., :3] ** 2)).item())
            root_rot6d_error = root_pose[..., 3:]
        metrics = {
            "loss": float(loss.detach().item()),
            "command_mse_norm": float(command_loss.detach().item()),
            "anchor_mse_norm": float(anchor_loss.detach().item()),
            "root_pos_mse_norm": float(root_pos_loss.detach().item()),
            "root_rot_mse_norm": float(root_rot_loss.detach().item()),
            "joint_pos_rmse_raw": float(torch.sqrt(torch.mean(joint_pos_error**2)).item()),
            "joint_vel_rmse_raw": float(torch.sqrt(torch.mean(joint_vel_error**2)).item()),
            "anchor_rmse_raw": float(torch.sqrt(torch.mean(anchor_raw**2)).item()),
            "root_pos_rmse_raw": root_pos_rmse,
            "root_rot6d_rmse_raw": float(torch.sqrt(torch.mean(root_rot6d_error**2)).item()),
        }
    return loss, metrics


def validate(
    model: nn.Module,
    loader: DataLoader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    command_dim: int,
    joint_dim: int,
    root_pose_dim: int,
) -> dict[str, float]:
    model.eval()
    rows = []
    max_batches = int(config["training"]["validation_batches"])
    command_weight = float(config["training"].get("command_loss_weight", 1.0))
    root_pos_weight = float(
        config["training"].get(
            "root_pos_loss_weight",
            config["training"].get("anchor_loss_weight", 1.0),
        )
    )
    root_rot_weight = float(
        config["training"].get(
            "root_rot_loss_weight",
            config["training"].get("anchor_loss_weight", 1.0),
        )
    )
    with torch.no_grad():
        for batch_idx, (motion, skeleton, target) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            motion = motion.to(device, non_blocking=True)
            skeleton = skeleton.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            motion_n, skeleton_n, target_n = normalize_batch(motion, skeleton, target, stats, config)
            pred_n = model(motion_n, skeleton_n)
            _, metrics = loss_and_metrics(
                pred_n,
                target_n,
                target,
                stats,
                command_dim,
                joint_dim,
                root_pose_dim,
                include_root_pos_target(config),
                command_weight,
                root_pos_weight,
                root_rot_weight,
            )
            rows.append(metrics)
    model.train()
    if not rows:
        return {}
    return {f"validation/{key}": float(np.mean([row[key] for row in rows])) for key in rows[0]}


def visual_validation_due(
    config: dict[str, Any],
    step: int,
    *,
    now: float | None = None,
    last_time: float | None = None,
) -> bool:
    cfg = config.get("visual_validation", {})
    if not cfg.get("enabled", False):
        return False
    every_steps = int(cfg.get("every_steps", 0))
    if every_steps > 0 and step > 0 and step % every_steps == 0:
        return True
    every_seconds = visual_validation_interval_seconds(config)
    if every_seconds is None or every_seconds <= 0 or now is None or last_time is None:
        return False
    return step > 0 and now - last_time >= every_seconds


def run_visual_validation(
    *,
    model: nn.Module,
    validation_rows: Sequence[Mapping[str, Any]],
    stats: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    output_dir: Path,
    step: int,
    joint_dim: int,
    wandb_run: Any,
    skeleton_feature_lookup: SkeletonAEFeatureLookup | None = None,
) -> dict[str, float]:
    cfg = config.get("visual_validation", {})
    started = time.perf_counter()
    vis_dir = output_dir / "visual_validation" / f"step_{step:08d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    report_path = vis_dir / "summary.json"
    rows = _select_visual_rows(
        validation_rows,
        count=int(cfg.get("num_videos", 8)),
        salt=f"{config['variant']['name']}:{config['training']['seed']}",
    )
    if not rows:
        summary = {
            "step": step,
            "status": "blocked",
            "message": "no validation rows were available for visual validation",
            "reports": [],
        }
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"visual_validation/videos_ok": 0.0, "visual_validation/videos_failed": 0.0}

    try:
        render_deps = _load_visual_render_deps()
    except Exception as exc:
        summary = {
            "step": step,
            "status": "blocked",
            "message": f"visual rendering dependencies are unavailable: {exc}",
            "reports": [],
        }
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"visual_validation/videos_ok": 0.0, "visual_validation/videos_failed": float(len(rows))}

    model_xml = Path(str(cfg.get("g1_model_xml", "")))
    if not model_xml.exists():
        summary = {
            "step": step,
            "status": "blocked",
            "message": f"g1_model_xml is missing: {model_xml}",
            "reports": [],
        }
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"visual_validation/videos_ok": 0.0, "visual_validation/videos_failed": float(len(rows))}

    g1_model = render_deps["load_g1_kinematic_model"](model_xml)
    was_training = model.training
    model.eval()
    reports: list[dict[str, Any]] = []
    wandb_payload: dict[str, Any] = {}
    try:
        for index, row in enumerate(rows):
            try:
                report = _render_visual_validation_clip(
                    model=model,
                    row=row,
                    stats=stats,
                    device=device,
                    config=config,
                    output_dir=vis_dir,
                    index=index,
                    step=step,
                    joint_dim=joint_dim,
                    render_deps=render_deps,
                    g1_model=g1_model,
                    skeleton_feature_lookup=skeleton_feature_lookup,
                )
            except Exception as exc:
                report = {
                    "index": index,
                    "filename": row.get("filename", ""),
                    "relative_path": row.get("relative_path", ""),
                    "combined_status": "failed",
                    "message": str(exc),
                }
            reports.append(report)
            if wandb_run is not None and report.get("combined_status") == "ok":
                try:
                    import wandb

                    video_path = Path(str(report["combined_video"]))
                    fps = float(report.get("fps") or cfg.get("fps") or 50.0)
                    key = f"visual_validation/{index:02d}_{_safe_metric_name(str(report.get('filename', index)))}"
                    wandb_payload[key] = wandb.Video(str(video_path), fps=int(round(fps)), format="mp4")
                except Exception as exc:
                    report["wandb_video_status"] = "failed"
                    report["wandb_video_message"] = str(exc)
    finally:
        if was_training:
            model.train()

    if wandb_run is not None and wandb_payload:
        wandb_run.log(wandb_payload, step=step)

    ok_count = sum(1 for report in reports if report.get("combined_status") == "ok")
    failed_count = len(reports) - ok_count
    summary = {
        "step": step,
        "status": "ok" if ok_count else "failed",
        "variant": config["variant"]["name"],
        "duration_sec": float(cfg.get("duration_sec", 4.0)),
        "requested_videos": int(cfg.get("num_videos", 8)),
        "videos_ok": ok_count,
        "videos_failed": failed_count,
        "elapsed_sec": time.perf_counter() - started,
        "reports": reports,
    }
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "visual_validation/videos_ok": float(ok_count),
        "visual_validation/videos_failed": float(failed_count),
        "visual_validation/elapsed_sec": float(summary["elapsed_sec"]),
    }


def _load_visual_render_deps() -> dict[str, Any]:
    _ensure_ffmpeg_on_path()

    from online_retarget.data.g1_quality import g1_fk_body_positions, load_g1_kinematic_model
    from online_retarget.data.review_clips import (
        G1_CAPSULE_IGNORE_BODIES,
        ReviewClipExportConfig,
        _SourceCapsuleRenderer,
        _source_capsule_body_names,
        _source_capsule_edges_from_motion,
        _render_capsule_3d_video,
    )
    from online_retarget.data.sonic_review_clips import (
        SONIC_BODY_NAMES,
        SONIC_PRUNED_BODY_NAMES,
        SONIC_PRUNED_CAPSULE_EDGES,
    )
    from online_retarget.data.windowed_builder import (
        global_body_position_maps_from_bvh,
        parse_bvh_motion,
    )

    return {
        "g1_fk_body_positions": g1_fk_body_positions,
        "load_g1_kinematic_model": load_g1_kinematic_model,
        "G1_CAPSULE_IGNORE_BODIES": G1_CAPSULE_IGNORE_BODIES,
        "ReviewClipExportConfig": ReviewClipExportConfig,
        "_SourceCapsuleRenderer": _SourceCapsuleRenderer,
        "_source_capsule_body_names": _source_capsule_body_names,
        "_source_capsule_edges_from_motion": _source_capsule_edges_from_motion,
        "_render_capsule_3d_video": _render_capsule_3d_video,
        "global_body_position_maps_from_bvh": global_body_position_maps_from_bvh,
        "parse_bvh_motion": parse_bvh_motion,
        "SONIC_BODY_NAMES": SONIC_BODY_NAMES,
        "SONIC_PRUNED_BODY_NAMES": SONIC_PRUNED_BODY_NAMES,
        "SONIC_PRUNED_CAPSULE_EDGES": SONIC_PRUNED_CAPSULE_EDGES,
    }


def _ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg") is not None:
        return
    try:
        import imageio_ffmpeg

        ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return
    if not ffmpeg_exe.exists():
        return
    shim_dir = Path(os.environ.get("ONLINE_RETARGET_FFMPEG_SHIM", "/tmp/online_retarget_ffmpeg_bin"))
    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / "ffmpeg"
        if not shim_path.exists():
            shim_path.symlink_to(ffmpeg_exe)
        current_path = os.environ.get("PATH", "")
        path_parts = current_path.split(os.pathsep) if current_path else []
        if str(shim_dir) not in path_parts:
            os.environ["PATH"] = str(shim_dir) + (os.pathsep + current_path if current_path else "")
    except OSError:
        return


def _render_visual_validation_clip(
    *,
    model: nn.Module,
    row: Mapping[str, Any],
    stats: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    output_dir: Path,
    index: int,
    step: int,
    joint_dim: int,
    render_deps: Mapping[str, Any],
    g1_model: Any,
    skeleton_feature_lookup: SkeletonAEFeatureLookup | None = None,
) -> dict[str, Any]:
    if config["input_data"].get("format") == "soma_motionlib":
        return _render_motionlib_visual_validation_clip(
            model=model,
            row=row,
            stats=stats,
            device=device,
            config=config,
            output_dir=output_dir,
            index=index,
            step=step,
            joint_dim=joint_dim,
            render_deps=render_deps,
            g1_model=g1_model,
            skeleton_feature_lookup=skeleton_feature_lookup,
        )

    cfg = config.get("visual_validation", {})
    clip_name = _safe_filename(str(row.get("filename") or Path(str(row["relative_path"])).stem))
    clip_dir = output_dir / f"{index:02d}_{clip_name}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    source_video = clip_dir / "source_soma_bvh_capsules.mp4"
    dataset_video = clip_dir / "dataset_g1_capsules.mp4"
    inference_video = clip_dir / "inference_g1_capsules.mp4"
    combined_video = clip_dir / "source_dataset_inference.mp4"
    metadata_path = clip_dir / "metadata.json"

    arrays, fps = _load_visual_npz(Path(config["input_data"]["data_root"]) / str(row["relative_path"]))
    total_frames = min(
        arrays["joint_pos"].shape[0],
        arrays["joint_vel"].shape[0],
        arrays["body_pos_w"].shape[0],
        arrays["body_quat_w"].shape[0],
    )
    requested_frames = max(1, int(round(float(cfg.get("duration_sec", 4.0)) * fps)))
    frame_count = min(total_frames, requested_frames)
    render_width = int(cfg.get("width", 640))
    render_height = int(cfg.get("height", 360))
    render_config = render_deps["ReviewClipExportConfig"](
        render_max_frames=frame_count,
        render_width=render_width,
        render_height=render_height,
        fps=fps,
        source_position_scale=float(cfg.get("source_position_scale", 0.01)),
        model_xml=Path(str(cfg.get("g1_model_xml", ""))),
    )

    source_bvh = _resolve_source_bvh(row, config, output_dir)
    if source_bvh is not None:
        source_report = _render_time_aligned_source_bvh(
            source_bvh,
            source_video,
            render_config=render_config,
            target_fps=fps,
            frame_count=frame_count,
            render_deps=render_deps,
        )
    else:
        source_report = _render_missing_panel(
            render_deps=render_deps,
            video_path=source_video,
            render_config=render_config,
            frame_count=frame_count,
            label="source bvh unavailable",
        )

    target_body_pos = arrays["body_pos_w"][:frame_count]
    dataset_report = render_deps["_render_capsule_3d_video"](
        frames=_sonic_body_pos_frames(
            target_body_pos,
            render_deps["SONIC_BODY_NAMES"],
            render_deps["SONIC_PRUNED_BODY_NAMES"],
        ),
        edges=_sonic_edges(render_deps["SONIC_BODY_NAMES"], render_deps["SONIC_PRUNED_CAPSULE_EDGES"]),
        video_path=dataset_video,
        config=render_config,
        label="dataset g1 body_pos_w",
        up_axis=2,
        capsule_color=(61, 107, 160),
        key_color=(139, 91, 41),
    )

    prediction = _predict_visual_g1_state(
        model=model,
        arrays=arrays,
        frame_count=frame_count,
        stats=stats,
        device=device,
        config=config,
        joint_dim=joint_dim,
        fallback_root_pos=arrays["body_pos_w"][:frame_count, 0],
        fallback_root_quat=arrays["body_quat_w"][:frame_count, 0],
        row=row,
        skeleton_feature_lookup=skeleton_feature_lookup,
    )
    inference_frames = _g1_prediction_frames(
        prediction["joint_pos"],
        root_pos=prediction["root_pos"],
        root_quat=prediction.get("root_quat"),
        root_euler=prediction.get("root_euler"),
        g1_model=g1_model,
        render_deps=render_deps,
    )
    inference_report = render_deps["_render_capsule_3d_video"](
        frames=inference_frames,
        edges=_g1_edges(g1_model, render_deps["G1_CAPSULE_IGNORE_BODIES"]),
        video_path=inference_video,
        config=render_config,
        label="inference g1 fk",
        up_axis=2,
        capsule_color=(142, 77, 117),
        key_color=(122, 89, 35),
    )

    combine_report = _combine_panel_videos(
        (source_video, dataset_video, inference_video),
        combined_video,
        fps=int(round(fps)),
    )
    metadata = {
        "step": step,
        "index": index,
        "filename": row.get("filename", ""),
        "relative_path": row.get("relative_path", ""),
        "source_soma_proportional_path": row.get("source_soma_proportional_path", ""),
        "source_bvh": str(source_bvh) if source_bvh is not None else "",
        "fps": fps,
        "frames": frame_count,
        "duration_sec": frame_count / fps if fps > 0 else 0.0,
        "source_render": source_report,
        "dataset_render": dataset_report,
        "inference_render": inference_report,
        "combine": combine_report,
        "combined_video": str(combined_video),
        "note": (
            "Panels are source SOMA proportional BVH capsules, paired dataset G1 body_pos_w capsules, "
            "and model inference rendered by G1 MJCF FK from predicted joint_pos."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "index": index,
        "filename": row.get("filename", ""),
        "relative_path": row.get("relative_path", ""),
        "fps": fps,
        "frames": frame_count,
        "source_status": source_report.get("status"),
        "dataset_status": dataset_report.get("status"),
        "inference_status": inference_report.get("status"),
        "combined_status": combine_report.get("status"),
        "combined_video": str(combined_video),
        "metadata": str(metadata_path),
    }


def _render_motionlib_visual_validation_clip(
    *,
    model: nn.Module,
    row: Mapping[str, Any],
    stats: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    output_dir: Path,
    index: int,
    step: int,
    joint_dim: int,
    render_deps: Mapping[str, Any],
    g1_model: Any,
    skeleton_feature_lookup: SkeletonAEFeatureLookup | None = None,
) -> dict[str, Any]:
    cfg = config.get("visual_validation", {})
    clip_name = _safe_filename(str(row.get("filename") or Path(str(row["relative_path"])).stem))
    clip_dir = output_dir / f"{index:02d}_{clip_name}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    source_video = clip_dir / "source_soma_bvh_capsules.mp4"
    dataset_video = clip_dir / "dataset_g1_capsules.mp4"
    inference_video = clip_dir / "inference_g1_capsules.mp4"
    combined_video = clip_dir / "source_dataset_inference.mp4"
    metadata_path = clip_dir / "metadata.json"

    arrays = load_soma_motionlib_arrays(row, config)
    robot_root = _load_motionlib_robot_root(row, config)
    fps = float(arrays["fps"])
    total_frames = min(arrays["joint_pos"].shape[0], robot_root["root_pos"].shape[0], robot_root["root_quat"].shape[0])
    requested_frames = max(1, int(round(float(cfg.get("duration_sec", 4.0)) * fps)))
    frame_count = min(total_frames, requested_frames)
    render_width = int(cfg.get("width", 640))
    render_height = int(cfg.get("height", 360))
    render_config = render_deps["ReviewClipExportConfig"](
        render_max_frames=frame_count,
        render_width=render_width,
        render_height=render_height,
        fps=fps,
        source_position_scale=float(cfg.get("source_position_scale", 0.01)),
        model_xml=Path(str(cfg.get("g1_model_xml", ""))),
    )

    source_bvh = _resolve_source_bvh(row, config, output_dir)
    if source_bvh is not None:
        source_report = _render_time_aligned_source_bvh(
            source_bvh,
            source_video,
            render_config=render_config,
            target_fps=fps,
            frame_count=frame_count,
            render_deps=render_deps,
        )
    else:
        source_report = render_deps["_render_capsule_3d_video"](
            frames=_soma_motionlib_source_frames(arrays["soma_joints"][:frame_count], arrays.get("joint_names")),
            edges=_soma_edges(arrays.get("joint_names")),
            video_path=source_video,
            config=render_config,
            label="source soma motionlib",
            up_axis=2,
            capsule_color=(48, 132, 83),
            key_color=(132, 103, 34),
        )

    root_pos = robot_root["root_pos"][:frame_count]
    root_quat = robot_root["root_quat"][:frame_count]
    dataset_frames = _g1_prediction_frames(
        arrays["joint_pos"][:frame_count],
        root_pos=root_pos,
        root_quat=root_quat,
        g1_model=g1_model,
        render_deps=render_deps,
    )
    dataset_report = render_deps["_render_capsule_3d_video"](
        frames=dataset_frames,
        edges=_g1_edges(g1_model, render_deps["G1_CAPSULE_IGNORE_BODIES"]),
        video_path=dataset_video,
        config=render_config,
        label="dataset g1 fk",
        up_axis=2,
        capsule_color=(61, 107, 160),
        key_color=(139, 91, 41),
    )

    prediction = _predict_motionlib_visual_g1_state(
        model=model,
        arrays=arrays,
        frame_count=frame_count,
        stats=stats,
        device=device,
        config=config,
        joint_dim=joint_dim,
        fallback_root_pos=root_pos,
        fallback_root_quat=root_quat,
        row=row,
        skeleton_feature_lookup=skeleton_feature_lookup,
    )
    inference_frames = _g1_prediction_frames(
        prediction["joint_pos"],
        root_pos=prediction["root_pos"],
        root_quat=prediction.get("root_quat"),
        root_euler=prediction.get("root_euler"),
        g1_model=g1_model,
        render_deps=render_deps,
    )
    inference_report = render_deps["_render_capsule_3d_video"](
        frames=inference_frames,
        edges=_g1_edges(g1_model, render_deps["G1_CAPSULE_IGNORE_BODIES"]),
        video_path=inference_video,
        config=render_config,
        label="inference g1 fk",
        up_axis=2,
        capsule_color=(142, 77, 117),
        key_color=(122, 89, 35),
    )

    combine_report = _combine_panel_videos(
        (source_video, dataset_video, inference_video),
        combined_video,
        fps=int(round(fps)),
    )
    metadata = {
        "step": step,
        "index": index,
        "filename": row.get("filename", ""),
        "relative_path": row.get("relative_path", ""),
        "source_bvh": str(source_bvh) if source_bvh is not None else "",
        "fps": fps,
        "frames": frame_count,
        "duration_sec": frame_count / fps if fps > 0 else 0.0,
        "source_render": source_report,
        "dataset_render": dataset_report,
        "inference_render": inference_report,
        "combine": combine_report,
        "combined_video": str(combined_video),
        "note": (
            "Panels are source SOMA proportional BVH capsules, paired dataset G1 FK from robot motionlib, "
            "and model inference rendered by G1 MJCF FK from predicted joint_pos."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "index": index,
        "filename": row.get("filename", ""),
        "relative_path": row.get("relative_path", ""),
        "fps": fps,
        "frames": frame_count,
        "source_status": source_report.get("status"),
        "dataset_status": dataset_report.get("status"),
        "inference_status": inference_report.get("status"),
        "combined_status": combine_report.get("status"),
        "combined_video": str(combined_video),
        "metadata": str(metadata_path),
    }


def _render_time_aligned_source_bvh(
    bvh_path: Path,
    video_path: Path,
    *,
    render_config: Any,
    target_fps: float,
    frame_count: int,
    render_deps: Mapping[str, Any],
) -> dict[str, object]:
    try:
        text = bvh_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        motion = render_deps["parse_bvh_motion"](text)
        body_names = render_deps["_source_capsule_body_names"](motion)
        edges = render_deps["_source_capsule_edges_from_motion"](motion, body_names)
        source_frames = render_deps["global_body_position_maps_from_bvh"](
            motion,
            body_names=body_names,
            position_scale=render_config.source_position_scale,
        )
        source_fps = 1.0 / float(motion.frame_time) if float(motion.frame_time) > 0 else target_fps
        aligned_frames, source_indices = _time_align_frame_maps(
            source_frames,
            source_fps=source_fps,
            target_fps=target_fps,
            frame_count=frame_count,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"status": "failed", "message": f"Could not load BVH source motion: {exc}"}

    report = render_deps["_render_capsule_3d_video"](
        frames=aligned_frames,
        edges=edges,
        video_path=video_path,
        config=render_config,
        label="source bvh 3d capsules",
        up_axis=1,
        capsule_color=(48, 132, 83),
        key_color=(132, 103, 34),
    )
    report.update(
        {
            "alignment": "source_bvh_time_to_target_fps",
            "source_fps": source_fps,
            "target_fps": target_fps,
            "source_total_frames": len(source_frames),
            "target_frames": frame_count,
            "source_index_first": int(source_indices[0]) if source_indices else "",
            "source_index_last": int(source_indices[-1]) if source_indices else "",
            "source_time_span_sec": (source_indices[-1] / source_fps) if source_indices and source_fps > 0 else 0.0,
            "target_time_span_sec": ((frame_count - 1) / target_fps) if frame_count > 0 and target_fps > 0 else 0.0,
        }
    )
    return report


def _time_align_frame_maps(
    frames: Sequence[Mapping[str, tuple[float, float, float]]],
    *,
    source_fps: float,
    target_fps: float,
    frame_count: int,
) -> tuple[list[Mapping[str, tuple[float, float, float]]], list[int]]:
    if not frames or frame_count <= 0:
        return [], []
    if source_fps <= 0 or target_fps <= 0:
        indices = list(range(min(frame_count, len(frames))))
    else:
        ratio = source_fps / target_fps
        indices = [
            min(len(frames) - 1, max(0, int(math.floor(frame_index * ratio + 1e-6))))
            for frame_index in range(frame_count)
        ]
    return [frames[index] for index in indices], indices


def _predict_visual_g1_state(
    *,
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    frame_count: int,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    joint_dim: int,
    fallback_root_pos: np.ndarray,
    fallback_root_quat: np.ndarray,
    row: Mapping[str, Any] | None = None,
    skeleton_feature_lookup: SkeletonAEFeatureLookup | None = None,
) -> dict[str, np.ndarray]:
    frame_indices = np.arange(frame_count, dtype=np.int64)
    motion, skeleton, _ = build_features(
        dict(arrays),
        frame_indices,
        int(config["features"]["future_window_frames"]),
        int(config["features"]["future_step"]),
        config,
    )
    if skeleton_feature_lookup is not None:
        if row is None:
            raise ValueError("Skeleton AE visual prediction requires the source row")
        embedding = skeleton_feature_lookup.embedding_for_row(row)
        skeleton = np.repeat(embedding[None, :], motion.shape[0], axis=0).astype(np.float32, copy=False)
    return _predict_g1_state_from_features(
        model=model,
        motion=motion,
        skeleton=skeleton,
        stats=stats,
        device=device,
        config=config,
        joint_dim=joint_dim,
        fallback_root_pos=fallback_root_pos,
        fallback_root_quat=fallback_root_quat,
    )


def _predict_motionlib_visual_g1_state(
    *,
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    frame_count: int,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    joint_dim: int,
    fallback_root_pos: np.ndarray,
    fallback_root_quat: np.ndarray,
    row: Mapping[str, Any] | None = None,
    skeleton_feature_lookup: SkeletonAEFeatureLookup | None = None,
) -> dict[str, np.ndarray]:
    frame_indices = np.arange(frame_count, dtype=np.int64)
    motion, skeleton, _ = build_soma_motionlib_features(
        dict(arrays),
        frame_indices,
        int(config["features"]["future_window_frames"]),
        int(config["features"]["future_step"]),
        config,
    )
    if skeleton_feature_lookup is not None:
        if row is None:
            raise ValueError("Skeleton AE visual prediction requires the source row")
        embedding = skeleton_feature_lookup.embedding_for_row(row)
        skeleton = np.repeat(embedding[None, :], motion.shape[0], axis=0).astype(np.float32, copy=False)
    return _predict_g1_state_from_features(
        model=model,
        motion=motion,
        skeleton=skeleton,
        stats=stats,
        device=device,
        config=config,
        joint_dim=joint_dim,
        fallback_root_pos=fallback_root_pos,
        fallback_root_quat=fallback_root_quat,
    )


def _predict_g1_state_from_features(
    *,
    model: nn.Module,
    motion: np.ndarray,
    skeleton: np.ndarray,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    joint_dim: int,
    fallback_root_pos: np.ndarray,
    fallback_root_quat: np.ndarray,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        motion_t = torch.from_numpy(motion).to(device)
        skeleton_t = torch.from_numpy(skeleton).to(device)
        motion_n = (motion_t - stats["motion_mean"]) / stats["motion_std"]
        skeleton_mean, skeleton_std = skeleton_stats_pair(stats, config)
        skeleton_n = (skeleton_t - skeleton_mean) / skeleton_std
        pred_n = model(motion_n, skeleton_n)
        pred_raw = pred_n * stats["target_std"] + stats["target_mean"]
    pred = pred_raw.detach().cpu().numpy().astype(np.float32, copy=False)
    state = {
        "joint_pos": pred[:, :joint_dim],
        "root_pos": fallback_root_pos.astype(np.float32, copy=False),
        "root_quat": fallback_root_quat.astype(np.float32, copy=False),
    }
    if include_root_pos_target(config):
        window = int(config["features"]["future_window_frames"])
        command_dim = window * joint_dim * 2
        root_pos = pred[:, command_dim : command_dim + window * 3].reshape(pred.shape[0], window, 3)
        root_rot_start = command_dim + window * 3
        root_rot6d = pred[:, root_rot_start : root_rot_start + window * 6].reshape(
            pred.shape[0],
            window,
            6,
        )
        state = {
            "joint_pos": pred[:, :joint_dim],
            "root_pos": root_pos[:, 0].astype(np.float32, copy=False),
            "root_euler": _rot6d_to_euler_xyz_batch(root_rot6d[:, 0]),
        }
    return state


def _load_motionlib_robot_root(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    robot_path = Path(config["input_data"]["robot_motion_dir"]) / str(row["robot_relative_path"])
    _, robot = _single_motionlib_entry(robot_path)
    dof = np.asarray(robot["dof"], dtype=np.float32)
    root_pos = np.asarray(robot.get("root_trans_offset", np.zeros((dof.shape[0], 3), dtype=np.float32)), dtype=np.float32)
    root_quat = robot_root_rot_to_wxyz(np.asarray(robot["root_rot"], dtype=np.float32), config)
    target_len = min(dof.shape[0], root_pos.shape[0], root_quat.shape[0])
    return {"root_pos": root_pos[:target_len], "root_quat": root_quat[:target_len]}


def _load_visual_npz(path: Path) -> tuple[dict[str, np.ndarray], float]:
    with np.load(path) as loaded:
        arrays = {key: loaded[key].astype(np.float32, copy=False) for key in REQUIRED_NPZ_KEYS}
        fps = _scalar_float(loaded["fps"]) if "fps" in loaded else 50.0
    if fps <= 0 or not math.isfinite(fps):
        fps = 50.0
    return arrays, fps


def _select_visual_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    salt: str,
) -> list[Mapping[str, Any]]:
    eligible = [row for row in rows if int(row.get("frame_count", 0)) > 1]
    ranked = sorted(
        eligible,
        key=lambda row: stable_hash_int(f"{salt}:{row.get('relative_path', '')}:{row.get('filename', '')}"),
    )
    return ranked[: max(0, count)]


def _resolve_source_bvh(row: Mapping[str, Any], config: dict[str, Any], output_dir: Path) -> Path | None:
    rel_text = str(row.get("source_soma_proportional_path") or "")
    if not rel_text:
        return None
    rel = Path(rel_text)
    cfg = config.get("visual_validation", {})
    if rel.is_absolute() and rel.exists():
        return rel

    roots = list(cfg.get("source_bvh_roots", []))
    if cfg.get("source_bvh_root"):
        roots.append(cfg["source_bvh_root"])
    for root_text in roots:
        root = Path(str(root_text))
        candidates = [root / rel_text]
        if rel_text.startswith("soma_proportional/"):
            candidates.append(root / rel_text[len("soma_proportional/") :])
        for candidate in candidates:
            if candidate.exists():
                return candidate

    tar_text = str(cfg.get("source_bvh_tar", ""))
    tar_path = Path(tar_text) if tar_text else None
    if tar_path is not None and tar_path.exists():
        cache_root = Path(str(cfg.get("source_bvh_cache", output_dir / "source_bvh_cache")))
        extracted = _extract_source_bvh_from_tar(tar_path, rel_text, cache_root)
        if extracted is not None:
            return extracted
    return None


def _extract_source_bvh_from_tar(tar_path: Path, member_name: str, cache_root: Path) -> Path | None:
    safe_member = Path(member_name)
    if safe_member.is_absolute() or ".." in safe_member.parts:
        return None
    out_path = cache_root / safe_member
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path) as tar:
            extracted = tar.extractfile(member_name)
            if extracted is None:
                return None
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            tmp_path.replace(out_path)
        return out_path
    except (OSError, KeyError, tarfile.TarError):
        return None


def _sonic_body_pos_frames(
    body_pos: np.ndarray,
    body_names: Sequence[str],
    selected_names: Sequence[str],
) -> list[dict[str, tuple[float, float, float]]]:
    selected = [
        (index, name)
        for index, name in enumerate(body_names)
        if index < body_pos.shape[1] and name in set(selected_names)
    ]
    return [
        {
            name: (
                float(frame[index, 0]),
                float(frame[index, 1]),
                float(frame[index, 2]),
            )
            for index, name in selected
        }
        for frame in body_pos
    ]


def _sonic_edges(
    body_names: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    available = set(body_names)
    return tuple((start, end) for start, end in edges if start in available and end in available)


def _soma_motionlib_source_frames(
    soma_joints: np.ndarray,
    joint_names: Any,
) -> list[dict[str, tuple[float, float, float]]]:
    names = list(joint_names) if joint_names else list(SOMA_JOINT_NAMES)
    usable = min(len(names), soma_joints.shape[1])
    return [
        {
            names[index]: (
                float(frame[index, 0]),
                float(frame[index, 1]),
                float(frame[index, 2]),
            )
            for index in range(usable)
        }
        for frame in soma_joints
    ]


def _soma_edges(joint_names: Any) -> tuple[tuple[str, str], ...]:
    names = set(joint_names) if joint_names else set(SOMA_JOINT_NAMES)
    return tuple((start, end) for start, end in SOMA_CAPSULE_EDGES if start in names and end in names)


def _g1_prediction_frames(
    predicted_joint_pos: np.ndarray,
    *,
    root_pos: np.ndarray,
    root_quat: np.ndarray | None = None,
    root_euler: np.ndarray | None = None,
    g1_model: Any,
    render_deps: Mapping[str, Any],
) -> list[dict[str, tuple[float, float, float]]]:
    frames: list[dict[str, tuple[float, float, float]]] = []
    ignored = set(render_deps["G1_CAPSULE_IGNORE_BODIES"])
    fk = render_deps["g1_fk_body_positions"]
    if root_euler is None:
        if root_quat is None:
            raise ValueError("root_quat or root_euler is required")
        root_euler = np.asarray([_quat_wxyz_to_euler_xyz(quat) for quat in root_quat], dtype=np.float32)
    for joints, root, euler in zip(predicted_joint_pos, root_pos, root_euler):
        body_points = fk(
            g1_model,
            [float(value) for value in joints],
            root_position=[float(value) for value in root],
            root_euler=[float(value) for value in euler],
            include_empty_body_origin=True,
        )
        frame: dict[str, tuple[float, float, float]] = {}
        for name, points in body_points.items():
            if name in ignored or not points:
                continue
            frame[name] = _centroid(points)
        frames.append(frame)
    return frames


def _g1_edges(g1_model: Any, ignored_names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    ignored = set(ignored_names)
    edges: list[tuple[str, str]] = []
    for body in g1_model.bodies:
        if body.name in ignored or body.parent is None:
            continue
        parent_name = g1_model.bodies[body.parent].name
        if parent_name in ignored:
            continue
        edges.append((parent_name, body.name))
    return tuple(edges)


def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    count = max(1, len(points))
    return (
        sum(float(point[0]) for point in points) / count,
        sum(float(point[1]) for point in points) / count,
        sum(float(point[2]) for point in points) / count,
    )


def _rot6d_to_euler_xyz_batch(rot6d: np.ndarray) -> np.ndarray:
    matrix = _rot6d_to_matrix(rot6d)
    return np.asarray([_matrix_to_euler_xyz(item) for item in matrix], dtype=np.float32)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    a1 = values[:, [0, 2, 4]]
    a2 = values[:, [1, 3, 5]]
    b1 = _normalize_vectors(a1)
    b2 = _normalize_vectors(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.where(norm < 1e-8, 1.0, norm)


def _quat_wxyz_to_euler_xyz(quat: Sequence[float]) -> list[float]:
    matrix = quat_to_matrix(np.asarray(quat, dtype=np.float64).reshape(1, 4))[0]
    return _matrix_to_euler_xyz(matrix)


def _matrix_to_euler_xyz(matrix: np.ndarray) -> list[float]:
    sy = float(np.clip(matrix[0, 2], -1.0, 1.0))
    y = math.asin(sy)
    cy = math.cos(y)
    if abs(cy) > 1e-6:
        x = math.atan2(-float(matrix[1, 2]), float(matrix[2, 2]))
        z = math.atan2(-float(matrix[0, 1]), float(matrix[0, 0]))
    else:
        x = math.atan2(float(matrix[2, 1]), float(matrix[1, 1]))
        z = 0.0
    return [x, y, z]


def _render_missing_panel(
    *,
    render_deps: Mapping[str, Any],
    video_path: Path,
    render_config: Any,
    frame_count: int,
    label: str,
) -> dict[str, object]:
    frames = [{"missing": (0.0, 0.0, 1.0)} for _ in range(frame_count)]
    report = render_deps["_render_capsule_3d_video"](
        frames=frames,
        edges=(),
        video_path=video_path,
        config=render_config,
        label=label,
        up_axis=2,
        capsule_color=(145, 73, 68),
        key_color=(145, 73, 68),
    )
    if report.get("status") == "ok":
        report["status"] = "missing"
        report["message"] = "Source BVH was unavailable; rendered a placeholder panel."
    return report


def _combine_panel_videos(
    inputs: Sequence[Path],
    output: Path,
    *,
    fps: int,
) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"status": "blocked", "message": "ffmpeg is required to combine visual validation videos"}
    missing = [str(path) for path in inputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        return {"status": "failed", "message": "one or more panel videos are missing", "missing": missing}
    command = [ffmpeg, "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            f"[0:v][1:v][2:v]hstack=inputs=3,fps={max(1, fps)}[v]",
            "-map",
            "[v]",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {
            "status": "failed",
            "message": "ffmpeg failed while combining panels",
            "ffmpeg_tail": result.stderr[-800:],
        }
    if not output.exists() or output.stat().st_size == 0:
        return {"status": "failed", "message": "combined visual validation video was not written"}
    return {
        "status": "ok",
        "message": "combined source/dataset/inference visual validation video",
        "video_path": str(output),
        "bytes": output.stat().st_size,
        "fps": max(1, fps),
    }


def _scalar_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _safe_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return safe[:96] or "clip"


def _safe_metric_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return safe[:64] or "clip"


def write_loss_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "elapsed_sec",
                "train_loss",
                "train_command_mse_norm",
                "train_anchor_mse_norm",
                "train_root_pos_mse_norm",
                "train_root_rot_mse_norm",
                "train_joint_pos_rmse_raw",
                "train_joint_vel_rmse_raw",
                "train_anchor_rmse_raw",
                "train_root_pos_rmse_raw",
                "train_root_rot6d_rmse_raw",
            ]
        )


def append_loss_row(path: Path, step: int, elapsed: float, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                step,
                f"{elapsed:.4f}",
                f"{metrics['loss']:.10f}",
                f"{metrics['command_mse_norm']:.10f}",
                f"{metrics['anchor_mse_norm']:.10f}",
                f"{metrics.get('root_pos_mse_norm', float('nan')):.10f}",
                f"{metrics.get('root_rot_mse_norm', float('nan')):.10f}",
                f"{metrics['joint_pos_rmse_raw']:.10f}",
                f"{metrics['joint_vel_rmse_raw']:.10f}",
                f"{metrics['anchor_rmse_raw']:.10f}",
                f"{metrics.get('root_pos_rmse_raw', float('nan')):.10f}",
                f"{metrics.get('root_rot6d_rmse_raw', float('nan')):.10f}",
            ]
        )


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, Any],
    keep_last: int,
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "saved_at": utc_now(),
    }
    step_path = ckpt_dir / f"step_{step:08d}.pt"
    torch.save(payload, step_path)
    tmp_latest = ckpt_dir / "latest.tmp.pt"
    torch.save(payload, tmp_latest)
    tmp_latest.replace(ckpt_dir / "latest.pt")
    for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep_last]:
        old.unlink(missing_ok=True)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def write_manifest(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    run_group: str,
    rows: list[dict[str, Any]],
    train_dataset: KinWindowDataset,
    validation_dataset: KinWindowDataset,
    motion_dim: int,
    skeleton_dim: int,
    target_dim: int,
    skipped_index_rows: int,
    optimizer_parameter_names: Sequence[str],
    skeleton_feature_lookup: SkeletonAEFeatureLookup | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_root = Path(config["source_repo"])
    control_root = Path.cwd()
    if config["input_data"].get("format") == "soma_motionlib":
        data_snapshot: dict[str, Any] = {
            "format": "soma_motionlib",
            "robot_motion_dir": config["input_data"]["robot_motion_dir"],
            "soma_motion_dir": config["input_data"]["soma_motion_dir"],
            "row_count": len(rows),
            "skipped_index_rows": skipped_index_rows,
            "train_chunks": len(train_dataset),
            "validation_chunks": len(validation_dataset),
        }
    else:
        data_snapshot = {
            "format": "npz",
            "data_root": config["input_data"]["data_root"],
            "index": file_stats(index_path_from_config(config)),
            "row_count": len(rows),
            "skipped_index_rows": skipped_index_rows,
            "train_chunks": len(train_dataset),
            "validation_chunks": len(validation_dataset),
        }
    if skeleton_feature_lookup is not None:
        feature_dims = {
            "motion": motion_dim,
            "raw_skeleton_geometry": SKELETON_GEOMETRY_DIM,
            "skeleton_embedding": skeleton_dim,
            "model_input": motion_dim + skeleton_dim,
            "target": target_dim,
        }
    else:
        feature_dims = {
            "motion": motion_dim,
            "skeleton": skeleton_dim,
            "model_input": motion_dim + skeleton_dim,
            "target": target_dim,
        }
    optimizer_has_encoder_params = any("skeleton_encoder" in name or ".encoder" in name for name in optimizer_parameter_names)
    ddp_runtime = runtime or {"rank": 0, "world_size": 1, "local_rank": 0, "device": torch.device("cpu"), "backend": "none"}
    ddp_device = ddp_runtime.get("device", torch.device("cpu"))
    if not isinstance(ddp_device, torch.device):
        ddp_device = torch.device(str(ddp_device))
    _ddp_kwargs, ddp_settings = ddp_constructor_kwargs(ddp_runtime, ddp_device, config)
    manifest = {
        "run_id": f"sonic-kin-skeleton-{config['variant']['name']}-{timestamp_compact()}",
        "run_group": run_group,
        "host": socket.gethostname(),
        "timestamp": utc_now(),
        "environment_prefix": sys.prefix,
        "command_line": " ".join(sys.argv),
        "config_path": str(config_path),
        "source_repo": str(source_root),
        "source_revision_declared": config["source_rev"],
        "source_revision_actual": git_revision(source_root),
        "source_status_short": git_status_short(source_root),
        "control_repo": str(control_root),
        "control_revision_actual": git_revision(control_root),
        "control_status_short": git_status_short(control_root),
        "variant": config["variant"],
        "feature_dims": feature_dims,
        "optimizer": {
            "parameter_names": list(optimizer_parameter_names),
            "contains_skeleton_encoder_params": optimizer_has_encoder_params,
        },
        "distributed": {
            "rank": int(runtime.get("rank", 0)) if runtime is not None else 0,
            "world_size": int(runtime.get("world_size", 1)) if runtime is not None else 1,
            "local_rank": int(runtime.get("local_rank", 0)) if runtime is not None else 0,
            "backend": str(runtime.get("backend", "none")) if runtime is not None else "none",
        },
        "ddp": ddp_settings,
        "data_snapshot": data_snapshot,
        "metrics_path": str(output_dir / "loss_curve.csv"),
        "notes": config["purpose"],
    }
    if skeleton_feature_lookup is not None:
        manifest["skeleton_ae"] = {
            "enabled": True,
            "skeleton_encoder_frozen": True,
            "embedding_cache": skeleton_feature_lookup.cache_path,
            "artifact_info": skeleton_feature_lookup.artifact_info,
            "mapping_report": skeleton_feature_lookup.mapping_report,
            "normalization_keys": list(skeleton_normalization_keys(config)),
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def init_wandb(config: dict[str, Any], manifest: dict[str, Any], output_dir: Path, run_group: str):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    import wandb

    wandb_dir = Path(resolve_text(wandb_cfg.get("dir", str(output_dir / "wandb")), config, run_group))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    name = resolve_text(wandb_cfg.get("name", config["variant"]["name"]), config, run_group)
    group = resolve_text(wandb_cfg.get("group", run_group), config, run_group)
    mode = wandb_cfg.get("mode") or os.environ.get("WANDB_MODE")
    init_kwargs = {"mode": mode} if mode else {}
    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity"),
        name=name,
        group=group,
        dir=wandb_dir,
        config={**config, "manifest": manifest},
        tags=wandb_cfg.get("tags", []),
        resume="allow",
        **init_kwargs,
    )
    if run is not None:
        run.summary["git_commit"] = manifest["control_revision_actual"]
        run.summary["source_repo_git_commit"] = manifest["source_revision_actual"]
    return run


def validate_runtime(
    config: dict[str, Any],
    output_dir: Path,
    runtime: Mapping[str, Any] | None = None,
    *,
    check_data_artifacts: bool = True,
    index_only: bool = False,
) -> None:
    write_root = Path(config["runtime"]["write_root"])
    if output_dir != write_root and write_root not in output_dir.parents:
        raise ValueError(f"output_dir must be under write_root {write_root}: {output_dir}")
    for forbidden in config["runtime"].get("forbid_write_roots", []):
        forbidden_path = Path(forbidden)
        if output_dir == forbidden_path or forbidden_path in output_dir.parents:
            raise ValueError(f"output_dir must not be under {forbidden_path}: {output_dir}")
    requested_device = str(config["runtime"].get("device", "cuda")).lower()
    if index_only:
        required_gpu_count = 0
    elif requested_device == "cpu":
        required_gpu_count = 0
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    else:
        required_gpu_count = int(config["runtime"].get("required_gpu_count", 1))
    if torch.cuda.device_count() < required_gpu_count:
        raise RuntimeError(f"expected at least {required_gpu_count} visible GPU(s), found {torch.cuda.device_count()}")
    world_size = int(runtime.get("world_size", 1)) if runtime is not None else 1
    if required_gpu_count > 1 and world_size != required_gpu_count:
        raise RuntimeError(
            f"expected torchrun WORLD_SIZE={required_gpu_count} for this config, got {world_size}"
        )
    if check_data_artifacts:
        if config["input_data"].get("format") == "soma_motionlib":
            for key in ("robot_motion_dir", "soma_motion_dir"):
                path = Path(config["input_data"][key])
                if not path.exists():
                    raise FileNotFoundError(f"{key} is missing: {path}")
        else:
            data_root = Path(config["input_data"]["data_root"])
            if not data_root.exists():
                raise FileNotFoundError(f"data_root is missing: {data_root}")
            if not index_path_from_config(config).exists():
                raise FileNotFoundError(f"index is missing: {index_path_from_config(config)}")
        cfg = skeleton_ae_config(config)
        if cfg is not None:
            for key in ("checkpoint", "normalization", "registry_csv"):
                path = Path(str(cfg[key]))
                if not path.exists():
                    raise FileNotFoundError(f"skeleton_ae.{key} is missing: {path}")
    if runtime is not None and not is_main_process(runtime):
        return
    if config["runtime"].get("require_committed_code", True):
        control_root = Path.cwd()
        if git_revision(control_root) is None:
            raise RuntimeError(f"control repo is not a git worktree: {control_root}")
        if git_has_tracked_changes(control_root):
            raise RuntimeError(f"control repo has uncommitted tracked changes: {control_root}")
    if config["runtime"].get("require_latest_code", True):
        require_latest_git(Path.cwd(), "control repo")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_model_init_seed(config: Mapping[str, Any], stage_trace: StageTracer, stage_name: str) -> int:
    seed = int(config["training"]["seed"])
    stage_trace.log(stage_name, "before", seed=seed, rank_independent=True)
    set_seed(seed)
    stage_trace.log(stage_name, "after")
    return seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override training.max_steps for short supervised smoke checks.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        help="Override W&B mode; disabled prevents W&B initialization.",
    )
    parser.add_argument(
        "--disable-visual-validation",
        action="store_true",
        help="Disable visual validation for short smoke checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data/model/artifacts, write manifest and dry-run summary, then exit before training.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Build rows_from_index on one CPU process, write index_only_summary.json, then exit before model/data-loader setup.",
    )
    parser.add_argument(
        "--stage-trace",
        action="store_true",
        help="Write per-rank A0 dry-run stage diagnostics under logs/a0_stage_trace.",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.local_rank is not None and "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    config = apply_cli_overrides(read_config(args.config), args)
    run_group = os.environ.get("KIN_RUN_GROUP", timestamp_compact())
    output_dir = resolve_output_dir(config, run_group)
    stage_trace_on, stage_trace_reason = should_enable_a0_stage_trace(config, args)
    stage_trace = StageTracer(enabled=stage_trace_on, env=distributed_env(), reason=stage_trace_reason)
    stage_trace.log(
        "main_entry",
        "after_config",
        config=str(args.config),
        dry_run=bool(args.dry_run),
        index_only=bool(args.index_only),
        output_dir=str(output_dir),
    )
    if args.index_only:
        index_runtime = {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "distributed": False,
            "device": torch.device("cpu"),
            "backend": "none",
        }
        stage_trace.update_runtime(index_runtime)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        stage_trace.attach(output_dir)
        with stage_trace.span("validate_runtime", output_dir=str(output_dir), index_only=True):
            validate_runtime(config, output_dir, index_runtime, check_data_artifacts=True, index_only=True)
        data_root = Path(config["input_data"].get("data_root", config["input_data"].get("robot_motion_dir", ".")))
        with stage_trace.span(
            "rows_from_index",
            data_root=str(data_root),
            index_path=str(index_path_from_config(config)),
            index_only=True,
        ):
            rows, skipped = rows_from_index(
                config,
                data_root,
                stage_trace=stage_trace,
                output_dir=output_dir,
                runtime=index_runtime,
            )
        summary = {
            "event": "index_only_preflight",
            "variant": config["variant"],
            "config_path": str(args.config),
            "output_dir": str(output_dir),
            "data_root": str(data_root),
            "index_path": str(index_path_from_config(config)),
            "robot_motion_dir": config["input_data"].get("robot_motion_dir", ""),
            "soma_motion_dir": config["input_data"].get("soma_motion_dir", ""),
            "rows_cache": str(rows_from_index_cache_path(output_dir)),
            "row_count": len(rows),
            "skipped_count": int(skipped),
            "sample_rows": rows[:5],
        }
        (output_dir / "index_only_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trace_summary = dict(summary)
        trace_summary["summary_event"] = trace_summary.pop("event")
        stage_trace.log("index_only_preflight", "details", **trace_summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return
    with stage_trace.span("distributed_runtime_setup"):
        runtime = setup_distributed_runtime(stage_trace)
    try:
        stage_trace.update_runtime(runtime)
        stage_trace.log(
            "distributed_runtime_setup",
            "details",
            backend=str(runtime.get("backend")),
            device=str(runtime.get("device")),
            distributed=bool(runtime.get("distributed")),
        )
        is_main = is_main_process(runtime)
        rank = int(runtime["rank"])
        world_size = int(runtime["world_size"])
        probe_only_enabled, probe_only_source = should_run_a0_ddp_probe_only(config)
        with stage_trace.span(
            "validate_runtime",
            output_dir=str(output_dir),
            ddp_probe_only=probe_only_enabled,
        ):
            validate_runtime(config, output_dir, runtime, check_data_artifacts=not probe_only_enabled)
        if is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        stage_trace.attach(output_dir)
        with stage_trace.span("post_output_dir_barrier"):
            distributed_barrier(runtime)

        seed = int(config["training"]["seed"])
        with stage_trace.span("set_seed", seed=seed):
            set_seed(seed)
        device = runtime["device"]
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        stage_trace.log(
            "ddp_probe_only",
            "configured",
            enabled=probe_only_enabled,
            source=probe_only_source,
        )
        if probe_only_enabled:
            probe_backward_enabled, probe_backward_source = should_run_a0_ddp_probe_backward(
                config,
                probe_only=True,
            )
            stage_trace.log(
                "ddp_probe_backward",
                "configured",
                enabled=probe_backward_enabled,
                source=probe_backward_source,
            )
            motion_dim, skeleton_dim, target_dim = a0_probe_expected_dims(config)
            run_a0_ddp_probe_suite(
                stage_trace=stage_trace,
                runtime=runtime,
                device=device,
                config=config,
                motion_dim=motion_dim,
                skeleton_dim=skeleton_dim,
                target_dim=target_dim,
                output_dir=output_dir,
                run_backward=probe_backward_enabled,
            )
            if is_main:
                summary = {
                    "event": "a0_ddp_probe_only",
                    "variant": config["variant"],
                    "output_dir": str(output_dir),
                    "feature_dims": {
                        "motion": motion_dim,
                        "skeleton_embedding": skeleton_dim,
                        "model_input": motion_dim + skeleton_dim,
                        "target": target_dim,
                    },
                    "probe_only_source": probe_only_source,
                    "probe_backward_enabled": probe_backward_enabled,
                    "probe_backward_source": probe_backward_source,
                }
                (output_dir / "dry_run_summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(summary, sort_keys=True), flush=True)
            with stage_trace.span("ddp_probe_only_barrier"):
                distributed_barrier(runtime)
            return

        skeleton_feature_lookup = build_skeleton_ae_feature_lookup(config, device, stage_trace)
        data_root = Path(config["input_data"].get("data_root", config["input_data"].get("robot_motion_dir", ".")))
        with stage_trace.span(
            "rows_from_index",
            data_root=str(data_root),
            index_path=str(index_path_from_config(config)),
        ):
            rows, skipped = rows_from_index(
                config,
                data_root,
                stage_trace=stage_trace,
                output_dir=output_dir,
                runtime=runtime,
            )
        stage_trace.log("rows_from_index", "details", row_count=len(rows), skipped_count=int(skipped))
        with stage_trace.span("deterministic_split", row_count=len(rows)):
            split_rows(rows, float(config["split"]["validation_ratio"]), str(config["split"]["hash_salt"]))
        if skeleton_feature_lookup is not None:
            with stage_trace.span("skeleton_ae_row_mapping", row_count=len(rows)):
                skeleton_feature_lookup.validate_and_annotate_rows(rows)
            stage_trace.log(
                "skeleton_ae_row_mapping",
                "details",
                **skeleton_feature_lookup.mapping_report,
            )
            if is_main:
                with stage_trace.span("skeleton_ae_cache_write", output_dir=str(output_dir)):
                    skeleton_feature_lookup.write_cache(output_dir)
        with stage_trace.span("train_dataset_build"):
            train_dataset = KinWindowDataset(rows, "train", config, skeleton_feature_lookup)
        with stage_trace.span("validation_dataset_build"):
            validation_dataset = KinWindowDataset(rows, "validation", config, skeleton_feature_lookup)
        stats_load_device = torch.device("cpu") if is_skeleton_ae_enabled(config) else device
        with stage_trace.span("normalization_stats_motion_z", stats_load_device=str(stats_load_device)):
            stats = compute_or_load_stats(output_dir, train_dataset, config, stats_load_device, runtime)

        feature_cfg = config["features"]
        window = int(feature_cfg["future_window_frames"])
        motion_dim = int(stats["motion_mean"].numel())
        skeleton_dim = skeleton_feature_dim(stats, config)
        target_dim = int(stats["target_mean"].numel())
        stage_trace.log(
            "normalization_stats_motion_z",
            "details",
            motion_dim=motion_dim,
            skeleton_dim=skeleton_dim,
            target_dim=target_dim,
            skeleton_mean_key=skeleton_normalization_keys(config)[0],
            skeleton_std_key=skeleton_normalization_keys(config)[1],
        )
        root_pose_dim = root_pose_target_dim(config, window)
        command_dim = target_command_dim(target_dim, window, config)
        if command_dim <= 0 or command_dim % (window * 2) != 0:
            raise ValueError(
                f"target_dim={target_dim} is incompatible with window={window} and root pose target"
            )
        joint_dim = command_dim // (window * 2)
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            if runtime["distributed"]
            else None
        )
        val_sampler = (
            DistributedSampler(
                validation_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if runtime["distributed"]
            else None
        )
        with stage_trace.span("train_loader_build"):
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=int(config["training"]["num_workers"]),
                collate_fn=collate_chunks,
                pin_memory=True,
                persistent_workers=int(config["training"]["num_workers"]) > 0,
            )
        with stage_trace.span("validation_loader_build"):
            val_loader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=False,
                sampler=val_sampler,
                num_workers=max(0, int(config["training"]["num_workers"]) // 2),
                collate_fn=collate_chunks,
                pin_memory=True,
            )
        diagnostic_batch = None
        if stage_trace.enabled:
            with stage_trace.span("first_batch_collation", loader="validation"):
                diagnostic_batch = next(iter(val_loader))
            stage_trace.log(
                "first_batch_collation",
                "details",
                motion_shape=list(diagnostic_batch[0].shape),
                skeleton_shape=list(diagnostic_batch[1].shape),
                target_shape=list(diagnostic_batch[2].shape),
            )
        set_model_init_seed(config, stage_trace, "model_init_seed")
        with stage_trace.span("model_construct", motion_dim=motion_dim, skeleton_dim=skeleton_dim, target_dim=target_dim):
            raw_model = make_model(motion_dim, skeleton_dim, target_dim, config)
        model_parameter_report = module_parameter_report(raw_model)
        stage_trace.log("model_parameter_checksum", "details", **model_parameter_report)
        if runtime["distributed"] and is_skeleton_ae_enabled(config):
            model_parameter_consistency = verify_rank_parameter_report(
                output_dir=output_dir,
                runtime=runtime,
                stage_trace=stage_trace,
                stage_name="model_parameter_checksum",
                report=model_parameter_report,
            )
        else:
            model_parameter_consistency = {
                "all_rank_parameter_checksums": [model_parameter_report["parameter_sha256"]],
                "all_rank_parameter_checksums_equal": True,
            }
            stage_trace.log(
                "model_parameter_checksum_all_rank_parameter_checksums",
                "details",
                **model_parameter_consistency,
            )
        with stage_trace.span("model_to_device", device=str(device), motion_dim=motion_dim, skeleton_dim=skeleton_dim):
            raw_model = raw_model.to(device)
        stage_trace.log(
            "model_ddp_preflight",
            "details",
            parameter_checksum=model_parameter_report,
            parameter_checksum_consistency=model_parameter_consistency,
            **module_ddp_preflight_snapshot(raw_model),
        )
        if device.type == "cuda":
            with stage_trace.span("model_to_device_cuda_synchronize", device=str(device)):
                torch.cuda.synchronize(device)
        if runtime["distributed"]:
            probe_enabled, probe_source = should_run_a0_ddp_probe(config, args, stage_trace)
            probe_backward_enabled, probe_backward_source = should_run_a0_ddp_probe_backward(
                config,
                probe_only=False,
            )
            stage_trace.log(
                "ddp_wrap_probe_minimal_mlp",
                "configured",
                enabled=probe_enabled,
                source=probe_source,
            )
            stage_trace.log(
                "ddp_probe_backward",
                "configured",
                enabled=probe_backward_enabled,
                source=probe_backward_source,
            )
            if probe_enabled:
                run_a0_ddp_probe_suite(
                    stage_trace=stage_trace,
                    runtime=runtime,
                    device=device,
                    config=config,
                    motion_dim=motion_dim,
                    skeleton_dim=skeleton_dim,
                    target_dim=target_dim,
                    output_dir=output_dir,
                    run_backward=probe_backward_enabled,
                )
            ddp_kwargs, ddp_kwargs_log = ddp_constructor_kwargs(runtime, device, config)
            stage_trace.log("ddp_wrap", "details", kwargs=ddp_kwargs_log)
            with stage_trace.span("ddp_wrap", backend=str(runtime["backend"]), device=str(device), **ddp_kwargs_log):
                model = nn.parallel.DistributedDataParallel(raw_model, **ddp_kwargs)
        else:
            model = raw_model
        if stats_load_device != device:
            with stage_trace.span("normalization_stats_to_device", device=str(device)):
                stats = {key: value.to(device) for key, value in stats.items()}
        if diagnostic_batch is not None:
            with stage_trace.span("first_forward", device=str(device)):
                motion, skeleton, target = diagnostic_batch
                motion = motion.to(device, non_blocking=True)
                skeleton = skeleton.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                motion_n, skeleton_n, _target_n = normalize_batch(motion, skeleton, target, stats, config)
                pred_n = model(motion_n, skeleton_n)
            stage_trace.log("first_forward", "details", output_shape=list(pred_n.shape), output_device=str(pred_n.device))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        optimizer_parameter_names = trainable_parameter_names(raw_model)
        loss_curve = output_dir / "loss_curve.csv"
        if is_main:
            write_loss_header(loss_curve)
            manifest = write_manifest(
                output_dir,
                args.config,
                config,
                run_group,
                rows,
                train_dataset,
                validation_dataset,
                motion_dim,
                skeleton_dim,
                target_dim,
                skipped,
                optimizer_parameter_names,
                skeleton_feature_lookup,
                runtime,
            )
        with stage_trace.span("manifest_barrier"):
            distributed_barrier(runtime)
        if not is_main:
            with stage_trace.span("manifest_load_non_main"):
                manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if args.dry_run:
            with stage_trace.span("dry_run_validate"):
                val_metrics = validate(
                    model,
                    val_loader,
                    stats,
                    device,
                    config,
                    command_dim,
                    joint_dim,
                    root_pose_dim,
                )
            with stage_trace.span("dry_run_metric_all_reduce"):
                val_metrics = average_metric_dict(val_metrics, runtime, device)
            if is_main:
                summary = {
                    "event": "dry_run",
                    "variant": config["variant"],
                    "output_dir": str(output_dir),
                    "manifest": str(output_dir / "manifest.json"),
                    "normalization": str(output_dir / "stats" / "normalization.pt"),
                    "feature_dims": manifest["feature_dims"],
                    "skeleton_encoder_frozen": bool(
                        manifest.get("skeleton_ae", {}).get("skeleton_encoder_frozen", False)
                    ),
                    "optimizer_contains_skeleton_encoder_params": bool(
                        manifest["optimizer"]["contains_skeleton_encoder_params"]
                    ),
                    "optimizer_parameter_count": len(manifest["optimizer"]["parameter_names"]),
                    "mapping_report": manifest.get("skeleton_ae", {}).get("mapping_report", {}),
                    **val_metrics,
                }
                (output_dir / "dry_run_summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(summary, sort_keys=True), flush=True)
            with stage_trace.span("dry_run_summary_barrier"):
                distributed_barrier(runtime)
            return
        wandb_run = init_wandb(config, manifest, output_dir, run_group) if is_main else None

        max_steps = int(config["training"]["max_steps"])
        per_batch_frames = int(config["training"]["batch_frames"])
        log_every = int(config["training"]["log_every"])
        validate_every = int(config["training"]["validate_every"])
        checkpoint_every = int(config["training"]["checkpoint_every"])
        keep_last = int(config["training"]["keep_last_checkpoints"])
        grad_clip = float(config["training"]["grad_clip_norm"])
        command_weight = float(config["training"].get("command_loss_weight", 1.0))
        root_pos_weight = float(
            config["training"].get(
                "root_pos_loss_weight",
                config["training"].get("anchor_loss_weight", 1.0),
            )
        )
        root_rot_weight = float(
            config["training"].get(
                "root_rot_loss_weight",
                config["training"].get("anchor_loss_weight", 1.0),
            )
        )
        precision = config["training"].get("precision", "bf16")
        use_amp = precision in {"bf16", "fp16"}
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        recent: deque[dict[str, float]] = deque(maxlen=log_every)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed + 20260520 + rank)
        start = time.perf_counter()
        last_visual_validation_time = start
        step = 0
        epoch = 0

        if is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "variant": config["variant"],
                        "run_group": run_group,
                        "output_dir": str(output_dir),
                        "control_commit": manifest["control_revision_actual"],
                        "source_repo_commit": manifest["source_revision_actual"],
                        "train_chunks": len(train_dataset),
                        "validation_chunks": len(validation_dataset),
                        "world_size": world_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        while step < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for motion, skeleton, target in train_loader:
                if step >= max_steps:
                    break
                motion = motion.to(device, non_blocking=True)
                skeleton = skeleton.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if motion.shape[0] > per_batch_frames:
                    select = torch.randint(0, motion.shape[0], (per_batch_frames,), generator=rng, device=device)
                    motion = motion[select]
                    skeleton = skeleton[select]
                    target = target[select]
                motion_n, skeleton_n, target_n = normalize_batch(motion, skeleton, target, stats, config)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pred_n = model(motion_n, skeleton_n)
                    loss, metrics = loss_and_metrics(
                        pred_n,
                        target_n,
                        target,
                        stats,
                        command_dim,
                        joint_dim,
                        root_pose_dim,
                        include_root_pos_target(config),
                        command_weight,
                        root_pos_weight,
                        root_rot_weight,
                    )
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                step += 1
                metrics = average_metric_dict(metrics, runtime, device)
                recent.append(metrics)

                if is_main and (step % log_every == 0 or step == 1):
                    elapsed = time.perf_counter() - start
                    avg = {key: float(np.mean([row[key] for row in recent])) for key in metrics}
                    append_loss_row(loss_curve, step, elapsed, avg)
                    log_payload = {f"train/{key}": value for key, value in avg.items()}
                    log_payload["elapsed_sec"] = elapsed
                    log_payload["step"] = step
                    print(json.dumps({"event": "train", **log_payload}, sort_keys=True), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(log_payload, step=step)

                if step % validate_every == 0 or step == 1:
                    if val_sampler is not None:
                        val_sampler.set_epoch(step)
                    val_metrics = validate(
                        model,
                        val_loader,
                        stats,
                        device,
                        config,
                        command_dim,
                        joint_dim,
                        root_pose_dim,
                    )
                    val_metrics = average_metric_dict(val_metrics, runtime, device)
                    if is_main and val_metrics:
                        print(json.dumps({"event": "validation", "step": step, **val_metrics}, sort_keys=True), flush=True)
                        if wandb_run is not None:
                            wandb_run.log(val_metrics, step=step)

                now = time.perf_counter()
                if visual_validation_due(config, step, now=now, last_time=last_visual_validation_time):
                    visual_metrics: dict[str, float] = {}
                    if is_main:
                        try:
                            visual_metrics = run_visual_validation(
                                model=unwrap_model(model),
                                validation_rows=validation_dataset.rows,
                                stats=stats,
                                device=device,
                                config=config,
                                output_dir=output_dir,
                                step=step,
                                joint_dim=joint_dim,
                                wandb_run=wandb_run,
                                skeleton_feature_lookup=skeleton_feature_lookup,
                            )
                        except Exception as exc:
                            visual_metrics = {
                                "visual_validation/videos_ok": 0.0,
                                "visual_validation/videos_failed": float(config.get("visual_validation", {}).get("num_videos", 8)),
                            }
                            visual_error = {
                                "event": "visual_validation",
                                "step": step,
                                "status": "failed",
                                "message": str(exc),
                            }
                            print(json.dumps(visual_error, sort_keys=True), flush=True)
                        else:
                            print(
                                json.dumps({"event": "visual_validation", "step": step, **visual_metrics}, sort_keys=True),
                                flush=True,
                            )
                        if wandb_run is not None and visual_metrics:
                            wandb_run.log(visual_metrics, step=step)
                    distributed_barrier(runtime)
                    last_visual_validation_time = now

                if step % checkpoint_every == 0 or step == 1:
                    if is_main:
                        save_checkpoint(output_dir, unwrap_model(model), optimizer, step, metrics, keep_last)
                    distributed_barrier(runtime)
            epoch += 1

        final_metrics = {"step": step, "elapsed_sec": time.perf_counter() - start, "finished": True}
        if is_main:
            save_checkpoint(output_dir, unwrap_model(model), optimizer, step, final_metrics, keep_last)
            if wandb_run is not None:
                wandb_run.summary.update(final_metrics)
                wandb_run.finish()
            print(json.dumps({"event": "finished", **final_metrics}, sort_keys=True), flush=True)
        distributed_barrier(runtime)
    finally:
        cleanup_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
