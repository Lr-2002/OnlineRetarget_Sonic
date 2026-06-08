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

from online_retarget.a0_visual_validation import (  # noqa: E402
    ACCEPTANCE_G1_BACKEND,
    ACCEPTANCE_ROW2_DATA_SOURCE,
    ACCEPTANCE_ROW2_ROLE,
    ACCEPTANCE_ROW3_DATA_SOURCE,
    ACCEPTANCE_ROW3_ROLE,
    ACCEPTANCE_SOURCE_BACKEND,
    ACCEPTANCE_SOURCE_RENDERER,
    DEBUG_CAPSULE_BACKEND,
    A0VisualValidationRenderer,
    SOMA_DISPLAY_TRANSFORM,
    accepted_vertical_v2_artifact_paths,
    build_accepted_vertical_v2_metadata,
)
from online_retarget.data.bones_sonic import SONIC_JOINT_NAMES as G1_SONIC_JOINT_NAMES  # noqa: E402
from online_retarget.data.skeleton_ae_registry import SKELETON_GEOMETRY_DIM  # noqa: E402
from online_retarget.models.skeleton_geometry_ae import (  # noqa: E402
    SKELETON_GEOMETRY_AE_ARCHITECTURE,
    SKELETON_GEOMETRY_LATENT_DIM,
    load_skeleton_geometry_ae_checkpoint,
    load_skeleton_geometry_ae_stats,
)
from online_retarget.metric_validation_artifacts import (  # noqa: E402
    load_metric_validation_artifact,
    metric_validation_due,
    metric_validation_wandb_payload,
    visual_validation_wandb_payload,
    write_metric_validation_artifact,
)
from online_retarget.metrics import compute_metric_bundle  # noqa: E402


REQUIRED_NPZ_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
REQUIRED_ROBOT_MOTIONLIB_KEYS = ("dof", "root_rot", "fps")
REQUIRED_SOMA_MOTIONLIB_KEYS = ("soma_joints", "soma_root_quat", "fps")
A0_TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
A0_TRACKING_WEIGHT_POLICY = "uniform_14_tracking_bodies"
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
NO_SKELETON_ENCODER_FEATURE = "no_skeleton_encoder_zero_dim"
EVAL_METRIC_CONTRACT: dict[str, Any] = {
    "primary": "g1_joint_pos_rmse_rad",
    "aliases": [
        "joint_pos_rmse_raw",
    ],
    "metric_family": "G1 joint-angle command RMSE",
    "unit": "radian",
    "joint_set": "G1 29-DoF joint position command targets over the future window",
    "space": "joint_angle_command",
    "root_align": False,
    "scale_align": False,
    "loss_usage": "eval_metric_only_not_training_objective",
    "logged_keys": [
        "train/g1_joint_pos_rmse_rad",
        "validation/g1_joint_pos_rmse_rad",
    ],
    "body_position_mpjpe": {
        "status": "not_available_from_a0_joint_angle_target",
        "reason": "A0 targets are G1 joint-angle command windows and do not contain FK/body-position targets.",
        "requires_supplemental_evaluator_artifact": True,
        "supplemental_evaluator_artifact": "body_position_mpjpe_supplemental.json",
        "required_run_families": [
            "A0_frozen_skeleton_ae_uniform",
            "A0_frozen_skeleton_ae_proportional",
            "A0_no_skeleton_encoder_uniform",
            "A0_no_skeleton_encoder_proportional",
        ],
        "training_objective_changed": False,
    },
}
TEMPORAL_CONSISTENCY_LOSS_WEIGHT_DEFAULT = 0.01
RAW_SONIC_DATASET_KIN = "kin"
RAW_SONIC_DATASET_PHY = "phy"
RAW_SONIC_DATASETS = (RAW_SONIC_DATASET_KIN, RAW_SONIC_DATASET_PHY)
DEFAULT_RAW_SONIC_DATASET_ROOTS = {
    RAW_SONIC_DATASET_KIN: "/mnt/data_cpfs/bones_sonic",
    RAW_SONIC_DATASET_PHY: "/mnt/data_cpfs/phsical_bones_sonic",
}
DEFAULT_RAW_SONIC_SOURCE_PATH_PREFIX = "/home/user/data/motion_data/bones_sonic"


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
    validate_raw_sonic_dataset_config(data)
    return data


def input_data_format(config: Mapping[str, Any]) -> str:
    input_cfg = config.get("input_data", {})
    if not isinstance(input_cfg, Mapping):
        return "npz"
    return str(input_cfg.get("format", "npz")).strip() or "npz"


def is_raw_sonic_npz_config(config: Mapping[str, Any]) -> bool:
    return input_data_format(config) != "soma_motionlib"


def raw_sonic_dataset(config: Mapping[str, Any]) -> str:
    input_cfg = config.get("input_data", {})
    dataset = str(input_cfg.get("dataset", RAW_SONIC_DATASET_KIN)).strip().lower()
    if dataset not in RAW_SONIC_DATASETS:
        raise ValueError(f"input_data.dataset must be one of: {', '.join(RAW_SONIC_DATASETS)}")
    return dataset


def raw_sonic_dataset_roots(config: Mapping[str, Any]) -> dict[str, str]:
    input_cfg = config.get("input_data", {})
    roots = dict(DEFAULT_RAW_SONIC_DATASET_ROOTS)
    configured_roots = input_cfg.get("dataset_roots", {})
    if configured_roots in ("", None):
        configured_roots = {}
    if not isinstance(configured_roots, Mapping):
        raise ValueError("input_data.dataset_roots must map kin/phy to dataset roots")
    for name, root in configured_roots.items():
        dataset = str(name).strip().lower()
        if dataset not in RAW_SONIC_DATASETS:
            raise ValueError(f"input_data.dataset_roots only supports: {', '.join(RAW_SONIC_DATASETS)}")
        root_text = str(root).strip()
        if not root_text:
            raise ValueError(f"input_data.dataset_roots.{dataset} must be a non-empty path")
        roots[dataset] = root_text
    if not configured_roots and input_cfg.get("data_root"):
        roots[raw_sonic_dataset(config)] = str(input_cfg["data_root"])
    return roots


def raw_sonic_dataset_manifests(config: Mapping[str, Any]) -> dict[str, str]:
    input_cfg = config.get("input_data", {})
    configured_manifests = input_cfg.get("dataset_manifests", {})
    if configured_manifests in ("", None):
        configured_manifests = {}
    if not isinstance(configured_manifests, Mapping):
        raise ValueError("input_data.dataset_manifests must map kin/phy to manifest paths")
    manifests: dict[str, str] = {}
    for name, manifest in configured_manifests.items():
        dataset = str(name).strip().lower()
        if dataset not in RAW_SONIC_DATASETS:
            raise ValueError(f"input_data.dataset_manifests only supports: {', '.join(RAW_SONIC_DATASETS)}")
        manifest_text = str(manifest).strip()
        if not manifest_text:
            raise ValueError(f"input_data.dataset_manifests.{dataset} must be a non-empty path")
        manifests[dataset] = manifest_text
    return manifests


def raw_sonic_data_root(config: Mapping[str, Any]) -> Path:
    dataset = raw_sonic_dataset(config)
    roots = raw_sonic_dataset_roots(config)
    return Path(roots[dataset])


def raw_sonic_dataset_manifest_path(config: Mapping[str, Any]) -> Path | None:
    dataset = raw_sonic_dataset(config)
    manifests = raw_sonic_dataset_manifests(config)
    if dataset in manifests:
        return Path(manifests[dataset])
    if dataset == RAW_SONIC_DATASET_PHY:
        return raw_sonic_data_root(config) / "data.txt"
    return None


def raw_sonic_dataset_relative_paths(config: Mapping[str, Any]) -> set[str] | None:
    manifest_path = raw_sonic_dataset_manifest_path(config)
    if manifest_path is None:
        return None
    data_root = raw_sonic_data_root(config)
    relative_paths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            path = Path(text)
            if path.is_absolute():
                try:
                    text = str(path.relative_to(data_root))
                except ValueError:
                    text = str(path)
            else:
                text = str(path)
            relative_paths.add(text.lstrip("./"))
    return relative_paths


def resolved_raw_sonic_indexing(config: Mapping[str, Any]) -> dict[str, Any]:
    input_cfg = config.get("input_data", {})
    raw_indexing = input_cfg.get("indexing", {})
    if raw_indexing in ("", None):
        raw_indexing = {}
    if not isinstance(raw_indexing, Mapping):
        raise ValueError("input_data.indexing must be a mapping")
    indexing = dict(raw_indexing)
    if not indexing.get("source_path_prefix"):
        indexing["source_path_prefix"] = DEFAULT_RAW_SONIC_SOURCE_PATH_PREFIX
    if input_cfg.get("dataset") or input_cfg.get("dataset_roots") or not indexing.get("target_path_prefix"):
        indexing["target_path_prefix"] = str(raw_sonic_data_root(config))
    return indexing


def data_root_from_config(config: Mapping[str, Any]) -> Path:
    if not is_raw_sonic_npz_config(config):
        return Path(config["input_data"]["robot_motion_dir"])
    return raw_sonic_data_root(config)


def validate_raw_sonic_dataset_config(config: Mapping[str, Any]) -> None:
    if not is_raw_sonic_npz_config(config):
        return
    raw_sonic_dataset(config)
    raw_sonic_dataset_roots(config)
    raw_sonic_dataset_manifests(config)
    resolved_raw_sonic_indexing(config)


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
    expected_dims = expected_feature_dims(config, required=True)
    if expected_dims is None:
        raise ValueError("features.expected_dims is required for A0 DDP probe-only mode")
    motion_dim = int(expected_dims["motion_token"])
    skeleton_dim = int(expected_dims["z_skel"])
    target_dim = int(expected_dims["target"])
    return motion_dim, skeleton_dim, target_dim


def expected_feature_dims(
    config: Mapping[str, Any],
    *,
    required: bool = False,
) -> dict[str, int] | None:
    features = config.get("features", {})
    expected_dims = features.get("expected_dims") if isinstance(features, Mapping) else None
    if expected_dims is None:
        if required:
            raise ValueError("features.expected_dims is required for this A0 path")
        return None
    if not isinstance(expected_dims, Mapping):
        raise ValueError("features.expected_dims must be a mapping")
    required_keys = ("motion_token", "z_skel", "model_input", "target")
    missing = [key for key in required_keys if key not in expected_dims]
    if missing:
        raise ValueError(f"features.expected_dims missing required key(s): {', '.join(missing)}")
    parsed = {key: int(expected_dims[key]) for key in required_keys}
    if "x_skel" in expected_dims:
        parsed["x_skel"] = int(expected_dims["x_skel"])
    return parsed


def assert_expected_feature_dims(
    config: Mapping[str, Any],
    *,
    motion_dim: int,
    skeleton_dim: int,
    target_dim: int,
    skeleton_feature_lookup: Any | None,
) -> None:
    expected = expected_feature_dims(config)
    if expected is None:
        return
    actual = {
        "motion_token": int(motion_dim),
        "z_skel": int(skeleton_dim),
        "model_input": int(motion_dim + skeleton_dim),
        "target": int(target_dim),
    }
    if "x_skel" in expected:
        actual["x_skel"] = SKELETON_GEOMETRY_DIM if skeleton_feature_lookup is not None else int(skeleton_dim)
    mismatches = [
        f"{key}: expected {expected[key]}, got {actual.get(key)}"
        for key in expected
        if actual.get(key) != expected[key]
    ]
    if mismatches:
        raise ValueError("features.expected_dims mismatch: " + "; ".join(mismatches))


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


RANK0_STAGE_TERMINAL_STATUSES = {"ok", "failed"}


def rank0_stage_status_path(output_dir: Path, stage: str, *, step: int | None = None) -> Path:
    suffix = f"step_{int(step):08d}" if step is not None else "final"
    return output_dir / "logs" / "a0_rank0_stage_status" / f"{stage}_{suffix}.json"


def write_rank0_stage_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def wait_for_rank0_stage_status(
    path: Path,
    *,
    timeout_sec: float,
    poll_sec: float = 5.0,
    stage_trace: StageTracer | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_error = ""
    while True:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                last_error = repr(exc)
            else:
                if str(payload.get("status", "")) in RANK0_STAGE_TERMINAL_STATUSES:
                    return dict(payload)
                last_error = f"non_terminal_status={payload.get('status', '')}"
        elapsed = time.perf_counter() - started
        if elapsed >= timeout_sec:
            if stage_trace is not None:
                stage_trace.log(
                    "rank0_stage_status_wait",
                    "timeout",
                    path=str(path),
                    timeout_sec=timeout_sec,
                    last_error=last_error,
                )
            raise TimeoutError(
                f"rank0 stage status did not reach a terminal state within {timeout_sec:g}s: "
                f"{path}; last_error={last_error}"
            )
        if stage_trace is not None and (not path.exists() or last_error):
            stage_trace.log(
                "rank0_stage_status_wait",
                "poll",
                path=str(path),
                elapsed_sec=round(elapsed, 3),
                last_error=last_error,
            )
        time.sleep(max(0.1, float(poll_sec)))


def rank0_stage_sync_timeout(config: Mapping[str, Any], stage: str) -> float:
    visual_cfg = config.get("visual_validation", {})
    if not isinstance(visual_cfg, Mapping):
        visual_cfg = {}
    stage_key = f"{stage}_sync_timeout_sec"
    if stage_key in visual_cfg:
        return float(visual_cfg[stage_key])
    if "distributed_sync_timeout_sec" in visual_cfg:
        return float(visual_cfg["distributed_sync_timeout_sec"])
    if stage == "visual_validation":
        per_clip_timeout = float(visual_cfg.get("somamesh_render_timeout_sec", 900.0)) + float(
            visual_cfg.get("isaaclab_render_timeout_sec", 900.0)
        )
        requested = max(1, int(visual_cfg.get("num_videos", 8)))
        return max(3600.0, per_clip_timeout * requested + 600.0)
    return 1800.0


def rank0_stage_sync_poll(config: Mapping[str, Any]) -> float:
    visual_cfg = config.get("visual_validation", {})
    if isinstance(visual_cfg, Mapping) and "distributed_sync_poll_sec" in visual_cfg:
        return float(visual_cfg["distributed_sync_poll_sec"])
    return 5.0


def finish_wandb_run(wandb_run: Any, *, exit_code: int = 0) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.finish(exit_code=exit_code)
    except TypeError:
        wandb_run.finish()


def accepted_visual_metrics_failed(visual_metrics: Mapping[str, Any], visual_cfg: Mapping[str, Any]) -> bool:
    if not bool(visual_cfg.get("acceptance_backend", False)):
        return False
    try:
        failed = float(visual_metrics.get("visual_validation/videos_failed", 0.0))
    except (TypeError, ValueError):
        failed = 1.0
    try:
        ok = float(visual_metrics.get("visual_validation/videos_ok", 0.0))
    except (TypeError, ValueError):
        ok = 0.0
    return failed > 0.0 or ok <= 0.0


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


def no_skeleton_encoder_feature_enabled(config: Mapping[str, Any] | None) -> bool:
    if config is None:
        return False
    features = config.get("features", {})
    if not isinstance(features, Mapping):
        return False
    return str(features.get("skeleton_feature", "")).strip() == NO_SKELETON_ENCODER_FEATURE


def maybe_zero_skeleton_feature(
    skeleton: np.ndarray,
    frame_count: int,
    config: Mapping[str, Any] | None,
) -> np.ndarray:
    if no_skeleton_encoder_feature_enabled(config):
        return np.zeros((int(frame_count), 0), dtype=np.float32)
    return skeleton


def eval_metric_contract() -> dict[str, Any]:
    return copy.deepcopy(EVAL_METRIC_CONTRACT)


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
    skeleton = maybe_zero_skeleton_feature(skeleton, frame_indices.shape[0], config)

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
    skeleton = maybe_zero_skeleton_feature(skeleton, frame_indices.shape[0], config)

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
    if not is_raw_sonic_npz_config(config):
        return Path(config["input_data"]["robot_motion_dir"])
    indexing = resolved_raw_sonic_indexing(config)
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
    indexing = resolved_raw_sonic_indexing(config)
    dataset_relative_paths = raw_sonic_dataset_relative_paths(config)
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
            if dataset_relative_paths is not None and relative_path not in dataset_relative_paths:
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
    indexing = resolved_raw_sonic_indexing(config)
    dataset_relative_paths = raw_sonic_dataset_relative_paths(config)
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
            if dataset_relative_paths is not None and relative_path not in dataset_relative_paths:
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
ROWS_FROM_INDEX_CACHE_WAIT_TIMEOUT_SEC = 7200.0
ROWS_FROM_INDEX_CACHE_POLL_SEC = 2.0


def rows_from_index_cache_path(output_dir: Path) -> Path:
    return output_dir / ROWS_FROM_INDEX_CACHE_SUBDIR / "rows_from_index_cache.json"


def rows_from_index_cache_payload(
    config: Mapping[str, Any],
    data_root: Path,
    rows: list[dict[str, Any]],
    skipped: int,
) -> dict[str, Any]:
    raw_indexing = resolved_raw_sonic_indexing(config) if is_raw_sonic_npz_config(config) else {}
    return {
        "cache_version": 1,
        "created_at": utc_now(),
        "config": {
            "format": input_data_format(config),
            "dataset": raw_sonic_dataset(config) if is_raw_sonic_npz_config(config) else "",
            "robot_motion_dir": config["input_data"].get("robot_motion_dir"),
            "soma_motion_dir": config["input_data"].get("soma_motion_dir"),
            "data_root": str(data_root),
            "manifest_path": (
                str(raw_sonic_dataset_manifest_path(config) or "") if is_raw_sonic_npz_config(config) else ""
            ),
            "max_clips": int(config["input_data"].get("max_clips", raw_indexing.get("max_clips", 0))),
            "max_duration_delta_sec": float(
                config["input_data"].get("max_duration_delta_sec", raw_indexing.get("max_duration_delta_sec", 0.05))
            ),
        },
        "rows": rows,
        "row_count": len(rows),
        "skipped_count": int(skipped),
    }


def write_rows_from_index_cache(cache_path: Path, payload: Mapping[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(cache_path)


def read_rows_from_index_cache(cache_path: Path) -> tuple[list[dict[str, Any]], int]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    skipped = int(payload["skipped_count"])
    if len(rows) != int(payload["row_count"]):
        raise ValueError(
            f"rows_from_index cache row_count mismatch: payload={payload['row_count']} rows={len(rows)}"
        )
    return rows, skipped


def wait_for_rows_from_index_cache(
    cache_path: Path,
    *,
    stage_trace: StageTracer | None = None,
    timeout_sec: float = ROWS_FROM_INDEX_CACHE_WAIT_TIMEOUT_SEC,
    poll_sec: float = ROWS_FROM_INDEX_CACHE_POLL_SEC,
) -> tuple[list[dict[str, Any]], int]:
    deadline = time.monotonic() + timeout_sec
    next_log = time.monotonic()
    last_error = ""
    if stage_trace:
        stage_trace.log(
            "rows_from_index_cache_wait",
            "start",
            cache_path=str(cache_path),
            timeout_sec=timeout_sec,
            poll_sec=poll_sec,
        )
    while True:
        try:
            if cache_path.exists():
                rows, skipped = read_rows_from_index_cache(cache_path)
                if stage_trace:
                    stage_trace.log(
                        "rows_from_index_cache_wait",
                        "read",
                        cache_path=str(cache_path),
                        row_count=len(rows),
                        skipped_count=skipped,
                    )
                return rows, skipped
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
            last_error = repr(exc)
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"timed out waiting for rows_from_index cache at {cache_path}; last_error={last_error}"
            )
        if stage_trace and now >= next_log:
            stage_trace.log(
                "rows_from_index_cache_wait",
                "progress",
                cache_path=str(cache_path),
                remaining_sec=round(deadline - now, 3),
                last_error=last_error,
            )
            next_log = now + max(30.0, poll_sec)
        time.sleep(poll_sec)


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
    if not is_raw_sonic_npz_config(config):
        if use_cache and runtime is not None and runtime.get("distributed"):
            if is_main_process(runtime):
                if cache_path.exists():
                    rows, skipped = read_rows_from_index_cache(cache_path)
                    if stage_trace:
                        stage_trace.log(
                            "rows_from_index_cache",
                            "reused",
                            cache_path=str(cache_path),
                            row_count=len(rows),
                            skipped_count=int(skipped),
                        )
                else:
                    rows, skipped = rows_from_soma_motionlib_pair(config, stage_trace=stage_trace)
                    payload = rows_from_index_cache_payload(config, data_root, rows, skipped)
                    write_rows_from_index_cache(cache_path, payload)
                    if stage_trace:
                        stage_trace.log(
                            "rows_from_index_cache",
                            "written",
                            cache_path=str(cache_path),
                            row_count=len(rows),
                            skipped_count=int(skipped),
                        )
            if not is_main_process(runtime):
                if stage_trace:
                    stage_trace.log("rows_from_index_cache", "wait_before_read", cache_path=str(cache_path))
                rows, skipped = wait_for_rows_from_index_cache(cache_path, stage_trace=stage_trace)
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
            payload = rows_from_index_cache_payload(config, data_root, rows, skipped)
            write_rows_from_index_cache(cache_path, payload)
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


DEFAULT_EVAL_COHORT_MANIFEST = "eval_cohort_manifest.json"
EVAL_COHORT_EXCLUDED_SAMPLING_FIELDS = (
    "variant.name",
    "wandb.name",
    "wandb.group",
)


def evaluation_cohort_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    cfg = config.get("evaluation_cohort", {})
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        return {}
    return cfg


def _positive_int_config(cfg: Mapping[str, Any], key: str, default: int) -> int:
    try:
        value = int(cfg.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evaluation_cohort.{key} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"evaluation_cohort.{key} must be a positive integer")
    return value


def evaluation_cohort_counts(config: Mapping[str, Any]) -> tuple[int, int]:
    visual_cfg = config.get("visual_validation", {})
    if not isinstance(visual_cfg, Mapping):
        visual_cfg = {}
    visual_default = max(1, int(visual_cfg.get("num_videos", 8)))
    cfg = evaluation_cohort_config(config)
    if not cfg:
        return visual_default, visual_default
    visual_count = _positive_int_config(cfg, "visual_num_samples", visual_default)
    metric_count = _positive_int_config(cfg, "metric_num_samples", max(visual_count, visual_default))
    if metric_count < visual_count:
        raise ValueError("evaluation_cohort.metric_num_samples must be >= visual_num_samples")
    return visual_count, metric_count


def evaluation_cohort_sampling(config: Mapping[str, Any], run_group: str = "") -> dict[str, Any]:
    cfg = evaluation_cohort_config(config)
    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    visual_cfg = config.get("visual_validation", {})
    if not isinstance(visual_cfg, Mapping):
        visual_cfg = {}
    if cfg:
        cohort_id = str(cfg.get("id") or cfg.get("cohort_id") or "default").strip() or "default"
        seed = int(cfg.get("seed", training_cfg.get("seed", 0)))
        include_run_group = bool(cfg.get("include_run_group", True))
    else:
        cohort_id = str(visual_cfg.get("cohort_id") or "legacy_visual_validation").strip()
        seed = int(training_cfg.get("seed", 0))
        include_run_group = False
    run_group_text = str(run_group or "") if include_run_group else ""
    salt_parts = [
        "evaluation_cohort",
        f"id={cohort_id}",
        f"seed={seed}",
    ]
    if include_run_group:
        salt_parts.append(f"run_group={run_group_text}")
    salt = "|".join(salt_parts)
    return {
        "cohort_id": cohort_id,
        "seed": seed,
        "run_group": run_group_text,
        "include_run_group": include_run_group,
        "salt_sha256": hashlib.sha256(salt.encode("utf-8")).hexdigest(),
        "salt": salt,
        "excluded_config_fields": list(EVAL_COHORT_EXCLUDED_SAMPLING_FIELDS),
    }


def evaluation_row_stable_key(row: Mapping[str, Any]) -> str:
    fields = (
        "sample_id",
        "relative_path",
        "filename",
        "robot_relative_path",
        "soma_relative_path",
        "source_soma_proportional_path",
        "source_bvh",
        "path",
    )
    parts: list[str] = []
    for key in fields:
        value = row.get(key)
        if value not in ("", None):
            parts.append(f"{key}={value}")
    if not parts:
        parts.append(f"row={json.dumps(dict(row), sort_keys=True, default=str)}")
    return "\n".join(parts)


def _row_frame_count(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("frame_count", 0))
    except (TypeError, ValueError):
        return 0


def build_evaluation_cohort(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    run_group: str = "",
) -> dict[str, Any]:
    visual_count, metric_count = evaluation_cohort_counts(config)
    sampling = evaluation_cohort_sampling(config, run_group)
    eligible = [row for row in rows if _row_frame_count(row) > 1]
    ranked = sorted(
        eligible,
        key=lambda row: (
            stable_hash_int(f"{sampling['salt']}:{evaluation_row_stable_key(row)}"),
            evaluation_row_stable_key(row),
        ),
    )
    metric_rows = list(ranked[:metric_count])
    visual_rows = list(metric_rows[:visual_count])
    return {
        "enabled": bool(evaluation_cohort_config(config)),
        "cohort_id": sampling["cohort_id"],
        "seed": sampling["seed"],
        "run_group": sampling["run_group"],
        "include_run_group": sampling["include_run_group"],
        "salt_sha256": sampling["salt_sha256"],
        "excluded_config_fields": sampling["excluded_config_fields"],
        "visual_num_samples": visual_count,
        "metric_num_samples": metric_count,
        "eligible_row_count": len(eligible),
        "metric_rows": metric_rows,
        "visual_rows": visual_rows,
        "metric_row_count": len(metric_rows),
        "visual_row_count": len(visual_rows),
        "visual_subset_of_metric": [
            evaluation_row_stable_key(row) for row in visual_rows
        ]
        == [
            evaluation_row_stable_key(row) for row in metric_rows[: len(visual_rows)]
        ],
    }


def evaluation_cohort_manifest_path(output_dir: Path, config: Mapping[str, Any]) -> Path:
    cfg = evaluation_cohort_config(config)
    configured = Path(str(cfg.get("manifest_path", DEFAULT_EVAL_COHORT_MANIFEST)))
    if configured.is_absolute():
        return configured
    return output_dir / configured


def _evaluation_cohort_manifest_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    stable_key = evaluation_row_stable_key(row)
    source_path = (
        row.get("source_soma_proportional_path")
        or row.get("source_bvh")
        or row.get("path")
        or ""
    )
    return {
        "index": int(index),
        "row_id": str(row.get("sample_id") or row.get("relative_path") or row.get("filename") or stable_key),
        "filename": str(row.get("filename", "")),
        "relative_path": str(row.get("relative_path", "")),
        "robot_relative_path": str(row.get("robot_relative_path", "")),
        "soma_relative_path": str(row.get("soma_relative_path", "")),
        "source_path": str(source_path),
        "path": str(row.get("path", "")),
        "frame_count": _row_frame_count(row),
        "stable_key": stable_key,
        "stable_key_sha256": hashlib.sha256(stable_key.encode("utf-8")).hexdigest(),
    }


def evaluation_cohort_manifest_payload(
    cohort: Mapping[str, Any],
    manifest_path: Path | str,
) -> dict[str, Any]:
    metric_rows = list(cohort.get("metric_rows", []))
    visual_rows = list(cohort.get("visual_rows", []))
    visual_manifest_rows = [
        _evaluation_cohort_manifest_row(row, index) for index, row in enumerate(visual_rows)
    ]
    metric_manifest_rows = [
        _evaluation_cohort_manifest_row(row, index) for index, row in enumerate(metric_rows)
    ]
    return {
        "artifact_type": "evaluation_cohort",
        "version": 1,
        "created_at": utc_now(),
        "path": str(manifest_path),
        "cohort_id": str(cohort.get("cohort_id", "")),
        "seed": int(cohort.get("seed", 0)),
        "run_group": str(cohort.get("run_group", "")),
        "include_run_group": bool(cohort.get("include_run_group", False)),
        "visual_num_samples": int(cohort.get("visual_num_samples", 0)),
        "metric_num_samples": int(cohort.get("metric_num_samples", 0)),
        "eligible_row_count": int(cohort.get("eligible_row_count", 0)),
        "visual_row_count": len(visual_rows),
        "metric_row_count": len(metric_rows),
        "visual_subset_of_metric": bool(cohort.get("visual_subset_of_metric", False)),
        "visual_rows_sha256": hashlib.sha256(
            json.dumps(visual_manifest_rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "metric_rows_sha256": hashlib.sha256(
            json.dumps(metric_manifest_rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sampling": {
            "salt_sha256": str(cohort.get("salt_sha256", "")),
            "salt_fields": [
                "evaluation_cohort.id",
                "evaluation_cohort.seed",
                "run_group",
                "row.stable_key",
            ],
            "excluded_config_fields": list(cohort.get("excluded_config_fields", [])),
        },
        "visual_rows": visual_manifest_rows,
        "metric_rows": metric_manifest_rows,
    }


def evaluation_cohort_artifact_summary(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        return {}
    keys = (
        "path",
        "cohort_id",
        "seed",
        "run_group",
        "visual_num_samples",
        "metric_num_samples",
        "visual_row_count",
        "metric_row_count",
        "visual_subset_of_metric",
        "visual_rows_sha256",
        "metric_rows_sha256",
        "sampling",
    )
    return {key: copy.deepcopy(manifest[key]) for key in keys if key in manifest}


def write_evaluation_cohort_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


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
        self.input_format = input_data_format(config)
        self.data_root = data_root_from_config(config)
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


def temporal_consistency_loss_weight(config: Mapping[str, Any]) -> float:
    training = config.get("training")
    if not isinstance(training, Mapping) or not training.get("temporal_consistency_loss_enabled", False):
        return 0.0
    return float(
        training.get(
            "temporal_consistency_loss_weight",
            TEMPORAL_CONSISTENCY_LOSS_WEIGHT_DEFAULT,
        )
    )


def command_temporal_consistency_loss(
    pred_command: torch.Tensor,
    target_command: torch.Tensor,
    *,
    joint_dim: int,
) -> torch.Tensor:
    command_frame_dim = int(joint_dim) * 2
    if command_frame_dim <= 0 or pred_command.shape[-1] % command_frame_dim != 0:
        raise ValueError(
            f"command_dim={pred_command.shape[-1]} is incompatible with joint_dim={joint_dim}"
        )
    window = pred_command.shape[-1] // command_frame_dim
    if window < 2:
        return pred_command.new_tensor(0.0)
    pred_frames = pred_command.reshape(pred_command.shape[0], window, command_frame_dim)
    target_frames = target_command.reshape(target_command.shape[0], window, command_frame_dim)
    pred_delta = pred_frames[:, 1:, :] - pred_frames[:, :-1, :]
    target_delta = target_frames[:, 1:, :] - target_frames[:, :-1, :]
    return torch.mean((pred_delta - target_delta) ** 2)


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
    temporal_consistency_weight: float = 0.0,
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
    temporal_consistency_loss = pred_norm.new_tensor(0.0)
    if temporal_consistency_weight > 0.0:
        temporal_consistency_loss = command_temporal_consistency_loss(
            pred_command,
            target_command,
            joint_dim=joint_dim,
        )
        loss = loss + temporal_consistency_weight * temporal_consistency_loss

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
        joint_pos_rmse = float(torch.sqrt(torch.mean(joint_pos_error**2)).item())
        metrics = {
            "loss": float(loss.detach().item()),
            "command_mse_norm": float(command_loss.detach().item()),
            "anchor_mse_norm": float(anchor_loss.detach().item()),
            "root_pos_mse_norm": float(root_pos_loss.detach().item()),
            "root_rot_mse_norm": float(root_rot_loss.detach().item()),
            "temporal_consistency_mse_norm": float(temporal_consistency_loss.detach().item()),
            "temporal_consistency_loss_weight": float(temporal_consistency_weight),
            "joint_pos_rmse_raw": joint_pos_rmse,
            "g1_joint_pos_rmse_rad": joint_pos_rmse,
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
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    rows = []
    if max_batches is None:
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
    temporal_consistency_weight = temporal_consistency_loss_weight(config)
    with torch.no_grad():
        for batch_idx, (motion, skeleton, target) in enumerate(loader):
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
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
                temporal_consistency_weight,
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
    acceptance_backend: bool = False,
    isaac_python_bin: Path | str | None = None,
    isaac_render_script: Path | str | None = None,
    execute_isaaclab: bool = True,
    run_group: str = "",
    evaluation_cohort_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config.get("visual_validation", {})
    started = time.perf_counter()
    vis_dir = output_dir / "visual_validation" / f"step_{step:08d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    report_path = vis_dir / "summary.json"
    rows, cohort_summary = select_visual_validation_rows(
        validation_rows,
        config,
        run_group=run_group,
        evaluation_cohort_manifest=evaluation_cohort_manifest,
    )
    requested_videos = (
        evaluation_cohort_counts(config)[0]
        if evaluation_cohort_config(config)
        else int(cfg.get("num_videos", 8))
    )
    if not rows:
        summary = {
            "step": step,
            "status": "blocked",
            "message": "no validation rows were available for visual validation",
            "evaluation_cohort": cohort_summary,
            "reports": [],
        }
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"visual_validation/videos_ok": 0.0, "visual_validation/videos_failed": 0.0}

    if acceptance_backend:
        if is_raw_sonic_npz_config(config):
            summary = {
                "step": step,
                "status": "blocked",
                "message": "acceptance rerender currently requires input_data.format=soma_motionlib",
                "evaluation_cohort": cohort_summary,
                "reports": [],
            }
            report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return {"visual_validation/videos_ok": 0.0, "visual_validation/videos_failed": float(len(rows))}
        try:
            render_deps = _load_metric_fk_deps()
            model_xml = Path(str(cfg.get("g1_model_xml", "")))
            if not model_xml.exists():
                raise FileNotFoundError(f"visual_validation.g1_model_xml is missing: {model_xml}")
            g1_model = render_deps["load_g1_kinematic_model"](model_xml)
        except Exception as exc:
            summary = {
                "step": step,
                "status": "blocked",
                "message": f"body-position metric FK dependencies are unavailable: {exc}",
                "evaluation_cohort": cohort_summary,
                "reports": [],
            }
            report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return {"visual_validation/videos_ok": 0.0, "visual_validation/videos_failed": float(len(rows))}
    else:
        try:
            render_deps = _load_visual_render_deps()
        except Exception as exc:
            summary = {
                "step": step,
                "status": "blocked",
                "message": f"visual rendering dependencies are unavailable: {exc}",
                "evaluation_cohort": cohort_summary,
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
                "evaluation_cohort": cohort_summary,
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
                    acceptance_backend=acceptance_backend,
                    isaac_python_bin=isaac_python_bin,
                    isaac_render_script=isaac_render_script,
                    execute_isaaclab=execute_isaaclab,
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
    body_position_metrics = _aggregate_body_position_metrics(reports)
    body_metrics_available = body_position_metrics.get("status") == "available"
    status = "ok" if ok_count and (body_metrics_available or not acceptance_backend) else "failed"
    summary = {
        "step": step,
        "status": status,
        "variant": config["variant"]["name"],
        "duration_sec": float(cfg.get("duration_sec", 4.0)),
        "requested_videos": int(requested_videos),
        "videos_ok": ok_count,
        "videos_failed": failed_count,
        "elapsed_sec": time.perf_counter() - started,
        "evaluation_cohort": cohort_summary,
        "body_position_metrics": body_position_metrics,
        "reports": reports,
    }
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    visual_wandb_payload = visual_validation_wandb_payload(summary, summary_path=report_path)
    return {
        "visual_validation/videos_ok": float(ok_count),
        "visual_validation/videos_failed": float(failed_count),
        "visual_validation/elapsed_sec": float(summary["elapsed_sec"]),
        **visual_wandb_payload,
    }


def _load_metric_fk_deps() -> dict[str, Any]:
    from online_retarget.data.g1_quality import g1_fk_body_positions, load_g1_kinematic_model

    return {
        "g1_fk_body_positions": g1_fk_body_positions,
        "load_g1_kinematic_model": load_g1_kinematic_model,
    }


def _aggregate_body_position_metrics(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available: list[Mapping[str, Any]] = []
    for report in reports:
        body_metrics = report.get("body_position_metrics")
        if not isinstance(body_metrics, Mapping):
            continue
        metric_results = body_metrics.get("metric_results")
        if not isinstance(metric_results, Mapping):
            continue
        mpjpe_result = metric_results.get("mpjpe")
        w_mpjpe_result = metric_results.get("w_mpjpe")
        if (
            isinstance(mpjpe_result, Mapping)
            and isinstance(w_mpjpe_result, Mapping)
            and mpjpe_result.get("status") == "available"
            and w_mpjpe_result.get("status") == "available"
        ):
            available.append(body_metrics)
    if not available:
        return {
            "status": "unavailable",
            "reason": "no visual validation report produced numeric body-position metrics",
            "metric_results": {},
        }

    count_total = sum(float(item.get("sample_count") or 0.0) for item in available)
    weight_total = sum(float(item.get("weighted_sample_weight") or 0.0) for item in available)
    if count_total <= 0.0 or weight_total <= 0.0:
        return {
            "status": "unavailable",
            "reason": "body-position metric reports have zero sample weight",
            "metric_results": {},
        }

    first = available[0]
    first_results = first["metric_results"]
    mpjpe = sum(
        float(item["metric_results"]["mpjpe"]["value"]) * float(item.get("sample_count") or 0.0)
        for item in available
    ) / count_total
    w_mpjpe = sum(
        float(item["metric_results"]["w_mpjpe"]["value"]) * float(item.get("weighted_sample_weight") or 0.0)
        for item in available
    ) / weight_total
    source_artifact_paths = [
        artifact
        for item in available
        for artifact in _metric_source_artifacts(item).values()
        if artifact
    ]
    return {
        "status": "available",
        "sample_count": count_total,
        "weighted_sample_weight": weight_total,
        "report_count": len(available),
        "body_count": int(first.get("body_count") or len(A0_TRACKING_BODY_NAMES)),
        "frame_count": int(sum(int(item.get("frame_count") or 0) for item in available)),
        "body_names": list(first.get("body_names") or A0_TRACKING_BODY_NAMES),
        "body_position_weights": list(first.get("body_position_weights") or [1.0] * len(A0_TRACKING_BODY_NAMES)),
        "weight_policy": str(first.get("weight_policy") or A0_TRACKING_WEIGHT_POLICY),
        "metric_contract": dict(first.get("metric_contract") or {}),
        "source_artifact_paths": source_artifact_paths,
        "metric_results": {
            "mpjpe": {
                **dict(first_results["mpjpe"]),
                "value": mpjpe,
                "status": "available",
                "reason": "",
            },
            "w_mpjpe": {
                **dict(first_results["w_mpjpe"]),
                "value": w_mpjpe,
                "status": "available",
                "reason": "",
            },
        },
    }


def _metric_source_artifacts(body_metrics: Mapping[str, Any]) -> dict[str, str]:
    artifacts = body_metrics.get("source_artifacts")
    if not isinstance(artifacts, Mapping):
        return {}
    return {str(key): str(value) for key, value in artifacts.items() if value}


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
    acceptance_backend: bool = False,
    isaac_python_bin: Path | str | None = None,
    isaac_render_script: Path | str | None = None,
    execute_isaaclab: bool = True,
) -> dict[str, Any]:
    if not is_raw_sonic_npz_config(config):
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
            acceptance_backend=acceptance_backend,
            isaac_python_bin=isaac_python_bin,
            isaac_render_script=isaac_render_script,
            execute_isaaclab=execute_isaaclab,
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
    visual_renderer = A0VisualValidationRenderer(config)

    arrays, fps = _load_visual_npz(data_root_from_config(config) / str(row["relative_path"]))
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
        "visual_backend": visual_renderer.backend_manifest(active_backend=DEBUG_CAPSULE_BACKEND),
        "root_composition": visual_renderer.root_composition_metadata(),
        "note": (
            "Panels are debug fallback capsules. Primary acceptance backend is SomaMesh/global-SOMA source "
            "plus IsaacLab G1 kinematic playback."
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
    acceptance_backend: bool = False,
    isaac_python_bin: Path | str | None = None,
    isaac_render_script: Path | str | None = None,
    execute_isaaclab: bool = True,
) -> dict[str, Any]:
    if acceptance_backend:
        return _render_motionlib_acceptance_visual_validation_clip(
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
            isaac_python_bin=isaac_python_bin,
            isaac_render_script=isaac_render_script,
            execute_isaaclab=execute_isaaclab,
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
    visual_renderer = A0VisualValidationRenderer(config)

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
        joint_names = list(arrays.get("joint_names", SOMA_JOINT_NAMES))
        source_report = render_deps["_render_capsule_3d_video"](
            frames=visual_renderer.soma_motionlib_source_frames(arrays["soma_joints"][:frame_count], joint_names),
            edges=_soma_edges(arrays.get("joint_names")),
            video_path=source_video,
            config=render_config,
            label="source soma global display capsules debug fallback",
            up_axis=2,
            capsule_color=(48, 132, 83),
            key_color=(132, 103, 34),
        )
        source_report.update(
            {
                "source_display_transform": SOMA_DISPLAY_TRANSFORM,
                "debug_fallback_backend": DEBUG_CAPSULE_BACKEND,
            }
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
        "visual_backend": visual_renderer.backend_manifest(active_backend=DEBUG_CAPSULE_BACKEND),
        "root_composition": visual_renderer.root_composition_metadata(),
        "note": (
            "Panels are debug fallback capsules. Primary acceptance backend is SomaMesh/global-SOMA source "
            "plus IsaacLab G1 kinematic playback."
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


def _render_motionlib_acceptance_visual_validation_clip(
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
    isaac_python_bin: Path | str | None = None,
    isaac_render_script: Path | str | None = None,
    execute_isaaclab: bool = True,
) -> dict[str, Any]:
    cfg = config.get("visual_validation", {})
    clip_name = _safe_filename(str(row.get("filename") or Path(str(row["relative_path"])).stem))
    sample_id = clip_name
    paths = accepted_vertical_v2_artifact_paths(output_dir, sample_id=sample_id, step=step)
    clip_dir = paths["artifact_dir"]
    clip_dir.mkdir(parents=True, exist_ok=True)

    source_video = paths["row1_video"]
    target_video = paths["row2_video"]
    inference_video = paths["row3_video"]
    target_motion = paths["row2_motion_npz"]
    inference_motion = paths["row3_motion_npz"]
    combined_video = paths["combined_video"]
    metadata_path = paths["manifest_json"]
    visual_renderer = A0VisualValidationRenderer(config)

    arrays = load_soma_motionlib_arrays(row, config)
    robot_root = _load_motionlib_robot_root(row, config)
    fps = float(arrays["fps"])
    total_frames = min(arrays["joint_pos"].shape[0], robot_root["root_pos"].shape[0], robot_root["root_quat"].shape[0])
    requested_frames = max(1, int(round(float(cfg.get("duration_sec", 4.0)) * fps)))
    frame_count = min(total_frames, requested_frames)
    render_width = int(cfg.get("width", 640))
    render_height = int(cfg.get("height", 360))
    source_bvh = _resolve_source_bvh(row, config, output_dir)
    source_report = _render_somamesh_shapes_source_video(
        cfg=cfg,
        source_bvh=source_bvh,
        video_path=source_video,
        report_path=source_video.with_suffix(".json"),
        fps=fps,
        frame_count=frame_count,
        width=render_width,
        height=render_height,
        sample_id=sample_id,
    )

    root_pos = robot_root["root_pos"][:frame_count]
    root_quat = robot_root["root_quat"][:frame_count]
    target_motion_asset_report = visual_renderer.write_g1_motion_npz(
        path=target_motion,
        joint_pos=arrays["joint_pos"][:frame_count],
        root_pos=root_pos,
        root_quat=root_quat,
        fps=fps,
        joint_names=G1_SONIC_JOINT_NAMES[:joint_dim],
    )
    target_motion_asset_report.update({"data_source": ACCEPTANCE_ROW2_DATA_SOURCE, "row_role": ACCEPTANCE_ROW2_ROLE})
    isaac_python = isaac_python_bin or cfg.get("isaac_python_bin") or "/workspace/isaaclab/_isaac_sim/python.sh"
    isaac_script = isaac_render_script or cfg.get("isaac_render_script") or ROOT / "scripts" / "render_g1_isaac_pair.py"
    target_report = visual_renderer.render_g1_isaaclab_playback(
        python_bin=isaac_python,
        script_path=isaac_script,
        motion_path=target_motion,
        output_path=target_video,
        duration_sec=frame_count / fps if fps > 0 else float(cfg.get("duration_sec", 4.0)),
        width=render_width,
        height=render_height,
        execute=execute_isaaclab,
        cwd=ROOT,
    )
    target_report.update(
        {
            "panel": "G1 Target Playback",
            "sample_id": sample_id,
            "backend": "IsaacLab",
            "render_backend": ACCEPTANCE_G1_BACKEND,
            "data_source": ACCEPTANCE_ROW2_DATA_SOURCE,
            "target_motion_path": target_motion_asset_report["path"],
            "target_motion_sha256": target_motion_asset_report["sha256"],
            "capsule_renderer_used": False,
        }
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
    inference_root_quat = _prediction_root_quat_wxyz(prediction, frame_count=frame_count)
    motion_asset_report = visual_renderer.write_g1_motion_npz(
        path=inference_motion,
        joint_pos=prediction["joint_pos"][:frame_count],
        root_pos=prediction["root_pos"][:frame_count],
        root_quat=inference_root_quat[:frame_count],
        fps=fps,
        joint_names=G1_SONIC_JOINT_NAMES[:joint_dim],
    )
    motion_asset_report.update({"data_source": ACCEPTANCE_ROW3_DATA_SOURCE, "row_role": ACCEPTANCE_ROW3_ROLE})
    inference_report = visual_renderer.render_g1_isaaclab_playback(
        python_bin=isaac_python,
        script_path=isaac_script,
        motion_path=inference_motion,
        output_path=inference_video,
        duration_sec=frame_count / fps if fps > 0 else float(cfg.get("duration_sec", 4.0)),
        width=render_width,
        height=render_height,
        execute=execute_isaaclab,
        cwd=ROOT,
    )
    inference_report.update(
        {
            "panel": "G1 Kinematics Playback",
            "sample_id": sample_id,
            "backend": "IsaacLab",
            "render_backend": ACCEPTANCE_G1_BACKEND,
            "data_source": ACCEPTANCE_ROW3_DATA_SOURCE,
            "prediction_motion_path": motion_asset_report["path"],
            "prediction_motion_sha256": motion_asset_report["sha256"],
            "checkpoint": str(cfg.get("checkpoint_path", "")),
            "checkpoint_step": int(cfg.get("checkpoint_step", step)),
            "capsule_renderer_used": False,
        }
    )
    body_position_metrics = _accepted_body_position_metric_report(
        target_joint_pos=arrays["joint_pos"][:frame_count],
        target_root_pos=root_pos,
        target_root_quat=root_quat,
        predicted_joint_pos=prediction["joint_pos"][:frame_count],
        predicted_root_pos=prediction["root_pos"][:frame_count],
        predicted_root_quat=inference_root_quat[:frame_count],
        fps=fps,
        g1_model=g1_model,
        render_deps=render_deps,
        target_motion_path=target_motion,
        prediction_motion_path=inference_motion,
    )

    combine_report = _combine_panel_videos(
        (source_video, target_video, inference_video),
        combined_video,
        fps=int(round(fps)),
        layout="vertical",
    )
    metadata, acceptance_ok, failure_reasons = build_accepted_vertical_v2_metadata(
        visual_renderer=visual_renderer,
        step=step,
        index=index,
        row=row,
        sample_id=sample_id,
        source_bvh=source_bvh,
        fps=fps,
        frame_count=frame_count,
        clip_dir=clip_dir,
        source_video=source_video,
        target_video=target_video,
        inference_video=inference_video,
        combined_video=combined_video,
        source_report=source_report,
        target_report=target_report,
        inference_report=inference_report,
        target_motion_asset_report=target_motion_asset_report,
        motion_asset_report=motion_asset_report,
        combine_report=combine_report,
        checkpoint_path=cfg.get("checkpoint_path", ""),
        checkpoint_step=int(cfg.get("checkpoint_step", step)),
    )
    metadata["body_position_metrics"] = body_position_metrics
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gated_combined_status = metadata["combine"].get("status", "failed")
    return {
        "index": index,
        "filename": row.get("filename", ""),
        "relative_path": row.get("relative_path", ""),
        "fps": fps,
        "frames": frame_count,
        "source_status": source_report.get("status"),
        "dataset_status": target_report.get("status"),
        "inference_status": inference_report.get("status"),
        "combined_status": gated_combined_status,
        "combined_video": str(combined_video),
        "metadata": str(metadata_path),
        "body_position_metrics": body_position_metrics,
        "acceptance_ok": bool(acceptance_ok),
        "accepted_vertical_v2_status": metadata["visual_backend"].get("accepted_vertical_v2_status"),
        "acceptance_failure_reasons": failure_reasons,
        "active_backend_is_acceptance_backend": bool(
            metadata["visual_backend"]["active_backend_is_acceptance_backend"]
        ),
    }


def _render_somamesh_shapes_source_video(
    *,
    cfg: Mapping[str, Any],
    source_bvh: Path | None,
    video_path: Path,
    report_path: Path,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    sample_id: str,
) -> dict[str, Any]:
    if source_bvh is None or not source_bvh.exists():
        return _failed_somamesh_source_report(
            "accepted SOMA Shapes Row 1 requires a resolvable source BVH for SomaMesh LBS rendering",
            ["source_bvh_missing"],
        )
    try:
        retargeter_root = _required_somamesh_path(cfg, ("soma_retargeter_root",), "soma_retargeter_root")
        soma_usd = _required_somamesh_path(cfg, ("somamesh_usd", "soma_usd"), "SOMA USD")
        script_path = _somamesh_renderer_script_path(cfg)
    except (RuntimeError, FileNotFoundError) as exc:
        return _failed_somamesh_source_report(str(exc), ["somamesh_dependency_missing"])

    python_bin = str(cfg.get("soma_python_bin") or cfg.get("somamesh_python_bin") or sys.executable)
    command = [
        python_bin,
        str(script_path),
        "--bvh",
        str(source_bvh),
        "--output",
        str(video_path),
        "--report",
        str(report_path),
        "--retargeter-root",
        str(retargeter_root),
        "--soma-usd",
        str(soma_usd),
        "--fps",
        f"{float(fps):g}",
        "--frame-count",
        str(int(frame_count)),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--stride-triangles",
        str(int(cfg.get("somamesh_triangle_stride", 3))),
        "--title",
        str(sample_id),
    ]

    timeout = float(cfg.get("somamesh_render_timeout_sec", 900.0))
    env = _somamesh_renderer_env(cfg, retargeter_root=retargeter_root)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return _failed_somamesh_source_report(
            "SomaMesh LBS renderer timed out",
            ["somamesh_renderer_timeout"],
            command=command,
            stdout_tail=str(exc.stdout or "")[-1000:],
            stderr_tail=str(exc.stderr or "")[-1000:],
        )
    except OSError as exc:
        return _failed_somamesh_source_report(
            f"SomaMesh LBS renderer could not start: {exc}",
            ["somamesh_renderer_start_failed"],
            command=command,
        )

    if result.returncode != 0:
        return _failed_somamesh_source_report(
            "SomaMesh LBS renderer failed",
            [f"somamesh_renderer_returncode={result.returncode}"],
            command=command,
            returncode=int(result.returncode),
            stdout_tail=result.stdout[-1000:],
            stderr_tail=result.stderr[-1000:],
        )
    if not report_path.exists():
        return _failed_somamesh_source_report(
            "SomaMesh LBS renderer did not write a report",
            ["somamesh_report_missing"],
            command=command,
            returncode=int(result.returncode),
        )

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _failed_somamesh_source_report(
            f"SomaMesh LBS renderer report is unreadable: {exc}",
            ["somamesh_report_unreadable"],
            command=command,
            returncode=int(result.returncode),
        )

    report.update(
        {
            "panel": "SOMA Shapes / SomaMesh",
            "sample_id": sample_id,
            "source_renderer": ACCEPTANCE_SOURCE_RENDERER,
            "backend": ACCEPTANCE_SOURCE_BACKEND,
            "render_backend": ACCEPTANCE_SOURCE_BACKEND,
            "soma_backend": "SomaMeshShapes",
            "source_provenance": {
                "source_type": "source_bvh",
                "source_bvh": str(source_bvh),
                "source_bvh_sha256": _path_sha256(source_bvh),
                "soma_usd": str(soma_usd),
                "retargeter_root": str(retargeter_root),
            },
            "source_bvh": str(source_bvh),
            "source_bvh_sha256": _path_sha256(source_bvh),
            "soma_usd": str(soma_usd),
            "retargeter_root": str(retargeter_root),
            "video_path": str(video_path),
            "report_path": str(report_path),
            "command": command,
            "returncode": int(result.returncode),
        }
    )
    return report


def _failed_somamesh_source_report(
    message: str,
    failure_reasons: Sequence[str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "message": message,
        "backend": ACCEPTANCE_SOURCE_BACKEND,
        "render_backend": ACCEPTANCE_SOURCE_BACKEND,
        "source_renderer": ACCEPTANCE_SOURCE_RENDERER,
        "soma_backend": "SomaMeshShapes",
        "source_provenance": {},
        "failure_reasons": list(failure_reasons),
        **extra,
    }


def preflight_acceptance_somamesh_visual_validation(
    config: Mapping[str, Any],
    output_dir: Path,
    runtime: Mapping[str, Any] | None = None,
) -> None:
    cfg = config.get("visual_validation", {})
    if not isinstance(cfg, Mapping) or not cfg.get("enabled", False) or not cfg.get("acceptance_backend", False):
        return
    if config.get("input_data", {}).get("format") != "soma_motionlib":
        raise RuntimeError("accepted SomaMesh/SOMA Shapes visual validation requires input_data.format=soma_motionlib")

    retargeter_root = _required_somamesh_path(cfg, ("soma_retargeter_root",), "soma_retargeter_root")
    soma_usd = _required_somamesh_path(cfg, ("somamesh_usd", "soma_usd"), "SOMA USD")
    script_path = _somamesh_renderer_script_path(cfg)
    python_bin = str(cfg.get("soma_python_bin") or cfg.get("somamesh_python_bin") or sys.executable)
    rank = int(runtime.get("rank", 0)) if runtime is not None else 0
    preflight_dir = output_dir / "logs" / "somamesh_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    report_path = preflight_dir / f"rank_{rank:02d}.json"
    output_path = preflight_dir / f"rank_{rank:02d}.mp4"
    command = [
        python_bin,
        str(script_path),
        "--preflight-only",
        "--output",
        str(output_path),
        "--report",
        str(report_path),
        "--retargeter-root",
        str(retargeter_root),
        "--soma-usd",
        str(soma_usd),
    ]
    timeout = float(cfg.get("somamesh_preflight_timeout_sec", 60.0))
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=_somamesh_renderer_env(cfg, retargeter_root=retargeter_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "accepted SomaMesh/SOMA Shapes renderer preflight timed out before training; "
            f"command={command}; stdout_tail={str(exc.stdout or '')[-1000:]}; "
            f"stderr_tail={str(exc.stderr or '')[-1000:]}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "accepted SomaMesh/SOMA Shapes renderer preflight could not start before training; "
            f"command={command}; error={exc}"
        ) from exc
    if result.returncode != 0:
        report: dict[str, Any] = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {}
        raise RuntimeError(
            "accepted SomaMesh/SOMA Shapes renderer preflight blocked before training; "
            f"returncode={result.returncode}; command={command}; report={report}; "
            f"stdout_tail={result.stdout[-1000:]}; stderr_tail={result.stderr[-1000:]}"
        )


def preflight_acceptance_skeleton_visual_validation(
    config: Mapping[str, Any],
    output_dir: Path,
    runtime: Mapping[str, Any] | None = None,
) -> None:
    """Compatibility alias; accepted Row 1 is now SomaMesh/SOMA Shapes."""

    preflight_acceptance_somamesh_visual_validation(config, output_dir, runtime)


def _is_unresolved_somamesh_path(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        not normalized
        or "must_configure" in normalized
        or normalized.startswith("<")
        or normalized.startswith("${")
        or normalized in {"todo", "required", "none", "null"}
    )


def _required_somamesh_path(cfg: Mapping[str, Any], keys: Sequence[str], label: str) -> Path:
    selected_key = ""
    selected_value: Any = None
    for key in keys:
        if cfg.get(key):
            selected_key = key
            selected_value = cfg[key]
            break
    if selected_value is None:
        names = " or ".join(f"visual_validation.{key}" for key in keys)
        raise RuntimeError(f"{names} is required for accepted SomaMesh/SOMA Shapes visual validation")

    value = str(selected_value)
    if _is_unresolved_somamesh_path(value):
        raise RuntimeError(f"visual_validation.{selected_key} is an unresolved {label} placeholder: {value}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"visual_validation.{selected_key} is missing: {path}")
    return path


def _somamesh_renderer_script_path(cfg: Mapping[str, Any]) -> Path:
    script_path = Path(
        str(cfg.get("soma_render_script") or cfg.get("somamesh_render_script") or ROOT / "scripts" / "render_somamesh_source.py")
    )
    if not script_path.is_absolute():
        script_path = ROOT / script_path
    if not script_path.exists():
        raise FileNotFoundError(f"visual_validation.soma_render_script is missing: {script_path}")
    return script_path


def _somamesh_renderer_env(cfg: Mapping[str, Any], *, retargeter_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    paths: list[str] = []
    if retargeter_root is None and cfg.get("soma_retargeter_root"):
        retargeter_root = Path(str(cfg["soma_retargeter_root"]))
    if retargeter_root is not None:
        for candidate in (retargeter_root, retargeter_root / "src"):
            if candidate.exists():
                paths.append(str(candidate))
    paths.extend([str(ROOT), str(SRC_ROOT)])
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        paths.extend([path for path in existing_pythonpath.split(os.pathsep) if path])
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        display_frames = A0VisualValidationRenderer.soma_frame_maps_to_display(aligned_frames)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"status": "failed", "message": f"Could not load BVH source motion: {exc}"}

    report = render_deps["_render_capsule_3d_video"](
        frames=display_frames,
        edges=edges,
        video_path=video_path,
        config=render_config,
        label="source soma global display capsules debug fallback",
        up_axis=2,
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
            "source_display_transform": SOMA_DISPLAY_TRANSFORM,
            "debug_fallback_backend": DEBUG_CAPSULE_BACKEND,
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
        pred_root = A0VisualValidationRenderer(config).compose_prediction_root(root_pos[:, 0], fallback_root_pos)
        state = {
            "joint_pos": pred[:, :joint_dim],
            "root_pos": pred_root,
            "root_euler": _rot6d_to_euler_xyz_batch(root_rot6d[:, 0]),
        }
    return state


def _prediction_root_quat_wxyz(prediction: Mapping[str, np.ndarray], *, frame_count: int) -> np.ndarray:
    if prediction.get("root_quat") is not None:
        return np.asarray(prediction["root_quat"], dtype=np.float32)[:frame_count]
    if prediction.get("root_euler") is None:
        quat = np.zeros((frame_count, 4), dtype=np.float32)
        quat[:, 0] = 1.0
        return quat
    return _euler_xyz_to_quat_wxyz_batch(np.asarray(prediction["root_euler"], dtype=np.float32)[:frame_count])


def _euler_xyz_to_quat_wxyz_batch(euler: np.ndarray) -> np.ndarray:
    return np.asarray([_euler_xyz_to_quat_wxyz(row) for row in euler], dtype=np.float32)


def _euler_xyz_to_quat_wxyz(euler: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z = (float(euler[index]) if index < len(euler) else 0.0 for index in range(3))
    cx, sx = math.cos(x * 0.5), math.sin(x * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cz, sz = math.cos(z * 0.5), math.sin(z * 0.5)
    return (
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    )


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


def select_visual_validation_rows(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    run_group: str = "",
    evaluation_cohort_manifest: Mapping[str, Any] | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if evaluation_cohort_config(config):
        cohort = build_evaluation_cohort(rows, config, run_group=run_group)
        return list(cohort["visual_rows"]), evaluation_cohort_artifact_summary(evaluation_cohort_manifest)
    visual_cfg = config.get("visual_validation", {})
    if not isinstance(visual_cfg, Mapping):
        visual_cfg = {}
    selected = _select_visual_rows(
        rows,
        count=int(visual_cfg.get("num_videos", 8)),
        salt=f"{config['variant']['name']}:{config['training']['seed']}",
    )
    return selected, {}


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


def _accepted_body_position_metric_report(
    *,
    target_joint_pos: np.ndarray,
    target_root_pos: np.ndarray,
    target_root_quat: np.ndarray,
    predicted_joint_pos: np.ndarray,
    predicted_root_pos: np.ndarray,
    predicted_root_quat: np.ndarray,
    fps: float,
    g1_model: Any,
    render_deps: Mapping[str, Any],
    target_motion_path: Path,
    prediction_motion_path: Path,
) -> dict[str, Any]:
    weights = [1.0] * len(A0_TRACKING_BODY_NAMES)
    contract = {
        "pinned": True,
        "name": "a0_accepted_v2_world_g1_fk_14_tracking_bodies",
        "units": "m",
        "coordinate_frame": "world_z_up",
        "root_alignment": "world_g1_root_no_pelvis_subtraction",
        "scale_align": False,
        "frame_alignment": "accepted_v2_clip_common_frame_range",
        "root_quat_format": "wxyz",
        "body_order": list(A0_TRACKING_BODY_NAMES),
        "weight_policy": A0_TRACKING_WEIGHT_POLICY,
    }
    source_artifacts = {
        "target_motion_npz": str(target_motion_path),
        "prediction_motion_npz": str(prediction_motion_path),
    }
    base = {
        "body_names": list(A0_TRACKING_BODY_NAMES),
        "body_position_weights": weights,
        "weight_policy": A0_TRACKING_WEIGHT_POLICY,
        "metric_contract": contract,
        "source_artifacts": source_artifacts,
    }
    try:
        target = _g1_tracking_body_frames(
            target_joint_pos,
            root_pos=target_root_pos,
            root_quat=target_root_quat,
            g1_model=g1_model,
            render_deps=render_deps,
        )
        predicted = _g1_tracking_body_frames(
            predicted_joint_pos,
            root_pos=predicted_root_pos,
            root_quat=predicted_root_quat,
            g1_model=g1_model,
            render_deps=render_deps,
        )
        frame_count = min(len(target), len(predicted))
        if frame_count <= 0:
            raise ValueError("accepted-v2 body-position metrics require at least one frame")
        target = target[:frame_count]
        predicted = predicted[:frame_count]
        fields = {
            "target_g1_body_pos": target,
            "predicted_g1_body_pos": predicted,
            "body_position_weights": weights,
            "g1_body_position_contract": contract,
        }
        bundle = compute_metric_bundle(fields, ("mpjpe", "w_mpjpe"))
        metric_results = {name: result.to_dict() for name, result in bundle.items()}
        status = (
            "available"
            if all(result.get("status") == "available" for result in metric_results.values())
            else "unavailable"
        )
        reasons = [
            str(result.get("reason"))
            for result in metric_results.values()
            if isinstance(result, Mapping) and result.get("reason")
        ]
        return {
            **base,
            "status": status,
            "reason": "; ".join(reasons),
            "metric_results": metric_results,
            "frame_count": int(frame_count),
            "body_count": len(A0_TRACKING_BODY_NAMES),
            "sample_count": float(frame_count * len(A0_TRACKING_BODY_NAMES)),
            "weighted_sample_weight": float(frame_count * sum(weights)),
            "fps": float(fps),
        }
    except Exception as exc:
        return {
            **base,
            "status": "unavailable",
            "reason": str(exc),
            "metric_results": {},
        }


def _g1_tracking_body_frames(
    joint_pos: np.ndarray,
    *,
    root_pos: np.ndarray,
    root_quat: np.ndarray | None = None,
    root_euler: np.ndarray | None = None,
    g1_model: Any,
    render_deps: Mapping[str, Any],
) -> list[list[list[float]]]:
    fk = render_deps["g1_fk_body_positions"]
    if root_euler is None:
        if root_quat is None:
            raise ValueError("root_quat or root_euler is required")
        root_euler = np.asarray([_quat_wxyz_to_euler_xyz(quat) for quat in root_quat], dtype=np.float32)
    frames: list[list[list[float]]] = []
    for joints, root, euler in zip(joint_pos, root_pos, root_euler):
        body_points = fk(
            g1_model,
            [float(value) for value in joints],
            root_position=[float(value) for value in root],
            root_euler=[float(value) for value in euler],
            include_empty_body_origin=True,
        )
        frame: list[list[float]] = []
        missing: list[str] = []
        for name in A0_TRACKING_BODY_NAMES:
            points = body_points.get(name, ())
            if not points:
                missing.append(name)
                continue
            frame.append([float(value) for value in _centroid(points)])
        if missing:
            raise ValueError(f"G1 FK output is missing tracking bodies: {', '.join(missing)}")
        frames.append(frame)
    return frames


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
    layout: str = "horizontal",
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
    stack_inputs = max(1, len(inputs))
    command.extend(
        [
            "-filter_complex",
            "".join(f"[{index}:v]" for index in range(stack_inputs))
            + f"{'vstack' if layout == 'vertical' else 'hstack'}=inputs={stack_inputs},fps={max(1, fps)}[v]",
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
        "message": f"combined source/dataset/inference visual validation video ({layout})",
        "video_path": str(output),
        "bytes": output.stat().st_size,
        "fps": max(1, fps),
        "layout": layout,
        "panel_count": stack_inputs,
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
                "train_temporal_consistency_mse_norm",
                "train_temporal_consistency_loss_weight",
                "train_joint_pos_rmse_raw",
                "train_g1_joint_pos_rmse_rad",
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
                f"{metrics.get('temporal_consistency_mse_norm', 0.0):.10f}",
                f"{metrics.get('temporal_consistency_loss_weight', 0.0):.10f}",
                f"{metrics['joint_pos_rmse_raw']:.10f}",
                f"{metrics['g1_joint_pos_rmse_rad']:.10f}",
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
    evaluation_cohort_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_root = Path(config["source_repo"])
    control_root = Path.cwd()
    if not is_raw_sonic_npz_config(config):
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
            "dataset": raw_sonic_dataset(config),
            "data_root": str(data_root_from_config(config)),
            "manifest_path": str(raw_sonic_dataset_manifest_path(config) or ""),
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
        "eval_metrics": eval_metric_contract(),
        "metrics_path": str(output_dir / "loss_curve.csv"),
        "notes": config["purpose"],
    }
    eval_cohort_summary = evaluation_cohort_artifact_summary(evaluation_cohort_manifest)
    if eval_cohort_summary:
        manifest["evaluation_cohort"] = eval_cohort_summary
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
        if not is_raw_sonic_npz_config(config):
            for key in ("robot_motion_dir", "soma_motion_dir"):
                path = Path(config["input_data"][key])
                if not path.exists():
                    raise FileNotFoundError(f"{key} is missing: {path}")
        else:
            data_root = data_root_from_config(config)
            if not data_root.exists():
                raise FileNotFoundError(f"data_root is missing: {data_root}")
            manifest_path = raw_sonic_dataset_manifest_path(config)
            if manifest_path is not None and not manifest_path.exists():
                raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
            if not index_path_from_config(config).exists():
                raise FileNotFoundError(f"index is missing: {index_path_from_config(config)}")
        cfg = skeleton_ae_config(config)
        if cfg is not None:
            for key in ("checkpoint", "normalization", "registry_csv"):
                path = Path(str(cfg[key]))
                if not path.exists():
                    raise FileNotFoundError(f"skeleton_ae.{key} is missing: {path}")
    if not index_only:
        preflight_acceptance_skeleton_visual_validation(config, output_dir, runtime)
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
        data_root = data_root_from_config(config)
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
            "dataset": raw_sonic_dataset(config) if is_raw_sonic_npz_config(config) else "",
            "data_root": str(data_root),
            "manifest_path": (
                str(raw_sonic_dataset_manifest_path(config) or "") if is_raw_sonic_npz_config(config) else ""
            ),
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
        data_root = data_root_from_config(config)
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
        eval_cohort_manifest: dict[str, Any] | None = None
        eval_cohort_manifest_file: Path | None = None
        metric_validation_dataset: KinWindowDataset | None = None
        with stage_trace.span("evaluation_cohort_build"):
            eval_cohort = build_evaluation_cohort(validation_dataset.rows, config, run_group=run_group)
            stage_trace.log(
                "evaluation_cohort_build",
                "details",
                enabled=bool(eval_cohort["enabled"]),
                cohort_id=str(eval_cohort["cohort_id"]),
                seed=int(eval_cohort["seed"]),
                run_group=str(eval_cohort["run_group"]),
                visual_row_count=int(eval_cohort["visual_row_count"]),
                metric_row_count=int(eval_cohort["metric_row_count"]),
                eligible_row_count=int(eval_cohort["eligible_row_count"]),
            )
            if eval_cohort["enabled"]:
                if int(eval_cohort["visual_row_count"]) < int(eval_cohort["visual_num_samples"]):
                    raise ValueError(
                        "evaluation_cohort could not select requested visual rows: "
                        f"requested={eval_cohort['visual_num_samples']} selected={eval_cohort['visual_row_count']}"
                    )
                if int(eval_cohort["metric_row_count"]) < int(eval_cohort["metric_num_samples"]):
                    raise ValueError(
                        "evaluation_cohort could not select requested metric rows: "
                        f"requested={eval_cohort['metric_num_samples']} selected={eval_cohort['metric_row_count']}"
                    )
                eval_cohort_manifest_file = evaluation_cohort_manifest_path(output_dir, config)
                eval_cohort_manifest = evaluation_cohort_manifest_payload(
                    eval_cohort,
                    eval_cohort_manifest_file,
                )
                if is_main:
                    write_evaluation_cohort_manifest(eval_cohort_manifest_file, eval_cohort_manifest)
                distributed_barrier(runtime)
                metric_validation_dataset = KinWindowDataset(
                    [dict(row) for row in eval_cohort["metric_rows"]],
                    "validation",
                    config,
                    skeleton_feature_lookup,
                )
        stats_load_device = torch.device("cpu") if is_skeleton_ae_enabled(config) else device
        with stage_trace.span("normalization_stats_motion_z", stats_load_device=str(stats_load_device)):
            stats = compute_or_load_stats(output_dir, train_dataset, config, stats_load_device, runtime)

        feature_cfg = config["features"]
        window = int(feature_cfg["future_window_frames"])
        motion_dim = int(stats["motion_mean"].numel())
        skeleton_dim = skeleton_feature_dim(stats, config)
        target_dim = int(stats["target_mean"].numel())
        with stage_trace.span("features_expected_dims_validate"):
            assert_expected_feature_dims(
                config,
                motion_dim=motion_dim,
                skeleton_dim=skeleton_dim,
                target_dim=target_dim,
                skeleton_feature_lookup=skeleton_feature_lookup,
            )
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
        metric_val_sampler = (
            DistributedSampler(
                metric_validation_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if runtime["distributed"] and metric_validation_dataset is not None
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
        metric_val_loader = None
        if metric_validation_dataset is not None:
            with stage_trace.span("metric_validation_loader_build"):
                metric_val_loader = DataLoader(
                    metric_validation_dataset,
                    batch_size=1,
                    shuffle=False,
                    sampler=metric_val_sampler,
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
                evaluation_cohort_manifest=eval_cohort_manifest,
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
                    "eval_metrics": manifest["eval_metrics"],
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
        temporal_consistency_weight = temporal_consistency_loss_weight(config)
        precision = config["training"].get("precision", "bf16")
        use_amp = precision in {"bf16", "fp16"}
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        recent: deque[dict[str, float]] = deque(maxlen=log_every)
        latest_train_metrics: dict[str, float] = {}
        latest_validation_metrics: dict[str, float] = {}
        latest_validation_step: int | None = None
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
                        temporal_consistency_weight,
                    )
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                step += 1
                metrics = average_metric_dict(metrics, runtime, device)
                latest_train_metrics = {f"train/{key}": float(value) for key, value in metrics.items()}
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
                    latest_validation_metrics = val_metrics
                    latest_validation_step = step
                    if is_main and val_metrics:
                        print(json.dumps({"event": "validation", "step": step, **val_metrics}, sort_keys=True), flush=True)
                        if wandb_run is not None:
                            wandb_run.log(val_metrics, step=step)

                now = time.perf_counter()
                if visual_validation_due(config, step, now=now, last_time=last_visual_validation_time):
                    visual_metrics: dict[str, Any] = {}
                    visual_status_path = rank0_stage_status_path(output_dir, "visual_validation", step=step)
                    if is_main:
                        visual_cfg = config.get("visual_validation", {})
                        if not isinstance(visual_cfg, Mapping):
                            visual_cfg = {}
                        write_rank0_stage_status(
                            visual_status_path,
                            {
                                "status": "running",
                                "stage": "visual_validation",
                                "step": int(step),
                                "rank": int(rank),
                                "pid": os.getpid(),
                                "started_at": utc_now(),
                            },
                        )
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
                                acceptance_backend=bool(visual_cfg.get("acceptance_backend", False)),
                                isaac_python_bin=visual_cfg.get("isaac_python_bin"),
                                isaac_render_script=visual_cfg.get("isaac_render_script"),
                                run_group=run_group,
                                evaluation_cohort_manifest=eval_cohort_manifest,
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
                            wandb_finish_error = ""
                            if wandb_run is not None and visual_metrics:
                                try:
                                    wandb_run.log(visual_metrics, step=step)
                                    finish_wandb_run(wandb_run, exit_code=1)
                                except Exception as finish_exc:
                                    wandb_finish_error = repr(finish_exc)
                                    visual_error["wandb_finish_error"] = wandb_finish_error
                            write_rank0_stage_status(
                                visual_status_path,
                                {
                                    "status": "failed",
                                    "stage": "visual_validation",
                                    "step": int(step),
                                    "rank": int(rank),
                                    "pid": os.getpid(),
                                    "finished_at": utc_now(),
                                    "error_type": type(exc).__name__,
                                    "message": str(exc),
                                    "metrics": visual_metrics,
                                    "wandb_finish_error": wandb_finish_error,
                                },
                            )
                            raise RuntimeError(f"rank0 visual validation failed at step {step}: {exc}") from exc
                        else:
                            print(
                                json.dumps({"event": "visual_validation", "step": step, **visual_metrics}, sort_keys=True),
                                flush=True,
                            )
                            wandb_log_error = ""
                            if wandb_run is not None and visual_metrics:
                                try:
                                    wandb_run.log(visual_metrics, step=step)
                                except Exception as exc:
                                    wandb_log_error = repr(exc)
                            accepted_failed = accepted_visual_metrics_failed(visual_metrics, visual_cfg)
                            stage_failed = accepted_failed or bool(wandb_log_error)
                            status_payload = {
                                "status": "failed" if stage_failed else "ok",
                                "stage": "visual_validation",
                                "step": int(step),
                                "rank": int(rank),
                                "pid": os.getpid(),
                                "finished_at": utc_now(),
                                "metrics": visual_metrics,
                            }
                            if accepted_failed:
                                status_payload["message"] = "accepted visual validation returned failed metrics"
                            if wandb_log_error:
                                status_payload["wandb_log_error"] = wandb_log_error
                                status_payload["message"] = "visual validation W&B metric log failed"
                            if stage_failed and wandb_run is not None:
                                try:
                                    finish_wandb_run(wandb_run, exit_code=1)
                                except Exception as exc:
                                    status_payload["wandb_finish_error"] = repr(exc)
                            write_rank0_stage_status(visual_status_path, status_payload)
                            if stage_failed:
                                raise RuntimeError(
                                    f"rank0 visual validation finalize failed at step {step}: {status_payload}"
                                )
                    else:
                        rank0_status = wait_for_rank0_stage_status(
                            visual_status_path,
                            timeout_sec=rank0_stage_sync_timeout(config, "visual_validation"),
                            poll_sec=rank0_stage_sync_poll(config),
                            stage_trace=stage_trace,
                        )
                        if rank0_status.get("status") != "ok":
                            raise RuntimeError(
                                f"rank0 visual validation failed at step {step}: "
                                f"{rank0_status.get('message', rank0_status)}"
                            )
                    distributed_barrier(runtime)
                    last_visual_validation_time = now

                if metric_validation_due(config, step):
                    metric_validation_metrics = latest_validation_metrics
                    metric_validation_source = "latest_validation"
                    if metric_val_loader is not None:
                        metric_validation_source = "evaluation_cohort"
                        metric_validation_metrics = validate(
                            model,
                            metric_val_loader,
                            stats,
                            device,
                            config,
                            command_dim,
                            joint_dim,
                            root_pose_dim,
                            max_batches=0,
                        )
                        metric_validation_metrics = average_metric_dict(metric_validation_metrics, runtime, device)
                    elif latest_validation_step != step or not latest_validation_metrics:
                        raise RuntimeError(
                            "metric_validation.every_steps must align with training.validate_every "
                            f"for same-step validation metrics: step={step}, "
                            f"latest_validation_step={latest_validation_step}"
                        )
                    if is_main:
                        metric_artifact_path = write_metric_validation_artifact(
                            output_dir=output_dir,
                            step=step,
                            config=config,
                            validation_metrics=metric_validation_metrics,
                            train_metrics=latest_train_metrics,
                            manifest=manifest,
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "metric_validation_artifact",
                                    "step": step,
                                    "path": str(metric_artifact_path),
                                    "primary_metric": EVAL_METRIC_CONTRACT["primary"],
                                    "validation_source": metric_validation_source,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        if wandb_run is not None:
                            metric_payload = load_metric_validation_artifact(metric_artifact_path)
                            wandb_run.log(
                                metric_validation_wandb_payload(
                                    metric_payload,
                                    artifact_path=metric_artifact_path,
                                ),
                                step=step,
                            )
                    distributed_barrier(runtime)

                if step % checkpoint_every == 0 or step == 1:
                    if is_main:
                        save_checkpoint(output_dir, unwrap_model(model), optimizer, step, metrics, keep_last)
                    distributed_barrier(runtime)
            epoch += 1

        final_metrics = {"step": step, "elapsed_sec": time.perf_counter() - start, "finished": True}
        finalize_status_path = rank0_stage_status_path(output_dir, "training_finalize")
        if is_main:
            write_rank0_stage_status(
                finalize_status_path,
                {
                    "status": "running",
                    "stage": "training_finalize",
                    "step": int(step),
                    "rank": int(rank),
                    "pid": os.getpid(),
                    "started_at": utc_now(),
                },
            )
            try:
                save_checkpoint(output_dir, unwrap_model(model), optimizer, step, final_metrics, keep_last)
                if wandb_run is not None:
                    wandb_run.summary.update(final_metrics)
                    finish_wandb_run(wandb_run, exit_code=0)
                print(json.dumps({"event": "finished", **final_metrics}, sort_keys=True), flush=True)
            except Exception as exc:
                wandb_finish_error = ""
                if wandb_run is not None:
                    try:
                        finish_wandb_run(wandb_run, exit_code=1)
                    except Exception as finish_exc:
                        wandb_finish_error = repr(finish_exc)
                write_rank0_stage_status(
                    finalize_status_path,
                    {
                        "status": "failed",
                        "stage": "training_finalize",
                        "step": int(step),
                        "rank": int(rank),
                        "pid": os.getpid(),
                        "finished_at": utc_now(),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "wandb_finish_error": wandb_finish_error,
                    },
                )
                raise
            write_rank0_stage_status(
                finalize_status_path,
                {
                    "status": "ok",
                    "stage": "training_finalize",
                    "step": int(step),
                    "rank": int(rank),
                    "pid": os.getpid(),
                    "finished_at": utc_now(),
                    "metrics": final_metrics,
                },
            )
        else:
            rank0_status = wait_for_rank0_stage_status(
                finalize_status_path,
                timeout_sec=rank0_stage_sync_timeout(config, "training_finalize"),
                poll_sec=rank0_stage_sync_poll(config),
                stage_trace=stage_trace,
            )
            if rank0_status.get("status") != "ok":
                raise RuntimeError(
                    f"rank0 training finalize failed: {rank0_status.get('message', rank0_status)}"
                )
        distributed_barrier(runtime)
    finally:
        cleanup_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
