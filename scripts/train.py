#!/usr/bin/env python3
"""Training entry point for the direct-output baseline."""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import replace
import hashlib
import html
import inspect
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - environment blocker path
    yaml = None

from online_retarget.config_presets import (
    apply_config_preset as _apply_config_preset,
    configured_model_family as _configured_model_family,
)
from online_retarget.data.schema import ObservationSpec, OutputSpec, iter_motion_pair_refs
from online_retarget.evaluation import EvaluationConfig, evaluate_jsonl


FORMAL_SAMPLE_BUILDER = "bvh_fk_30body_window"
TEMPORAL_MODEL_CONDITION_SAMPLE_FIELDS = (
    "source_body_tokens",
    "source_skeleton",
    "morphology",
    "prev_target_joints",
    "previous_target_joints",
    "prev_g1_joints",
)
TEMPORAL_ROBOT_STATE_SAMPLE_FIELDS = ("robot_state",)
TEMPORAL_FORBIDDEN_CONDITION_SAMPLE_FIELDS = (
    "target_joints",
    "future_target_joints",
    "target_frame",
    "target_frame_indices",
    "target_g1_path",
    "actor_uid",
)
TEMPORAL_BATCH_KEYS = (
    "source_body_tokens",
    "source_skeleton",
    "morphology",
    "robot_state",
    "prev_action",
    "fps",
    "target_action",
)
TEMPORAL_STEP_PROFILER_PHASES = (
    "dataloader_next",
    "cpu_batch_materialize_or_cache",
    "h2d_to_device",
    "forward",
    "backward",
    "optimizer",
    "logging_checkpoint",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--index-csv", type=Path)
    parser.add_argument("--samples-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quality-policy-id")
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--quality-policy-audit", type=Path)
    parser.add_argument("--action-column")
    parser.add_argument(
        "--allow-debug-data",
        action="store_true",
        help="Allow non-formal training on debug samples without the full M2Q quality gate.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--wandb-mode",
        help="Override tracking.wandb_mode, e.g. disabled, offline, or online.",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Load a checkpoint and write predictions/eval for --samples-jsonl without optimizing.",
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    git_sha = _git_sha()
    config = _apply_wandb_mode_override(_load_config(args.config), args.wandb_mode)
    index_csv = args.index_csv or _nested_get(config, ("data", "index_csv"), None)
    samples_jsonl = args.samples_jsonl or _nested_get(config, ("data", "samples_jsonl"), None)
    sample_manifest = _load_sample_manifest(Path(samples_jsonl) if samples_jsonl else None)
    observation = _observation_spec_from_config_and_manifest(config, sample_manifest)
    output = OutputSpec(target=str(_nested_get(config, ("model", "output"), "g1_joint_position")))
    reported_output_dim = _reported_output_dim(
        config,
        output_spec=output,
        sample_manifest=sample_manifest,
    )
    action_column = args.action_column or str(_nested_get(config, ("data", "action_column"), "curation_action"))
    quality_gate = _quality_gate_context(
        config,
        index_csv=Path(index_csv) if index_csv else None,
        samples_jsonl=Path(samples_jsonl) if samples_jsonl else None,
        quality_policy_id=args.quality_policy_id,
        quality_report=args.quality_report,
        quality_policy_audit=args.quality_policy_audit,
        action_column=action_column,
        allow_debug_data=args.allow_debug_data,
    )

    print(f"config={args.config}")
    print(f"rank={rank} world_size={world_size}")
    print(f"git_sha={git_sha}")
    print(f"git_dirty={_git_dirty()}")
    print(f"observation_dim={observation.flattened_dim()}")
    print(f"output_dim={reported_output_dim}")
    print(f"quality_gate={json.dumps(quality_gate, sort_keys=True)}")
    if index_csv:
        index_path = Path(index_csv)
        if not _index_supports_motion_pair_refs(index_path, action_column=action_column):
            print(
                "train_refs=not_applicable "
                f"index_csv={index_csv} reason=index lacks split/{action_column} motion-pair columns"
            )
        else:
            ref_count = 0
            ref_samples = []
            for ref in iter_motion_pair_refs(
                index_path,
                splits=("train",),
                actions=("keep", "downweight"),
                action_column=action_column,
            ):
                ref_count += 1
                if len(ref_samples) < args.limit:
                    ref_samples.append(ref)
            print(f"train_refs={ref_count} index_csv={index_csv}")
            for ref in ref_samples:
                print(json.dumps(ref.to_dict(), sort_keys=True))
    else:
        print("index_csv=unset")
    if samples_jsonl:
        print(f"samples_jsonl={samples_jsonl}")

    if args.dry_run:
        return

    _validate_quality_gate(quality_gate)
    _validate_sample_manifest_contract(
        config,
        sample_manifest,
        Path(samples_jsonl) if samples_jsonl else None,
    )

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Training requires the conda environment from environment.yml with torch installed."
        ) from exc
    runtime = _setup_torch_runtime(torch, config=config, rank=rank, world_size=world_size)

    if not samples_jsonl:
        raise SystemExit(
            "Set --samples-jsonl or data.samples_jsonl to train. The current training loop "
            "consumes supervised JSONL artifacts produced by build-supervised-jsonl."
        )
    output_dir = args.output_dir or (
        Path(str(_nested_get(config, ("experiment", "output_root"), "runs")))
        / "train"
        / str(_nested_get(config, ("experiment", "name"), "baseline_mlp_direct_g1"))
    )
    if args.predict_only:
        if args.checkpoint is None:
            raise SystemExit("--predict-only requires --checkpoint")
        _predict_jsonl(
            torch=torch,
            config=config,
            samples_jsonl=Path(samples_jsonl),
            checkpoint=args.checkpoint,
            output_dir=output_dir,
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
        _cleanup_torch_runtime(torch, runtime)
        return

    try:
        _train_jsonl(
            torch=torch,
            config=config,
            samples_jsonl=Path(samples_jsonl),
            output_dir=output_dir,
            resume_checkpoint=args.checkpoint,
            max_steps=args.max_steps or int(_nested_get(config, ("train", "max_steps"), 1000)),
            batch_size=args.batch_size or int(_nested_get(config, ("train", "batch_size"), 64)),
            learning_rate=float(_nested_get(config, ("train", "learning_rate"), 3e-4)),
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
    finally:
        _cleanup_torch_runtime(torch, runtime)


def _index_supports_motion_pair_refs(index_csv: Path, *, action_column: str) -> bool:
    if not index_csv.exists():
        return False
    with index_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False
    columns = set(header)
    required = {
        "split",
        "actor_uid",
        "filename",
        "move_soma_proportional_path",
        "move_g1_path",
        action_column,
    }
    return required.issubset(columns)


def _load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read --config. Install the project environment "
            f"from environment.yml or pass a Python environment with pyyaml available: {path}"
        )
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return _apply_config_preset(payload)


def _nested_get(mapping: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _reported_output_dim(
    config: dict[str, Any],
    *,
    output_spec: OutputSpec,
    sample_manifest: dict[str, Any],
) -> int:
    manifest_output_dim = sample_manifest.get("output_dim")
    if isinstance(manifest_output_dim, int) and manifest_output_dim > 0:
        return manifest_output_dim
    try:
        horizon = int(_nested_get(config, ("data", "target_horizon_frames"), 1))
    except (TypeError, ValueError):
        horizon = 1
    return output_spec.output_dim() * max(1, horizon)


def _apply_wandb_mode_override(config: dict[str, Any], wandb_mode: str | None) -> dict[str, Any]:
    if not wandb_mode:
        return config
    updated = dict(config)
    tracking = updated.get("tracking", {})
    tracking = dict(tracking) if isinstance(tracking, dict) else {}
    tracking["wandb_mode"] = wandb_mode
    updated["tracking"] = tracking
    return updated


def _quality_gate_context(
    config: dict[str, Any],
    *,
    index_csv: Path | None,
    samples_jsonl: Path | None,
    quality_policy_id: str | None = None,
    quality_report: Path | None = None,
    quality_policy_audit: Path | None = None,
    action_column: str = "curation_action",
    allow_debug_data: bool = False,
) -> dict[str, Any]:
    """Collect the M2Q training gate inputs from config, CLI, and sample manifest."""

    manifest = _load_sample_manifest(samples_jsonl)
    manifest_config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    manifest_index = _path_or_none(str(manifest.get("index_csv", "")) if manifest else "")
    effective_index = index_csv or manifest_index
    effective_action_column = action_column or str(manifest_config.get("action_column", "curation_action"))
    if action_column == "curation_action" and manifest_config.get("action_column"):
        effective_action_column = str(manifest_config["action_column"])

    policy_id = quality_policy_id or _nested_get(config, ("data", "quality_policy_id"), "")
    report_path = quality_report or _path_or_none(str(_nested_get(config, ("data", "quality_report"), "")))
    audit_path = quality_policy_audit or _path_or_none(
        str(_nested_get(config, ("data", "quality_policy_audit"), ""))
    )
    audit_payload = _load_policy_audit(audit_path)
    allow_debug = allow_debug_data or bool(_nested_get(config, ("data", "allow_debug_data"), False))

    index_text = str(effective_index or "")
    manifest_path = samples_jsonl.parent / "manifest.json" if samples_jsonl else None
    uses_curated_index = (
        "curated_index.csv" in Path(index_text).name
        or "/curated/" in index_text.replace("\\", "/")
    )
    uses_merged_action = effective_action_column == "merged_quality_action"
    report_exists = bool(report_path and report_path.exists())
    audit_exists = bool(audit_path and audit_path.exists())
    audit_error = str(audit_payload.get("_error", "")) if audit_payload else ""
    audit_policy_id = str(audit_payload.get("policy_id", "")) if audit_payload else ""
    audit_promotable = bool(audit_payload.get("promotable", False)) if audit_payload else False
    audit_status = str(audit_payload.get("status", "")) if audit_payload else ""
    audit_blockers = audit_payload.get("blockers", []) if audit_payload else []
    if not isinstance(audit_blockers, list):
        audit_blockers = []
    return {
        "policy_id": str(policy_id),
        "quality_report": str(report_path or ""),
        "quality_report_exists": report_exists,
        "quality_policy_audit": str(audit_path or ""),
        "quality_policy_audit_exists": audit_exists,
        "quality_policy_audit_error": audit_error,
        "quality_policy_audit_policy_id": audit_policy_id,
        "quality_policy_audit_promotable": audit_promotable,
        "quality_policy_audit_status": audit_status,
        "quality_policy_audit_blockers": audit_blockers[:5],
        "index_csv": index_text,
        "samples_jsonl": str(samples_jsonl or ""),
        "samples_manifest": str(manifest_path or ""),
        "samples_manifest_exists": bool(manifest_path and manifest_path.exists()),
        "samples_builder": str(manifest.get("builder", "")) if manifest else "",
        "samples_builder_is_formal": manifest.get("builder") == FORMAL_SAMPLE_BUILDER if manifest else False,
        "formal_sample_builder": FORMAL_SAMPLE_BUILDER,
        "action_column": effective_action_column,
        "allow_debug_data": allow_debug,
        "uses_curated_index": uses_curated_index,
        "uses_merged_action": uses_merged_action,
        "status": "debug_override" if allow_debug else "formal_required",
    }


def _validate_quality_gate(context: dict[str, Any]) -> None:
    """Refuse formal optimization when M2Q provenance is missing."""

    if context.get("allow_debug_data"):
        print(
            "quality_gate_warning=debug data override enabled; "
            "this run is not a formal M2Q-gated training run."
        )
        return

    missing = []
    if not context.get("policy_id"):
        missing.append("data.quality_policy_id or --quality-policy-id")
    if not context.get("quality_report"):
        missing.append("data.quality_report or --quality-report")
    elif not context.get("quality_report_exists"):
        missing.append(f"existing quality report at {context.get('quality_report')}")
    if not context.get("quality_policy_audit"):
        missing.append("data.quality_policy_audit or --quality-policy-audit")
    elif not context.get("quality_policy_audit_exists"):
        missing.append(f"existing quality policy audit at {context.get('quality_policy_audit')}")
    elif context.get("quality_policy_audit_error"):
        missing.append(
            "readable quality policy audit "
            f"({context.get('quality_policy_audit_error')})"
        )
    elif not context.get("quality_policy_audit_promotable"):
        blockers = context.get("quality_policy_audit_blockers") or []
        blocker_text = f": {'; '.join(str(item) for item in blockers)}" if blockers else ""
        missing.append(f"promotable quality policy audit{blocker_text}")
    elif (
        context.get("policy_id")
        and context.get("quality_policy_audit_policy_id")
        and context.get("quality_policy_audit_policy_id") != context.get("policy_id")
    ):
        missing.append(
            "quality policy audit matching policy_id "
            f"{context.get('policy_id')} "
            f"(found {context.get('quality_policy_audit_policy_id')})"
        )
    if not context.get("uses_curated_index"):
        missing.append("curated index path generated by merge-quality")
    if not context.get("uses_merged_action"):
        missing.append("data.action_column=merged_quality_action or --action-column merged_quality_action")
    if context.get("samples_jsonl"):
        if not context.get("samples_manifest_exists"):
            missing.append("samples manifest next to supervised JSONL")
        elif not context.get("samples_builder_is_formal"):
            missing.append(
                "formal samples built by "
                f"{context.get('formal_sample_builder')} "
                f"(found {context.get('samples_builder') or 'unknown'})"
            )

    if missing:
        joined = "; ".join(missing)
        raise SystemExit(
            "Formal training quality gate failed. Missing: "
            f"{joined}. Use --allow-debug-data only for explicitly labeled debug runs."
        )


def _load_sample_manifest(samples_jsonl: Path | None) -> dict[str, Any]:
    if samples_jsonl is None:
        return {}
    manifest_path = samples_jsonl.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _validate_sample_manifest_contract(
    config: dict[str, Any],
    manifest: dict[str, Any],
    samples_jsonl: Path | None,
) -> None:
    expected = _nested_get(config, ("data", "target_future_step"), None)
    if expected is None or samples_jsonl is None:
        return
    try:
        expected_step = int(expected)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"data.target_future_step must be an integer: {expected}") from exc
    manifest_config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    actual = manifest.get("target_future_step", manifest_config.get("target_future_step"))
    manifest_path = samples_jsonl.parent / "manifest.json"
    if actual is None:
        if expected_step == 1:
            return
        raise SystemExit(
            "Sample manifest lacks target_future_step for configured "
            f"data.target_future_step={expected_step}: {manifest_path}. "
            "Rebuild samples with the matching --target-future-step."
        )
    try:
        actual_step = int(actual)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Sample manifest target_future_step must be an integer: {manifest_path}"
        ) from exc
    if actual_step != expected_step:
        raise SystemExit(
            "Sample manifest target_future_step mismatch: "
            f"config data.target_future_step={expected_step}, "
            f"manifest target_future_step={actual_step}, manifest={manifest_path}. "
            "Rebuild samples or point data.samples_jsonl at the matching artifact."
        )


def _load_policy_audit(audit_path: Path | None) -> dict[str, Any]:
    if audit_path is None or not audit_path.exists():
        return {}
    try:
        with audit_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_error": "audit JSON must be an object"}


def _path_or_none(value: str) -> Path | None:
    return Path(value) if value else None


def _observation_spec_from_config_and_manifest(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    input_dim: int | None = None,
) -> ObservationSpec:
    spec_payload = manifest.get("observation_spec", {})
    if not isinstance(spec_payload, dict):
        spec_payload = {}
    robot_payload = spec_payload.get("robot_state", {})
    robot_state = ObservationSpec().robot_state
    if isinstance(robot_payload, dict):
        robot_state = replace(
            robot_state,
            joint_dim=int(robot_payload.get("joint_dim", robot_state.joint_dim)),
            include_joint_position=bool(
                robot_payload.get("include_joint_position", robot_state.include_joint_position)
            ),
            include_joint_velocity=bool(
                robot_payload.get("include_joint_velocity", robot_state.include_joint_velocity)
            ),
            include_previous_action=bool(
                robot_payload.get("include_previous_action", robot_state.include_previous_action)
            ),
            include_imu_orientation=bool(
                robot_payload.get("include_imu_orientation", robot_state.include_imu_orientation)
            ),
            include_base_angular_velocity=bool(
                robot_payload.get(
                    "include_base_angular_velocity",
                    robot_state.include_base_angular_velocity,
                )
            ),
        )
    spec = ObservationSpec(
        history_frames=int(
            spec_payload.get(
                "history_frames",
                _nested_get(config, ("data", "history_frames"), 8),
            )
        ),
        source_body_count=int(
            spec_payload.get(
                "source_body_count",
                _nested_get(config, ("data", "source_body_count"), 30),
            )
        ),
        source_position_dim=int(spec_payload.get("source_position_dim", 3)),
        include_source_velocity=bool(spec_payload.get("include_source_velocity", True)),
        include_morphology=bool(spec_payload.get("include_morphology", True)),
        robot_state=robot_state,
    )
    if input_dim is None or spec.flattened_dim() == input_dim:
        return spec
    inferred = _infer_source_body_count(spec, input_dim)
    return replace(spec, source_body_count=inferred) if inferred else spec


def _infer_source_body_count(spec: ObservationSpec, input_dim: int) -> int | None:
    side_dim = spec.morphology_dim() + spec.robot_state_dim()
    source_dim = input_dim - side_dim
    per_body_per_frame = spec.source_position_dim * (2 if spec.include_source_velocity else 1)
    denom = spec.history_frames * per_body_per_frame
    if source_dim <= 0 or denom <= 0 or source_dim % denom != 0:
        return None
    return source_dim // denom


def _setup_torch_runtime(
    torch,
    *,
    config: dict[str, Any] | None = None,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ddp_enabled = _train_ddp_enabled(config or {})
    launch_world_size = int(world_size)
    distributed = launch_world_size > 1 and ddp_enabled
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    backend = _train_distributed_backend(config or {}, device_type=device_type)
    if device_type == "cuda":
        device_index = local_rank if launch_world_size > 1 else 0
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cpu")
    if distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
    return {
        "distributed": distributed,
        "rank": rank,
        "world_size": launch_world_size,
        "local_rank": local_rank,
        "device": device,
        "device_type": device_type,
        "distributed_backend": backend if distributed else "",
        "ddp_enabled": ddp_enabled,
        "sample_sharded": launch_world_size > 1,
    }


def _cleanup_torch_runtime(torch, runtime: dict[str, Any]) -> None:
    if runtime.get("distributed") and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _runtime_report(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "distributed": bool(runtime.get("distributed")),
        "ddp_enabled": bool(runtime.get("ddp_enabled", runtime.get("distributed"))),
        "sample_sharded": bool(runtime.get("sample_sharded", runtime.get("distributed"))),
        "rank": int(runtime.get("rank", 0)),
        "world_size": int(runtime.get("world_size", 1)),
        "local_rank": int(runtime.get("local_rank", 0)),
        "device_type": str(runtime.get("device_type", "cpu")),
        "distributed_backend": str(runtime.get("distributed_backend", "")),
    }


def _train_section(config: dict[str, Any]) -> dict[str, Any]:
    payload = config.get("train", {})
    return dict(payload) if isinstance(payload, dict) else {}


def _train_ddp_enabled(config: dict[str, Any]) -> bool:
    env_override = os.environ.get("ONLINE_RETARGET_DDP")
    if env_override is not None:
        return _bool_value(env_override, default=True)
    return _bool_value(_train_section(config).get("ddp", True), default=True)


def _train_distributed_backend(config: dict[str, Any], *, device_type: str) -> str:
    for env_name in ("ONLINE_RETARGET_DISTRIBUTED_BACKEND", "ONLINE_RETARGET_DDP_BACKEND"):
        env_override = os.environ.get(env_name)
        if env_override:
            return env_override.strip()
    train_cfg = _train_section(config)
    ddp_cfg = _ddp_options_section(config)
    backend = (
        ddp_cfg.get("backend")
        or train_cfg.get("distributed_backend")
        or train_cfg.get("ddp_backend")
    )
    if backend:
        return str(backend)
    return "nccl" if device_type == "cuda" else "gloo"


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _ddp_options_section(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = _train_section(config)
    for key in ("ddp_options", "ddp_kwargs", "distributed"):
        payload = train_cfg.get(key, {})
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _optional_bool_option(
    options: dict[str, Any],
    key: str,
    env_name: str,
) -> bool | None:
    env_override = os.environ.get(env_name)
    if env_override is not None:
        return _bool_value(env_override, default=False)
    if key in options:
        return _bool_value(options.get(key), default=False)
    return None


def _optional_float_option(
    options: dict[str, Any],
    key: str,
    env_name: str,
) -> float | None:
    env_override = os.environ.get(env_name)
    value = env_override if env_override is not None else options.get(key)
    if value is None:
        return None
    return float(value)


def _supported_ddp_kwargs(torch) -> set[str]:
    try:
        return set(inspect.signature(torch.nn.parallel.DistributedDataParallel).parameters)
    except (TypeError, ValueError, AttributeError):
        return set()


def _ddp_constructor_kwargs(
    torch,
    config: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    options = _ddp_options_section(config)
    device_type = str(runtime.get("device_type", "cpu"))
    broadcast_buffers = _optional_bool_option(
        options,
        "broadcast_buffers",
        "ONLINE_RETARGET_DDP_BROADCAST_BUFFERS",
    )
    kwargs: dict[str, Any] = {}
    if broadcast_buffers is not None:
        kwargs["broadcast_buffers"] = broadcast_buffers
    if device_type == "cuda":
        local_rank = int(runtime.get("local_rank", 0))
        kwargs["device_ids"] = [local_rank]
        kwargs["output_device"] = local_rank

    optional_bools = (
        ("find_unused_parameters", "ONLINE_RETARGET_DDP_FIND_UNUSED_PARAMETERS"),
        ("static_graph", "ONLINE_RETARGET_DDP_STATIC_GRAPH"),
        ("gradient_as_bucket_view", "ONLINE_RETARGET_DDP_GRADIENT_AS_BUCKET_VIEW"),
        ("init_sync", "ONLINE_RETARGET_DDP_INIT_SYNC"),
    )
    for key, env_name in optional_bools:
        value = _optional_bool_option(options, key, env_name)
        if value is not None:
            kwargs[key] = value
    bucket_cap_mb = _optional_float_option(
        options,
        "bucket_cap_mb",
        "ONLINE_RETARGET_DDP_BUCKET_CAP_MB",
    )
    if bucket_cap_mb is not None:
        kwargs["bucket_cap_mb"] = bucket_cap_mb

    supported = _supported_ddp_kwargs(torch)
    unsupported = []
    if supported:
        for key in list(kwargs):
            if key not in supported:
                unsupported.append(key)
                kwargs.pop(key)
    elif "init_sync" in kwargs:
        unsupported.append("init_sync")
        kwargs.pop("init_sync")

    report = {
        "enabled": bool(runtime.get("distributed")),
        "backend": str(runtime.get("distributed_backend", "")),
        "kwargs": _jsonable_ddp_kwargs(kwargs),
        "unsupported_kwargs": unsupported,
    }
    return kwargs, report


def _jsonable_ddp_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    serializable = {}
    for key, value in kwargs.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            serializable[key] = value
        elif isinstance(value, (list, tuple)):
            serializable[key] = list(value)
        else:
            serializable[key] = str(value)
    return serializable


def _train_dataloader_kwargs(
    config: dict[str, Any],
    runtime: dict[str, Any],
    *,
    dataset_length: int,
    batch_size: int,
    sampler=None,
    shuffle: bool = True,
    force_single_process: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_cfg = _train_section(config)
    loader_cfg = train_cfg.get("dataloader", train_cfg.get("data_loader", {}))
    loader_cfg = dict(loader_cfg) if isinstance(loader_cfg, dict) else {}
    device_type = str(runtime.get("device_type", "cpu"))
    requested_workers = max(0, int(loader_cfg.get("num_workers", 0)))
    num_workers = 0 if force_single_process else requested_workers
    effective_batch_size = min(max(1, int(batch_size)), max(1, int(dataset_length)))
    pin_memory_default = device_type == "cuda"
    pin_memory = bool(loader_cfg.get("pin_memory", pin_memory_default))
    if force_single_process:
        pin_memory = False
    drop_last = bool(loader_cfg.get("drop_last", False))
    shuffle_enabled = bool(loader_cfg.get("shuffle", shuffle)) and sampler is None
    persistent_requested = bool(loader_cfg.get("persistent_workers", False))
    persistent_workers = persistent_requested and num_workers > 0
    prefetch_factor = loader_cfg.get("prefetch_factor", None)
    if prefetch_factor is not None:
        prefetch_factor = max(1, int(prefetch_factor))

    kwargs: dict[str, Any] = {
        "batch_size": effective_batch_size,
        "sampler": sampler,
        "shuffle": shuffle_enabled,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

    report = {
        "batch_size": effective_batch_size,
        "requested_batch_size": int(batch_size),
        "dataset_length": int(dataset_length),
        "sampler": type(sampler).__name__ if sampler is not None else "none",
        "shuffle": shuffle_enabled,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "requested_num_workers": requested_workers,
        "persistent_workers": persistent_workers,
        "requested_persistent_workers": persistent_requested,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "force_single_process": bool(force_single_process),
    }
    if force_single_process and requested_workers > 0:
        report["forced_single_process_reason"] = "materialized_tensors_preloaded_to_device"
    return kwargs, report


def _temporal_feed_config(
    config: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    train_cfg = _train_section(config)
    feed_cfg = train_cfg.get("batch_to_device", train_cfg.get("feeding", {}))
    feed_cfg = dict(feed_cfg) if isinstance(feed_cfg, dict) else {}
    device_type = str(runtime.get("device_type", "cpu"))
    non_blocking = bool(feed_cfg.get("non_blocking", device_type == "cuda"))
    preload_requested = bool(feed_cfg.get("preload_tensors", False))
    pin_requested = bool(feed_cfg.get("pin_materialized_tensors", False))
    preload_active = preload_requested and device_type == "cuda"
    pin_active = pin_requested and device_type == "cuda" and not preload_active
    prebatch_requested = bool(feed_cfg.get("prebatch_in_memory", False))
    prebatch_active = prebatch_requested and preload_active
    report = {
        "non_blocking": non_blocking,
        "preload_tensors": preload_active,
        "preload_tensors_requested": preload_requested,
        "pin_materialized_tensors": pin_active,
        "pin_materialized_tensors_requested": pin_requested,
        "prebatch_in_memory": prebatch_active,
        "prebatch_in_memory_requested": prebatch_requested,
    }
    if preload_requested and not preload_active:
        report["preload_tensors_skip_reason"] = f"device_type={device_type}"
    if pin_requested and not pin_active:
        reason = "preload_tensors_enabled" if preload_active else f"device_type={device_type}"
        report["pin_materialized_tensors_skip_reason"] = reason
    if prebatch_requested and not prebatch_active:
        reason = "preload_tensors_disabled" if not preload_active else f"device_type={device_type}"
        report["prebatch_in_memory_skip_reason"] = reason
    return report


def _prepare_temporal_tensors_for_training(
    tensors: dict[str, Any],
    *,
    device,
    feed: dict[str, Any],
) -> dict[str, Any]:
    prepared = tensors
    if feed.get("pin_materialized_tensors"):
        prepared = {key: value.pin_memory() for key, value in prepared.items()}
    if feed.get("preload_tensors"):
        non_blocking = bool(feed.get("non_blocking", True))
        prepared = {
            key: value.to(device, non_blocking=non_blocking)
            for key, value in prepared.items()
        }
    return prepared


def _forward_microbatch_config(
    config: dict[str, Any],
    *,
    logical_batch_size: int,
) -> dict[str, Any]:
    train_cfg = _train_section(config)
    requested = train_cfg.get(
        "forward_microbatch_size",
        train_cfg.get("microbatch_size", 0),
    )
    requested_size = max(0, int(requested or 0))
    logical_size = max(1, int(logical_batch_size))
    enabled = 0 < requested_size < logical_size
    return {
        "enabled": enabled,
        "requested_size": requested_size,
        "size": requested_size if enabled else logical_size,
        "logical_batch_size": logical_size,
    }


def _checkpointing_config(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = _train_section(config)
    checkpoint_cfg = train_cfg.get("checkpoint", train_cfg.get("checkpointing", {}))
    checkpoint_cfg = dict(checkpoint_cfg) if isinstance(checkpoint_cfg, dict) else {}
    every_steps = checkpoint_cfg.get(
        "every_steps",
        train_cfg.get("checkpoint_every_steps", train_cfg.get("checkpoint_every", 0)),
    )
    keep_last = checkpoint_cfg.get(
        "keep_last",
        train_cfg.get("keep_last_checkpoints", train_cfg.get("checkpoint_keep_last", 0)),
    )
    dir_name = str(checkpoint_cfg.get("dir", train_cfg.get("checkpoint_dir", "checkpoints")))
    every_steps = max(0, int(every_steps or 0))
    keep_last = max(0, int(keep_last or 0))
    return {
        "enabled": every_steps > 0,
        "every_steps": every_steps,
        "keep_last": keep_last,
        "dir": dir_name,
        "latest_manifest": str(checkpoint_cfg.get("latest_manifest", "latest_checkpoint.json")),
    }


def _step_profiler_config(config: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    train_cfg = _train_section(config)
    profiler_cfg = train_cfg.get("step_profiler", train_cfg.get("profiler", {}))
    profiler_cfg = dict(profiler_cfg) if isinstance(profiler_cfg, dict) else {}
    enabled = _bool_value(profiler_cfg.get("enabled", False), default=False)
    active_steps = max(
        0,
        int(profiler_cfg.get("active_steps", profiler_cfg.get("steps", 0)) or 0),
    )
    warmup_steps = max(0, int(profiler_cfg.get("warmup_steps", 0) or 0))
    sync_requested = profiler_cfg.get("synchronize_cuda", None)
    if sync_requested is None:
        synchronize_cuda = enabled and str(runtime.get("device_type", "cpu")) == "cuda"
    else:
        synchronize_cuda = _bool_value(sync_requested, default=False)
    return {
        "enabled": enabled,
        "active_steps": active_steps,
        "warmup_steps": warmup_steps,
        "synchronize_cuda": synchronize_cuda,
        "summary_filename": str(
            profiler_cfg.get("summary_filename", "step_profiler_rank{rank}.json")
        ),
    }


def _init_temporal_step_profiler(config: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    payload = _step_profiler_config(config, runtime)
    return {
        **payload,
        "seen_steps": 0,
        "recorded_steps": 0,
        "recorded_step_numbers": [],
        "phase_samples": {
            **{phase: [] for phase in TEMPORAL_STEP_PROFILER_PHASES},
            "step_total": [],
        },
    }


def _begin_temporal_step_profile(step_profiler: dict[str, Any], step: int) -> dict[str, float] | None:
    if not step_profiler.get("enabled", False):
        return None
    step_profiler["seen_steps"] += 1
    if step_profiler["seen_steps"] <= int(step_profiler.get("warmup_steps", 0)):
        return None
    active_steps = int(step_profiler.get("active_steps", 0))
    if active_steps > 0 and int(step_profiler.get("recorded_steps", 0)) >= active_steps:
        return None
    step_profiler["recorded_steps"] += 1
    step_profiler["recorded_step_numbers"].append(int(step))
    return {}


def _temporal_step_profiler_sync(torch, runtime: dict[str, Any], step_profiler: dict[str, Any]) -> None:
    if not step_profiler.get("synchronize_cuda", False):
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "synchronize"):
        return
    device = runtime.get("device", None)
    try:
        if device is None:
            cuda.synchronize()
        else:
            cuda.synchronize(device)
    except TypeError:
        cuda.synchronize()


@contextlib.contextmanager
def _temporal_step_profile_phase(
    step_profiler: dict[str, Any],
    step_timings: dict[str, float] | None,
    *,
    phase: str,
    torch,
    runtime: dict[str, Any],
):
    if step_timings is None:
        yield
        return
    _temporal_step_profiler_sync(torch, runtime, step_profiler)
    started_at = time.perf_counter()
    try:
        yield
    finally:
        _temporal_step_profiler_sync(torch, runtime, step_profiler)
        step_timings[phase] = step_timings.get(phase, 0.0) + (time.perf_counter() - started_at)


def _finish_temporal_step_profile(
    step_profiler: dict[str, Any],
    step_timings: dict[str, float] | None,
) -> None:
    if step_timings is None:
        return
    phase_samples = step_profiler.get("phase_samples", {})
    total_seconds = 0.0
    for phase in TEMPORAL_STEP_PROFILER_PHASES:
        elapsed_seconds = float(step_timings.get(phase, 0.0))
        phase_samples[phase].append(elapsed_seconds)
        total_seconds += elapsed_seconds
    phase_samples["step_total"].append(total_seconds)


def _step_profiler_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(100.0, float(percentile))) / 100.0 * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * weight


def _step_profiler_phase_summary(
    values: list[float],
    *,
    total_seconds: float,
    share_override: float | None = None,
) -> dict[str, Any]:
    total_value = sum(float(value) for value in values)
    count = len(values)
    mean_seconds = total_value / count if count else 0.0
    share_percent = 0.0
    if share_override is not None:
        share_percent = float(share_override)
    elif total_seconds > 0.0:
        share_percent = total_value / total_seconds * 100.0
    return {
        "count": count,
        "mean_ms": round(mean_seconds * 1000.0, 3),
        "p50_ms": round(_step_profiler_percentile(values, 50.0) * 1000.0, 3),
        "p95_ms": round(_step_profiler_percentile(values, 95.0) * 1000.0, 3),
        "share_percent": round(share_percent, 3),
    }


def _temporal_step_profiler_summary(step_profiler: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "enabled": bool(step_profiler.get("enabled", False)),
        "active_steps": int(step_profiler.get("active_steps", 0)),
        "warmup_steps": int(step_profiler.get("warmup_steps", 0)),
        "synchronize_cuda": bool(step_profiler.get("synchronize_cuda", False)),
        "seen_steps": int(step_profiler.get("seen_steps", 0)),
        "recorded_steps": int(step_profiler.get("recorded_steps", 0)),
        "first_profiled_step": None,
        "last_profiled_step": None,
        "phases": {},
        "step_total": _step_profiler_phase_summary([], total_seconds=0.0, share_override=0.0),
    }
    recorded_step_numbers = step_profiler.get("recorded_step_numbers", [])
    if recorded_step_numbers:
        summary["first_profiled_step"] = int(recorded_step_numbers[0])
        summary["last_profiled_step"] = int(recorded_step_numbers[-1])
    if not summary["enabled"]:
        return summary
    phase_samples = step_profiler.get("phase_samples", {})
    step_totals = list(phase_samples.get("step_total", []))
    total_seconds = sum(float(value) for value in step_totals)
    summary["step_total"] = _step_profiler_phase_summary(
        step_totals,
        total_seconds=total_seconds,
        share_override=100.0 if step_totals else 0.0,
    )
    summary["phases"] = {
        phase: _step_profiler_phase_summary(
            list(phase_samples.get(phase, [])),
            total_seconds=total_seconds,
        )
        for phase in TEMPORAL_STEP_PROFILER_PHASES
    }
    return summary


def _step_profiler_summary_path(
    output_dir: Path,
    *,
    rank: int,
    step_profiler: dict[str, Any],
) -> Path:
    filename = str(step_profiler.get("summary_filename", "step_profiler_rank{rank}.json"))
    return output_dir / filename.format(rank=int(rank))


def _emit_temporal_step_profiler_summary(
    *,
    output_dir: Path,
    rank: int,
    step_profiler: dict[str, Any],
) -> dict[str, Any]:
    summary = _temporal_step_profiler_summary(step_profiler)
    if not summary["enabled"]:
        return summary
    summary_path = _step_profiler_summary_path(output_dir, rank=rank, step_profiler=step_profiler)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("step_profiler_summary=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _should_save_periodic_checkpoint(
    step: int,
    total_steps: int,
    checkpointing: dict[str, Any],
) -> bool:
    every_steps = int(checkpointing.get("every_steps", 0))
    return every_steps > 0 and step > 0 and step < total_steps and step % every_steps == 0


def _resume_training_position(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"resumed": False, "step": 0, "epoch": 0}
    has_position = "step" in payload or "epoch" in payload
    return {
        "resumed": bool(has_position),
        "step": _nonnegative_int(payload.get("step", 0), name="checkpoint step"),
        "epoch": _nonnegative_int(payload.get("epoch", 0), name="checkpoint epoch"),
    }


def _nonnegative_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _save_periodic_training_checkpoint(
    torch,
    *,
    output_dir: Path,
    model,
    optimizer,
    step: int,
    epoch: int,
    loss: float,
    checkpointing: dict[str, Any],
    sample_loader: dict[str, Any],
    data_loader: dict[str, Any],
    feed: dict[str, Any],
    runtime: dict[str, Any],
) -> Path:
    checkpoint_dir = output_dir / str(checkpointing.get("dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
    tmp_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    payload = {
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "loss": float(loss),
        "sample_loader": sample_loader,
        "data_loader": data_loader,
        "batch_to_device": feed,
        "distributed_runtime": _runtime_report(runtime),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    torch.save(payload, tmp_checkpoint)
    tmp_checkpoint.replace(checkpoint)
    _write_latest_checkpoint_manifest(
        output_dir,
        checkpoint=checkpoint,
        step=step,
        loss=loss,
        checkpointing=checkpointing,
    )
    _prune_periodic_checkpoints(checkpoint_dir, keep_last=int(checkpointing.get("keep_last", 0)))
    return checkpoint


def _write_latest_checkpoint_manifest(
    output_dir: Path,
    *,
    checkpoint: Path,
    step: int,
    loss: float,
    checkpointing: dict[str, Any],
) -> None:
    manifest_path = output_dir / str(
        checkpointing.get("latest_manifest", "latest_checkpoint.json")
    )
    manifest = {
        "checkpoint": str(checkpoint),
        "step": int(step),
        "loss": float(loss),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prune_periodic_checkpoints(checkpoint_dir: Path, *, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink(missing_ok=True)


def _train_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples_jsonl: Path,
    output_dir: Path,
    resume_checkpoint: Path | None,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    from online_retarget.models.registry import build_model

    sample_rank = rank if runtime["sample_sharded"] else 0
    sample_world_size = world_size if runtime["sample_sharded"] else 1
    samples, sample_loader = _load_supervised_samples_with_report(
        samples_jsonl,
        rank=sample_rank,
        world_size=sample_world_size,
    )
    if not samples:
        raise SystemExit(f"no supervised samples found in {samples_jsonl}")
    if rank == 0:
        print(f"sample_loader={json.dumps(sample_loader, sort_keys=True)}", flush=True)
    model_family = _configured_model_family(config)
    if model_family == "temporal_diffusion_policy":
        _print_temporal_startup_stage(
            rank=rank,
            stage="sample_loader_done",
            runtime=runtime,
            sample_loader=sample_loader,
            sample_count=len(samples),
            samples_jsonl=samples_jsonl,
        )
        _train_temporal_diffusion_jsonl(
            torch=torch,
            config=config,
            samples=samples,
            sample_loader=sample_loader,
            samples_jsonl=samples_jsonl,
            output_dir=output_dir,
            resume_checkpoint=resume_checkpoint,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
        return
    input_dim = len(samples[0]["observation"])
    output_dim = len(_target_vector(samples[0]))
    observation_spec = _observation_spec_from_config_and_manifest(
        config,
        _load_sample_manifest(samples_jsonl),
        input_dim=input_dim,
    )
    device = runtime["device"]
    x = torch.tensor([sample["observation"] for sample in samples], dtype=torch.float32)
    y = torch.tensor([_target_vector(sample) for sample in samples], dtype=torch.float32)
    prev_y = torch.tensor(
        [_previous_target_vector(sample, output_dim) for sample in samples],
        dtype=torch.float32,
    )
    samples, x, y, prev_y, sample_filter = _filter_finite_supervised_tensors(
        torch,
        samples=samples,
        x=x,
        y=y,
        prev_y=prev_y,
    )
    if not samples:
        raise SystemExit(f"all supervised samples contain non-finite values: {samples_jsonl}")
    if rank == 0:
        print(f"sample_filter={json.dumps(sample_filter, sort_keys=True)}")

    seed = int(_nested_get(config, ("experiment", "seed"), 17))
    torch.manual_seed(seed)
    model_build = build_model(
        config,
        input_dim=input_dim,
        output_dim=output_dim,
        observation_spec=observation_spec,
    )
    model = model_build.model.to(device)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=device)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if state_dict is None:
            raise SystemExit(f"checkpoint lacks model_state_dict: {resume_checkpoint}")
        model.load_state_dict(state_dict)
    if runtime["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime["local_rank"]] if runtime["device_type"] == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    wandb_run = _init_wandb(
        config=config,
        quality_gate=quality_gate,
        output_dir=output_dir,
        enabled=rank == 0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_every = max(1, int(_nested_get(config, ("train", "log_every"), 100)))
    steps = min(max_steps, max_steps if max_steps > 0 else 1)
    dataset = torch.utils.data.TensorDataset(x, y, prev_y)
    sampler = None
    loader_kwargs, data_loader = _train_dataloader_kwargs(
        config,
        runtime,
        dataset_length=len(dataset),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
    )
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    step = 0
    epoch = 0
    while step < steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_x, batch_y, batch_prev_y in loader:
            step += 1
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_prev_y = batch_prev_y.to(device, non_blocking=True)
            loss = _training_loss(
                torch,
                model,
                model_build.family,
                batch_x,
                batch_y,
                config,
                prev_target=batch_prev_y,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if rank == 0 and (step == 1 or step == steps or step % log_every == 0):
                step_log = {"step": step, "loss": float(loss.detach().cpu())}
                print(json.dumps(step_log, sort_keys=True))
                _wandb_log(wandb_run, step_log, step=step)
            if step >= steps:
                break
        epoch += 1

    if runtime["distributed"]:
        torch.distributed.barrier()
    if rank != 0:
        _wandb_finish(wandb_run)
        return

    with torch.no_grad():
        full_pred = _predict_tensor(
            torch,
            model,
            x,
            family=model_build.family,
            config=config,
            batch_size=batch_size,
            device=device,
            prev_y=prev_y,
        )
        final_loss = float(torch.nn.functional.mse_loss(full_pred, y.to(device)).detach().cpu())
    checkpoint = output_dir / "checkpoint.pt"
    predictions_jsonl = output_dir / "train_predictions.jsonl"
    _write_prediction_jsonl(
        predictions_jsonl,
        samples=samples,
        predictions=full_pred.detach().cpu().tolist(),
    )
    eval_result = None
    if bool(_nested_get(config, ("tracking", "auto_offline_eval"), True)):
        eval_result = evaluate_jsonl(
            input_jsonl=predictions_jsonl,
            output_root=output_dir,
            config=_evaluation_config(config, run_name="train_offline_eval"),
        )
    visualization = _write_visualization_artifacts(
        config=config,
        predictions_jsonl=predictions_jsonl,
        output_dir=output_dir,
        eval_result=eval_result,
        run_name="train_visualization",
        checkpoint=checkpoint,
        checkpoint_step=steps,
    )
    report = _build_train_report(
        samples_jsonl=samples_jsonl,
        output_dir=output_dir,
        checkpoint=checkpoint,
        predictions_jsonl=predictions_jsonl,
        offline_eval=eval_result.to_dict() if eval_result is not None else {},
        visualization=visualization,
        sample_count=len(samples),
        input_dim=input_dim,
        output_dim=output_dim,
        max_steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dims=tuple(int(value) for value in model_build.config.get("hidden_dims", [])),
        dropout=float(model_build.config.get("dropout", 0.0)),
        quality_gate=quality_gate,
        device=str(device),
        world_size=world_size,
        rank=rank,
        final_train_mse=final_loss,
        wandb_summary=_wandb_summary(wandb_run),
        model_family=model_build.family,
        model_config=model_build.config,
        loss_config=_loss_config(config),
        evaluation_config=_evaluation_config(config, run_name="train_offline_eval").to_dict(),
        distributed_runtime=_runtime_report(runtime),
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else "",
        sample_filter=sample_filter,
        sample_loader=sample_loader,
        data_loader=data_loader,
        checkpointing=_checkpointing_config(config),
    )
    torch.save(
        {
            "model_state_dict": _unwrap_model(model).state_dict(),
            "report": report,
        },
        checkpoint,
    )
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _wandb_log(wandb_run, {"final_train_mse": final_loss})
    _wandb_save(wandb_run, checkpoint)
    _wandb_save(wandb_run, output_dir / "train_report.json")
    if eval_result is not None:
        _wandb_save(wandb_run, eval_result.summary_json)
    _wandb_log_visualization(wandb_run, visualization, config)
    _wandb_finish(wandb_run)
    print(json.dumps(report, indent=2, sort_keys=True))


def _train_temporal_diffusion_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    sample_loader: dict[str, Any],
    samples_jsonl: Path,
    output_dir: Path,
    resume_checkpoint: Path | None,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    from online_retarget.models.registry import build_model

    _print_temporal_startup_stage(
        rank=rank,
        stage="temporal_entry",
        runtime=runtime,
        sample_count=len(samples),
        samples_jsonl=samples_jsonl,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        max_steps=max_steps,
        batch_size=batch_size,
    )
    input_dim = len(samples[0]["observation"])
    target_shape = _target_action_shape(samples[0])
    action_horizon, action_dim = target_shape
    _print_temporal_startup_stage(
        rank=rank,
        stage="shape_infer_done",
        runtime=runtime,
        input_dim=input_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    manifest_path = samples_jsonl.parent / "manifest.json"
    _print_temporal_startup_stage(
        rank=rank,
        stage="manifest_load_begin",
        runtime=runtime,
        manifest=manifest_path,
    )
    sample_manifest = _load_sample_manifest(samples_jsonl)
    _print_temporal_startup_stage(
        rank=rank,
        stage="manifest_load_done",
        runtime=runtime,
        manifest=manifest_path,
        manifest_loaded=bool(sample_manifest),
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="observation_spec_begin",
        runtime=runtime,
        input_dim=input_dim,
    )
    observation_spec = _observation_spec_from_config_and_manifest(
        config,
        sample_manifest,
        input_dim=input_dim,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="observation_spec_done",
        runtime=runtime,
        flattened_dim=observation_spec.flattened_dim(),
        history_frames=observation_spec.history_frames,
        source_body_count=observation_spec.source_body_count,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="tensorize_begin",
        runtime=runtime,
        sample_count=len(samples),
    )
    tensorize_started_at = time.monotonic()

    def _tensorize_progress(stage: str, **fields: Any) -> None:
        progress_fields = dict(fields)
        progress_fields.setdefault("elapsed_seconds", round(time.monotonic() - tensorize_started_at, 3))
        _print_temporal_startup_stage(
            rank=rank,
            stage=f"tensorize_{stage}",
            runtime=runtime,
            sample_count=len(samples),
            **progress_fields,
        )

    tensors = _temporal_condition_tensors(torch, samples, progress=_tensorize_progress)
    _print_temporal_startup_stage(
        rank=rank,
        stage="tensorize_done",
        runtime=runtime,
        elapsed_seconds=round(time.monotonic() - tensorize_started_at, 3),
        tensor_shapes=_tensor_collection_report(tensors),
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="finite_filter_begin",
        runtime=runtime,
        sample_count=len(samples),
    )
    samples, tensors, sample_filter = _filter_finite_temporal_tensors(
        torch,
        samples=samples,
        tensors=tensors,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="finite_filter_done",
        runtime=runtime,
        sample_filter=sample_filter,
        sample_count=len(samples),
        tensor_shapes=_tensor_collection_report(tensors),
    )
    if not samples:
        raise SystemExit(f"all temporal samples contain non-finite values: {samples_jsonl}")
    _print_temporal_startup_stage(
        rank=rank,
        stage="feature_contract_begin",
        runtime=runtime,
        sample_count=len(samples),
    )
    feature_contract = _temporal_feature_contract_report(config, samples, tensors)
    _print_temporal_startup_stage(
        rank=rank,
        stage="feature_contract_done",
        runtime=runtime,
        feature_contract=feature_contract,
    )
    if rank == 0:
        print(f"sample_filter={json.dumps(sample_filter, sort_keys=True)}", flush=True)
        print(f"feature_contract={json.dumps(feature_contract, sort_keys=True)}", flush=True)

    seed = int(_nested_get(config, ("experiment", "seed"), 17))
    _print_temporal_startup_stage(
        rank=rank,
        stage="manual_seed_begin",
        runtime=runtime,
        seed=seed,
    )
    torch.manual_seed(seed)
    _print_temporal_startup_stage(
        rank=rank,
        stage="manual_seed_done",
        runtime=runtime,
        seed=seed,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="output_dir_mkdir_begin",
        runtime=runtime,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_temporal_startup_stage(
        rank=rank,
        stage="output_dir_mkdir_done",
        runtime=runtime,
        output_dir=output_dir,
    )
    log_every = max(1, int(_nested_get(config, ("train", "log_every"), 100)))
    steps = min(max_steps, max_steps if max_steps > 0 else 1)
    feed = _temporal_feed_config(config, runtime)
    checkpointing = _checkpointing_config(config)
    step_profiler = _init_temporal_step_profiler(config, runtime)
    _print_temporal_startup_stage(
        rank=rank,
        stage="train_config_done",
        runtime=runtime,
        log_every=log_every,
        steps=steps,
        feed=feed,
        checkpointing=checkpointing,
        step_profiler={
            key: value
            for key, value in step_profiler.items()
            if key in ("enabled", "active_steps", "warmup_steps", "synchronize_cuda", "summary_filename")
        },
    )
    sampler = None
    loader_kwargs, data_loader = _train_dataloader_kwargs(
        config,
        runtime,
        dataset_length=len(samples),
        batch_size=batch_size,
        sampler=None,
        shuffle=True,
        force_single_process=bool(feed.get("preload_tensors", False)),
    )
    data_loader["ddp_shard_strategy"] = (
        "jsonl_nonempty_row_index_mod_world_size"
        if sample_loader.get("sharded")
        else "none"
    )
    microbatching = _forward_microbatch_config(
        config,
        logical_batch_size=int(loader_kwargs["batch_size"]),
    )
    _print_temporal_pre_cuda_diagnostics(
        rank=rank,
        runtime=runtime,
        tensors=tensors,
        feed=feed,
        data_loader=data_loader,
        microbatching=microbatching,
        checkpointing=checkpointing,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="ddp_kwargs_begin",
        runtime=runtime,
    )
    ddp_kwargs, ddp_report = _ddp_constructor_kwargs(torch, config, runtime)
    _print_temporal_startup_stage(
        rank=rank,
        stage="ddp_kwargs_done",
        runtime=runtime,
        ddp=ddp_report,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="model_build_begin",
        runtime=runtime,
        input_dim=input_dim,
        output_dim=action_dim,
    )
    model_build = build_model(
        config,
        input_dim=input_dim,
        output_dim=action_dim,
        observation_spec=observation_spec,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="model_build_done",
        runtime=runtime,
        model_family=model_build.family,
        model_config=model_build.config,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="model_to_device_begin",
        runtime=runtime,
        device=runtime["device"],
    )
    model = model_build.model.to(runtime["device"])
    _print_temporal_startup_stage(
        rank=rank,
        stage="model_to_device_done",
        runtime=runtime,
        model=_module_tensor_report(model),
    )
    _print_temporal_ddp_diagnostics(
        rank=rank,
        stage="post_model_to",
        runtime=runtime,
        model=model,
        tensors=tensors,
        ddp=ddp_report,
    )
    resume_payload = None
    resume_position = {"resumed": False, "step": 0, "epoch": 0}
    if resume_checkpoint is not None:
        _print_temporal_startup_stage(
            rank=rank,
            stage="checkpoint_load_begin",
            runtime=runtime,
            checkpoint=resume_checkpoint,
            map_location=runtime["device"],
        )
        resume_payload = torch.load(resume_checkpoint, map_location=runtime["device"])
        _print_temporal_startup_stage(
            rank=rank,
            stage="checkpoint_load_done",
            runtime=runtime,
            checkpoint=resume_checkpoint,
            payload_type=type(resume_payload).__name__,
            payload_keys=(
                sorted(str(key) for key in resume_payload)
                if isinstance(resume_payload, dict)
                else []
            ),
        )
        state_dict = (
            resume_payload.get("model_state_dict") if isinstance(resume_payload, dict) else None
        )
        if state_dict is None:
            raise SystemExit(f"checkpoint lacks model_state_dict: {resume_checkpoint}")
        _print_temporal_startup_stage(
            rank=rank,
            stage="checkpoint_model_state_load_begin",
            runtime=runtime,
            checkpoint=resume_checkpoint,
            state_dict_entries=len(state_dict),
        )
        model.load_state_dict(state_dict)
        _print_temporal_startup_stage(
            rank=rank,
            stage="checkpoint_model_state_load_done",
            runtime=runtime,
            checkpoint=resume_checkpoint,
            state_dict_entries=len(state_dict),
        )
        try:
            resume_position = _resume_training_position(resume_payload)
        except ValueError as exc:
            raise SystemExit(
                f"invalid checkpoint training position: {resume_checkpoint}: {exc}"
            ) from exc
    _print_temporal_startup_stage(
        rank=rank,
        stage="resume_position",
        runtime=runtime,
        resume_position=resume_position,
    )
    if runtime["distributed"]:
        _print_temporal_startup_stage(
            rank=rank,
            stage="ddp_wrap_begin",
            runtime=runtime,
            ddp=ddp_report,
        )
        _print_temporal_ddp_diagnostics(
            rank=rank,
            stage="pre_ddp_wrap",
            runtime=runtime,
            model=model,
            tensors=tensors,
            ddp=ddp_report,
        )
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        _print_temporal_startup_stage(
            rank=rank,
            stage="ddp_wrap_done",
            runtime=runtime,
            ddp=ddp_report,
        )
        _print_temporal_ddp_diagnostics(
            rank=rank,
            stage="post_ddp_wrap",
            runtime=runtime,
            model=_unwrap_model(model),
            tensors=tensors,
            ddp=ddp_report,
        )
    _print_temporal_startup_stage(
        rank=rank,
        stage="optimizer_init_begin",
        runtime=runtime,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    _print_temporal_startup_stage(
        rank=rank,
        stage="optimizer_init_done",
        runtime=runtime,
        learning_rate=learning_rate,
    )
    if isinstance(resume_payload, dict) and resume_payload.get("optimizer_state_dict"):
        _print_temporal_startup_stage(
            rank=rank,
            stage="optimizer_state_load_begin",
            runtime=runtime,
        )
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        _print_temporal_startup_stage(
            rank=rank,
            stage="optimizer_state_load_done",
            runtime=runtime,
        )
    _print_temporal_startup_stage(
        rank=rank,
        stage="wandb_init_begin",
        runtime=runtime,
        enabled=rank == 0,
        wandb_mode=_nested_get(config, ("tracking", "wandb_mode"), ""),
    )
    wandb_run = _init_wandb(
        config=config,
        quality_gate=quality_gate,
        output_dir=output_dir,
        enabled=rank == 0,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="wandb_init_done",
        runtime=runtime,
        enabled=rank == 0,
        run_present=wandb_run is not None,
        run_id=getattr(wandb_run, "id", "") if wandb_run is not None else "",
        run_url=getattr(wandb_run, "url", "") if wandb_run is not None else "",
    )

    _print_temporal_startup_stage(
        rank=rank,
        stage="training_tensors_prepare_begin",
        runtime=runtime,
        feed=feed,
    )
    tensors = _prepare_temporal_tensors_for_training(
        tensors,
        device=runtime["device"],
        feed=feed,
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="training_tensors_prepare_done",
        runtime=runtime,
        tensor_shapes=_tensor_collection_report(tensors),
    )
    _print_temporal_startup_stage(
        rank=rank,
        stage="dataset_build_begin",
        runtime=runtime,
    )
    dataset = torch.utils.data.TensorDataset(*_temporal_training_dataset_tensors(tensors))
    _print_temporal_startup_stage(
        rank=rank,
        stage="dataset_build_done",
        runtime=runtime,
        dataset_length=len(dataset),
    )
    prebatched_epoch_enabled = bool(feed.get("prebatch_in_memory", False))
    data_loader["prebatched_in_memory"] = prebatched_epoch_enabled
    if prebatched_epoch_enabled:
        data_loader["prebatched_epoch_seed"] = seed
    if rank == 0:
        print(f"batch_to_device={json.dumps(feed, sort_keys=True)}", flush=True)
        print(f"data_loader={json.dumps(data_loader, sort_keys=True)}", flush=True)
        print(f"forward_microbatch={json.dumps(microbatching, sort_keys=True)}", flush=True)
        print(f"checkpointing={json.dumps(checkpointing, sort_keys=True)}", flush=True)
        print(f"ddp={json.dumps(ddp_report, sort_keys=True)}", flush=True)
        if resume_position["resumed"]:
            print(f"resume_position={json.dumps(resume_position, sort_keys=True)}", flush=True)
    step = int(resume_position["step"])
    epoch = int(resume_position["epoch"])
    first_train_step = step + 1
    while step < steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        if prebatched_epoch_enabled:
            loader_iter = iter(
                _temporal_prebatched_epoch(
                    tensors,
                    batch_size=int(loader_kwargs["batch_size"]),
                    seed=seed,
                    epoch=epoch,
                    shuffle=bool(data_loader.get("shuffle", False)),
                    drop_last=bool(data_loader.get("drop_last", False)),
                )
            )
        else:
            loader_iter = iter(torch.utils.data.DataLoader(dataset, **loader_kwargs))
        while True:
            fetch_started_at = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            fetch_elapsed_seconds = time.perf_counter() - fetch_started_at
            step += 1
            step_timings = _begin_temporal_step_profile(step_profiler, step)
            if step_timings is not None:
                step_timings["dataloader_next"] = fetch_elapsed_seconds
            if step == first_train_step:
                _print_temporal_ddp_diagnostics(
                    rank=rank,
                    stage="first_batch_cpu",
                    runtime=runtime,
                    batch=batch,
                    ddp=ddp_report,
                )
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            with _temporal_step_profile_phase(
                step_profiler,
                step_timings,
                phase="cpu_batch_materialize_or_cache",
                torch=torch,
                runtime=runtime,
            ):
                microbatches = list(
                    _iter_temporal_microbatches(
                        batch,
                        int(microbatching["size"]),
                    )
                )
            for micro_index, (microbatch, microbatch_count, logical_batch_count) in enumerate(
                microbatches
            ):
                is_last_microbatch = micro_index == len(microbatches) - 1
                sync_context = (
                    model.no_sync()
                    if (
                        runtime["distributed"]
                        and hasattr(model, "no_sync")
                        and not is_last_microbatch
                    )
                    else contextlib.nullcontext()
                )
                with sync_context:
                    if step == first_train_step:
                        _print_temporal_ddp_diagnostics(
                            rank=rank,
                            stage="first_microbatch_to_device_begin",
                            runtime=runtime,
                            batch=microbatch,
                            ddp=ddp_report,
                            microbatching={
                                **microbatching,
                                "microbatch_index": micro_index,
                                "microbatch_count": microbatch_count,
                                "sync": is_last_microbatch,
                            },
                        )
                    with _temporal_step_profile_phase(
                        step_profiler,
                        step_timings,
                        phase="h2d_to_device",
                        torch=torch,
                        runtime=runtime,
                    ):
                        condition = _temporal_batch_to_device(
                            microbatch,
                            runtime["device"],
                            non_blocking=bool(feed.get("non_blocking", True)),
                        )
                    if step == first_train_step:
                        _print_temporal_ddp_diagnostics(
                            rank=rank,
                            stage="first_microbatch_forward_begin",
                            runtime=runtime,
                            condition=condition,
                            ddp=ddp_report,
                            microbatching={
                                **microbatching,
                                "microbatch_index": micro_index,
                                "microbatch_count": microbatch_count,
                                "sync": is_last_microbatch,
                            },
                        )
                    with _temporal_step_profile_phase(
                        step_profiler,
                        step_timings,
                        phase="forward",
                        torch=torch,
                        runtime=runtime,
                    ):
                        batch_y = condition.pop("target_action")
                        loss = _training_loss(
                            torch,
                            model,
                            model_build.family,
                            condition,
                            batch_y,
                            config,
                            prev_target=condition.get("prev_action"),
                        )
                        loss_scale = float(microbatch_count) / float(logical_batch_count)
                    if step == first_train_step:
                        _print_temporal_ddp_diagnostics(
                            rank=rank,
                            stage="first_microbatch_backward_begin",
                            runtime=runtime,
                            ddp=ddp_report,
                            microbatching={
                                **microbatching,
                                "microbatch_index": micro_index,
                                "microbatch_count": microbatch_count,
                                "sync": is_last_microbatch,
                            },
                        )
                    with _temporal_step_profile_phase(
                        step_profiler,
                        step_timings,
                        phase="backward",
                        torch=torch,
                        runtime=runtime,
                    ):
                        (loss * loss_scale).backward()
                    if step == first_train_step:
                        _print_temporal_ddp_diagnostics(
                            rank=rank,
                            stage="first_microbatch_backward_done",
                            runtime=runtime,
                            ddp=ddp_report,
                            microbatching={
                                **microbatching,
                                "microbatch_index": micro_index,
                                "microbatch_count": microbatch_count,
                                "sync": is_last_microbatch,
                            },
                        )
                    loss_value += float(loss.detach().cpu()) * loss_scale
            with _temporal_step_profile_phase(
                step_profiler,
                step_timings,
                phase="optimizer",
                torch=torch,
                runtime=runtime,
            ):
                optimizer.step()
            with _temporal_step_profile_phase(
                step_profiler,
                step_timings,
                phase="logging_checkpoint",
                torch=torch,
                runtime=runtime,
            ):
                if rank == 0 and (step == 1 or step == steps or step % log_every == 0):
                    step_log = {"step": step, "loss": loss_value}
                    print(json.dumps(step_log, sort_keys=True))
                    _wandb_log(wandb_run, step_log, step=step)
                if rank == 0 and _should_save_periodic_checkpoint(step, steps, checkpointing):
                    periodic_checkpoint = _save_periodic_training_checkpoint(
                        torch,
                        output_dir=output_dir,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        epoch=epoch,
                        loss=loss_value,
                        checkpointing=checkpointing,
                        sample_loader=sample_loader,
                        data_loader=data_loader,
                        feed=feed,
                        runtime=runtime,
                    )
                    print(
                        json.dumps(
                            {
                                "checkpoint": str(periodic_checkpoint),
                                "checkpoint_reason": "periodic",
                                "step": step,
                            },
                            sort_keys=True,
                        )
                    )
            _finish_temporal_step_profile(step_profiler, step_timings)
            if step >= steps:
                break
        epoch += 1

    step_profiler_summary = _emit_temporal_step_profiler_summary(
        output_dir=output_dir,
        rank=rank,
        step_profiler=step_profiler,
    )
    if runtime["distributed"]:
        torch.distributed.barrier()
    if rank != 0:
        _wandb_finish(wandb_run)
        return

    with torch.no_grad():
        full_pred = _predict_tensor(
            torch,
            model,
            tensors,
            family=model_build.family,
            config=config,
            batch_size=batch_size,
            device=runtime["device"],
        )
        final_loss = float(
            torch.nn.functional.mse_loss(
                full_pred,
                tensors["target_action"].to(runtime["device"]),
            )
            .detach()
            .cpu()
        )
    checkpoint = output_dir / "checkpoint.pt"
    predictions_jsonl = output_dir / "train_predictions.jsonl"
    _write_prediction_jsonl(
        predictions_jsonl,
        samples=samples,
        predictions=full_pred.detach().cpu().tolist(),
    )
    eval_result = None
    if bool(_nested_get(config, ("tracking", "auto_offline_eval"), True)):
        eval_result = evaluate_jsonl(
            input_jsonl=predictions_jsonl,
            output_root=output_dir,
            config=_evaluation_config(config, run_name="train_offline_eval"),
        )
    visualization = _write_visualization_artifacts(
        config=config,
        predictions_jsonl=predictions_jsonl,
        output_dir=output_dir,
        eval_result=eval_result,
        run_name="train_visualization",
        checkpoint=checkpoint,
        checkpoint_step=steps,
    )
    report = _build_train_report(
        samples_jsonl=samples_jsonl,
        output_dir=output_dir,
        checkpoint=checkpoint,
        predictions_jsonl=predictions_jsonl,
        offline_eval=eval_result.to_dict() if eval_result is not None else {},
        visualization=visualization,
        sample_count=len(samples),
        input_dim=input_dim,
        output_dim=action_dim,
        max_steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dims=(),
        dropout=float(model_build.config.get("dropout", 0.0)),
        quality_gate=quality_gate,
        device=str(runtime["device"]),
        world_size=world_size,
        rank=rank,
        final_train_mse=final_loss,
        wandb_summary=_wandb_summary(wandb_run),
        model_family=model_build.family,
        model_config={
            **model_build.config,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
        },
        loss_config=_loss_config(config),
        evaluation_config=_evaluation_config(config, run_name="train_offline_eval").to_dict(),
        feature_contract=feature_contract,
        distributed_runtime=_runtime_report(runtime),
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else "",
        sample_filter=sample_filter,
        sample_loader=sample_loader,
        data_loader=data_loader,
        batch_to_device=feed,
        forward_microbatch=microbatching,
        ddp=ddp_report,
        checkpointing=checkpointing,
        step_profiler=step_profiler_summary,
    )
    torch.save(
        {
            "model_state_dict": _unwrap_model(model).state_dict(),
            "report": report,
        },
        checkpoint,
    )
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _wandb_log(wandb_run, {"final_train_mse": final_loss})
    _wandb_save(wandb_run, checkpoint)
    _wandb_save(wandb_run, output_dir / "train_report.json")
    if eval_result is not None:
        _wandb_save(wandb_run, eval_result.summary_json)
    _wandb_log_visualization(wandb_run, visualization, config)
    _wandb_finish(wandb_run)
    print(json.dumps(report, indent=2, sort_keys=True))


def _training_loss(
    torch,
    model,
    family: str,
    observation,
    target,
    config: dict[str, Any],
    *,
    prev_target=None,
):
    if family == "flow_matching":
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], 1, device=target.device, dtype=target.dtype)
        state = (1.0 - time) * noise + time * target
        target_velocity = target - noise
        pred_velocity = model(observation, state, time)
        return torch.nn.functional.mse_loss(pred_velocity, target_velocity)
    if family == "diffusion_policy":
        weight = float(_loss_config(config).get("diffusion_policy", 1.0))
        return weight * _unwrap_model(model).diffusion_loss(observation, target)
    if family == "temporal_diffusion_policy":
        loss_cfg = _loss_config(config)
        weight = float(loss_cfg.get("temporal_diffusion_policy", 1.0))
        return weight * _unwrap_model(model).diffusion_loss(
            observation["source_body_tokens"],
            target,
            source_skeleton=observation.get("source_skeleton"),
            morphology=observation.get("morphology"),
            robot_state=observation.get("robot_state"),
            prev_action=observation.get("prev_action"),
            loss_config=loss_cfg,
            fps=observation.get("fps"),
        )
    if family == "token_transformer":
        unwrapped = _unwrap_model(model)
        if prev_target is None:
            prev_target = torch.zeros_like(target)
        prediction, aux = unwrapped.forward_with_aux(observation, prev_state=prev_target)
        return _supervised_loss(torch, prediction, target, config) + _auxiliary_token_loss(
            torch,
            aux,
            config,
        )
    return _supervised_loss(torch, model(observation), target, config)


def _supervised_loss(torch, prediction, target, config: dict[str, Any]):
    loss_cfg = _loss_config(config)
    total = prediction.new_tensor(0.0)
    total_weight = 0.0
    has_explicit_loss = any(key in loss_cfg for key in ("mse", "joint_position", "l1", "smooth_l1"))
    mse_weight = float(loss_cfg.get("mse", loss_cfg.get("joint_position", 0.0 if has_explicit_loss else 1.0)))
    if mse_weight:
        total = total + mse_weight * torch.nn.functional.mse_loss(prediction, target)
        total_weight += abs(mse_weight)
    l1_weight = float(loss_cfg.get("l1", 0.0))
    if l1_weight:
        total = total + l1_weight * torch.nn.functional.l1_loss(prediction, target)
        total_weight += abs(l1_weight)
    smooth_l1_weight = float(loss_cfg.get("smooth_l1", 0.0))
    if smooth_l1_weight:
        total = total + smooth_l1_weight * torch.nn.functional.smooth_l1_loss(prediction, target)
        total_weight += abs(smooth_l1_weight)
    if total_weight == 0.0:
        raise ValueError("loss config must enable at least one of mse, l1, smooth_l1")
    return total


def _auxiliary_token_loss(torch, aux: dict[str, Any], config: dict[str, Any]):
    loss_cfg = _loss_config(config)
    token_cfg = loss_cfg.get("token_autoencoder", {})
    if not isinstance(token_cfg, dict):
        token_cfg = {}
    total = aux["source"].new_tensor(0.0)
    weights = {
        "skeleton": float(token_cfg.get("skeleton", 0.0)),
        "motion": float(token_cfg.get("motion", 0.0)),
        "state": float(token_cfg.get("state", 0.0)),
        "latent_alignment": float(token_cfg.get("latent_alignment", 0.0)),
    }
    if weights["skeleton"] and aux["skeleton"].shape[-1] > 0:
        total = total + weights["skeleton"] * torch.nn.functional.mse_loss(
            aux["skeleton_reconstruction"],
            aux["skeleton"],
        )
    if weights["motion"]:
        total = total + weights["motion"] * torch.nn.functional.mse_loss(
            aux["motion_reconstruction"],
            aux["source"],
        )
    if weights["state"]:
        total = total + weights["state"] * torch.nn.functional.mse_loss(
            aux["state_reconstruction"],
            aux["prev_state"],
        )
    if weights["latent_alignment"]:
        total = total + weights["latent_alignment"] * torch.nn.functional.mse_loss(
            aux["z_motion"],
            aux["z_state"],
        )
    return total


def _predict_tensor(
    torch,
    model,
    x,
    *,
    family: str,
    config: dict[str, Any],
    batch_size: int,
    device,
    prev_y=None,
):
    model.eval()
    predictions = []
    if family == "temporal_diffusion_policy":
        count = x["source_body_tokens"].shape[0]
        diffusion_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
        for start in range(0, count, max(1, batch_size)):
            batch = {
                key: value[start : start + batch_size].to(device, non_blocking=True)
                for key, value in x.items()
                if key != "target_action"
            }
            pred = _unwrap_model(model).sample(
                batch["source_body_tokens"],
                source_skeleton=batch.get("source_skeleton"),
                morphology=batch.get("morphology"),
                robot_state=batch.get("robot_state"),
                prev_action=batch.get("prev_action"),
                steps=int(
                    diffusion_cfg.get(
                        "inference_steps",
                        diffusion_cfg.get("diffusion_steps", 32),
                    )
                ),
                start=str(diffusion_cfg.get("inference_start", "zeros")),
            )
            predictions.append(pred.detach())
        return torch.cat(predictions, dim=0)
    for start in range(0, x.shape[0], max(1, batch_size)):
        batch = x[start : start + batch_size].to(device, non_blocking=True)
        if family == "flow_matching":
            flow_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
            pred = _unwrap_model(model).sample(
                batch,
                steps=int(flow_cfg.get("inference_steps", 8)),
                start=str(flow_cfg.get("inference_start", "zeros")),
            )
        elif family == "diffusion_policy":
            diffusion_cfg = (
                config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
            )
            pred = _unwrap_model(model).sample(
                batch,
                steps=int(
                    diffusion_cfg.get(
                        "inference_steps",
                        diffusion_cfg.get("diffusion_steps", 32),
                    )
                ),
                start=str(diffusion_cfg.get("inference_start", "zeros")),
            )
        elif family == "token_transformer":
            if prev_y is None:
                prev_batch = torch.zeros(
                    batch.shape[0],
                    _unwrap_model(model).output_dim,
                    device=device,
                    dtype=batch.dtype,
                )
            else:
                prev_batch = prev_y[start : start + batch_size].to(device, non_blocking=True)
            pred = _unwrap_model(model)(batch, prev_state=prev_batch)
        else:
            pred = model(batch)
        predictions.append(pred.detach())
    return torch.cat(predictions, dim=0)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _target_vector(sample: dict[str, Any]) -> list[float]:
    future = sample.get("future_target_joints")
    if isinstance(future, list) and future and all(isinstance(row, list) for row in future):
        return [float(value) for row in future for value in row]
    target = sample.get("target_joints")
    if isinstance(target, list):
        return [float(value) for value in target]
    raise ValueError("sample lacks target_joints or future_target_joints")


def _previous_target_vector(sample: dict[str, Any], output_dim: int) -> list[float]:
    for key in ("prev_target_joints", "previous_target_joints", "prev_g1_joints"):
        value = sample.get(key)
        if isinstance(value, list) and len(value) == output_dim:
            return [float(item) for item in value]
        if isinstance(value, list) and value and output_dim % len(value) == 0:
            repeats = output_dim // len(value)
            return [float(item) for _ in range(repeats) for item in value]
    return [0.0] * output_dim


def _previous_target_joints(sample: dict[str, Any], output_dim: int) -> list[float]:
    return _previous_target_vector(sample, output_dim)


def _target_action_sequence(sample: dict[str, Any]) -> list[list[float]]:
    future = sample.get("future_target_joints")
    if isinstance(future, list) and future and all(isinstance(row, list) for row in future):
        return [[float(value) for value in row] for row in future]
    target = sample.get("target_joints")
    if isinstance(target, list):
        return [[float(value) for value in target]]
    raise ValueError("sample lacks target_joints or future_target_joints")


def _target_action_shape(sample: dict[str, Any]) -> tuple[int, int]:
    sequence = _target_action_sequence(sample)
    if not sequence or not sequence[0]:
        raise ValueError("target action sequence must be non-empty")
    return len(sequence), len(sequence[0])


def _temporal_condition_tensors(torch, samples: list[dict[str, Any]], *, progress=None) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must be non-empty")
    started_at = time.monotonic()

    def _progress(stage: str, **fields: Any) -> None:
        if progress is None:
            return
        progress(stage, elapsed_seconds=round(time.monotonic() - started_at, 3), **fields)

    first_target_shape = _target_action_shape(samples[0])
    source_body_tokens = _source_body_token_sequence(samples[0])
    source_skeleton_dim = len(_source_skeleton_vector(samples[0], source_body_tokens))
    morphology_dim = len(_morphology_condition_vector(samples[0]))
    robot_state_dim = len(_robot_state_vector(samples[0]))
    action_dim = first_target_shape[1]
    _progress(
        "shape_infer_done",
        total_count=len(samples),
        target_horizon=first_target_shape[0],
        action_dim=action_dim,
        source_skeleton_dim=source_skeleton_dim,
        morphology_dim=morphology_dim,
        robot_state_dim=robot_state_dim,
    )
    rows = {
        "source_body_tokens": [],
        "source_skeleton": [],
        "morphology": [],
        "robot_state": [],
        "prev_action": [],
        "fps": [],
        "target_action": [],
    }
    progress_interval = max(1, min(8192, len(samples)))
    _progress("rows_begin", total_count=len(samples), progress_interval=progress_interval)
    for index, sample in enumerate(samples):
        target_action = _target_action_sequence(sample)
        shape = (len(target_action), len(target_action[0]) if target_action else 0)
        if shape != first_target_shape:
            raise ValueError(
                "temporal samples must share target shape "
                f"{first_target_shape}; sample {index} has {shape}"
            )
        source_tokens = _source_body_token_sequence(sample)
        if len(source_tokens) != first_target_shape[0]:
            raise ValueError(
                "source_body_tokens horizon must match future_target_joints; "
                f"sample {index} has {len(source_tokens)} vs {first_target_shape[0]}"
            )
        rows["source_body_tokens"].append(source_tokens)
        rows["source_skeleton"].append(
            _pad_or_trim(_source_skeleton_vector(sample, source_tokens), source_skeleton_dim)
        )
        rows["morphology"].append(_pad_or_trim(_morphology_condition_vector(sample), morphology_dim))
        rows["robot_state"].append(_pad_or_trim(_robot_state_vector(sample), robot_state_dim))
        rows["prev_action"].append(_pad_or_trim(_prev_action_vector(sample), action_dim))
        rows["fps"].append(float(sample.get("fps", 30.0)))
        rows["target_action"].append(target_action)
        processed_count = index + 1
        if processed_count == len(samples) or processed_count % progress_interval == 0:
            _progress("rows_progress", processed_count=processed_count, total_count=len(samples))
    _progress("rows_done", total_count=len(samples))

    tensors: dict[str, Any] = {}
    for key, value in rows.items():
        _progress("tensor_convert_begin", key=key, row_count=len(value))
        tensors[key] = torch.tensor(value, dtype=torch.float32)
        _progress("tensor_convert_key_done", key=key, tensor=_tensor_report(tensors[key]))
    _progress("tensor_convert_done", keys=list(tensors.keys()))
    return tensors


def _feature_contract_config(config: dict[str, Any]) -> dict[str, Any]:
    payload = config.get("feature_contract", {})
    return dict(payload) if isinstance(payload, dict) else {}


def _temporal_feature_contract_report(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    tensors: dict[str, Any],
) -> dict[str, Any]:
    cfg = _feature_contract_config(config)
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    evaluation_cfg = (
        config.get("evaluation", {}) if isinstance(config.get("evaluation", {}), dict) else {}
    )
    metrics = evaluation_cfg.get("metrics", ())
    if isinstance(metrics, str):
        metric_names = [metrics]
    else:
        metric_names = [str(metric) for metric in metrics]
    model_robot_state_dim = int(model_cfg.get("robot_state_dim", tensors["robot_state"].shape[1]))
    actual_condition_sample_fields = list(
        _temporal_condition_sample_fields(model_robot_state_dim=model_robot_state_dim)
    )
    actual_model_condition_tensor_keys = list(
        _temporal_model_condition_tensor_keys(model_robot_state_dim=model_robot_state_dim)
    )
    condition_keys = cfg.get("condition_sample_keys")
    condition_keys = (
        [str(key) for key in condition_keys] if isinstance(condition_keys, list) else []
    )
    forbidden_keys = cfg.get(
        "forbid_condition_sample_keys",
        list(TEMPORAL_FORBIDDEN_CONDITION_SAMPLE_FIELDS),
    )
    forbidden_keys = (
        [str(key) for key in forbidden_keys] if isinstance(forbidden_keys, list) else []
    )
    dimensions = {
        "sample_count": len(samples),
        "source_body_count": int(tensors["source_body_tokens"].shape[2]),
        "source_body_token_dim": int(tensors["source_body_tokens"].shape[3]),
        "target_horizon_frames": int(tensors["target_action"].shape[1]),
        "action_dim": int(tensors["target_action"].shape[2]),
        "source_skeleton_dim": int(tensors["source_skeleton"].shape[1]),
        "morphology_dim": int(tensors["morphology"].shape[1]),
        "robot_state_tensor_dim": int(tensors["robot_state"].shape[1]),
        "model_robot_state_dim": model_robot_state_dim,
    }
    violations = []
    forbidden_declared_condition_keys = sorted(set(condition_keys) & set(forbidden_keys))
    if forbidden_declared_condition_keys:
        violations.append(
            "condition_sample_keys include target-only or identity keys: "
            + ", ".join(forbidden_declared_condition_keys)
        )
    forbidden_actual_condition_keys = sorted(
        set(actual_condition_sample_fields) & set(forbidden_keys)
    )
    if forbidden_actual_condition_keys:
        violations.append(
            "actual temporal condition source fields include target-only or identity keys: "
            + ", ".join(forbidden_actual_condition_keys)
        )
    if condition_keys and set(condition_keys) != set(actual_condition_sample_fields):
        missing_actual = sorted(set(actual_condition_sample_fields) - set(condition_keys))
        stale_declared = sorted(set(condition_keys) - set(actual_condition_sample_fields))
        detail = []
        if missing_actual:
            detail.append("missing actual fields: " + ", ".join(missing_actual))
        if stale_declared:
            detail.append("declares unused fields: " + ", ".join(stale_declared))
        violations.append(
            "condition_sample_keys must match actual condition source fields ("
            + "; ".join(detail)
            + ")"
        )
    expected = cfg.get("expected", {})
    if isinstance(expected, dict):
        for key, expected_value in expected.items():
            actual_key = "model_robot_state_dim" if key == "robot_state_dim" else str(key)
            actual_value = dimensions.get(actual_key)
            if actual_value is None:
                continue
            try:
                expected_int = int(expected_value)
            except (TypeError, ValueError):
                continue
            if int(actual_value) != expected_int:
                violations.append(f"{key} expected {expected_int}, got {actual_value}")
    robot_state_policy = str(cfg.get("robot_state_policy", "allow_zero"))
    robot_state_abs_sum = float(tensors["robot_state"].abs().sum().detach().cpu())
    robot_state_nonzero = robot_state_abs_sum > 0.0
    if robot_state_policy in {"disabled", "none"} and model_robot_state_dim != 0:
        violations.append("robot_state_policy=disabled requires model.robot_state_dim=0")
    if robot_state_policy == "require_nonzero" and not robot_state_nonzero:
        violations.append("robot_state_policy=require_nonzero but robot_state tensor is all zero")
    required_metrics = cfg.get("required_eval_metrics", ())
    if isinstance(required_metrics, str):
        required_metric_names = [required_metrics]
    elif isinstance(required_metrics, list):
        required_metric_names = [str(metric) for metric in required_metrics]
    else:
        required_metric_names = []
    missing_metrics = sorted(set(required_metric_names) - set(metric_names))
    if missing_metrics:
        violations.append("evaluation.metrics missing: " + ", ".join(missing_metrics))
    output_contract = {
        "model_output_mode": str(model_cfg.get("output_mode", "absolute")),
        "prediction_export": "absolute_g1_joint_position_future_window",
        "target_format": str(_nested_get(config, ("data", "target_format"), "")),
        "model_output": str(model_cfg.get("output", "")),
    }
    digest_payload = {
        "condition_sample_keys": condition_keys,
        "actual_condition_sample_fields": actual_condition_sample_fields,
        "actual_model_condition_tensor_keys": actual_model_condition_tensor_keys,
        "dimensions": dimensions,
        "output_contract": output_contract,
        "required_eval_metrics": required_metric_names,
        "robot_state_policy": robot_state_policy,
    }
    report = {
        "enabled": True,
        "name": str(cfg.get("name", "temporal_diffusion_policy_feature_eval_v1")),
        "status": "pass" if not violations else "fail",
        "condition_sample_keys": condition_keys,
        "actual_condition_sample_fields": actual_condition_sample_fields,
        "actual_model_condition_tensor_keys": actual_model_condition_tensor_keys,
        "forbidden_condition_sample_keys": forbidden_keys,
        "dimensions": dimensions,
        "output_contract": output_contract,
        "required_eval_metrics": required_metric_names,
        "missing_eval_metrics": missing_metrics,
        "robot_state_policy": robot_state_policy,
        "robot_state_nonzero": robot_state_nonzero,
        "actor_uid_used_as_input": "actor_uid" in actual_condition_sample_fields,
        "violations": violations,
        "digest": hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    if violations and bool(cfg.get("enforce", True)):
        raise SystemExit("Temporal feature contract failed: " + "; ".join(violations))
    return report


def _temporal_condition_sample_fields(*, model_robot_state_dim: int) -> tuple[str, ...]:
    fields = list(TEMPORAL_MODEL_CONDITION_SAMPLE_FIELDS)
    if model_robot_state_dim > 0:
        fields.extend(TEMPORAL_ROBOT_STATE_SAMPLE_FIELDS)
    return tuple(fields)


def _temporal_model_condition_tensor_keys(*, model_robot_state_dim: int) -> tuple[str, ...]:
    keys = ["source_body_tokens", "source_skeleton", "morphology", "prev_action"]
    if model_robot_state_dim > 0:
        keys.append("robot_state")
    return tuple(keys)


def _source_body_token_sequence(sample: dict[str, Any]) -> list[list[list[float]]]:
    tokens = sample.get("source_body_tokens")
    if (
        isinstance(tokens, list)
        and tokens
        and all(isinstance(step, list) and step for step in tokens)
        and all(isinstance(body, list) for step in tokens for body in step)
    ):
        return [
            [[float(value) for value in body] for body in step]
            for step in tokens
        ]
    raise ValueError("temporal_diffusion_policy samples require source_body_tokens [T,N,D]")


def _source_skeleton_vector(
    sample: dict[str, Any],
    source_body_tokens: list[list[list[float]]] | None = None,
) -> list[float]:
    value = sample.get("source_skeleton")
    if isinstance(value, list):
        return [float(item) for item in value]
    body_count = len(source_body_tokens[0]) if source_body_tokens else 0
    return [0.0] * body_count * 4


def _morphology_condition_vector(sample: dict[str, Any]) -> list[float]:
    value = sample.get("morphology")
    if isinstance(value, list):
        return [float(item) for item in value]
    return [0.0] * 13


def _robot_state_vector(sample: dict[str, Any]) -> list[float]:
    value = sample.get("robot_state")
    if isinstance(value, list):
        return [float(item) for item in value]
    return [0.0] * 94


def _prev_action_vector(sample: dict[str, Any]) -> list[float]:
    for key in ("prev_target_joints", "previous_target_joints", "prev_g1_joints"):
        value = sample.get(key)
        if isinstance(value, list):
            return [float(item) for item in value]
    return []


def _pad_or_trim(values: list[float], width: int) -> list[float]:
    if len(values) >= width:
        return values[:width]
    return values + [0.0] * (width - len(values))


def _filter_finite_temporal_tensors(
    torch,
    *,
    samples: list[dict[str, Any]],
    tensors: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Drop JSONL rows with NaN/Inf values in any temporal tensor."""

    keep_mask = None
    finite_by_key = {}
    for key, tensor in tensors.items():
        finite = torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1)
        finite_by_key[key] = finite
        keep_mask = finite if keep_mask is None else keep_mask & finite
    if keep_mask is None:
        raise ValueError("no tensors to filter")
    dropped_indices = (~keep_mask).nonzero(as_tuple=False).flatten().tolist()
    report = {
        "input_count": len(samples),
        "filtered_count": int(keep_mask.sum().item()),
        "dropped_count": len(dropped_indices),
        "dropped_examples": [],
    }
    if not dropped_indices:
        return samples, tensors, report

    for index in dropped_indices[:20]:
        reasons = [
            f"{key}_nonfinite"
            for key, finite in finite_by_key.items()
            if not bool(finite[index])
        ]
        sample = samples[index]
        report["dropped_examples"].append(
            {
                "index": int(index),
                "sample_id": str(sample.get("sample_id", "")),
                "source_motion_path": str(sample.get("source_motion_path", "")),
                "target_g1_path": str(sample.get("target_g1_path", "")),
                "reasons": reasons,
            }
        )

    keep_indices = keep_mask.nonzero(as_tuple=False).flatten()
    filtered_samples = [samples[int(index)] for index in keep_indices.tolist()]
    filtered_tensors = {
        key: tensor.index_select(0, keep_indices)
        for key, tensor in tensors.items()
    }
    return filtered_samples, filtered_tensors, report


def _temporal_batch_to_device(batch, device, *, non_blocking: bool = True) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in zip(TEMPORAL_BATCH_KEYS, batch)
    }


def _iter_temporal_microbatches(batch, microbatch_size: int):
    logical_batch_count = int(batch[0].shape[0])
    if logical_batch_count <= 0:
        raise ValueError("temporal batch must be non-empty")
    chunk_size = logical_batch_count
    if microbatch_size > 0:
        chunk_size = min(int(microbatch_size), logical_batch_count)
    for start in range(0, logical_batch_count, chunk_size):
        end = min(start + chunk_size, logical_batch_count)
        yield tuple(value[start:end] for value in batch), end - start, logical_batch_count


def _print_temporal_pre_cuda_diagnostics(
    *,
    rank: int,
    runtime: dict[str, Any],
    tensors: dict[str, Any],
    feed: dict[str, Any],
    data_loader: dict[str, Any],
    microbatching: dict[str, Any],
    checkpointing: dict[str, Any],
) -> None:
    tensor_shapes = {key: [int(dim) for dim in value.shape] for key, value in tensors.items()}
    print(
        " ".join(
            (
                f"rank={rank}",
                f"data_loader.num_workers={data_loader.get('num_workers')}",
                f"data_loader.pin_memory={str(data_loader.get('pin_memory')).lower()}",
            )
        ),
        flush=True,
    )
    print(
        " ".join(
            (
                f"rank={rank}",
                f"forward_microbatch enabled={str(microbatching.get('enabled')).lower()}",
                f"size={microbatching.get('size')}",
                f"logical_batch_size={microbatching.get('logical_batch_size')}",
            )
        ),
        flush=True,
    )
    print(
        "pre_cuda_train_state="
        + json.dumps(
            {
                "rank": rank,
                "runtime": _runtime_report(runtime),
                "tensor_shapes": tensor_shapes,
                "batch_to_device": feed,
                "data_loader": data_loader,
                "forward_microbatch": microbatching,
                "checkpointing": checkpointing,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _print_temporal_startup_stage(
    *,
    rank: int,
    stage: str,
    runtime: dict[str, Any],
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {
        "rank": int(rank),
        "stage": str(stage),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "runtime": _runtime_report(runtime),
        "device": str(runtime.get("device", "")),
        "env": {
            key: os.environ.get(key, "")
            for key in (
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "CUDA_VISIBLE_DEVICES",
                "ONLINE_RETARGET_DDP",
                "WANDB_MODE",
            )
        },
    }
    for key, value in fields.items():
        payload[str(key)] = _jsonable_log_value(value)
    print("temporal_startup_state=" + json.dumps(payload, sort_keys=True), flush=True)


def _jsonable_log_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_log_value(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _tensor_report(value)
    return str(value)


def _print_temporal_ddp_diagnostics(
    *,
    rank: int,
    stage: str,
    runtime: dict[str, Any],
    model=None,
    tensors: dict[str, Any] | None = None,
    batch=None,
    condition: dict[str, Any] | None = None,
    ddp: dict[str, Any] | None = None,
    microbatching: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "rank": int(rank),
        "stage": str(stage),
        "runtime": _runtime_report(runtime),
    }
    if model is not None:
        payload["model"] = _module_tensor_report(model)
    if tensors is not None:
        payload["tensor_shapes"] = _tensor_collection_report(tensors)
    if batch is not None:
        payload["batch_shapes"] = _temporal_batch_report(batch)
    if condition is not None:
        payload["condition_shapes"] = _tensor_collection_report(condition)
    if ddp is not None:
        payload["ddp"] = ddp
    if microbatching is not None:
        payload["forward_microbatch"] = microbatching
    print("temporal_ddp_state=" + json.dumps(payload, sort_keys=True), flush=True)


def _tensor_collection_report(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _tensor_report(value) for key, value in values.items()}


def _temporal_batch_report(batch) -> dict[str, Any]:
    return {
        key: _tensor_report(value)
        for key, value in zip(TEMPORAL_BATCH_KEYS, batch)
    }


def _tensor_report(value) -> dict[str, Any]:
    shape = getattr(value, "shape", ())
    return {
        "shape": [int(dim) for dim in shape],
        "dtype": str(getattr(value, "dtype", "")),
        "device": str(getattr(value, "device", "")),
    }


def _module_tensor_report(model) -> dict[str, Any]:
    named_parameters = list(model.named_parameters())
    named_buffers = list(model.named_buffers())
    signature = hashlib.sha256()
    trainable_numel = 0
    frozen_numel = 0
    parameter_numel = 0
    buffer_numel = 0
    for name, parameter in named_parameters:
        numel = int(parameter.numel())
        requires_grad = bool(getattr(parameter, "requires_grad", False))
        parameter_numel += numel
        trainable_numel += numel if requires_grad else 0
        frozen_numel += 0 if requires_grad else numel
        signature.update(
            f"param:{name}:{tuple(int(dim) for dim in parameter.shape)}:{requires_grad};".encode()
        )
    for name, buffer in named_buffers:
        numel = int(buffer.numel())
        buffer_numel += numel
        signature.update(
            f"buffer:{name}:{tuple(int(dim) for dim in buffer.shape)};".encode()
        )
    return {
        "module_type": type(model).__name__,
        "parameter_count": len(named_parameters),
        "buffer_count": len(named_buffers),
        "parameter_numel": parameter_numel,
        "trainable_parameter_numel": trainable_numel,
        "frozen_parameter_numel": frozen_numel,
        "buffer_numel": buffer_numel,
        "tensor_signature_sha256": signature.hexdigest(),
    }


def _temporal_training_dataset_tensors(tensors: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(tensors[key] for key in TEMPORAL_BATCH_KEYS)


def _temporal_prebatched_epoch(
    tensors: dict[str, Any],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    drop_last: bool,
):
    dataset = _temporal_training_dataset_tensors(tensors)
    if not dataset:
        return
    logical_batch_count = int(dataset[0].shape[0])
    if logical_batch_count <= 0:
        return
    effective_batch_size = min(max(1, int(batch_size)), logical_batch_count)
    indices = list(range(logical_batch_count))
    if shuffle and logical_batch_count > 1:
        random.Random(int(seed) + int(epoch)).shuffle(indices)
    for start in range(0, logical_batch_count, effective_batch_size):
        end = min(start + effective_batch_size, logical_batch_count)
        if drop_last and end - start < effective_batch_size:
            break
        batch_indices = indices[start:end]
        yield tuple(value[batch_indices] for value in dataset)


def _filter_finite_supervised_tensors(
    torch,
    *,
    samples: list[dict[str, Any]],
    x,
    y,
    prev_y,
) -> tuple[list[dict[str, Any]], Any, Any, Any, dict[str, Any]]:
    """Drop JSONL rows with NaN/Inf tensors before they can poison optimization."""

    finite_x = torch.isfinite(x).all(dim=1)
    finite_y = torch.isfinite(y).all(dim=1)
    finite_prev_y = torch.isfinite(prev_y).all(dim=1)
    keep_mask = finite_x & finite_y & finite_prev_y
    dropped_indices = (~keep_mask).nonzero(as_tuple=False).flatten().tolist()
    report = {
        "input_count": len(samples),
        "filtered_count": int(keep_mask.sum().item()),
        "dropped_count": len(dropped_indices),
        "dropped_examples": [],
    }
    if not dropped_indices:
        return samples, x, y, prev_y, report

    for index in dropped_indices[:20]:
        reasons = []
        if not bool(finite_x[index]):
            reasons.append("observation_nonfinite")
        if not bool(finite_y[index]):
            reasons.append("target_joints_nonfinite")
        if not bool(finite_prev_y[index]):
            reasons.append("prev_target_joints_nonfinite")
        sample = samples[index]
        report["dropped_examples"].append(
            {
                "index": int(index),
                "sample_id": str(sample.get("sample_id", "")),
                "source_motion_path": str(sample.get("source_motion_path", "")),
                "target_g1_path": str(sample.get("target_g1_path", "")),
                "reasons": reasons,
            }
        )

    keep_indices = keep_mask.nonzero(as_tuple=False).flatten()
    filtered_samples = [samples[int(index)] for index in keep_indices.tolist()]
    return (
        filtered_samples,
        x.index_select(0, keep_indices),
        y.index_select(0, keep_indices),
        prev_y.index_select(0, keep_indices),
        report,
    )


def _loss_config(config: dict[str, Any]) -> dict[str, Any]:
    payload = config.get("loss", {})
    return dict(payload) if isinstance(payload, dict) else {}


def _evaluation_config(config: dict[str, Any], *, run_name: str) -> EvaluationConfig:
    payload = config.get("evaluation", {})
    if not isinstance(payload, dict):
        payload = {}
    metrics = payload.get("metrics", ())
    if isinstance(metrics, str):
        metric_tuple = (metrics,)
    else:
        metric_tuple = tuple(str(metric) for metric in metrics)
    return EvaluationConfig(
        metrics=metric_tuple,
        fps=float(payload.get("fps", 30.0)),
        joint_jump_velocity=float(payload.get("joint_jump_velocity", 20.0)),
        ground_height=float(payload.get("ground_height", 0.0)),
        up_axis=payload.get("up_axis", 2),
        contact_height_threshold=float(payload.get("contact_height_threshold", 0.04)),
        max_contact_slide_speed=float(payload.get("max_contact_slide_speed", 0.25)),
        max_contact_skate_distance=float(payload.get("max_contact_skate_distance", 0.02)),
        failure_metric=str(payload.get("failure_metric", "joint_rmse")),
        max_failures=int(payload.get("max_failures", 50)),
        run_name=run_name,
    )


def _load_supervised_samples(
    samples_jsonl: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> list[dict[str, Any]]:
    samples, _report = _load_supervised_samples_with_report(
        samples_jsonl,
        rank=rank,
        world_size=world_size,
    )
    return samples


def _load_supervised_samples_with_report(
    samples_jsonl: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    samples = []
    total_nonempty_rows = 0
    parsed_count = 0
    with samples_jsonl.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            sample_index = total_nonempty_rows
            total_nonempty_rows += 1
            if world_size > 1 and sample_index % world_size != rank:
                continue
            sample = json.loads(stripped)
            parsed_count += 1
            if "observation" not in sample:
                raise ValueError(f"sample on line {line_number} lacks observation")
            try:
                _target_vector(sample)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sample on line {line_number} lacks target_joints/future_target_joints"
                ) from exc
            samples.append(sample)

    dropped_uneven_tail_count = 0
    if world_size > 1:
        even_shard_count = total_nonempty_rows // world_size
        if len(samples) > even_shard_count:
            dropped_uneven_tail_count = len(samples) - even_shard_count
            samples = samples[:even_shard_count]

    report = {
        "path": str(samples_jsonl),
        "rank": int(rank),
        "world_size": int(world_size),
        "sharded": bool(world_size > 1),
        "assignment": "nonempty_jsonl_row_index_mod_world_size",
        "total_nonempty_rows_seen": int(total_nonempty_rows),
        "parsed_count": int(parsed_count),
        "materialized_count": int(len(samples)),
        "skipped_by_shard_count": int(total_nonempty_rows - parsed_count),
        "dropped_uneven_tail_count": int(dropped_uneven_tail_count),
    }
    return samples, report


def _predict_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples_jsonl: Path,
    checkpoint: Path,
    output_dir: Path,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    from online_retarget.models.registry import build_model

    if rank != 0:
        return
    samples = _load_supervised_samples(samples_jsonl)
    if not samples:
        raise SystemExit(f"no supervised samples found in {samples_jsonl}")
    if _configured_model_family(config) == "temporal_diffusion_policy":
        _predict_temporal_diffusion_jsonl(
            torch=torch,
            config=config,
            samples=samples,
            samples_jsonl=samples_jsonl,
            checkpoint=checkpoint,
            output_dir=output_dir,
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
        return
    input_dim = len(samples[0]["observation"])
    output_dim = len(_target_vector(samples[0]))
    observation_spec = _observation_spec_from_config_and_manifest(
        config,
        _load_sample_manifest(samples_jsonl),
        input_dim=input_dim,
    )
    device = runtime["device"]
    model_build = build_model(
        config,
        input_dim=input_dim,
        output_dim=output_dim,
        observation_spec=observation_spec,
    )
    model = model_build.model.to(device)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if state_dict is None:
        raise SystemExit(f"checkpoint lacks model_state_dict: {checkpoint}")
    model.load_state_dict(state_dict)
    model.eval()
    x = torch.tensor([sample["observation"] for sample in samples], dtype=torch.float32)
    y = torch.tensor([_target_vector(sample) for sample in samples], dtype=torch.float32)
    prev_y = torch.tensor(
        [_previous_target_vector(sample, output_dim) for sample in samples],
        dtype=torch.float32,
    )
    samples, x, y, prev_y, sample_filter = _filter_finite_supervised_tensors(
        torch,
        samples=samples,
        x=x,
        y=y,
        prev_y=prev_y,
    )
    if not samples:
        raise SystemExit(f"all supervised samples contain non-finite values: {samples_jsonl}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        pred = _predict_tensor(
            torch,
            model,
            x,
            family=model_build.family,
            config=config,
            batch_size=int(_nested_get(config, ("train", "batch_size"), 64)),
            device=device,
            prev_y=prev_y,
        )
        predictions = pred.detach().cpu().tolist()
        final_mse = float(torch.nn.functional.mse_loss(pred, y.to(device)).detach().cpu())
    predictions_jsonl = output_dir / "predictions.jsonl"
    _write_prediction_jsonl(predictions_jsonl, samples=samples, predictions=predictions)
    eval_result = None
    if bool(_nested_get(config, ("tracking", "auto_offline_eval"), True)):
        eval_result = evaluate_jsonl(
            input_jsonl=predictions_jsonl,
            output_root=output_dir,
            config=_evaluation_config(config, run_name="offline_eval"),
        )
    visualization = _write_visualization_artifacts(
        config=config,
        predictions_jsonl=predictions_jsonl,
        output_dir=output_dir,
        eval_result=eval_result,
        run_name="predict_visualization",
        checkpoint=checkpoint,
        checkpoint_step=_checkpoint_step_from_payload(payload),
    )
    report = {
        "mode": "predict_only",
        "samples_jsonl": str(samples_jsonl),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_jsonl),
        "offline_eval": eval_result.to_dict() if eval_result is not None else {},
        "visualization": visualization,
        "sample_count": len(samples),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "model_family": model_build.family,
        "model_config": model_build.config,
        "loss_config": _loss_config(config),
        "evaluation_config": _evaluation_config(config, run_name="offline_eval").to_dict(),
        "sample_filter": sample_filter,
        "quality_gate": quality_gate,
        "device": str(device),
        "world_size": world_size,
        "rank": rank,
        "mse": final_mse,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    (output_dir / "predict_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _predict_temporal_diffusion_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    samples_jsonl: Path,
    checkpoint: Path,
    output_dir: Path,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    from online_retarget.models.registry import build_model

    if rank != 0:
        return
    input_dim = len(samples[0]["observation"])
    action_horizon, action_dim = _target_action_shape(samples[0])
    observation_spec = _observation_spec_from_config_and_manifest(
        config,
        _load_sample_manifest(samples_jsonl),
        input_dim=input_dim,
    )
    device = runtime["device"]
    model_build = build_model(
        config,
        input_dim=input_dim,
        output_dim=action_dim,
        observation_spec=observation_spec,
    )
    model = model_build.model.to(device)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if state_dict is None:
        raise SystemExit(f"checkpoint lacks model_state_dict: {checkpoint}")
    model.load_state_dict(state_dict)
    model.eval()
    tensors = _temporal_condition_tensors(torch, samples)
    samples, tensors, sample_filter = _filter_finite_temporal_tensors(
        torch,
        samples=samples,
        tensors=tensors,
    )
    if not samples:
        raise SystemExit(f"all temporal samples contain non-finite values: {samples_jsonl}")
    feature_contract = _temporal_feature_contract_report(config, samples, tensors)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        pred = _predict_tensor(
            torch,
            model,
            tensors,
            family=model_build.family,
            config=config,
            batch_size=int(_nested_get(config, ("train", "batch_size"), 64)),
            device=device,
        )
        predictions = pred.detach().cpu().tolist()
        final_mse = float(
            torch.nn.functional.mse_loss(
                pred,
                tensors["target_action"].to(device),
            )
            .detach()
            .cpu()
        )
    predictions_jsonl = output_dir / "predictions.jsonl"
    _write_prediction_jsonl(predictions_jsonl, samples=samples, predictions=predictions)
    eval_result = None
    if bool(_nested_get(config, ("tracking", "auto_offline_eval"), True)):
        eval_result = evaluate_jsonl(
            input_jsonl=predictions_jsonl,
            output_root=output_dir,
            config=_evaluation_config(config, run_name="offline_eval"),
        )
    visualization = _write_visualization_artifacts(
        config=config,
        predictions_jsonl=predictions_jsonl,
        output_dir=output_dir,
        eval_result=eval_result,
        run_name="predict_visualization",
        checkpoint=checkpoint,
        checkpoint_step=_checkpoint_step_from_payload(payload),
    )
    report = {
        "mode": "predict_only",
        "samples_jsonl": str(samples_jsonl),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_jsonl),
        "offline_eval": eval_result.to_dict() if eval_result is not None else {},
        "visualization": visualization,
        "sample_count": len(samples),
        "input_dim": input_dim,
        "output_dim": action_dim,
        "model_family": model_build.family,
        "model_config": {
            **model_build.config,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
        },
        "loss_config": _loss_config(config),
        "evaluation_config": _evaluation_config(config, run_name="offline_eval").to_dict(),
        "feature_contract": feature_contract,
        "sample_filter": sample_filter,
        "quality_gate": quality_gate,
        "device": str(device),
        "world_size": world_size,
        "rank": rank,
        "mse": final_mse,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    (output_dir / "predict_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _write_prediction_jsonl(
    path: Path,
    *,
    samples: list[dict[str, Any]],
    predictions: list[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample, prediction in zip(samples, predictions):
            payload = {
                "sample_id": sample.get("sample_id", ""),
                "actor_uid": sample.get("actor_uid", ""),
                "category": sample.get("category", ""),
                "package": sample.get("package", ""),
                "quality_flags": sample.get("quality_flags", []),
                "predicted_joints": _prediction_sequence(sample, prediction),
                "target_joints": _target_sequence(sample),
            }
            for key in (
                "fps",
                "target_frame",
                "target_frame_indices",
                "target_horizon_frames",
                "source_motion_path",
                "target_g1_path",
                "sonic_relative_path",
                "target_joint_names",
            ):
                if key in sample:
                    payload[key] = sample[key]
            payload["sequence_id"] = str(
                sample.get(
                    "target_g1_path",
                    sample.get("sonic_relative_path", sample.get("source_motion_path", "")),
                )
            )
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")


def _target_sequence(sample: dict[str, Any]) -> list[list[float]]:
    future = sample.get("future_target_joints")
    if isinstance(future, list) and future and all(isinstance(row, list) for row in future):
        return [[float(value) for value in row] for row in future]
    return [[float(value) for value in sample["target_joints"]]]


def _prediction_sequence(sample: dict[str, Any], prediction: Any) -> list[list[float]]:
    if (
        isinstance(prediction, list)
        and prediction
        and all(isinstance(row, list) for row in prediction)
    ):
        return [[float(value) for value in row] for row in prediction]
    target = _target_sequence(sample)
    horizon = len(target)
    joint_dim = len(target[0]) if target else len(prediction)
    if horizon > 0 and joint_dim > 0 and len(prediction) == horizon * joint_dim:
        return [
            [float(value) for value in prediction[index * joint_dim : (index + 1) * joint_dim]]
            for index in range(horizon)
        ]
    return [[float(value) for value in prediction]]


def _write_visualization_artifacts(
    *,
    config: dict[str, Any],
    predictions_jsonl: Path,
    output_dir: Path,
    eval_result: Any,
    run_name: str,
    checkpoint: Path | None = None,
    checkpoint_step: int | None = None,
) -> dict[str, Any]:
    visual_cfg = config.get("visualization", {})
    if not isinstance(visual_cfg, dict):
        visual_cfg = {}
    accepted_vertical_cfg = _accepted_vertical_v2_config(config, visual_cfg)
    route_enabled = bool(visual_cfg.get("enabled", False))
    accepted_vertical_enabled = bool(accepted_vertical_cfg.get("enabled", False))
    if not route_enabled and not accepted_vertical_enabled:
        return {"enabled": False}
    artifact_name = str(visual_cfg.get("artifact_name") or visual_cfg.get("run_name") or run_name)
    configured_output = str(visual_cfg.get("output_dir", "") or "")
    if configured_output:
        artifact_dir = Path(configured_output).expanduser()
        if not artifact_dir.is_absolute():
            artifact_dir = output_dir / artifact_dir
    else:
        artifact_dir = output_dir / "visualization" / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    num_samples = max(1, int(visual_cfg.get("num_samples", visual_cfg.get("max_samples", 4))))
    max_joints = max(1, int(visual_cfg.get("max_joints", 8)))
    samples = _read_prediction_samples(predictions_jsonl, limit=num_samples)
    trajectory_csv = artifact_dir / "trajectory_preview.csv"
    svg_path = artifact_dir / "trajectory_preview.svg"
    html_path = artifact_dir / "trajectory_preview.html"
    if route_enabled:
        rows = _visualization_rows(samples, max_joints=max_joints)
        _write_visualization_csv(trajectory_csv, rows)
        _write_visualization_svg(svg_path, rows)
        _write_visualization_html(html_path, rows, svg_path=svg_path)
        capsule = _write_capsule_visualization_artifacts(
            visual_cfg=visual_cfg,
            samples=samples,
            artifact_dir=artifact_dir,
        )
    else:
        rows = []
        capsule = {"enabled": False}
    accepted_vertical = _write_accepted_vertical_v2_artifacts(
        config=config,
        visual_cfg=accepted_vertical_cfg,
        predictions_jsonl=predictions_jsonl,
        artifact_dir=artifact_dir,
        checkpoint=checkpoint,
        checkpoint_step=checkpoint_step,
    )
    summary_path = artifact_dir / "visual_manifest.json"
    eval_payload = eval_result.to_dict() if eval_result is not None else {}
    summary = {
        "enabled": True,
        "artifact_version": "route_b_joint_trajectory_v1",
        "artifact_name": artifact_name,
        "status": "ok" if rows else "empty",
        "predictions_jsonl": str(predictions_jsonl),
        "output_dir": str(artifact_dir),
        "trajectory_csv": str(trajectory_csv),
        "trajectory_svg": str(svg_path),
        "trajectory_html": str(html_path),
        "capsule_visualization": capsule,
        "accepted_vertical_v2": accepted_vertical,
        "summary_json": str(summary_path),
        "sample_count": len(samples),
        "trajectory_row_count": len(rows),
        "num_samples": num_samples,
        "max_joints": max_joints,
        "offline_eval": eval_payload,
        "wandb_upload": bool(visual_cfg.get("wandb_upload", False)),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _accepted_vertical_v2_config(config: dict[str, Any], visual_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit train-closeout accepted_vertical_v2 config.

    Top-level visual_validation remains the bridge CLI/config surface, but it is
    not an opt-in gate for train/predict closeout. The closeout bridge must be
    enabled under visualization.accepted_vertical_v2.enabled.
    """

    nested = visual_cfg.get("accepted_vertical_v2", {})
    return dict(nested) if isinstance(nested, dict) else {}


def _write_accepted_vertical_v2_artifacts(
    *,
    config: dict[str, Any],
    visual_cfg: dict[str, Any],
    predictions_jsonl: Path,
    artifact_dir: Path,
    checkpoint: Path | None,
    checkpoint_step: int | None,
) -> dict[str, Any]:
    if not bool(visual_cfg.get("enabled", False)):
        return {"enabled": False}
    output_name = str(
        visual_cfg.get("output_dir")
        or visual_cfg.get("artifact_name")
        or "accepted_vertical_v2"
    )
    output_dir = Path(output_name).expanduser()
    if not output_dir.is_absolute():
        output_dir = artifact_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "lr310_dp_visual_validation_summary.json"
    count_value = visual_cfg.get("count", visual_cfg.get("num_samples", visual_cfg.get("max_samples", 1)))
    count = max(1, int(count_value))
    execute_renderers = bool(visual_cfg.get("execute_renderers", False))
    skip_source_bvh_resolve = bool(visual_cfg.get("skip_source_bvh_resolve", not execute_renderers))
    continue_on_error = bool(visual_cfg.get("continue_on_error", True))
    root_source = str(visual_cfg.get("root_source", "auto"))
    root_body_index = int(visual_cfg.get("root_body_index", 0))
    root_body_name = str(visual_cfg.get("root_body_name", "pelvis"))
    root_quat_format = str(visual_cfg.get("root_quat_format", "wxyz"))
    allow_root_fixed_fallback = bool(visual_cfg.get("allow_root_fixed_fallback", False))
    step = int(0 if checkpoint_step is None else checkpoint_step)
    target_g1_roots = _visual_validation_path_list(
        visual_cfg.get("target_g1_roots", visual_cfg.get("target_g1_root", []))
    )
    bridge_config = dict(config)
    bridge_config["visual_validation"] = dict(visual_cfg)
    bridge = _load_lr310_dp_visual_bridge()
    rows = bridge.read_prediction_rows(predictions_jsonl, count=count)
    clips: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            clips.append(
                bridge.rerender_prediction_row(
                    row=row,
                    index=index,
                    predictions_jsonl=predictions_jsonl,
                    output_dir=output_dir,
                    config=bridge_config,
                    target_g1_roots=target_g1_roots,
                    step=step,
                    execute_renderers=execute_renderers,
                    root_source=root_source,
                    root_body_index=root_body_index,
                    root_body_name=root_body_name,
                    root_quat_format=root_quat_format,
                    allow_root_fixed_fallback=allow_root_fixed_fallback,
                    checkpoint_path=checkpoint,
                    checkpoint_step=checkpoint_step,
                    skip_source_bvh_resolve=skip_source_bvh_resolve,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            errors.append(
                {
                    "index": int(index),
                    "sample_id": str(row.get("sample_id") or row.get("sequence_id") or ""),
                    "error": str(exc),
                }
            )
    export_status = _accepted_vertical_v2_export_status(clips=clips, errors=errors)
    status = _accepted_vertical_v2_summary_status(
        clips=clips,
        errors=errors,
        requested_clip_count=len(rows),
        execute_renderers=execute_renderers,
    )
    accepted_count = sum(1 for clip in clips if bool(clip.get("acceptance_ok", False)))
    payload = {
        "enabled": True,
        "artifact_version": "lr310_dp_accepted_vertical_v2_train_bridge_v1",
        "status": status,
        "export_status": export_status,
        "predictions_jsonl": str(predictions_jsonl),
        "output_dir": str(output_dir),
        "summary_json": str(summary_path),
        "count": count,
        "requested_clip_count": len(rows),
        "step": step,
        "checkpoint": str(checkpoint or ""),
        "checkpoint_step": checkpoint_step,
        "execute_renderers": execute_renderers,
        "skip_source_bvh_resolve": skip_source_bvh_resolve,
        "continue_on_error": continue_on_error,
        "target_g1_roots": [str(path) for path in target_g1_roots],
        "visualization_core": "scripts.rerender_lr310_dp_visual_validation.rerender_prediction_row",
        "accepted_vertical_v2_ok_count": accepted_count,
        "clip_count": len(clips),
        "error_count": len(errors),
        "clips": clips,
        "errors": errors,
    }
    if not execute_renderers:
        payload["renderer_status"] = (
            "not_executed; accepted_vertical_v2 NPZ assets, commands, and metadata were exported only"
        )
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _accepted_vertical_v2_export_status(
    *,
    clips: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> str:
    if clips and errors:
        return "partial"
    if errors:
        return "failed"
    if clips:
        return "ok"
    return "empty"


def _accepted_vertical_v2_summary_status(
    *,
    clips: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    requested_clip_count: int,
    execute_renderers: bool,
) -> str:
    if requested_clip_count <= 0:
        return "empty"
    accepted_count = sum(1 for clip in clips if bool(clip.get("acceptance_ok", False)))
    if accepted_count == requested_clip_count and not errors:
        return "ok"
    if accepted_count > 0:
        return "partial"
    if clips and not execute_renderers:
        return "blocked"
    if clips or errors:
        return "failed"
    return "empty"


def _visual_validation_path_list(value: Any) -> list[Path]:
    if value in (None, ""):
        return []
    if isinstance(value, (str, os.PathLike)):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    return [Path(str(item)).expanduser() for item in values if str(item)]


def _checkpoint_step_from_payload(payload: Any) -> int | None:
    if not isinstance(payload, dict) or "step" not in payload:
        return None
    try:
        return int(payload["step"])
    except (TypeError, ValueError):
        return None


def _load_lr310_dp_visual_bridge():
    from scripts import rerender_lr310_dp_visual_validation

    return rerender_lr310_dp_visual_validation


def _read_prediction_samples(path: Path, *, limit: int) -> list[dict[str, Any]]:
    samples = []
    if not path.exists():
        return samples
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            samples.append(json.loads(stripped))
            if len(samples) >= limit:
                break
    return samples


def _visualization_rows(samples: list[dict[str, Any]], *, max_joints: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        predicted = sample.get("predicted_joints", [])
        target = sample.get("target_joints", [])
        if not isinstance(predicted, list) or not isinstance(target, list):
            continue
        joint_names = sample.get("target_joint_names", [])
        if not isinstance(joint_names, list):
            joint_names = []
        frame_indices = sample.get("target_frame_indices", [])
        if not isinstance(frame_indices, list):
            frame_indices = []
        for horizon_index, (pred_frame, target_frame) in enumerate(zip(predicted, target)):
            if not isinstance(pred_frame, list) or not isinstance(target_frame, list):
                continue
            joint_count = min(max_joints, len(pred_frame), len(target_frame))
            for joint_index in range(joint_count):
                predicted_value = float(pred_frame[joint_index])
                target_value = float(target_frame[joint_index])
                rows.append(
                    {
                        "sample_id": str(sample.get("sample_id", "")),
                        "sequence_id": str(sample.get("sequence_id", "")),
                        "actor_uid": str(sample.get("actor_uid", "")),
                        "category": str(sample.get("category", "")),
                        "package": str(sample.get("package", "")),
                        "horizon_index": horizon_index,
                        "target_frame": (
                            frame_indices[horizon_index]
                            if horizon_index < len(frame_indices)
                            else sample.get("target_frame", "")
                        ),
                        "joint_index": joint_index,
                        "joint_name": str(joint_names[joint_index]) if joint_index < len(joint_names) else f"joint_{joint_index}",
                        "predicted": predicted_value,
                        "target": target_value,
                        "abs_error": abs(predicted_value - target_value),
                    }
                )
    return rows


def _write_capsule_visualization_artifacts(
    *,
    visual_cfg: dict[str, Any],
    samples: list[dict[str, Any]],
    artifact_dir: Path,
) -> dict[str, Any]:
    capsule_cfg = visual_cfg.get("capsule", {})
    if capsule_cfg is True:
        capsule_cfg = {"enabled": True}
    if not isinstance(capsule_cfg, dict) or not bool(capsule_cfg.get("enabled", False)):
        return {"enabled": False}
    capsule_dir = artifact_dir / str(capsule_cfg.get("output_dir") or "capsule_preview")
    capsule_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = capsule_dir / "capsule_manifest.json"
    html_path = capsule_dir / "capsule_preview.html"
    num_samples = max(1, int(capsule_cfg.get("num_samples", visual_cfg.get("num_samples", 1))))
    max_frames = max(0, int(capsule_cfg.get("max_frames", capsule_cfg.get("render_max_frames", 120))))
    width = max(1, int(capsule_cfg.get("width", capsule_cfg.get("render_width", 640))))
    height = max(1, int(capsule_cfg.get("height", capsule_cfg.get("render_height", 360))))
    fps = float(capsule_cfg.get("fps", 50.0))
    model_xml = _optional_path(capsule_cfg.get("model_xml") or visual_cfg.get("g1_model_xml"))
    selected_samples = samples[:num_samples]
    sample_reports: list[dict[str, Any]] = []
    render_statuses: list[str] = []
    deps: dict[str, Any] | None = None
    deps_error = ""
    if model_xml is None or not model_xml.exists():
        deps_error = f"g1_model_xml is missing: {model_xml}" if model_xml else "g1_model_xml is not configured"
    else:
        try:
            deps = _load_route_b_capsule_render_deps()
        except Exception as exc:
            deps_error = f"Route B capsule render dependencies are unavailable: {exc}"

    model = None
    edges: Any = None
    render_config = None
    if deps is not None and model_xml is not None:
        try:
            model = deps["load_g1_kinematic_model"](model_xml)
            edges = deps["_g1_capsule_edges"](model)
            render_config = deps["ReviewClipExportConfig"](
                render_max_frames=max_frames,
                render_width=width,
                render_height=height,
                fps=fps,
                model_xml=model_xml,
            )
        except Exception as exc:
            deps_error = f"Could not initialize Route B G1 capsule renderer: {exc}"
            deps = None

    for index, sample in enumerate(selected_samples):
        sample_dir = capsule_dir / f"{index:02d}_{_safe_visual_name(str(sample.get('sample_id', 'sample')))}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        predicted_sequence = _joint_sequence_for_capsules(sample.get("predicted_joints"), max_frames=max_frames)
        target_sequence = _joint_sequence_for_capsules(sample.get("target_joints"), max_frames=max_frames)
        trajectory_path = sample_dir / "joint_trajectory.json"
        trajectory_payload = {
            "sample_id": str(sample.get("sample_id", "")),
            "sequence_id": str(sample.get("sequence_id", "")),
            "fps": fps,
            "target_joint_names": sample.get("target_joint_names", []),
            "predicted_joints": predicted_sequence,
            "target_joints": target_sequence,
            "note": "Route B capsule preview uses G1 FK with root fixed at origin; it is a train/eval visualization artifact, not Isaac Lab rollout evidence.",
        }
        trajectory_path.write_text(
            json.dumps(trajectory_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sample_report: dict[str, Any] = {
            "sample_id": str(sample.get("sample_id", "")),
            "sequence_id": str(sample.get("sequence_id", "")),
            "trajectory_json": str(trajectory_path),
            "predicted_frames": len(predicted_sequence),
            "target_frames": len(target_sequence),
            "target_video": "",
            "predicted_video": "",
            "target_render": {"status": "blocked", "message": deps_error or "capsule renderer unavailable"},
            "predicted_render": {"status": "blocked", "message": deps_error or "capsule renderer unavailable"},
        }
        if deps is not None and model is not None and render_config is not None and edges is not None:
            sample_report["target_render"] = _render_route_b_capsule_sequence(
                deps=deps,
                model=model,
                edges=edges,
                render_config=render_config,
                sequence=target_sequence,
                video_path=sample_dir / "target_g1_3d_capsules.mp4",
                label="Route B target G1 FK capsules",
                capsule_color=(61, 107, 160),
                key_color=(139, 91, 41),
            )
            sample_report["predicted_render"] = _render_route_b_capsule_sequence(
                deps=deps,
                model=model,
                edges=edges,
                render_config=render_config,
                sequence=predicted_sequence,
                video_path=sample_dir / "predicted_g1_3d_capsules.mp4",
                label="Route B predicted G1 FK capsules",
                capsule_color=(142, 77, 117),
                key_color=(122, 89, 35),
            )
            for key, render_key in (
                ("target_video", "target_render"),
                ("predicted_video", "predicted_render"),
            ):
                render = sample_report[render_key]
                if isinstance(render, dict) and render.get("status") == "ok":
                    sample_report[key] = str(render.get("video_path", ""))
        for render_key in ("target_render", "predicted_render"):
            render = sample_report.get(render_key)
            if isinstance(render, dict):
                render_statuses.append(str(render.get("status", "unknown")))
        sample_reports.append(sample_report)

    if not selected_samples:
        status = "empty"
    elif render_statuses and all(status == "ok" for status in render_statuses):
        status = "ok"
    elif any(status == "ok" for status in render_statuses):
        status = "partial"
    else:
        status = "blocked"
    manifest = {
        "enabled": True,
        "artifact_version": "route_b_g1_capsule_visualization_v1",
        "status": status,
        "message": deps_error,
        "backend": "online_retarget.data.review_clips._render_capsule_3d_video",
        "sonic_semantics": [
            "software_perspective_capsules",
            "g1_fk_body_capsule_edges",
            "target_vs_predicted_route_b_joint_sequences",
        ],
        "model_xml": str(model_xml) if model_xml is not None else "",
        "output_dir": str(capsule_dir),
        "manifest_json": str(manifest_path),
        "html": str(html_path),
        "num_samples": num_samples,
        "max_frames": max_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "samples": sample_reports,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_capsule_visualization_html(html_path, manifest)
    return manifest


def _load_route_b_capsule_render_deps() -> dict[str, Any]:
    from online_retarget.data.review_clips import (
        ReviewClipExportConfig,
        _g1_capsule_edges,
        _g1_capsule_frames,
        _render_capsule_3d_video,
    )
    from online_retarget.data.g1_quality import load_g1_kinematic_model

    return {
        "ReviewClipExportConfig": ReviewClipExportConfig,
        "_g1_capsule_edges": _g1_capsule_edges,
        "_g1_capsule_frames": _g1_capsule_frames,
        "_render_capsule_3d_video": _render_capsule_3d_video,
        "load_g1_kinematic_model": load_g1_kinematic_model,
    }


def _render_route_b_capsule_sequence(
    *,
    deps: dict[str, Any],
    model: Any,
    edges: Any,
    render_config: Any,
    sequence: list[list[float]],
    video_path: Path,
    label: str,
    capsule_color: tuple[int, int, int],
    key_color: tuple[int, int, int],
) -> dict[str, Any]:
    if not sequence:
        return {"status": "blocked", "message": "No Route B joint frames were available."}
    try:
        trajectory = _g1_joint_sequence_to_trajectory(sequence)
        frames = deps["_g1_capsule_frames"](model, trajectory)
        return deps["_render_capsule_3d_video"](
            frames=frames,
            edges=edges,
            video_path=video_path,
            config=render_config,
            label=label,
            up_axis=2,
            capsule_color=capsule_color,
            key_color=key_color,
        )
    except Exception as exc:
        return {"status": "failed", "message": f"Route B capsule render failed: {exc}"}


def _g1_joint_sequence_to_trajectory(sequence: list[list[float]]) -> list[dict[str, Any]]:
    from online_retarget.data.bones_seed import G1_JOINT_COLUMNS

    trajectory: list[dict[str, Any]] = []
    for frame, joints in enumerate(sequence):
        trajectory.append(
            {
                "frame": frame,
                "root": [0.0, 0.0, 0.0],
                "root_euler": [0.0, 0.0, 0.0],
                "joints": {
                    column: float(joints[index]) if index < len(joints) else 0.0
                    for index, column in enumerate(G1_JOINT_COLUMNS)
                },
            }
        )
    return trajectory


def _joint_sequence_for_capsules(value: Any, *, max_frames: int) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    if value and all(isinstance(row, list) for row in value):
        rows = value
    elif all(isinstance(item, (int, float)) for item in value):
        rows = [value]
    else:
        return []
    if max_frames > 0:
        rows = rows[:max_frames]
    sequence: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        try:
            sequence.append([float(item) for item in row])
        except (TypeError, ValueError):
            continue
    return sequence


def _write_capsule_visualization_html(path: Path, manifest: dict[str, Any]) -> None:
    sample_rows = []
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        target_video = _video_tag(sample.get("target_video", ""), base_dir=path.parent)
        predicted_video = _video_tag(sample.get("predicted_video", ""), base_dir=path.parent)
        target_render = sample.get("target_render") if isinstance(sample.get("target_render"), dict) else {}
        predicted_render = sample.get("predicted_render") if isinstance(sample.get("predicted_render"), dict) else {}
        sample_rows.append(
            "<section>"
            f"<h2>{_html_escape(sample.get('sample_id', 'sample'))}</h2>"
            f"<p>target: {_html_escape(target_render.get('status', ''))} - {_html_escape(target_render.get('message', ''))}</p>"
            f"{target_video}"
            f"<p>predicted: {_html_escape(predicted_render.get('status', ''))} - {_html_escape(predicted_render.get('message', ''))}</p>"
            f"{predicted_video}"
            "</section>"
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html><head><meta charset="utf-8"><title>Route B 3D capsule preview</title></head>',
                "<body>",
                "<h1>Route B 3D capsule preview</h1>",
                f"<p>Status: {_html_escape(manifest.get('status', ''))}</p>",
                f"<p>Backend: {_html_escape(manifest.get('backend', ''))}</p>",
                f"<p>Model XML: {_html_escape(manifest.get('model_xml', ''))}</p>",
                *sample_rows,
                "</body></html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _video_tag(path_text: Any, *, base_dir: Path) -> str:
    if not path_text:
        return ""
    try:
        src = os.path.relpath(Path(str(path_text)), base_dir)
    except ValueError:
        src = str(path_text)
    return (
        '<video controls muted playsinline width="640" '
        f'src="{_html_escape(src)}"></video>'
    )


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _safe_visual_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe[:96] or "sample"


def _write_visualization_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "sequence_id",
        "actor_uid",
        "category",
        "package",
        "horizon_index",
        "target_frame",
        "joint_index",
        "joint_name",
        "predicted",
        "target",
        "abs_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_visualization_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width = 720
    height = 260
    padding = 32
    series_rows = _first_visual_series(rows)
    if not series_rows:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="260">'
            '<text x="32" y="40">No trajectory rows available</text></svg>\n',
            encoding="utf-8",
        )
        return
    values = [float(row["predicted"]) for row in series_rows] + [float(row["target"]) for row in series_rows]
    low = min(values)
    high = max(values)
    if abs(high - low) < 1.0e-8:
        high = low + 1.0

    def point(value: float, index: int) -> tuple[float, float]:
        if len(series_rows) == 1:
            x = width * 0.5
        else:
            x = padding + (width - 2 * padding) * index / (len(series_rows) - 1)
        y = height - padding - (height - 2 * padding) * (value - low) / (high - low)
        return x, y

    pred_points = " ".join(
        f"{x:.2f},{y:.2f}"
        for index, row in enumerate(series_rows)
        for x, y in [point(float(row["predicted"]), index)]
    )
    target_points = " ".join(
        f"{x:.2f},{y:.2f}"
        for index, row in enumerate(series_rows)
        for x, y in [point(float(row["target"]), index)]
    )
    title = _html_escape(f"{series_rows[0]['sample_id']} {series_rows[0]['joint_name']}")
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="260" viewBox="0 0 720 260">',
                '<rect x="0" y="0" width="720" height="260" fill="white"/>',
                f'<text x="32" y="24" font-size="14" fill="#111">{title}</text>',
                f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#999"/>',
                f'<polyline points="{target_points}" fill="none" stroke="#2f6fed" stroke-width="2"/>',
                f'<polyline points="{pred_points}" fill="none" stroke="#d14b2f" stroke-width="2"/>',
                '<text x="32" y="248" font-size="12" fill="#2f6fed">target</text>',
                '<text x="96" y="248" font-size="12" fill="#d14b2f">predicted</text>',
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_visualization_html(path: Path, rows: list[dict[str, Any]], *, svg_path: Path) -> None:
    table_rows = "\n".join(
        "<tr>"
        f"<td>{_html_escape(row['sample_id'])}</td>"
        f"<td>{_html_escape(row['joint_name'])}</td>"
        f"<td>{row['horizon_index']}</td>"
        f"<td>{float(row['predicted']):.6f}</td>"
        f"<td>{float(row['target']):.6f}</td>"
        f"<td>{float(row['abs_error']):.6f}</td>"
        "</tr>"
        for row in rows[:200]
    )
    svg_text = svg_path.read_text(encoding="utf-8") if svg_path.exists() else ""
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html><head><meta charset="utf-8"><title>Route B trajectory preview</title></head>',
                "<body>",
                "<h1>Route B trajectory preview</h1>",
                svg_text,
                "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
                "<thead><tr><th>sample_id</th><th>joint</th><th>horizon</th><th>predicted</th><th>target</th><th>abs_error</th></tr></thead>",
                f"<tbody>{table_rows}</tbody>",
                "</table>",
                "</body></html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _first_visual_series(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    first = rows[0]
    sample_id = first["sample_id"]
    joint_index = first["joint_index"]
    return [
        row
        for row in rows
        if row["sample_id"] == sample_id and row["joint_index"] == joint_index
    ]


def _build_train_report(
    *,
    samples_jsonl: Path,
    output_dir: Path,
    checkpoint: Path,
    predictions_jsonl: Path,
    offline_eval: dict[str, Any],
    visualization: dict[str, Any] | None,
    sample_count: int,
    input_dim: int,
    output_dim: int,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: tuple[int, ...],
    dropout: float,
    quality_gate: dict[str, Any],
    device: str,
    world_size: int,
    rank: int,
    final_train_mse: float,
    wandb_summary: dict[str, Any],
    model_family: str = "temporal_mlp",
    model_config: dict[str, Any] | None = None,
    loss_config: dict[str, Any] | None = None,
    evaluation_config: dict[str, Any] | None = None,
    feature_contract: dict[str, Any] | None = None,
    distributed_runtime: dict[str, Any] | None = None,
    resume_checkpoint: str = "",
    sample_filter: dict[str, Any] | None = None,
    sample_loader: dict[str, Any] | None = None,
    data_loader: dict[str, Any] | None = None,
    batch_to_device: dict[str, Any] | None = None,
    forward_microbatch: dict[str, Any] | None = None,
    ddp: dict[str, Any] | None = None,
    checkpointing: dict[str, Any] | None = None,
    step_profiler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "samples_jsonl": str(samples_jsonl),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "predictions_jsonl": str(predictions_jsonl),
        "offline_eval": offline_eval,
        "visualization": visualization or {"enabled": False},
        "sample_count": sample_count,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "model_family": model_family,
        "model_config": model_config or {},
        "loss_config": loss_config or {},
        "evaluation_config": evaluation_config or {},
        "feature_contract": feature_contract or {"enabled": False},
        "quality_gate": quality_gate,
        "device": device,
        "world_size": world_size,
        "rank": rank,
        "distributed_runtime": distributed_runtime or {},
        "resume_checkpoint": resume_checkpoint,
        "sample_loader": sample_loader
        or {
            "rank": rank,
            "world_size": world_size,
            "sharded": False,
            "materialized_count": sample_count,
        },
        "data_loader": data_loader or {},
        "batch_to_device": batch_to_device or {},
        "forward_microbatch": forward_microbatch or {"enabled": False},
        "ddp": ddp or {"enabled": False},
        "checkpointing": checkpointing or {"enabled": False},
        "step_profiler": step_profiler or {"enabled": False},
        "sample_filter": sample_filter
        or {
            "input_count": sample_count,
            "filtered_count": sample_count,
            "dropped_count": 0,
            "dropped_examples": [],
        },
        "final_train_mse": final_train_mse,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "wandb": wandb_summary,
    }


def _init_wandb(
    *,
    config: dict[str, Any],
    quality_gate: dict[str, Any],
    output_dir: Path,
    enabled: bool,
):
    mode = str(_nested_get(config, ("tracking", "wandb_mode"), os.environ.get("WANDB_MODE", "disabled")))
    if not enabled or mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    project = str(_nested_get(config, ("experiment", "project"), "OnlineRetarget"))
    name = str(_nested_get(config, ("experiment", "name"), "baseline_mlp_direct_g1"))
    return wandb.init(
        project=project,
        name=name,
        mode=mode,
        dir=str(output_dir),
        config={
            "config": config,
            "quality_gate": quality_gate,
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
        },
    )


def _wandb_log(run, payload: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(payload, step=step)


def _wandb_save(run, path: Path) -> None:
    if run is not None and path.exists():
        run.save(str(path))


def _wandb_log_visualization(run, visualization: dict[str, Any], config: dict[str, Any]) -> None:
    if run is None or not visualization.get("enabled"):
        return
    if not bool(_nested_get(config, ("visualization", "wandb_upload"), False)):
        return
    capsule = visualization.get("capsule_visualization", {})
    if not isinstance(capsule, dict):
        capsule = {}
    for key in (
        "summary_json",
        "trajectory_csv",
        "trajectory_svg",
        "trajectory_html",
        "manifest_json",
        "html",
    ):
        source = capsule if key in {"manifest_json", "html"} else visualization
        path_text = source.get(key)
        if path_text:
            _wandb_save(run, Path(str(path_text)))
    for video_path in _capsule_video_paths(capsule):
        _wandb_save(run, video_path)
    payload = {
        "visualization/status": visualization.get("status", ""),
        "visualization/summary_json": visualization.get("summary_json", ""),
        "visualization/sample_count": visualization.get("sample_count", 0),
        "visualization/trajectory_row_count": visualization.get("trajectory_row_count", 0),
        "visualization/capsule_status": capsule.get("status", ""),
        "visualization/capsule_manifest": capsule.get("manifest_json", ""),
    }
    html_path = Path(str(visualization.get("trajectory_html", "")))
    if html_path.exists():
        try:
            import wandb

            payload["visualization/trajectory_preview"] = wandb.Html(
                html_path.read_text(encoding="utf-8")
            )
        except Exception:
            pass
    capsule_html = Path(str(capsule.get("html", "")))
    if capsule_html.exists():
        try:
            import wandb

            payload["visualization/capsule_preview"] = wandb.Html(
                capsule_html.read_text(encoding="utf-8")
            )
            for index, video_path in enumerate(_capsule_video_paths(capsule)[:4]):
                payload[f"visualization/capsule_video_{index}"] = wandb.Video(str(video_path))
        except Exception:
            pass
    _wandb_log(run, payload)


def _capsule_video_paths(capsule: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for sample in capsule.get("samples", []):
        if not isinstance(sample, dict):
            continue
        for key in ("target_video", "predicted_video"):
            path_text = sample.get(key)
            if path_text:
                path = Path(str(path_text))
                if path.exists():
                    paths.append(path)
    return paths


def _wandb_finish(run) -> None:
    if run is not None:
        run.finish()


def _wandb_summary(run) -> dict[str, Any]:
    if run is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "project": getattr(run, "project", ""),
        "name": getattr(run, "name", ""),
        "id": getattr(run, "id", ""),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(result.strip())
    except Exception:
        return False


if __name__ == "__main__":
    main()
