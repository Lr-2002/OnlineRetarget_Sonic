"""Training-time metric and visualization logging contracts."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .humanoid_eval_registry import (
    MEASURED,
    MISSING,
    NOT_APPLICABLE,
    EvaluationResult,
    make_result,
    missing_result,
    not_applicable_result,
    registry_manifest,
    result_dicts,
    status_counts,
)


REMOTE_LOGGING_SCHEMA = "online_retarget.remote_training_logging.v1"
A0_METHOD_ID = "online_retarget_a0"
A0_PROTOCOL_ID = "sonic_kin_soma_motionlib_training"
DEFAULT_WANDB_METRIC_NAMESPACE = "metric_registry"
DEFAULT_VISUAL_ARTIFACT_PREFIX = "lr177-a0-visual-validation"


def apply_remote_logging_visual_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply remote_logging visual controls to the runtime visual_validation config."""
    updated = copy.deepcopy(dict(config))
    remote_cfg = _mapping(updated.get("remote_logging", updated.get("training_time_logging", {})))
    if not remote_cfg:
        return updated

    visual_cfg = dict(_mapping(updated.get("visual_validation", {})))
    changed = False
    overrides = (
        ("visual_upload", "wandb_upload", bool),
        ("visual_every_n_steps", "every_steps", int),
        ("visual_every_minutes", "every_minutes", _optional_float),
        ("num_visual_samples", "num_videos", int),
        ("max_video_sec", "duration_sec", _optional_float),
    )
    for remote_key, visual_key, caster in overrides:
        if remote_key not in remote_cfg:
            continue
        visual_cfg[visual_key] = caster(remote_cfg[remote_key])
        changed = True
    if changed:
        updated["visual_validation"] = visual_cfg
    return updated


def remote_logging_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    remote_cfg = _mapping(config.get("remote_logging", config.get("training_time_logging", {})))
    visual_cfg = _mapping(config.get("visual_validation", {}))
    wandb_cfg = _mapping(config.get("wandb", {}))

    enabled = bool(remote_cfg.get("enabled", True)) and not bool(remote_cfg.get("disabled", False))
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    wandb_mode = str(
        remote_cfg.get("wandb_mode") or wandb_cfg.get("mode") or os.environ.get("WANDB_MODE", "")
    )
    if not wandb_mode:
        wandb_mode = "online" if wandb_enabled else "disabled"
    wandb_effective = wandb_enabled and wandb_mode != "disabled"
    visual_enabled = bool(visual_cfg.get("enabled", False))
    visual_upload = bool(remote_cfg.get("visual_upload", visual_cfg.get("wandb_upload", True)))

    return {
        "enabled": enabled,
        "probe": bool(remote_cfg.get("probe", False)),
        "log_scalars": bool(remote_cfg.get("log_scalars", remote_cfg.get("scalar_metrics", True))),
        "scalar_eval_only": bool(
            remote_cfg.get("scalar_eval_only", remote_cfg.get("eval_only", False))
        ),
        "wandb_enabled": wandb_enabled,
        "wandb_mode": wandb_mode,
        "wandb_effective": wandb_effective,
        "wandb_metric_namespace": str(
            remote_cfg.get("wandb_metric_namespace", DEFAULT_WANDB_METRIC_NAMESPACE)
        ),
        "visual_enabled": visual_enabled,
        "visual_wandb_upload_enabled": (
            enabled and wandb_effective and visual_enabled and visual_upload
        ),
        "visual_every_n_steps": int(
            remote_cfg.get("visual_every_n_steps", visual_cfg.get("every_steps", 0) or 0)
        ),
        "visual_every_minutes": _optional_float(
            remote_cfg.get("visual_every_minutes", visual_cfg.get("every_minutes"))
        ),
        "num_visual_samples": int(
            remote_cfg.get("num_visual_samples", visual_cfg.get("num_videos", 0) or 0)
        ),
        "max_video_sec": _optional_float(
            remote_cfg.get("max_video_sec", visual_cfg.get("duration_sec"))
        ),
        "artifact_prefix": str(
            remote_cfg.get("wandb_artifact_prefix", DEFAULT_VISUAL_ARTIFACT_PREFIX)
        ),
        "write_probe_summary": bool(remote_cfg.get("write_probe_summary", True)),
    }


def build_remote_logging_contract(
    config: Mapping[str, Any],
    *,
    output_dir: Path | str | None = None,
    run_group: str | None = None,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    settings = remote_logging_settings(config)
    visual_seconds = (
        settings["visual_every_minutes"] * 60.0
        if settings["visual_every_minutes"] is not None
        else None
    )
    return {
        "schema_version": REMOTE_LOGGING_SCHEMA,
        "enabled": bool(settings["enabled"]),
        "probe": bool(settings["probe"]),
        "run_group": run_group or "",
        "output_dir": str(output_dir) if output_dir is not None else "",
        "config_path": str(config_path) if config_path is not None else "",
        "non_invasive": {
            "changes_training_loss": False,
            "changes_training_objective": False,
            "changes_model_selection": False,
            "changes_ddp_collectives": False,
            "rank0_wandb_only": True,
        },
        "wandb": {
            "enabled_in_config": bool(settings["wandb_enabled"]),
            "mode": settings["wandb_mode"],
            "effective": bool(settings["wandb_effective"]),
            "metric_namespace": settings["wandb_metric_namespace"],
        },
        "scalars": {
            "enabled": bool(settings["enabled"] and settings["log_scalars"]),
            "scalar_eval_only": bool(settings["scalar_eval_only"]),
            "metric_registry_schema": registry_manifest()["schema_version"],
            "stable_metric_keys": [
                "train/g1_joint_pos_rmse_rad",
                "validation/g1_joint_pos_rmse_rad",
                "metric_registry/train/g1_joint_pos_rmse_rad",
                "metric_registry/validation/g1_joint_pos_rmse_rad",
            ],
            "body_position_mpjpe_source": "body_position_mpjpe_supplemental.json",
            "body_position_mpjpe_default_status": MISSING,
            "policy_success_default_status": NOT_APPLICABLE,
            "registry": registry_manifest(),
        },
        "visuals": {
            "enabled_in_config": bool(settings["visual_enabled"]),
            "wandb_upload_enabled": bool(settings["visual_wandb_upload_enabled"]),
            "controls_apply_to": {
                "visual_upload": "visual_validation.wandb_upload",
                "visual_every_n_steps": "visual_validation.every_steps",
                "visual_every_minutes": "visual_validation.every_minutes",
                "num_visual_samples": "visual_validation.num_videos",
                "max_video_sec": "visual_validation.duration_sec",
            },
            "visual_every_n_steps": int(settings["visual_every_n_steps"]),
            "visual_every_seconds": visual_seconds,
            "num_visual_samples": int(settings["num_visual_samples"]),
            "max_video_sec": settings["max_video_sec"],
            "artifact_prefix": settings["artifact_prefix"],
            "artifact_type": "online_retarget_visual_validation_video",
            "expected_artifact_template": (
                f"{settings['artifact_prefix']}-"
                "{variant}-step-{step:08d}-sample-{sample_index:02d}"
            ),
            "fallback_label": "fallback_not_final_somamesh",
        },
    }


def a0_metric_registry_results(
    metrics: Mapping[str, Any],
    *,
    source: str,
    step: int | None = None,
    sequence_id: str | None = None,
    method_id: str = A0_METHOD_ID,
    protocol_id: str = A0_PROTOCOL_ID,
) -> list[EvaluationResult]:
    sequence = sequence_id or _sequence_id(source, step)
    normalized = _normalize_metric_values(metrics)
    results: list[EvaluationResult] = []

    g1_value = _first_number(
        normalized,
        "g1_joint_pos_rmse_rad",
        "joint_pos_rmse_raw",
        f"{source}/g1_joint_pos_rmse_rad",
        f"{source}/joint_pos_rmse_raw",
    )
    if g1_value is None:
        results.append(
            missing_result(
                method_id=method_id,
                protocol_id=protocol_id,
                sequence_id=sequence,
                metric_id="g1_joint_pos_rmse_rad",
                source=source,
                notes=(
                    "Expected from A0 joint-angle validation metrics but absent from this payload."
                ),
            )
        )
    else:
        results.append(
            make_result(
                method_id=method_id,
                protocol_id=protocol_id,
                sequence_id=sequence,
                metric_id="g1_joint_pos_rmse_rad",
                value=g1_value,
                source=source,
                notes="G1 joint-angle RMSE in radians; not MPJPE.",
            )
        )

    body_mpjpe = _first_number(
        normalized,
        "body_position_mpjpe",
        "supplemental/body_position_mpjpe",
        f"{source}/body_position_mpjpe",
    )
    if body_mpjpe is None:
        results.append(
            missing_result(
                method_id=method_id,
                protocol_id=protocol_id,
                sequence_id=sequence,
                metric_id="body_position_mpjpe",
                source="supplemental_evaluator",
                notes=(
                    "Requires body_position_mpjpe_supplemental.json; not computed from "
                    "A0 joint-angle targets."
                ),
            )
        )
    else:
        results.append(
            make_result(
                method_id=method_id,
                protocol_id=protocol_id,
                sequence_id=sequence,
                metric_id="body_position_mpjpe",
                value=body_mpjpe,
                source="supplemental_evaluator",
                notes="Body-position MPJPE from supplemental FK/body-position evaluator.",
            )
        )

    results.append(
        not_applicable_result(
            method_id=method_id,
            protocol_id=protocol_id,
            sequence_id=sequence,
            metric_id="policy_success",
            source=source,
            notes="Kin-only supervised A0 training has no simulator policy rollout.",
        )
    )
    return results


def wandb_registry_payload(
    results: Sequence[EvaluationResult],
    *,
    namespace: str = DEFAULT_WANDB_METRIC_NAMESPACE,
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for result in results:
        if (
            result.status != MEASURED
            or result.value is None
            or not math.isfinite(float(result.value))
        ):
            continue
        source = _slug(result.source.replace("/", "_"))
        payload[f"{namespace}/{source}/{result.metric_id}"] = float(result.value)
    for status, count in status_counts(list(results)).items():
        payload[f"{namespace}/status_count/{status}"] = float(count)
    return payload


def build_remote_logging_probe_payload(
    config: Mapping[str, Any],
    metric_results: Sequence[EvaluationResult],
) -> dict[str, float]:
    settings = remote_logging_settings(config)
    return {
        "remote_logging/probe": 1.0,
        **wandb_registry_payload(
            metric_results,
            namespace=settings["wandb_metric_namespace"],
        ),
    }


def remote_logging_summary(
    config: Mapping[str, Any],
    *,
    output_dir: Path | str | None = None,
    run_group: str | None = None,
    config_path: Path | str | None = None,
    metric_results: Sequence[EvaluationResult] = (),
) -> dict[str, Any]:
    contract = build_remote_logging_contract(
        config,
        output_dir=output_dir,
        run_group=run_group,
        config_path=config_path,
    )
    contract["metric_registry_results"] = result_dicts(list(metric_results))
    contract["metric_registry_status_counts"] = status_counts(list(metric_results))
    contract["wandb_registry_payload_keys"] = sorted(wandb_registry_payload(metric_results))
    return contract


def visual_wandb_artifact_specs(
    reports: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    step: int,
    checkpoint_path: Path | str | None,
) -> list[dict[str, Any]]:
    settings = remote_logging_settings(config)
    if not settings["enabled"] or not settings["visual_enabled"]:
        return []
    variant = _variant_name(config)
    run_id = str((manifest or {}).get("run_id", ""))
    run_group = str((manifest or {}).get("run_group", ""))
    specs: list[dict[str, Any]] = []
    for report in reports:
        if report.get("combined_status") != "ok":
            continue
        video_text = str(report.get("combined_video", "")).strip()
        if not video_text:
            continue
        video_path = Path(video_text)
        index = int(report.get("index", len(specs)))
        sequence_id = _sequence_from_visual_report(report, index)
        artifact_name = _artifact_name(
            settings["artifact_prefix"],
            variant,
            step,
            index,
            sequence_id,
        )
        acceptance = bool(report.get("active_backend_is_acceptance_backend", False))
        metadata = {
            "schema_version": REMOTE_LOGGING_SCHEMA,
            "run_id": run_id,
            "run_group": run_group,
            "variant": variant,
            "step": int(step),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
            "sequence_id": sequence_id,
            "filename": str(report.get("filename", "")),
            "relative_path": str(report.get("relative_path", "")),
            "fps": _safe_float(report.get("fps")),
            "frames": int(report.get("frames", 0) or 0),
            "metadata_path": str(report.get("metadata", "")),
            "control_revision_actual": str((manifest or {}).get("control_revision_actual", "")),
            "source_revision_actual": str((manifest or {}).get("source_revision_actual", "")),
            "render_acceptance_state": (
                "acceptance" if acceptance else "fallback_not_final_somamesh"
            ),
            "soma_mesh_final_render": bool(acceptance),
            "wandb_upload_enabled": bool(settings["visual_wandb_upload_enabled"]),
        }
        specs.append(
            {
                "artifact_name": artifact_name,
                "artifact_type": "online_retarget_visual_validation_video",
                "wandb_key": f"visual_validation/{index:02d}_{_safe_metric_name(sequence_id)}",
                "video_path": str(video_path),
                "metadata_path": str(report.get("metadata", "")),
                "metadata": metadata,
                "upload_enabled": bool(settings["visual_wandb_upload_enabled"]),
            }
        )
    return specs


def log_visual_artifact_specs(
    wandb_run: Any,
    specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if wandb_run is None:
        return []
    import wandb

    logged: list[dict[str, str]] = []
    for spec in specs:
        if not spec.get("upload_enabled", False):
            continue
        artifact = wandb.Artifact(
            name=str(spec["artifact_name"]),
            type=str(spec["artifact_type"]),
            metadata=dict(spec.get("metadata", {})),
        )
        video_path = Path(str(spec["video_path"]))
        artifact.add_file(str(video_path), name=video_path.name)
        metadata_text = str(spec.get("metadata_path", ""))
        metadata_path = Path(metadata_text) if metadata_text else None
        if metadata_path is not None and metadata_path.is_file():
            artifact.add_file(str(metadata_path), name=f"{video_path.stem}_metadata.json")
        wandb_run.log_artifact(artifact)
        logged.append({"artifact_name": str(spec["artifact_name"]), "video_path": str(video_path)})
    return logged


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _normalize_metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        key_text = str(key)
        normalized[key_text] = numeric
        normalized[key_text.split("/")[-1]] = numeric
    return normalized


def _first_number(values: Mapping[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in values:
            return float(values[key])
    return None


def _sequence_id(source: str, step: int | None) -> str:
    if step is None:
        return source
    return f"{source}:step_{int(step)}"


def _variant_name(config: Mapping[str, Any]) -> str:
    variant = config.get("variant", {})
    if isinstance(variant, Mapping):
        return str(variant.get("name", "variant"))
    return "variant"


def _sequence_from_visual_report(report: Mapping[str, Any], index: int) -> str:
    for key in ("relative_path", "filename"):
        value = str(report.get(key, "")).strip()
        if value:
            return value
    return f"sample_{index:02d}"


def _artifact_name(prefix: str, variant: str, step: int, index: int, sequence_id: str) -> str:
    raw = f"{prefix}-{variant}-step-{int(step):08d}-sample-{index:02d}-{sequence_id}"
    return _slug(raw)[:128].strip("-_.") or "online-retarget-visual"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-_.")


def _safe_metric_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return safe[:64] or "sample"


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric
