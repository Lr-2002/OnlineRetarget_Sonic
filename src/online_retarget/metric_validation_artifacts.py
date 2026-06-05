from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PRIMARY_METRIC = "g1_joint_pos_rmse_rad"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def metric_validation_due(config: Mapping[str, Any], step: int) -> bool:
    cfg = config.get("metric_validation", {})
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        return False
    every_steps = int(cfg.get("every_steps", 0))
    return every_steps > 0 and step > 0 and step % every_steps == 0


def metric_validation_output_dir(output_dir: Path, config: Mapping[str, Any]) -> Path:
    cfg = config.get("metric_validation", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    configured = Path(str(cfg.get("output_dir", "metrics")))
    if configured.is_absolute():
        return configured
    return output_dir / configured


def _json_metric_value(value: Any) -> float | None:
    try:
        if hasattr(value, "reshape"):
            value = value.reshape(-1)[0]
        scalar = float(value)
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _json_metric_dict(metrics: Mapping[str, Any] | None) -> dict[str, float | None]:
    if metrics is None:
        return {}
    return {str(key): _json_metric_value(value) for key, value in sorted(metrics.items())}


def _visual_validation_artifact_status(output_dir: Path, step: int) -> dict[str, Any]:
    summary_path = output_dir / "visual_validation" / f"step_{step:08d}" / "summary.json"
    status: dict[str, Any] = {
        "path": str(summary_path),
        "summary_path": str(summary_path),
        "exists": summary_path.exists(),
    }
    if not summary_path.exists():
        status["status"] = "missing"
        return status
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        status["status"] = "unreadable"
        status["message"] = str(exc)
        return status
    status["status"] = str(summary.get("status", "unknown"))
    for key in ("requested_videos", "videos_ok", "videos_failed", "duration_sec", "elapsed_sec"):
        if key in summary:
            status[key] = _json_metric_value(summary[key])
    reports = summary.get("reports")
    if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes)):
        status["report_count"] = len(reports)
    return status


def write_metric_validation_artifact(
    *,
    output_dir: Path,
    step: int,
    config: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    train_metrics: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    if not validation_metrics:
        raise ValueError("metric validation artifact requires validation_metrics")
    configured_eval_metrics = config.get("evaluation_metrics", {})
    eval_metrics = copy.deepcopy(configured_eval_metrics) if isinstance(configured_eval_metrics, Mapping) else {}
    primary_metric = str(eval_metrics.get("primary") or DEFAULT_PRIMARY_METRIC)
    primary_metric_key = f"validation/{primary_metric}"
    validation_payload = _json_metric_dict(validation_metrics)
    visual_payload = _visual_validation_artifact_status(output_dir, step)
    metric_cfg = config.get("metric_validation", {})
    variant = config.get("variant", {})
    run_payload: dict[str, Any] = {}
    if manifest is not None:
        for key in (
            "run_id",
            "run_group",
            "config_path",
            "control_revision_actual",
            "source_revision_actual",
        ):
            if key in manifest:
                run_payload[key] = str(manifest[key])

    payload: dict[str, Any] = {
        "artifact_type": "metric_validation",
        "created_at": utc_now(),
        "step": int(step),
        "metric_family": str(eval_metrics.get("metric_family", "")),
        "metric_contract": eval_metrics,
        "primary_metric": primary_metric,
        "primary_metric_key": primary_metric_key,
        "primary_metric_value": validation_payload.get(primary_metric_key),
        primary_metric_key: validation_payload.get(primary_metric_key),
        "validation_metrics": validation_payload,
        "train_metrics": _json_metric_dict(train_metrics),
        "visual_validation": visual_payload,
        "associated_visual_status": visual_payload.get("status"),
        "associated_visual_path": visual_payload.get("path"),
        "metric_validation": dict(metric_cfg) if isinstance(metric_cfg, Mapping) else {},
        "variant": dict(variant) if isinstance(variant, Mapping) else {},
        "run": run_payload,
    }

    metric_dir = metric_validation_output_dir(output_dir, config)
    metric_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = metric_dir / f"step_{step:08d}.json"
    tmp_path = artifact_path.with_name(f".{artifact_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(artifact_path)
    return artifact_path
