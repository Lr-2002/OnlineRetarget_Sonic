from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from online_retarget.metrics import compute_metric_bundle, metric_metadata


DEFAULT_PRIMARY_METRIC = "g1_joint_pos_rmse_rad"
DEFAULT_REQUESTED_METRICS = ("mpjpe", "w_mpjpe", "context_compositing")
VISUAL_BODY_POSITION_METRICS = ("mpjpe", "w_mpjpe")
BODY_POSITION_RESULT_ALIASES = {
    "mpjpe": "mpjpe",
    "body_position_mpjpe": "mpjpe",
    "w_mpjpe": "w_mpjpe",
    "weighted_mpjpe": "w_mpjpe",
}


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


def body_position_metric_wandb_scalars(
    body_position_metrics: Mapping[str, Any] | None,
    *,
    prefix: str = "visual_validation",
) -> dict[str, float]:
    if not isinstance(body_position_metrics, Mapping):
        return {}
    metric_results = body_position_metrics.get("metric_results")
    if not isinstance(metric_results, Mapping):
        return {}
    scalars: dict[str, float] = {}
    for name in VISUAL_BODY_POSITION_METRICS:
        raw_result = metric_results.get(name)
        if not isinstance(raw_result, Mapping) or raw_result.get("status") != "available":
            continue
        scalar = _json_metric_value(raw_result.get("value"))
        if scalar is not None:
            scalars[f"{prefix}/{name}"] = scalar
    return scalars


def visual_validation_wandb_payload(
    summary: Mapping[str, Any],
    *,
    summary_path: Path | str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if summary_path is not None:
        payload["visual_validation/summary_path"] = str(summary_path)
    status = summary.get("status")
    if status is not None:
        payload["visual_validation/status"] = str(status)
    body_position_metrics = summary.get("body_position_metrics")
    if isinstance(body_position_metrics, Mapping):
        payload.update(body_position_metric_wandb_scalars(body_position_metrics))
    return payload


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
    body_position_metrics = summary.get("body_position_metrics")
    if isinstance(body_position_metrics, Mapping):
        status["body_position_metrics"] = copy.deepcopy(dict(body_position_metrics))
    reports = summary.get("reports")
    if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes)):
        status["report_count"] = len(reports)
        combined_ok = 0
        accepted_ok = 0
        accepted_failed = 0
        for report in reports:
            if not isinstance(report, Mapping):
                continue
            combined_ok += str(report.get("combined_status", "")).lower() == "ok"
            accepted_status = str(report.get("accepted_vertical_v2_status", "")).lower()
            accepted_ok += accepted_status == "ok"
            accepted_failed += accepted_status == "failed"
        status["context_compositing_ok_count"] = float(combined_ok)
        status["context_compositing_failed_count"] = float(max(0, len(reports) - combined_ok))
        status["accepted_vertical_v2_ok_count"] = float(accepted_ok)
        status["accepted_vertical_v2_failed_count"] = float(accepted_failed)
        if reports:
            status["context_compositing_status"] = "ok" if combined_ok == len(reports) else "failed"
    return status


def _requested_metric_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    metric_cfg = config.get("metric_validation", {})
    names: object = None
    if isinstance(metric_cfg, Mapping):
        names = metric_cfg.get("requested_metrics")
    if names is None:
        eval_cfg = config.get("evaluation_metrics", {})
        if isinstance(eval_cfg, Mapping):
            names = eval_cfg.get("requested_metrics")
    if names is None:
        return DEFAULT_REQUESTED_METRICS
    if isinstance(names, str):
        return (names,)
    if isinstance(names, Sequence):
        return tuple(str(name) for name in names)
    raise ValueError("metric_validation.requested_metrics must be a string or sequence")


def _metric_fields(
    validation_metrics: Mapping[str, Any],
    train_metrics: Mapping[str, Any] | None,
    visual_payload: Mapping[str, Any],
) -> dict[str, Any]:
    fields: dict[str, Any] = {"visual_validation": dict(visual_payload)}
    for metrics in (validation_metrics, train_metrics or {}):
        for key, value in metrics.items():
            key_text = str(key)
            fields[key_text] = value
            if "/" in key_text:
                fields[key_text.rsplit("/", 1)[-1]] = value
    return fields


def _context_compositing_metric_result(visual_payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "name": "context_compositing",
        "unit": "0/1",
        "direction": "higher_is_better",
        "required_fields": ["visual_validation/summary.json reports[].combined_status"],
        "mask_semantics": "not maskable; computed from visual artifact status",
        "reducer": "1 when every requested visual report has combined_status=ok, else 0",
        "description": "Context-compositing status for accepted vertical visual validation artifacts.",
        "source_ref": "LR-238/LR-254 accepted vertical-v2 visual artifact status",
    }
    status = str(visual_payload.get("status", "missing"))
    if status in {"missing", "unreadable"}:
        return {
            "name": "context_compositing",
            "status": "unavailable",
            "value": None,
            "reason": f"visual validation summary is {status}",
            "metadata": metadata,
        }
    report_count = int(visual_payload.get("report_count") or 0)
    if report_count <= 0:
        return {
            "name": "context_compositing",
            "status": "unavailable",
            "value": None,
            "reason": "visual validation summary contains no reports",
            "metadata": metadata,
        }
    ok_count = float(visual_payload.get("context_compositing_ok_count") or 0.0)
    value = 1.0 if ok_count == float(report_count) else 0.0
    reason = "" if value == 1.0 else "one or more visual reports did not compose successfully"
    return {
        "name": "context_compositing",
        "status": "available",
        "value": value,
        "reason": reason,
        "metadata": metadata,
    }


def _precomputed_body_position_metric_result(
    name: str,
    visual_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    canonical_name = BODY_POSITION_RESULT_ALIASES.get(name)
    if canonical_name is None:
        return None
    body_metrics = visual_payload.get("body_position_metrics")
    if not isinstance(body_metrics, Mapping):
        return None

    metric_results = body_metrics.get("metric_results")
    if isinstance(metric_results, Mapping):
        raw_result = metric_results.get(canonical_name)
        if isinstance(raw_result, Mapping):
            result = copy.deepcopy(dict(raw_result))
            result["name"] = name
            _attach_body_position_metric_context(result, body_metrics)
            return result

    metadata = metric_metadata((canonical_name,)).get(canonical_name, {})
    result = {
        "name": name,
        "status": "unavailable",
        "value": None,
        "reason": str(
            body_metrics.get("reason")
            or f"visual validation did not produce numeric {canonical_name}"
        ),
        "metadata": metadata,
    }
    _attach_body_position_metric_context(result, body_metrics)
    return result


def _attach_body_position_metric_context(
    result: dict[str, Any],
    body_metrics: Mapping[str, Any],
) -> None:
    for key in (
        "body_names",
        "body_position_weights",
        "weight_policy",
        "metric_contract",
        "source_artifact_paths",
        "sample_count",
        "weighted_sample_weight",
        "frame_count",
        "body_count",
        "report_count",
    ):
        if key in body_metrics:
            result[key] = copy.deepcopy(body_metrics[key])


def _requested_metric_results(
    *,
    config: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    train_metrics: Mapping[str, Any] | None,
    visual_payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    names = _requested_metric_names(config)
    fields = _metric_fields(validation_metrics, train_metrics, visual_payload)
    results: dict[str, dict[str, Any]] = {}
    registry_names: list[str] = []
    for name in names:
        if name == "context_compositing":
            results[name] = _context_compositing_metric_result(visual_payload)
        elif precomputed := _precomputed_body_position_metric_result(name, visual_payload):
            results[name] = precomputed
        else:
            registry_names.append(name)
    if registry_names:
        bundle = compute_metric_bundle(fields, registry_names)
        results.update({name: result.to_dict() for name, result in bundle.items()})
    return {name: results[name] for name in names}


def _requested_metric_metadata(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    registry_names: list[str] = []
    for name, result in results.items():
        if name == "context_compositing":
            metadata[name] = result.get("metadata", {})
        else:
            registry_names.append(name)
    if registry_names:
        metadata.update(metric_metadata(registry_names))
    return metadata


def build_metric_validation_payload(
    *,
    output_dir: Path,
    step: int,
    config: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    train_metrics: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not validation_metrics:
        raise ValueError("metric validation artifact requires validation_metrics")
    configured_eval_metrics = config.get("evaluation_metrics", {})
    eval_metrics = copy.deepcopy(configured_eval_metrics) if isinstance(configured_eval_metrics, Mapping) else {}
    metric_cfg = config.get("metric_validation", {})
    primary_metric = str(
        metric_cfg.get("primary")
        if isinstance(metric_cfg, Mapping) and metric_cfg.get("primary")
        else eval_metrics.get("primary") or DEFAULT_PRIMARY_METRIC
    )
    primary_metric_key = f"validation/{primary_metric}"
    validation_payload = _json_metric_dict(validation_metrics)
    visual_payload = _visual_validation_artifact_status(output_dir, step)
    body_position_metrics = visual_payload.get("body_position_metrics", {})
    requested_results = _requested_metric_results(
        config=config,
        validation_metrics=validation_metrics,
        train_metrics=train_metrics,
        visual_payload=visual_payload,
    )
    requested_metadata = _requested_metric_metadata(requested_results)
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
        "primary_metric_status": None,
        primary_metric_key: validation_payload.get(primary_metric_key),
        "validation_metrics": validation_payload,
        "train_metrics": _json_metric_dict(train_metrics),
        "requested_metric_names": list(_requested_metric_names(config)),
        "requested_metric_results": requested_results,
        "requested_metric_metadata": requested_metadata,
        "body_position_metrics": (
            copy.deepcopy(dict(body_position_metrics))
            if isinstance(body_position_metrics, Mapping)
            else {}
        ),
        "visual_validation": visual_payload,
        "associated_visual_status": visual_payload.get("status"),
        "associated_visual_path": visual_payload.get("path"),
        "metric_validation": dict(metric_cfg) if isinstance(metric_cfg, Mapping) else {},
        "variant": dict(variant) if isinstance(variant, Mapping) else {},
        "run": run_payload,
    }
    for name, result in requested_results.items():
        key = f"validation/{name}"
        value = result.get("value") if result.get("status") == "available" else None
        payload[key] = value
        payload[f"{key}_status"] = result.get("status")
        if result.get("reason"):
            payload[f"{key}_reason"] = result.get("reason")
        if key == primary_metric_key:
            payload["primary_metric_value"] = value
            payload["primary_metric_status"] = result.get("status")
    return payload


def write_metric_validation_artifact(
    *,
    output_dir: Path,
    step: int,
    config: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    train_metrics: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    payload = build_metric_validation_payload(
        output_dir=output_dir,
        step=step,
        config=config,
        validation_metrics=validation_metrics,
        train_metrics=train_metrics,
        manifest=manifest,
    )
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


def load_metric_validation_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_validation_wandb_payload(
    metric_payload: Mapping[str, Any],
    *,
    artifact_path: Path | str | None = None,
) -> dict[str, Any]:
    wandb_payload: dict[str, Any] = {}
    primary_metric = str(metric_payload.get("primary_metric", DEFAULT_PRIMARY_METRIC))
    primary_metric_key = str(metric_payload.get("primary_metric_key", f"validation/{primary_metric}"))
    primary_metric_value = metric_payload.get("primary_metric_value")
    wandb_payload["metric_validation/primary_metric"] = primary_metric
    wandb_payload["metric_validation/primary_metric_key"] = primary_metric_key
    if primary_metric_value is not None:
        wandb_payload["metric_validation/primary_metric_value"] = primary_metric_value
        wandb_payload[primary_metric_key] = primary_metric_value
        wandb_payload[f"metric_validation/{primary_metric_key}"] = primary_metric_value
    if artifact_path is not None:
        wandb_payload["metric_validation/artifact_path"] = str(artifact_path)

    validation_metrics = metric_payload.get("validation_metrics", {})
    if isinstance(validation_metrics, Mapping):
        for key, value in validation_metrics.items():
            if value is None:
                continue
            key_text = str(key)
            wandb_payload[key_text] = value
            wandb_payload[f"metric_validation/{key_text}"] = value

    requested_results = metric_payload.get("requested_metric_results", {})
    if isinstance(requested_results, Mapping):
        for name, raw_result in requested_results.items():
            if not isinstance(raw_result, Mapping):
                continue
            key_text = str(name)
            status = str(raw_result.get("status", "unknown"))
            value = raw_result.get("value")
            wandb_payload[f"metric_validation/{key_text}_status"] = status
            wandb_payload[f"metric_validation/{key_text}_available"] = (
                1.0 if status == "available" and value is not None else 0.0
            )
            if value is not None:
                scalar = _json_metric_value(value)
                if scalar is not None:
                    wandb_payload[f"validation/{key_text}"] = scalar
                    wandb_payload[f"metric_validation/validation/{key_text}"] = scalar

    train_metrics = metric_payload.get("train_metrics", {})
    if isinstance(train_metrics, Mapping):
        for key, value in train_metrics.items():
            if value is None:
                continue
            key_text = str(key)
            wandb_payload[key_text] = value
            wandb_payload[f"metric_validation/{key_text}"] = value

    visual_payload = metric_payload.get("visual_validation", {})
    if isinstance(visual_payload, Mapping):
        status = visual_payload.get("status")
        path = visual_payload.get("path") or visual_payload.get("summary_path")
        if status is not None:
            wandb_payload["metric_validation/associated_visual_status"] = str(status)
            wandb_payload["metric_validation/visual_validation/status"] = str(status)
        if path is not None:
            wandb_payload["metric_validation/associated_visual_path"] = str(path)
            wandb_payload["metric_validation/visual_validation/path"] = str(path)
            wandb_payload["visual_validation/summary_path"] = str(path)
        for key in ("requested_videos", "videos_ok", "videos_failed", "duration_sec", "elapsed_sec", "report_count"):
            value = visual_payload.get(key)
            if value is not None:
                wandb_payload[f"metric_validation/visual_validation/{key}"] = value

    body_metrics = metric_payload.get("body_position_metrics", {})
    if isinstance(body_metrics, Mapping):
        wandb_payload.update(body_position_metric_wandb_scalars(body_metrics))
        for key in ("sample_count", "weighted_sample_weight", "frame_count", "body_count", "report_count"):
            value = body_metrics.get(key)
            scalar = _json_metric_value(value)
            if scalar is not None:
                wandb_payload[f"metric_validation/body_position_metrics/{key}"] = scalar
        for key in ("status", "reason", "weight_policy"):
            value = body_metrics.get(key)
            if value is not None:
                wandb_payload[f"metric_validation/body_position_metrics/{key}"] = str(value)
        body_names = body_metrics.get("body_names")
        if isinstance(body_names, Sequence) and not isinstance(body_names, (str, bytes)):
            wandb_payload["metric_validation/body_position_metrics/body_names"] = "|".join(
                str(name) for name in body_names
            )
        source_artifact_paths = body_metrics.get("source_artifact_paths")
        if isinstance(source_artifact_paths, Sequence) and not isinstance(source_artifact_paths, (str, bytes)):
            wandb_payload["metric_validation/body_position_metrics/source_artifact_paths"] = "|".join(
                str(path) for path in source_artifact_paths
            )
        contract = body_metrics.get("metric_contract")
        if isinstance(contract, Mapping):
            for key in ("name", "units", "coordinate_frame", "root_alignment", "frame_alignment", "weight_policy"):
                value = contract.get(key)
                if value is not None:
                    wandb_payload[f"metric_validation/body_position_metrics/contract/{key}"] = str(value)
            if "scale_align" in contract:
                wandb_payload["metric_validation/body_position_metrics/contract/scale_align"] = (
                    1.0 if bool(contract["scale_align"]) else 0.0
                )

    return wandb_payload
