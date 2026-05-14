"""Offline evaluation report generation for retargeting outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Sequence

from .metrics import action_similarity, contact_artifact_metrics, joint_jump_rate, joint_rmse, mpjpe


@dataclass(frozen=True)
class EvaluationConfig:
    fps: float = 30.0
    joint_jump_velocity: float = 20.0
    ground_height: float = 0.0
    up_axis: int | str = 2
    contact_height_threshold: float = 0.04
    max_contact_slide_speed: float = 0.25
    failure_metric: str = "joint_rmse"
    max_failures: int = 50
    run_name: str = "offline_eval"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    summary_json: Path
    per_sample_csv: Path
    failure_manifest_csv: Path
    sample_count: int
    overall: dict[str, float]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["summary_json"] = str(self.summary_json)
        payload["per_sample_csv"] = str(self.per_sample_csv)
        payload["failure_manifest_csv"] = str(self.failure_manifest_csv)
        return payload


def evaluate_jsonl(
    input_jsonl: Path,
    output_root: Path,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Evaluate a JSONL file of predicted/target motion pairs."""

    config = config or EvaluationConfig()
    output_dir = output_root.expanduser() / "eval" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "eval_summary.json"
    per_sample_csv = output_dir / "per_sample_metrics.csv"
    failure_manifest_csv = output_dir / "failure_manifest.csv"

    samples = list(_iter_jsonl(input_jsonl))
    per_sample = [_sample_metrics(sample, config) for sample in samples]
    _write_csv(per_sample_csv, per_sample)
    failures = sorted(
        per_sample,
        key=lambda row: float(row.get(config.failure_metric, 0.0)),
        reverse=True,
    )[: config.max_failures]
    _write_csv(failure_manifest_csv, failures)

    overall = _aggregate_metrics(per_sample)
    summary = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "config": config.to_dict(),
        "sample_count": len(per_sample),
        "overall": overall,
        "by_actor": _grouped_aggregate(per_sample, "actor_uid"),
        "by_category": _grouped_aggregate(per_sample, "category"),
        "by_package": _grouped_aggregate(per_sample, "package"),
        "by_quality_flag": _aggregate_by_quality_flag(per_sample),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(summary_json, summary)

    return EvaluationResult(
        output_dir=output_dir,
        summary_json=summary_json,
        per_sample_csv=per_sample_csv,
        failure_manifest_csv=failure_manifest_csv,
        sample_count=len(per_sample),
        overall=overall,
        git_sha=summary["git_sha"],
        git_dirty=summary["git_dirty"],
    )


def _sample_metrics(sample: Mapping[str, object], config: EvaluationConfig) -> dict[str, object]:
    predicted_joints = _required(sample, "predicted_joints")
    target_joints = _required(sample, "target_joints")
    row = {
        "sample_id": str(sample.get("sample_id", "")),
        "actor_uid": str(sample.get("actor_uid", "")),
        "category": str(sample.get("category", "")),
        "package": str(sample.get("package", "")),
        "quality_flags": _quality_flags_string(sample.get("quality_flags", "")),
        "joint_rmse": joint_rmse(predicted_joints, target_joints),
        "action_similarity": action_similarity(predicted_joints, target_joints),
        "predicted_joint_jump_rate": joint_jump_rate(
            predicted_joints,
            fps=float(sample.get("fps", config.fps)),
            max_velocity=config.joint_jump_velocity,
        ),
    }
    if sample.get("predicted_body_pos") is not None and sample.get("target_body_pos") is not None:
        predicted_body_pos = sample["predicted_body_pos"]
        target_body_pos = sample["target_body_pos"]
        row["mpjpe"] = mpjpe(predicted_body_pos, target_body_pos)
        foot_indices = _foot_indices(sample)
        if foot_indices:
            row.update(
                {
                    f"predicted_{metric}": value
                    for metric, value in contact_artifact_metrics(
                        predicted_body_pos,
                        fps=float(sample.get("fps", config.fps)),
                        foot_indices=foot_indices,
                        ground_height=config.ground_height,
                        up_axis=config.up_axis,
                        contact_height_threshold=config.contact_height_threshold,
                        max_contact_slide_speed=config.max_contact_slide_speed,
                        contact_reference=target_body_pos,
                    ).items()
                }
            )
    return row


def _foot_indices(sample: Mapping[str, object]) -> tuple[int, ...]:
    explicit = sample.get("foot_indices")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("foot_indices must be a list when provided")
        return tuple(int(index) for index in explicit)
    body_names = sample.get("body_names")
    foot_names = sample.get("foot_body_names", sample.get("foot_names"))
    if body_names is None or foot_names is None:
        return ()
    if not isinstance(body_names, list) or not isinstance(foot_names, list):
        raise ValueError("body_names and foot_body_names must be lists when provided")
    name_to_index = {str(name): index for index, name in enumerate(body_names)}
    indices: list[int] = []
    for name in foot_names:
        key = str(name)
        if key in name_to_index:
            indices.append(name_to_index[key])
    return tuple(indices)


def _aggregate_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    metric_names = _metric_names(rows)
    return {metric: _mean_float(row.get(metric) for row in rows) for metric in metric_names}


def _grouped_aggregate(rows: Sequence[Mapping[str, object]], group_key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, ""))].append(row)
    return {group: _aggregate_metrics(group_rows) for group, group_rows in sorted(groups.items())}


def _aggregate_by_quality_flag(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        flags = str(row.get("quality_flags", ""))
        split_flags = [flag for flag in flags.split("|") if flag] or ["none"]
        for flag in split_flags:
            groups[flag].append(row)
    return {flag: _aggregate_metrics(group_rows) for flag, group_rows in sorted(groups.items())}


def _metric_names(rows: Sequence[Mapping[str, object]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                names.add(key)
    return sorted(names)


def _mean_float(values: Iterable[object]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)


def _required(sample: Mapping[str, object], key: str):
    if key not in sample:
        raise ValueError(f"evaluation sample missing required key: {key}")
    return sample[key]


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            yield payload


def _quality_flags_string(value: object) -> str:
    if isinstance(value, list):
        return "|".join(str(flag) for flag in value if flag)
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


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
