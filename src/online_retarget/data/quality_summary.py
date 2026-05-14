"""Summarize quality scanner JSONL artifacts for progress checkpoints."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


DEFAULT_SUMMARY_METRICS = (
    "contact_frame_ratio",
    "contact_slide_rate",
    "joint_limit_violation_rate",
    "max_abs_joint_velocity",
    "max_start_end_root_speed",
    "mean_foot_clearance",
    "penetration_depth",
    "self_collision_proxy_rate",
    "min_self_collision_distance",
)

DEFAULT_GROUP_BY = ("split", "category")

DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class QualitySummaryResult:
    input_jsonl: Path
    output_json: Path
    row_count: int
    action_counts: dict[str, int]
    flag_counts: dict[str, int]
    group_counts: dict[str, dict[str, int]]
    metric_summary: dict[str, dict[str, float | int | None]]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_jsonl"] = str(self.input_jsonl)
        payload["output_json"] = str(self.output_json)
        return payload


def summarize_quality_jsonl(
    *,
    stats_jsonl: Path,
    output_json: Path,
    metrics: Sequence[str] = DEFAULT_SUMMARY_METRICS,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> QualitySummaryResult:
    """Write an exact summary over a quality stats JSONL file."""

    stats_jsonl = stats_jsonl.expanduser()
    output_json = output_json.expanduser()
    metrics = tuple(metrics) or DEFAULT_SUMMARY_METRICS
    group_by = tuple(group_by) or DEFAULT_GROUP_BY
    quantiles = tuple(quantiles) or DEFAULT_QUANTILES
    _validate_quantiles(quantiles)
    action_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    group_counts: dict[str, Counter[str]] = {field: Counter() for field in group_by}
    metric_values: dict[str, list[float]] = {metric: [] for metric in metrics}
    row_count = 0

    with stats_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {stats_jsonl}") from exc
            row_count += 1
            action_counts[str(row.get("quality_action", ""))] += 1
            for flag in _split_flags(str(row.get("quality_flags", ""))):
                flag_counts[flag] += 1
            for field, counts in group_counts.items():
                counts[str(row.get(field, ""))] += 1
            for metric in metrics:
                value = _maybe_float(row.get(metric))
                if value is not None:
                    metric_values[metric].append(value)

    stat = stats_jsonl.stat()
    payload = {
        "input_jsonl": str(stats_jsonl),
        "input_size_bytes": stat.st_size,
        "input_mtime_epoch": stat.st_mtime,
        "row_count": row_count,
        "action_counts": dict(sorted(action_counts.items())),
        "flag_counts": dict(flag_counts.most_common()),
        "group_counts": {
            field: dict(counts.most_common()) for field, counts in group_counts.items()
        },
        "metrics": list(metrics),
        "quantiles": list(quantiles),
        "metric_summary": {
            metric: _summarize_values(values, quantiles)
            for metric, values in metric_values.items()
        },
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return QualitySummaryResult(
        input_jsonl=stats_jsonl,
        output_json=output_json,
        row_count=row_count,
        action_counts=dict(sorted(action_counts.items())),
        flag_counts=dict(flag_counts.most_common()),
        group_counts={field: dict(counts.most_common()) for field, counts in group_counts.items()},
        metric_summary=payload["metric_summary"],  # type: ignore[arg-type]
        git_sha=str(payload["git_sha"]),
        git_dirty=bool(payload["git_dirty"]),
    )


def _summarize_values(values: Sequence[float], quantiles: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        summary: dict[str, float | int | None] = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
        for quantile in quantiles:
            summary[_quantile_key(quantile)] = None
        return summary
    ordered = sorted(values)
    total = sum(ordered)
    summary = {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": total / len(ordered),
    }
    for quantile in quantiles:
        summary[_quantile_key(quantile)] = ordered[
            min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
        ]
    return summary


def _quantile_key(quantile: float) -> str:
    return f"p{int(round(quantile * 100))}"


def _validate_quantiles(quantiles: Sequence[float]) -> None:
    if not quantiles:
        raise ValueError("at least one quantile is required")
    for quantile in quantiles:
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")


def _split_flags(flags: str) -> list[str]:
    return [flag for flag in flags.split("|") if flag]


def _maybe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
