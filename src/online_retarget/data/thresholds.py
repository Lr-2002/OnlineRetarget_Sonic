"""Quality-threshold proposal helpers from scan statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ThresholdProposal:
    metric: str
    percentile: float
    value: float
    action: str
    tail: str
    comparison: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def propose_thresholds_from_jsonl(
    stats_jsonl: Path,
    metrics: Sequence[str],
    percentile: float = 0.99,
    action: str = "quarantine",
    group_by: Sequence[str] = (),
    min_group_size: int = 1,
    lower_metrics: Sequence[str] = (),
) -> dict[str, object]:
    """Propose percentile thresholds from scan stats."""

    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be within [0, 1]")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")

    rows = list(_iter_jsonl(stats_jsonl))
    proposals = _proposals_for_rows(rows, metrics, lower_metrics, percentile, action)
    groups = _group_proposals(
        rows,
        metrics,
        lower_metrics,
        percentile,
        action,
        group_by,
        min_group_size,
    )
    grouped_rows = {
        field: sum(group["sample_count"] for group in field_groups)
        for field, field_groups in groups.items()
    }
    return {
        "stats_jsonl": str(stats_jsonl),
        "sample_count": len(rows),
        "percentile": percentile,
        "proposals": [proposal.to_dict() for proposal in proposals],
        "group_by": list(group_by),
        "min_group_size": min_group_size,
        "lower_metrics": list(lower_metrics),
        "grouped_rows": grouped_rows,
        "groups": groups,
    }


def write_threshold_proposals(
    stats_jsonl: Path,
    output_json: Path,
    metrics: Sequence[str],
    percentile: float = 0.99,
    action: str = "quarantine",
    group_by: Sequence[str] = (),
    min_group_size: int = 1,
    lower_metrics: Sequence[str] = (),
) -> dict[str, object]:
    payload = propose_thresholds_from_jsonl(
        stats_jsonl=stats_jsonl,
        metrics=metrics,
        percentile=percentile,
        action=action,
        group_by=group_by,
        min_group_size=min_group_size,
        lower_metrics=lower_metrics,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_accepted_threshold_policy(
    proposal_jsons: Sequence[Path],
    output_json: Path,
    policy_id: str,
    accepted_by: str,
    rationale: str,
    representative: bool = False,
) -> dict[str, object]:
    """Write a named threshold policy artifact from reviewed proposal files."""

    if not proposal_jsons:
        raise ValueError("at least one threshold proposal JSON is required")
    policy_id = policy_id.strip()
    accepted_by = accepted_by.strip()
    rationale = rationale.strip()
    if not policy_id:
        raise ValueError("policy_id is required")
    if not accepted_by:
        raise ValueError("accepted_by is required")
    if not rationale:
        raise ValueError("rationale is required")

    proposal_reports = [_read_json(path) for path in proposal_jsons]
    proposal_summaries = [
        _threshold_report_summary(path, report)
        for path, report in zip(proposal_jsons, proposal_reports)
    ]
    total_samples = sum(int(summary["sample_count"]) for summary in proposal_summaries)
    payload = {
        "policy_id": policy_id,
        "status": "accepted",
        "accepted_by": accepted_by,
        "rationale": rationale,
        "representative": representative,
        "proposal_jsons": [str(path) for path in proposal_jsons],
        "proposal_summaries": proposal_summaries,
        "proposal_count": sum(int(summary["proposal_count"]) for summary in proposal_summaries),
        "total_samples": total_samples,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _proposals_for_rows(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
    lower_metrics: Sequence[str],
    percentile: float,
    action: str,
) -> list[ThresholdProposal]:
    proposals = []
    metric_specs = [(metric, "upper") for metric in metrics]
    metric_specs.extend((metric, "lower") for metric in lower_metrics)
    for metric, tail in metric_specs:
        values = _numeric_values(rows, metric)
        effective_percentile = 1.0 - percentile if tail == "lower" else percentile
        comparison = "<" if tail == "lower" else ">"
        direction = "minimum accepted" if tail == "lower" else "maximum accepted"
        proposals.append(
            ThresholdProposal(
                metric=metric,
                percentile=effective_percentile,
                value=round(_percentile(sorted(values), effective_percentile), 6) if values else 0.0,
                action=action,
                tail=tail,
                comparison=comparison,
                rationale=(
                    f"Set {metric} {direction} threshold to "
                    f"p{int(round(effective_percentile * 100))} of {len(values)} "
                    "scanned samples. Use as a proposal only; inspect category-specific "
                    "distributions before excluding data."
                ),
            )
        )
    return proposals


def _group_proposals(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
    lower_metrics: Sequence[str],
    percentile: float,
    action: str,
    group_by: Sequence[str],
    min_group_size: int,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for field in group_by:
        buckets: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            key = _group_key(row.get(field))
            buckets.setdefault(key, []).append(row)
        field_groups = []
        for key, group_rows in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
            if len(group_rows) < min_group_size:
                continue
            field_groups.append(
                {
                    "field": field,
                    "value": key,
                    "sample_count": len(group_rows),
                    "proposals": [
                        proposal.to_dict()
                        for proposal in _proposals_for_rows(
                            group_rows,
                            metrics,
                            lower_metrics,
                            percentile,
                            action,
                        )
                    ],
                }
            )
        grouped[field] = field_groups
    return grouped


def _iter_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _threshold_report_summary(path: Path, report: Mapping[str, object]) -> dict[str, object]:
    proposals = report.get("proposals", [])
    group_by = report.get("group_by", [])
    grouped_rows = report.get("grouped_rows", {})
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        proposals = []
    if not isinstance(group_by, Sequence) or isinstance(group_by, (str, bytes)):
        group_by = []
    if not isinstance(grouped_rows, Mapping):
        grouped_rows = {}
    metrics = []
    for proposal in proposals:
        if isinstance(proposal, Mapping):
            metric = proposal.get("metric")
            if metric is not None:
                metrics.append(str(metric))
    lower_metrics = report.get("lower_metrics", [])
    if not isinstance(lower_metrics, Sequence) or isinstance(lower_metrics, (str, bytes)):
        lower_metrics = []
    return {
        "path": str(path),
        "stats_jsonl": str(report.get("stats_jsonl", "")),
        "sample_count": _as_int(report.get("sample_count")),
        "proposal_count": len(proposals),
        "metrics": metrics,
        "group_by": [str(field) for field in group_by],
        "grouped_rows": {str(key): _as_int(value) for key, value in grouped_rows.items()},
        "lower_metrics": [str(metric) for metric in lower_metrics],
    }


def _numeric_values(rows: Sequence[Mapping[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _group_key(value: object) -> str:
    if value in (None, ""):
        return "unknown"
    return str(value)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


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
