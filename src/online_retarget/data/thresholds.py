"""Quality-threshold proposal helpers from scan statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ThresholdProposal:
    metric: str
    percentile: float
    value: float
    action: str
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
) -> dict[str, object]:
    """Propose percentile thresholds from scan stats."""

    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be within [0, 1]")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")

    rows = list(_iter_jsonl(stats_jsonl))
    proposals = _proposals_for_rows(rows, metrics, percentile, action)
    groups = _group_proposals(rows, metrics, percentile, action, group_by, min_group_size)
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
) -> dict[str, object]:
    payload = propose_thresholds_from_jsonl(
        stats_jsonl=stats_jsonl,
        metrics=metrics,
        percentile=percentile,
        action=action,
        group_by=group_by,
        min_group_size=min_group_size,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _proposals_for_rows(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
    percentile: float,
    action: str,
) -> list[ThresholdProposal]:
    proposals = []
    for metric in metrics:
        values = _numeric_values(rows, metric)
        proposals.append(
            ThresholdProposal(
                metric=metric,
                percentile=percentile,
                value=round(_percentile(sorted(values), percentile), 6) if values else 0.0,
                action=action,
                rationale=(
                    f"Set {metric} threshold to p{int(percentile * 100)} of "
                    f"{len(values)} scanned samples. Use as a proposal only; inspect "
                    "category-specific distributions before excluding data."
                ),
            )
        )
    return proposals


def _group_proposals(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
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
                        for proposal in _proposals_for_rows(group_rows, metrics, percentile, action)
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


def _numeric_values(rows: Sequence[Mapping[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


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
