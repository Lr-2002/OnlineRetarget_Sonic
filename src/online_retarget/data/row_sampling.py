"""Shared row selection helpers for bounded quality scans."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def select_rows_for_scan(
    rows: Sequence[Mapping[str, str]],
    limit: int | None,
    sample_by: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Select rows for a bounded scan.

    The default is intentionally the historical first-N behavior. When
    ``sample_by`` is set, rows are bucketed by the field tuple and selected in a
    deterministic round-robin order across buckets.
    """

    materialized = [dict(row) for row in rows]
    if limit is None or limit >= len(materialized):
        return materialized
    if limit <= 0:
        return []
    if not sample_by:
        return materialized[:limit]

    buckets: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in materialized:
        key = tuple(_bucket(row.get(field, "")) for field in sample_by)
        buckets.setdefault(key, []).append(row)

    selected: list[dict[str, str]] = []
    offsets = {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            offset = offsets[key]
            bucket = buckets[key]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[key] = offset + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def scan_sampling_report(
    candidate_rows: Sequence[Mapping[str, str]],
    selected_rows: Sequence[Mapping[str, str]],
    limit: int | None,
    sample_by: Sequence[str] = (),
) -> dict[str, object]:
    """Build a compact provenance report for scan row selection."""

    return {
        "mode": "stratified_round_robin" if sample_by and limit is not None else "first_n",
        "limit": limit,
        "sample_by": list(sample_by),
        "candidate_rows": len(candidate_rows),
        "selected_rows": len(selected_rows),
        "candidate_group_counts": _group_counts(candidate_rows, sample_by),
        "selected_group_counts": _group_counts(selected_rows, sample_by),
    }


def sampling_run_tag(limit: int | None, sample_by: Sequence[str] = ()) -> str:
    """Stable run-name tag for quality scan output directories."""

    limit_tag = "full" if limit is None else f"limit{limit}"
    if not sample_by or limit is None:
        return limit_tag
    fields = "-".join(_sanitize_field(field) for field in sample_by)
    return f"{limit_tag}_by-{fields}"


def _group_counts(
    rows: Sequence[Mapping[str, str]],
    sample_by: Sequence[str],
) -> dict[str, int]:
    if not sample_by:
        return {}
    counts = Counter(_group_label(row, sample_by) for row in rows)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _group_label(row: Mapping[str, str], sample_by: Sequence[str]) -> str:
    return "|".join(f"{field}={_bucket(row.get(field, ''))}" for field in sample_by)


def _bucket(value: str | None) -> str:
    if value in (None, ""):
        return "unknown"
    return str(value)


def _sanitize_field(field: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in field) or "field"
