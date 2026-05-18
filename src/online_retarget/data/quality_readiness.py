"""Readiness checks for multi-lane motion-quality scan artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


DEFAULT_REQUIRED_LANES = ("source", "source_fk", "g1", "pair")


@dataclass(frozen=True)
class QualityLaneInput:
    """Paths for one quality-scan lane."""

    name: str
    stats_jsonl: Path | None = None
    report_json: Path | None = None
    required: bool = True


@dataclass(frozen=True)
class QualityLaneStatus:
    """Readiness status for one quality-scan lane."""

    name: str
    required: bool
    ready: bool
    status: str
    row_count: int
    expected_rows: int
    coverage_ratio: float
    stats_jsonl: str
    report_json: str
    stats_exists: bool
    report_exists: bool
    report_scanned_rows: int | None = None
    report_limit: int | None = None
    report_git_sha: str = ""
    report_git_dirty: bool | None = None
    action_counts: dict[str, int] = field(default_factory=dict)
    flag_counts: dict[str, int] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReadinessResult:
    """Aggregate readiness status for required quality-scan lanes."""

    ready: bool
    status: str
    expected_rows: int
    index_csv: Path
    output_json: Path
    lanes: list[QualityLaneStatus]
    blockers: list[str]
    warnings: list[str]
    next_actions: list[str]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["index_csv"] = str(self.index_csv)
        payload["output_json"] = str(self.output_json)
        payload["lanes"] = [lane.to_dict() for lane in self.lanes]
        return payload


def check_quality_lane_readiness(
    *,
    index_csv: Path,
    output_json: Path,
    lanes: Sequence[QualityLaneInput],
    actions: Sequence[str] = ("keep", "downweight", "quarantine"),
) -> QualityReadinessResult:
    """Write a machine-readable readiness report for quality scan lanes."""

    index_csv = index_csv.expanduser()
    output_json = output_json.expanduser()
    expected_rows = _count_index_rows(index_csv, actions=actions)
    lane_statuses = [
        _inspect_lane(lane, expected_rows=expected_rows)
        for lane in lanes
    ]
    blockers: list[str] = []
    warnings: list[str] = []
    for lane_status in lane_statuses:
        if lane_status.required and not lane_status.ready:
            blockers.extend(f"{lane_status.name}: {blocker}" for blocker in lane_status.blockers)
        warnings.extend(f"{lane_status.name}: {warning}" for warning in lane_status.warnings)

    ready = not blockers and all(
        lane.ready for lane in lane_statuses if lane.required
    )
    result = QualityReadinessResult(
        ready=ready,
        status="ready" if ready else "blocked",
        expected_rows=expected_rows,
        index_csv=index_csv,
        output_json=output_json,
        lanes=lane_statuses,
        blockers=blockers,
        warnings=warnings,
        next_actions=_next_actions(lane_statuses),
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _inspect_lane(lane: QualityLaneInput, expected_rows: int) -> QualityLaneStatus:
    stats_jsonl = lane.stats_jsonl.expanduser() if lane.stats_jsonl else None
    report_json = lane.report_json.expanduser() if lane.report_json else None
    stats_exists = bool(stats_jsonl and stats_jsonl.exists())
    report_exists = bool(report_json and report_json.exists())
    row_count = _count_jsonl_rows(stats_jsonl) if stats_exists else 0
    report = _read_json(report_json) if report_exists else {}
    report_scanned_rows = _maybe_int(report.get("scanned_rows")) if report else None
    report_limit = _maybe_int(report.get("limit")) if report else None
    action_counts = _int_mapping(report.get("action_counts")) if report else {}
    flag_counts = _int_mapping(report.get("flag_counts")) if report else {}
    coverage_ratio = round((row_count / expected_rows) if expected_rows else 0.0, 6)

    blockers: list[str] = []
    warnings: list[str] = []
    if lane.required and not stats_exists:
        blockers.append("stats JSONL is missing")
    if lane.required and not report_exists:
        blockers.append("final report JSON is missing")
    if stats_exists and report_exists and report_scanned_rows is not None:
        if report_scanned_rows != row_count:
            blockers.append(
                f"report scanned_rows={report_scanned_rows} does not match stats row_count={row_count}"
            )
    if stats_exists and row_count < expected_rows:
        blockers.append(f"stats cover {row_count}/{expected_rows} expected rows")
    if report_exists and report_limit is not None:
        blockers.append(f"report limit is {report_limit}, not a full scan")
    if report and bool(report.get("git_dirty", False)):
        warnings.append("report was generated from a dirty git tree")
    if not lane.required and not stats_exists and not report_exists:
        warnings.append("optional lane has no artifacts")

    ready = not blockers and (stats_exists or not lane.required) and (report_exists or not lane.required)
    status = "ready" if ready else ("missing" if not stats_exists and not report_exists else "partial")
    return QualityLaneStatus(
        name=lane.name,
        required=lane.required,
        ready=ready,
        status=status,
        row_count=row_count,
        expected_rows=expected_rows,
        coverage_ratio=coverage_ratio,
        stats_jsonl=str(stats_jsonl or ""),
        report_json=str(report_json or ""),
        stats_exists=stats_exists,
        report_exists=report_exists,
        report_scanned_rows=report_scanned_rows,
        report_limit=report_limit,
        report_git_sha=str(report.get("git_sha", "")) if report else "",
        report_git_dirty=bool(report.get("git_dirty")) if report else None,
        action_counts=action_counts,
        flag_counts=flag_counts,
        blockers=blockers,
        warnings=warnings,
    )


def _count_index_rows(index_csv: Path, actions: Sequence[str]) -> int:
    action_filter = set(actions)
    with index_csv.open(newline="", encoding="utf-8") as handle:
        return sum(
            1
            for row in csv.DictReader(handle)
            if not action_filter or row.get("curation_action") in action_filter
        )


def _count_jsonl_rows(path: Path | None) -> int:
    if path is None:
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_json(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _maybe_int(item) or 0 for key, item in value.items()}


def _maybe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _next_actions(lanes: Sequence[QualityLaneStatus]) -> list[str]:
    actions: list[str] = []
    for lane in lanes:
        if lane.ready:
            continue
        if not lane.stats_exists:
            actions.append(f"run {lane.name} full scan to produce stats JSONL")
            continue
        if lane.row_count < lane.expected_rows:
            actions.append(f"wait for or rerun {lane.name} full scan; current coverage {lane.row_count}/{lane.expected_rows}")
        if not lane.report_exists:
            actions.append(f"wait for {lane.name} final report JSON")
        if lane.report_exists and lane.report_limit is not None:
            actions.append(f"rerun {lane.name} without --limit")
    if not actions:
        actions.append("merge full source/source-FK/G1/pair quality stats into a curated run")
    return actions


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
