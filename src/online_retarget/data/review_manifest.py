"""Manual review manifests for curated worst-clip outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


REVIEW_FIELD_KEYS = (
    "decision",
    "reviewer",
    "notes",
    "confirmed_issue",
    "recommended_action",
)
RECOMMENDED_ACTIONS = ("keep", "downweight", "quarantine", "exclude")
FAMILY_KEYWORDS = {
    "parser": ("nonfinite", "mismatch", "missing", "decode", "parse", "empty"),
    "mirror": ("mirror_variant",),
    "jump": ("jump", "discontinuity", "unstable_start_end"),
    "foot_slide": ("foot_slide", "slide"),
    "penetration": ("penetration", "ground_penetration"),
    "contact_correction": ("contact_correction",),
    "float": ("float", "low_foot_contact"),
    "joint_limit": ("joint_limit",),
    "self_collision": ("self_collision", "self_intersection"),
}


@dataclass(frozen=True)
class ReviewManifestResult:
    output_dir: Path
    manifest_jsonl: Path
    manifest_md: Path
    report_json: Path
    reviewed_rows: int
    family_counts: dict[str, int]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["manifest_jsonl"] = str(self.manifest_jsonl)
        payload["manifest_md"] = str(self.manifest_md)
        payload["report_json"] = str(self.report_json)
        return payload


@dataclass(frozen=True)
class ReviewDecisionMergeResult:
    output_jsonl: Path
    report_json: Path
    manifest_items: int
    decision_rows: int
    matched_decisions: int
    complete_decisions: int
    incomplete_decisions: int
    incomplete_review_ids: list[str]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_jsonl"] = str(self.output_jsonl)
        payload["report_json"] = str(self.report_json)
        return payload


def build_review_manifest(
    worst_clips_csv: Path,
    output_root: Path | None = None,
    run_name: str = "manual_review",
    max_per_family: int = 5,
) -> ReviewManifestResult:
    """Build JSONL and Markdown artifacts for manual inspection."""

    if max_per_family <= 0:
        raise ValueError("max_per_family must be positive")
    rows = _read_csv(worst_clips_csv)
    items = _review_items(rows, max_per_family=max_per_family)
    output_dir = (output_root or worst_clips_csv.parent).expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl = output_dir / "review_manifest.jsonl"
    manifest_md = output_dir / "review_manifest.md"
    report_json = output_dir / "review_report.json"
    _write_jsonl(manifest_jsonl, items)
    _write_text(manifest_md, _markdown_report(items, worst_clips_csv, max_per_family))

    family_counts = Counter(item["failure_family"] for item in items)
    report = {
        "worst_clips_csv": str(worst_clips_csv),
        "manifest_jsonl": str(manifest_jsonl),
        "manifest_md": str(manifest_md),
        "max_per_family": max_per_family,
        "input_rows": len(rows),
        "reviewed_rows": len(items),
        "family_counts": dict(sorted(family_counts.items())),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(report_json, report)
    return ReviewManifestResult(
        output_dir=output_dir,
        manifest_jsonl=manifest_jsonl,
        manifest_md=manifest_md,
        report_json=report_json,
        reviewed_rows=len(items),
        family_counts=dict(sorted(family_counts.items())),
        git_sha=report["git_sha"],
        git_dirty=report["git_dirty"],
    )


def merge_review_decisions(
    review_manifest_jsonl: Path,
    decisions_file: Path,
    output_jsonl: Path | None = None,
    output_report_json: Path | None = None,
) -> ReviewDecisionMergeResult:
    """Merge reviewer decisions into a new review manifest JSONL.

    The source manifest is left untouched. Decisions can be supplied as CSV or
    JSONL rows keyed by ``review_id``. JSONL rows may contain either flat
    review fields or a nested ``review_fields`` object.
    """

    items = _read_jsonl(review_manifest_jsonl)
    decisions = _read_decisions(decisions_file)
    by_review_id = _manifest_by_review_id(items)
    decision_by_id: dict[str, dict[str, str]] = {}

    for row in decisions:
        review_id = str(row.get("review_id", "")).strip()
        if not review_id:
            raise ValueError("decision row is missing review_id")
        if review_id in decision_by_id:
            raise ValueError(f"duplicate decision for review_id: {review_id}")
        if review_id not in by_review_id:
            raise ValueError(f"decision references unknown review_id: {review_id}")
        fields = _decision_fields(row)
        decision_by_id[review_id] = fields

    updated_items: list[dict[str, object]] = []
    for item in items:
        updated = dict(item)
        review_id = str(updated.get("review_id", "")).strip()
        review_fields = updated.get("review_fields", {})
        if not isinstance(review_fields, Mapping):
            review_fields = {}
        merged_fields = {key: str(review_fields.get(key, "")) for key in REVIEW_FIELD_KEYS}
        if review_id in decision_by_id:
            merged_fields.update(decision_by_id[review_id])
        updated["review_fields"] = merged_fields
        updated_items.append(updated)

    incomplete_ids = _incomplete_review_ids(updated_items)
    output_path = output_jsonl or review_manifest_jsonl.with_name(
        review_manifest_jsonl.stem + ".reviewed.jsonl"
    )
    report_path = output_report_json or output_path.with_name("review_decision_report.json")
    _write_jsonl(output_path, updated_items)

    result = ReviewDecisionMergeResult(
        output_jsonl=output_path,
        report_json=report_path,
        manifest_items=len(items),
        decision_rows=len(decisions),
        matched_decisions=len(decision_by_id),
        complete_decisions=len(items) - len(incomplete_ids),
        incomplete_decisions=len(incomplete_ids),
        incomplete_review_ids=incomplete_ids,
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
    )
    report = result.to_dict()
    report.update(
        {
            "review_manifest_jsonl": str(review_manifest_jsonl),
            "decisions_file": str(decisions_file),
            "allowed_recommended_actions": list(RECOMMENDED_ACTIONS),
        }
    )
    _write_json(report_path, report)
    return result


def _review_items(
    rows: Sequence[Mapping[str, str]],
    max_per_family: int,
) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    items: list[dict[str, object]] = []
    for row in rows:
        flags = _split_flags(row.get("merged_quality_flags", ""))
        families = _families_for_row(row, flags)
        if not families:
            families = ("other",)
        for family in families:
            if counts[family] >= max_per_family:
                continue
            row_index = row.get("row_index", "")
            key = (family, row_index)
            if key in seen:
                continue
            seen.add(key)
            counts[family] += 1
            items.append(_review_item(row, flags, family))
    return items


def _review_item(
    row: Mapping[str, str],
    flags: Sequence[str],
    family: str,
) -> dict[str, object]:
    return {
        "review_id": f"{family}:{row.get('row_index', '')}:{row.get('filename', '')}",
        "failure_family": family,
        "row_index": row.get("row_index", ""),
        "split": row.get("split", ""),
        "actor_uid": row.get("actor_uid", ""),
        "package": row.get("package", ""),
        "category": row.get("category", ""),
        "filename": row.get("filename", ""),
        "merged_quality_action": row.get("merged_quality_action", ""),
        "merged_quality_flags": list(flags),
        "motion_paths": {
            "source_bvh": row.get("move_soma_proportional_path", ""),
            "g1_csv": row.get("move_g1_path", ""),
        },
        "metrics": _review_metrics(row),
        "review_fields": {
            "decision": "",
            "reviewer": "",
            "notes": "",
            "confirmed_issue": "",
            "recommended_action": "",
        },
    }


def _review_metrics(row: Mapping[str, str]) -> dict[str, str]:
    keys = (
        "source_channel_jump_rate",
        "source_max_abs_channel_velocity",
        "source_fk_contact_frame_ratio",
        "source_fk_contact_slide_rate",
        "source_fk_max_contact_slide_speed",
        "source_fk_mean_foot_clearance",
        "source_fk_penetration_depth",
        "source_fk_contact_correction_candidate",
        "source_fk_contact_correction_reason",
        "source_fk_contact_correction_offset",
        "source_fk_contact_correction_abs_offset",
        "g1_joint_jump_rate",
        "g1_max_abs_joint_velocity",
        "g1_joint_limit_violation_rate",
        "g1_max_joint_limit_violation",
        "g1_contact_frame_ratio",
        "g1_contact_slide_rate",
        "g1_max_contact_slide_speed",
        "g1_mean_foot_clearance",
        "g1_penetration_depth",
        "g1_contact_correction_candidate",
        "g1_contact_correction_reason",
        "g1_contact_correction_offset",
        "g1_contact_correction_abs_offset",
        "g1_self_collision_proxy_rate",
        "g1_min_self_collision_distance",
        "g1_mean_min_self_collision_distance",
    )
    return {key: row.get(key, "") for key in keys if row.get(key, "") != ""}


def _read_decisions(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl(path)
    raise ValueError(f"decisions file must be CSV or JSONL: {path}")


def _manifest_by_review_id(
    items: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    by_id: dict[str, Mapping[str, object]] = {}
    for item in items:
        review_id = str(item.get("review_id", "")).strip()
        if not review_id:
            raise ValueError("review manifest item is missing review_id")
        if review_id in by_id:
            raise ValueError(f"duplicate review_id in manifest: {review_id}")
        by_id[review_id] = item
    return by_id


def _decision_fields(row: Mapping[str, object]) -> dict[str, str]:
    nested = row.get("review_fields", {})
    source: dict[str, object] = {}
    if isinstance(nested, Mapping):
        source.update(nested)
    source.update({key: value for key, value in row.items() if key in REVIEW_FIELD_KEYS})
    fields = {key: _text_field(source.get(key, "")) for key in REVIEW_FIELD_KEYS}
    if not fields["decision"]:
        raise ValueError("decision row is missing decision")
    if not fields["recommended_action"]:
        raise ValueError("decision row is missing recommended_action")
    if fields["recommended_action"] not in RECOMMENDED_ACTIONS:
        allowed = ", ".join(RECOMMENDED_ACTIONS)
        raise ValueError(
            f"invalid recommended_action {fields['recommended_action']!r}; "
            f"expected one of: {allowed}"
        )
    return fields


def _text_field(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _incomplete_review_ids(items: Sequence[Mapping[str, object]]) -> list[str]:
    incomplete = []
    for item in items:
        fields = item.get("review_fields", {})
        if not isinstance(fields, Mapping):
            incomplete.append(str(item.get("review_id", "")))
            continue
        decision = str(fields.get("decision", "")).strip()
        recommended_action = str(fields.get("recommended_action", "")).strip()
        if not decision or not recommended_action:
            incomplete.append(str(item.get("review_id", "")))
    return incomplete


def _families_for_row(row: Mapping[str, str], flags: Sequence[str]) -> tuple[str, ...]:
    families = []
    joined = " ".join(flags)
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(keyword in joined for keyword in keywords):
            families.append(family)
    if _is_contact_correction_candidate(row):
        families.append("contact_correction")
    return tuple(dict.fromkeys(families))


def _is_contact_correction_candidate(row: Mapping[str, str]) -> bool:
    for key in ("source_fk_contact_correction_candidate", "g1_contact_correction_candidate"):
        try:
            if float(row.get(key, "") or 0.0) > 0.0:
                return True
        except ValueError:
            continue
    return False


def _markdown_report(
    items: Sequence[Mapping[str, object]],
    worst_clips_csv: Path,
    max_per_family: int,
) -> str:
    lines = [
        "# Manual Motion Review Manifest",
        "",
        f"- Source worst clips: `{worst_clips_csv}`",
        f"- Max per family: {max_per_family}",
        f"- Review items: {len(items)}",
        "",
    ]
    family_counts = Counter(str(item["failure_family"]) for item in items)
    lines.append("## Failure Families")
    lines.append("")
    for family, count in sorted(family_counts.items()):
        lines.append(f"- `{family}`: {count}")
    lines.append("")
    for item in items:
        lines.extend(_markdown_item(item))
    return "\n".join(lines).rstrip() + "\n"


def _markdown_item(item: Mapping[str, object]) -> list[str]:
    metrics = item.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    flags = item.get("merged_quality_flags", [])
    flag_text = ", ".join(str(flag) for flag in flags) if isinstance(flags, Sequence) else str(flags)
    paths = item.get("motion_paths", {})
    source_bvh = paths.get("source_bvh", "") if isinstance(paths, Mapping) else ""
    g1_csv = paths.get("g1_csv", "") if isinstance(paths, Mapping) else ""
    lines = [
        f"## {item.get('review_id', '')}",
        "",
        f"- Family: `{item.get('failure_family', '')}`",
        f"- Actor/category: `{item.get('actor_uid', '')}` / `{item.get('package', '')}` / `{item.get('category', '')}` / `{item.get('split', '')}`",
        f"- Action: `{item.get('merged_quality_action', '')}`",
        f"- Flags: {flag_text}",
        f"- Source BVH: `{source_bvh}`",
        f"- G1 CSV: `{g1_csv}`",
        "- Metrics:",
    ]
    if metrics:
        for key, value in sorted(metrics.items()):
            lines.append(f"  - `{key}`: {value}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            "- Review:",
            "  - Decision:",
            "  - Confirmed issue:",
            "  - Recommended action:",
            "  - Notes:",
            "",
        ]
    )
    return lines


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must be a JSON object: {path}")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _split_flags(flags: str) -> list[str]:
    if not flags:
        return []
    return [flag for flag in flags.split("|") if flag]


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
