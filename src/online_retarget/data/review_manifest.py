"""Manual review manifests for curated worst-clip outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


FAMILY_KEYWORDS = {
    "parser": ("nonfinite", "mismatch", "missing", "decode", "parse", "empty"),
    "mirror": ("mirror_variant",),
    "jump": ("jump", "discontinuity", "unstable_start_end"),
    "foot_slide": ("foot_slide", "slide"),
    "penetration": ("penetration", "ground_penetration"),
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


def _review_items(
    rows: Sequence[Mapping[str, str]],
    max_per_family: int,
) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    items: list[dict[str, object]] = []
    for row in rows:
        flags = _split_flags(row.get("merged_quality_flags", ""))
        families = _families_for_flags(flags)
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
        "g1_joint_jump_rate",
        "g1_max_abs_joint_velocity",
        "g1_joint_limit_violation_rate",
        "g1_max_joint_limit_violation",
        "g1_contact_frame_ratio",
        "g1_contact_slide_rate",
        "g1_max_contact_slide_speed",
        "g1_mean_foot_clearance",
        "g1_penetration_depth",
        "g1_self_collision_proxy_rate",
        "g1_min_self_collision_distance",
        "g1_mean_min_self_collision_distance",
    )
    return {key: row.get(key, "") for key in keys if row.get(key, "") != ""}


def _families_for_flags(flags: Sequence[str]) -> tuple[str, ...]:
    families = []
    joined = " ".join(flags)
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(keyword in joined for keyword in keywords):
            families.append(family)
    return tuple(families)


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
