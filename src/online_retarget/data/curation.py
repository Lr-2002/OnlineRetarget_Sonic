"""Dataset curation, split, and index helpers.

This module stays in the standard library so the repo can build traceable split
artifacts before the full training environment is available.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
import subprocess
from typing import Literal, Mapping, Sequence

from .bones_seed import (
    SKELETON_MEASURE_COLUMNS,
    actor_skeletons,
    iter_metadata_rows,
    summarize_metadata,
)


CurateAction = Literal["keep", "downweight", "quarantine", "exclude"]
ThresholdDirection = Literal["min", "max"]


QUALITY_ACTION_ORDER: dict[CurateAction, int] = {
    "keep": 0,
    "downweight": 1,
    "quarantine": 2,
    "exclude": 3,
}

DEFAULT_REQUIRED_FIELDS = (
    "actor_uid",
    "move_soma_proportional_path",
    "move_soma_proportional_shape_path",
    "move_g1_path",
)


@dataclass(frozen=True)
class QualityThreshold:
    """Optional clip-stat threshold used for future quality scoring."""

    metric: str
    direction: ThresholdDirection
    value: float
    flag: str
    action: CurateAction = "quarantine"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QualityPolicy:
    """Policy for converting metadata and optional clip stats into a decision."""

    name: str = "metadata_only"
    required_fields: tuple[str, ...] = DEFAULT_REQUIRED_FIELDS
    min_duration_frames: int = 0
    downweight_mirrors: bool = True
    quarantine_missing_optional_metadata: bool = True
    thresholds: tuple[QualityThreshold, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required_fields": list(self.required_fields),
            "min_duration_frames": self.min_duration_frames,
            "downweight_mirrors": self.downweight_mirrors,
            "quarantine_missing_optional_metadata": self.quarantine_missing_optional_metadata,
            "thresholds": [threshold.to_dict() for threshold in self.thresholds],
        }


@dataclass(frozen=True)
class SplitConfig:
    """Actor-heldout split configuration."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 17
    group_by: str = "actor_uid"

    def test_ratio(self) -> float:
        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        if test_ratio < 0:
            raise ValueError("train_ratio + val_ratio must be <= 1.0")
        return test_ratio

    def to_dict(self) -> dict[str, object]:
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio(),
            "seed": self.seed,
            "group_by": self.group_by,
        }


@dataclass(frozen=True)
class QualityDecision:
    """Per-row quality result for traceable curation."""

    action: CurateAction
    score: float
    flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SplitIndexResult:
    """Paths and summary for a generated split index."""

    output_dir: Path
    index_csv: Path
    report_json: Path
    manifest_json: Path
    row_count: int
    actor_count: int
    split_counts: dict[str, int]
    action_counts: dict[str, int]
    flag_counts: dict[str, int]
    split_actor_counts: dict[str, int]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["index_csv"] = str(self.index_csv)
        payload["report_json"] = str(self.report_json)
        payload["manifest_json"] = str(self.manifest_json)
        return payload


def build_split_index(
    data_root: Path,
    output_root: Path,
    split_config: SplitConfig,
    quality_policy: QualityPolicy | None = None,
) -> SplitIndexResult:
    """Build a row-level split index and curation report from BONES-SEED metadata."""

    policy = quality_policy or QualityPolicy()
    output_root = output_root.expanduser()
    data_root = data_root.expanduser()
    _validate_output_root(data_root, output_root)

    actors = actor_skeletons(data_root)
    actor_splits = _assign_actor_splits(actors, split_config)
    quality_rows = {
        "rows": build_quality_rows(data_root, actor_splits, policy),
        "actor_splits": actor_splits,
    }
    split_name = _split_name(split_config, policy)
    output_dir = output_root / "indices" / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    index_csv = output_dir / "split_index.csv"
    report_json = output_dir / "split_report.json"
    manifest_json = output_dir / "split_manifest.json"

    _write_csv(index_csv, quality_rows["rows"])
    report = _build_report(data_root, output_dir, split_config, policy, actors, quality_rows)
    _write_json(report_json, report)
    manifest = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "metadata_csv": str(data_root / "metadata/seed_metadata_v003.csv"),
        "index_csv": str(index_csv),
        "report_json": str(report_json),
        "split_config": split_config.to_dict(),
        "quality_policy": policy.to_dict(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(manifest_json, manifest)

    split_counts = Counter(row["split"] for row in quality_rows["rows"])
    action_counts = Counter(row["curation_action"] for row in quality_rows["rows"])
    flag_counts = Counter()
    split_actor_counts = Counter(actor_splits.values())
    for row in quality_rows["rows"]:
        for flag in _split_flags(row["quality_flags"]):
            flag_counts[flag] += 1

    return SplitIndexResult(
        output_dir=output_dir,
        index_csv=index_csv,
        report_json=report_json,
        manifest_json=manifest_json,
        row_count=len(quality_rows["rows"]),
        actor_count=len(actors),
        split_counts=dict(sorted(split_counts.items())),
        action_counts=dict(sorted(action_counts.items())),
        flag_counts=dict(sorted(flag_counts.items())),
        split_actor_counts=dict(sorted(split_actor_counts.items())),
        git_sha=manifest["git_sha"],
        git_dirty=manifest["git_dirty"],
    )


def assess_row_quality(
    row: Mapping[str, str],
    policy: QualityPolicy | None = None,
    clip_stats: Mapping[str, float] | None = None,
) -> QualityDecision:
    """Assign a traceable quality decision from metadata and optional clip stats."""

    policy = policy or QualityPolicy()
    flags: list[str] = []
    action: CurateAction = "keep"
    score = 1.0

    for field_name in policy.required_fields:
        if not row.get(field_name):
            flags.append(f"missing_{field_name}")
            action = "exclude"
            return QualityDecision(action=action, score=0.0, flags=tuple(flags))

    duration = _maybe_int(row.get("move_duration_frames"))
    if policy.min_duration_frames > 0 and duration is not None and duration < policy.min_duration_frames:
        flags.append("short_clip")
        action = _worse_action(action, "quarantine")
        score -= 0.35

    if _is_true(row.get("is_mirror")) and policy.downweight_mirrors:
        flags.append("mirror_variant")
        action = _worse_action(action, "downweight")
        score -= 0.1

    if clip_stats:
        for threshold in policy.thresholds:
            value = clip_stats.get(threshold.metric)
            if value is None:
                continue
            violation = (
                value < threshold.value
                if threshold.direction == "min"
                else value > threshold.value
            )
            if violation:
                flags.append(threshold.flag)
                action = _worse_action(action, threshold.action)
                score -= _penalty_for_action(threshold.action)

    if policy.quarantine_missing_optional_metadata and _has_missing_optional(row):
        flags.append("missing_optional_metadata")
        action = _worse_action(action, "quarantine")
        score -= 0.15

    score = max(score, 0.0)
    return QualityDecision(action=action, score=score, flags=tuple(dict.fromkeys(flags)))


def build_quality_rows(
    data_root: Path,
    actor_splits: Mapping[str, str],
    policy: QualityPolicy | None = None,
) -> list[dict[str, object]]:
    """Return row records with split assignments and curation decisions."""

    policy = policy or QualityPolicy()
    rows: list[dict[str, object]] = []
    for row_index, row in enumerate(iter_metadata_rows(data_root), start=1):
        actor_uid = row["actor_uid"]
        split = actor_splits.get(actor_uid, "unassigned")
        decision = assess_row_quality(row, policy=policy)
        rows.append(
            {
                "row_index": row_index,
                "split": split if decision.action != "exclude" else "excluded",
                "actor_uid": actor_uid,
                "move_name": row.get("move_name", ""),
                "filename": row.get("filename", ""),
                "package": row.get("package", ""),
                "category": row.get("category", ""),
                "is_mirror": row.get("is_mirror", ""),
                "move_duration_frames": row.get("move_duration_frames", ""),
                "move_soma_proportional_path": row.get("move_soma_proportional_path", ""),
                "move_soma_proportional_shape_path": row.get(
                    "move_soma_proportional_shape_path", ""
                ),
                "move_g1_path": row.get("move_g1_path", ""),
                "curation_action": decision.action,
                "quality_score": f"{decision.score:.3f}",
                "quality_flags": "|".join(decision.flags),
                **_morphology_fields(row),
            }
        )
    return rows


def quality_report(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate split and curation statistics from row records."""

    split_counts = Counter(str(row["split"]) for row in rows)
    action_counts = Counter(str(row["curation_action"]) for row in rows)
    flag_counts = Counter()
    packages = Counter()
    categories = Counter()
    actor_uids = Counter()
    for row in rows:
        packages[str(row["package"])] += 1
        categories[str(row["category"])] += 1
        actor_uids[str(row["actor_uid"])] += 1
        for flag in _split_flags(str(row["quality_flags"])):
            flag_counts[flag] += 1

    return {
        "rows": len(rows),
        "actors": len(actor_uids),
        "split_counts": dict(sorted(split_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "package_counts": dict(sorted(packages.items(), key=lambda item: (-item[1], item[0]))),
        "category_counts": dict(sorted(categories.items(), key=lambda item: (-item[1], item[0]))),
    }


def _build_report(
    data_root: Path,
    output_dir: Path,
    split_config: SplitConfig,
    policy: QualityPolicy,
    actors: Sequence,
    quality_rows: Mapping[str, object],
) -> dict[str, object]:
    summary = summarize_metadata(data_root)
    row_report = quality_report(quality_rows["rows"])
    return {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "metadata_csv": summary.metadata_csv,
        "split_config": split_config.to_dict(),
        "quality_policy": policy.to_dict(),
        "inventory": summary.to_dict(),
        "actor_count": len(actors),
        "split_actor_counts": dict(sorted(Counter(quality_rows["actor_splits"].values()).items())),
        "quality_report": row_report,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }


def _assign_actor_splits(actors: Sequence, split_config: SplitConfig) -> dict[str, str]:
    actor_uids = [actor.actor_uid for actor in actors]
    rng = random.Random(split_config.seed)
    rng.shuffle(actor_uids)

    train_end = int(len(actor_uids) * split_config.train_ratio)
    val_end = train_end + int(len(actor_uids) * split_config.val_ratio)

    splits: dict[str, str] = {}
    for actor_uid in actor_uids[:train_end]:
        splits[actor_uid] = "train"
    for actor_uid in actor_uids[train_end:val_end]:
        splits[actor_uid] = "val"
    for actor_uid in actor_uids[val_end:]:
        splits[actor_uid] = "test"
    return splits


def _split_name(split_config: SplitConfig, policy: QualityPolicy) -> str:
    train = round(split_config.train_ratio * 100)
    val = round(split_config.val_ratio * 100)
    test = round(split_config.test_ratio() * 100)
    return f"actor_split_t{train:02d}_v{val:02d}_x{test:02d}_s{split_config.seed}_{policy.name}"


def _validate_output_root(data_root: Path, output_root: Path) -> None:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError(
            f"Output root {output_root} is inside read-only data root {data_root}; "
            "choose a repo-local or scratch output path."
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


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


def _is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _maybe_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _has_missing_optional(row: Mapping[str, str]) -> bool:
    optional_fields = ("actor_height_cm", "actor_weight_kg", "actor_age_yr", "actor_gender")
    return any(not row.get(field) for field in optional_fields)


def _morphology_fields(row: Mapping[str, str]) -> dict[str, str]:
    fields = {column: row.get(column, "") for column in SKELETON_MEASURE_COLUMNS}
    fields["actor_weight_kg"] = row.get("actor_weight_kg", "")
    fields["actor_age_yr"] = row.get("actor_age_yr", "")
    fields["actor_gender"] = row.get("actor_gender", "")
    return fields


def _worse_action(left: CurateAction, right: CurateAction) -> CurateAction:
    return left if QUALITY_ACTION_ORDER[left] >= QUALITY_ACTION_ORDER[right] else right


def _penalty_for_action(action: CurateAction) -> float:
    if action == "downweight":
        return 0.1
    if action == "quarantine":
        return 0.25
    if action == "exclude":
        return 1.0
    return 0.0


def _split_flags(flags: str) -> list[str]:
    if not flags:
        return []
    return [flag for flag in flags.split("|") if flag]
