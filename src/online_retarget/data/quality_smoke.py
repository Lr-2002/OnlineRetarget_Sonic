"""Small end-to-end motion-quality smoke runs.

This module intentionally orchestrates the existing M2Q pieces instead of
adding a second quality policy path. It is for bounded experiments and
regression checks; a smoke run is not a promotable curation policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .bvh_quality import BVHQualityConfig, scan_bvh_quality_from_index
from .curation import QualityPolicy, SplitConfig, build_split_index
from .g1_quality import G1QualityConfig, scan_g1_quality_from_index
from .pair_quality import PairQualityConfig, scan_pair_quality_from_index
from .policy_audit import CurationPolicyAuditConfig, preflight_curation_policy
from .quality_merge import merge_quality_stats
from .review_manifest import build_review_decision_template, build_review_manifest
from .source_fk_quality import SourceFKQualityConfig, scan_source_fk_quality_from_index
from .thresholds import write_threshold_proposals


SOURCE_THRESHOLD_METRICS = (
    "max_abs_channel_velocity",
    "max_root_speed",
    "max_abs_channel_acceleration",
    "max_root_acceleration",
    "max_root_jerk",
)
SOURCE_FK_THRESHOLD_METRICS = (
    "mean_foot_clearance",
    "penetration_depth",
    "max_contact_slide_speed",
    "contact_slide_rate",
    "root_height_range",
    "max_root_support_distance",
)
SOURCE_FK_LOWER_THRESHOLD_METRICS = (
    "contact_frame_ratio",
    "support_frame_ratio",
    "root_height_min",
)
G1_THRESHOLD_METRICS = (
    "max_abs_joint_velocity",
    "max_root_speed",
    "max_abs_joint_acceleration",
    "max_root_acceleration",
    "max_root_jerk",
    "max_start_end_root_speed",
    "root_height_range",
    "mean_foot_clearance",
    "penetration_depth",
    "max_contact_slide_speed",
    "contact_slide_rate",
    "joint_limit_violation_rate",
    "max_joint_limit_violation",
    "self_collision_proxy_rate",
)
G1_LOWER_THRESHOLD_METRICS = (
    "contact_frame_ratio",
    "support_frame_ratio",
    "root_height_min",
)
PAIR_THRESHOLD_METRICS = (
    "abs_frame_count_delta",
    "abs_duration_delta_sec",
)


@dataclass(frozen=True)
class QualitySmokeConfig:
    run_name: str
    limit: int = 24
    seed: int = 17
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    sample_by: tuple[str, ...] = ("category", "split")
    threshold_percentile: float = 0.95
    threshold_min_group_size: int = 1
    frame_stride: int = 2
    max_frames: int | None = 256
    model_xml: Path | None = None
    review_max_per_family: int = 3

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["sample_by"] = list(self.sample_by)
        payload["model_xml"] = str(self.model_xml) if self.model_xml is not None else ""
        return payload


@dataclass(frozen=True)
class QualitySmokeResult:
    output_dir: Path
    smoke_report_json: Path
    split_index_csv: Path
    source_stats_jsonl: Path
    source_fk_stats_jsonl: Path
    g1_stats_jsonl: Path
    pair_stats_jsonl: Path
    threshold_proposal_jsons: tuple[Path, ...]
    curated_index_csv: Path
    curated_report_json: Path
    worst_clips_csv: Path
    review_manifest_jsonl: Path
    review_decision_template_csv: Path
    policy_preflight_json: Path
    promotable: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "smoke_report_json": str(self.smoke_report_json),
            "split_index_csv": str(self.split_index_csv),
            "source_stats_jsonl": str(self.source_stats_jsonl),
            "source_fk_stats_jsonl": str(self.source_fk_stats_jsonl),
            "g1_stats_jsonl": str(self.g1_stats_jsonl),
            "pair_stats_jsonl": str(self.pair_stats_jsonl),
            "threshold_proposal_jsons": [str(path) for path in self.threshold_proposal_jsons],
            "curated_index_csv": str(self.curated_index_csv),
            "curated_report_json": str(self.curated_report_json),
            "worst_clips_csv": str(self.worst_clips_csv),
            "review_manifest_jsonl": str(self.review_manifest_jsonl),
            "review_decision_template_csv": str(self.review_decision_template_csv),
            "policy_preflight_json": str(self.policy_preflight_json),
            "promotable": self.promotable,
            "blockers": list(self.blockers),
        }


def run_quality_smoke(
    data_root: Path,
    output_root: Path,
    config: QualitySmokeConfig,
) -> QualitySmokeResult:
    """Run a bounded, non-promotable M2Q quality pipeline."""

    if config.limit <= 0:
        raise ValueError("quality smoke limit must be positive")

    split = build_split_index(
        data_root=data_root,
        output_root=output_root,
        split_config=SplitConfig(
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            seed=config.seed,
        ),
        quality_policy=QualityPolicy(name=f"{config.run_name}_metadata"),
    )
    limit = config.limit
    sample_by = config.sample_by
    actions = ("keep", "downweight", "quarantine")

    source = scan_bvh_quality_from_index(
        data_root=data_root,
        index_csv=split.index_csv,
        output_root=output_root,
        config=BVHQualityConfig(expected_frame_time=1.0 / 120.0),
        limit=limit,
        actions=actions,
        sample_by=sample_by,
    )
    source_fk = scan_source_fk_quality_from_index(
        data_root=data_root,
        index_csv=split.index_csv,
        output_root=output_root,
        config=SourceFKQualityConfig(
            frame_stride=config.frame_stride,
            max_frames=config.max_frames,
        ),
        limit=limit,
        actions=actions,
        sample_by=sample_by,
    )
    g1 = scan_g1_quality_from_index(
        data_root=data_root,
        index_csv=split.index_csv,
        output_root=output_root,
        config=G1QualityConfig(
            frame_stride=config.frame_stride,
            max_frames=config.max_frames,
            model_xml=config.model_xml,
        ),
        limit=limit,
        actions=actions,
        sample_by=sample_by,
    )
    pair = scan_pair_quality_from_index(
        data_root=data_root,
        index_csv=split.index_csv,
        output_root=output_root,
        config=PairQualityConfig(expected_source_frame_time=1.0 / 120.0),
        limit=limit,
        actions=actions,
        sample_by=sample_by,
    )

    threshold_proposals = _write_threshold_proposals(config, source, source_fk, g1, pair)
    merged = merge_quality_stats(
        split_index_csv=split.index_csv,
        source_stats_jsonl=source.stats_jsonl,
        source_fk_stats_jsonl=source_fk.stats_jsonl,
        g1_stats_jsonl=g1.stats_jsonl,
        pair_stats_jsonl=pair.stats_jsonl,
        output_root=output_root,
        run_name=config.run_name,
    )
    review = build_review_manifest(
        worst_clips_csv=merged.worst_clips_csv,
        output_root=merged.output_dir,
        run_name="manual_review",
        max_per_family=config.review_max_per_family,
    )
    review_template = build_review_decision_template(
        review_manifest_jsonl=review.manifest_jsonl,
        overwrite=True,
    )
    preflight = preflight_curation_policy(
        curated_run_dir=merged.output_dir,
        policy_id=config.run_name,
        config=CurationPolicyAuditConfig(
            policy_id=config.run_name,
            allow_representative=True,
            require_review_decisions=False,
            require_clean_report_git=False,
        ),
    )

    result = QualitySmokeResult(
        output_dir=merged.output_dir,
        smoke_report_json=merged.output_dir / "quality_smoke_report.json",
        split_index_csv=split.index_csv,
        source_stats_jsonl=source.stats_jsonl,
        source_fk_stats_jsonl=source_fk.stats_jsonl,
        g1_stats_jsonl=g1.stats_jsonl,
        pair_stats_jsonl=pair.stats_jsonl,
        threshold_proposal_jsons=tuple(threshold_proposals),
        curated_index_csv=merged.curated_index_csv,
        curated_report_json=merged.report_json,
        worst_clips_csv=merged.worst_clips_csv,
        review_manifest_jsonl=review.manifest_jsonl,
        review_decision_template_csv=review_template.output_csv,
        policy_preflight_json=preflight.audit_json,
        promotable=preflight.audit.promotable,
        blockers=tuple(preflight.audit.blockers),
    )
    _write_smoke_report(
        result.smoke_report_json,
        config=config.to_dict(),
        result=result.to_dict(),
        split=split.to_dict(),
        source=source.to_dict(),
        source_fk=source_fk.to_dict(),
        g1=g1.to_dict(),
        pair=pair.to_dict(),
        merged=merged.to_dict(),
        review=review.to_dict(),
        review_template=review_template.to_dict(),
        preflight=preflight.to_dict(),
    )
    return result


def _write_threshold_proposals(
    config: QualitySmokeConfig,
    source,
    source_fk,
    g1,
    pair,
) -> list[Path]:
    percentile_tag = f"p{int(round(config.threshold_percentile * 100))}"
    group_by = config.sample_by
    min_group_size = config.threshold_min_group_size
    outputs = [
        (
            source.stats_jsonl,
            source.stats_jsonl.with_name(f"source_threshold_proposals_grouped_{percentile_tag}.json"),
            SOURCE_THRESHOLD_METRICS,
            (),
        ),
        (
            source_fk.stats_jsonl,
            source_fk.stats_jsonl.with_name(
                f"source_fk_threshold_proposals_grouped_{percentile_tag}.json"
            ),
            SOURCE_FK_THRESHOLD_METRICS,
            SOURCE_FK_LOWER_THRESHOLD_METRICS,
        ),
        (
            g1.stats_jsonl,
            g1.stats_jsonl.with_name(f"g1_threshold_proposals_grouped_{percentile_tag}.json"),
            G1_THRESHOLD_METRICS,
            G1_LOWER_THRESHOLD_METRICS,
        ),
        (
            pair.stats_jsonl,
            pair.stats_jsonl.with_name(f"pair_threshold_proposals_grouped_{percentile_tag}.json"),
            PAIR_THRESHOLD_METRICS,
            (),
        ),
    ]
    written = []
    for stats_jsonl, output_json, metrics, lower_metrics in outputs:
        write_threshold_proposals(
            stats_jsonl=stats_jsonl,
            output_json=output_json,
            metrics=metrics,
            lower_metrics=lower_metrics,
            percentile=config.threshold_percentile,
            action="quarantine",
            group_by=group_by,
            min_group_size=min_group_size,
        )
        written.append(output_json)
    return written


def _write_smoke_report(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
