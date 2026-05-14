"""Audit whether a curated quality policy is ready for formal training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence


SCAN_COVERAGE_FIELDS = (
    "merged_source_rows",
    "merged_source_fk_rows",
    "merged_g1_rows",
    "merged_pair_rows",
)
RETAINED_ACTIONS = ("keep", "downweight")
RECOMMENDED_REVIEW_ACTIONS = ("keep", "downweight", "quarantine", "exclude")
DEFAULT_REQUIRED_GROUP_BY = ("category", "split")
DEFAULT_DIVERSITY_DIMENSIONS = ("actor_uid", "source_skeleton", "category", "split")


@dataclass(frozen=True)
class CurationPolicyAuditConfig:
    policy_id: str
    allow_representative: bool = False
    thresholds_accepted: bool = False
    require_review_decisions: bool = True
    required_group_by: tuple[str, ...] = DEFAULT_REQUIRED_GROUP_BY
    diversity_dimensions: tuple[str, ...] = DEFAULT_DIVERSITY_DIMENSIONS
    require_clean_report_git: bool = True


@dataclass(frozen=True)
class CurationPolicyAuditResult:
    policy_id: str
    promotable: bool
    status: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CurationPolicyPreflightResult:
    policy_id: str
    curated_run_dir: Path
    audit_json: Path
    audit: CurationPolicyAuditResult
    discovered: dict[str, object]
    next_actions: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "curated_run_dir": str(self.curated_run_dir),
            "audit_json": str(self.audit_json),
            "audit": self.audit.to_dict(),
            "discovered": self.discovered,
            "next_actions": list(self.next_actions),
        }


def audit_curation_policy(
    curated_report_json: Path,
    threshold_proposal_jsons: Sequence[Path],
    output_json: Path | None = None,
    threshold_policy_json: Path | None = None,
    review_report_json: Path | None = None,
    review_manifest_jsonl: Path | None = None,
    review_decision_report_json: Path | None = None,
    config: CurationPolicyAuditConfig | None = None,
) -> CurationPolicyAuditResult:
    """Audit a merged quality report before a curation policy is promoted."""

    cfg = config or CurationPolicyAuditConfig(policy_id=curated_report_json.stem)
    curated_report = _read_json(curated_report_json)
    threshold_reports = [_read_json(path) for path in threshold_proposal_jsons]
    threshold_policy = _read_json(threshold_policy_json) if threshold_policy_json else {}
    review_report = _read_json(review_report_json) if review_report_json else {}
    review_decision_report = (
        _read_json(review_decision_report_json) if review_decision_report_json else {}
    )
    review_items = _read_jsonl(review_manifest_jsonl) if review_manifest_jsonl else []

    blockers: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, object] = {
        "curated_report_json": str(curated_report_json),
        "threshold_proposal_jsons": [str(path) for path in threshold_proposal_jsons],
        "threshold_policy_json": str(threshold_policy_json) if threshold_policy_json else "",
        "review_report_json": str(review_report_json) if review_report_json else "",
        "review_manifest_jsonl": str(review_manifest_jsonl) if review_manifest_jsonl else "",
        "review_decision_report_json": str(review_decision_report_json)
        if review_decision_report_json
        else "",
        "allow_representative": cfg.allow_representative,
        "thresholds_accepted": cfg.thresholds_accepted,
    }

    _audit_curated_report(curated_report, cfg, blockers, warnings, evidence)
    _audit_threshold_reports(threshold_reports, threshold_policy, cfg, blockers, warnings, evidence)
    _audit_manual_review(
        review_report,
        review_items,
        review_decision_report,
        cfg,
        blockers,
        warnings,
        evidence,
    )

    promotable = not blockers
    result = CurationPolicyAuditResult(
        policy_id=cfg.policy_id,
        promotable=promotable,
        status="promotable" if promotable else "blocked",
        blockers=blockers,
        warnings=warnings,
        evidence=evidence,
    )
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def preflight_curation_policy(
    curated_run_dir: Path,
    policy_id: str | None = None,
    output_json: Path | None = None,
    threshold_policy_json: Path | None = None,
    review_decision_report_json: Path | None = None,
    config: CurationPolicyAuditConfig | None = None,
) -> CurationPolicyPreflightResult:
    """Run the standard promotion audit from a curated run directory.

    The preflight is a path-discovery wrapper around ``audit_curation_policy``.
    It does not relax the audit gates; it makes the current blockers and next
    actions reproducible from one stable directory.
    """

    run_dir = curated_run_dir.expanduser()
    curated_report_json = run_dir / "curated_report.json"
    if not curated_report_json.exists():
        raise FileNotFoundError(f"missing curated report: {curated_report_json}")
    curated_report = _read_json(curated_report_json)
    resolved_policy_id = policy_id or run_dir.name
    audit_json = output_json or run_dir / "policy_preflight.json"

    threshold_proposals = discover_threshold_proposals_from_report(curated_report)
    review_dir = run_dir / "manual_review"
    review_report_json = _existing_path(review_dir / "review_report.json")
    review_manifest_jsonl = _existing_path(review_dir / "review_manifest.jsonl")
    resolved_threshold_policy_json = _existing_path(threshold_policy_json) or _existing_path(
        run_dir / "threshold_policy.json"
    )
    resolved_review_decision_report_json = _existing_path(review_decision_report_json) or _existing_path(
        review_dir / "review_decision_report.json"
    )

    base_config = config or CurationPolicyAuditConfig(policy_id=resolved_policy_id)
    audit = audit_curation_policy(
        curated_report_json=curated_report_json,
        threshold_proposal_jsons=tuple(threshold_proposals),
        threshold_policy_json=resolved_threshold_policy_json,
        review_report_json=review_report_json,
        review_manifest_jsonl=review_manifest_jsonl,
        review_decision_report_json=resolved_review_decision_report_json,
        output_json=audit_json,
        config=base_config,
    )
    discovered = {
        "curated_report_json": str(curated_report_json),
        "threshold_proposal_jsons": [str(path) for path in threshold_proposals],
        "threshold_policy_json": str(resolved_threshold_policy_json or ""),
        "review_report_json": str(review_report_json or ""),
        "review_manifest_jsonl": str(review_manifest_jsonl or ""),
        "review_decision_report_json": str(resolved_review_decision_report_json or ""),
    }
    result = CurationPolicyPreflightResult(
        policy_id=resolved_policy_id,
        curated_run_dir=run_dir,
        audit_json=audit_json,
        audit=audit,
        discovered=discovered,
        next_actions=_next_actions_for_audit(audit),
    )
    audit_json.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _audit_curated_report(
    report: Mapping[str, object],
    cfg: CurationPolicyAuditConfig,
    blockers: list[str],
    warnings: list[str],
    evidence: dict[str, object],
) -> None:
    row_count = _as_int(report.get("row_count"))
    action_counts = report.get("action_counts", {})
    if not isinstance(action_counts, Mapping):
        action_counts = {}
    retained_count = sum(_as_int(action_counts.get(action)) for action in RETAINED_ACTIONS)
    evidence["row_count"] = row_count
    evidence["action_counts"] = dict(action_counts)
    evidence["retained_count"] = retained_count

    if row_count <= 0:
        blockers.append("curated report has no rows")
    if retained_count <= 0:
        blockers.append("curated report has no retained keep/downweight rows")

    scan_coverage: dict[str, dict[str, object]] = {}
    for field_name in SCAN_COVERAGE_FIELDS:
        count = _as_int(report.get(field_name))
        ratio = (count / row_count) if row_count else 0.0
        scan_coverage[field_name] = {"count": count, "ratio": round(ratio, 6)}
        if count <= 0:
            blockers.append(f"{field_name} is missing or zero")
        elif row_count and count < row_count:
            message = f"{field_name} covers {count}/{row_count} rows"
            if cfg.allow_representative:
                warnings.append(message + " because representative scan mode is allowed")
            else:
                blockers.append(message + "; full scan coverage is required")
    evidence["scan_coverage"] = scan_coverage

    if cfg.require_clean_report_git and bool(report.get("git_dirty", False)):
        blockers.append("curated report was generated from a dirty git tree")
    evidence["curated_git_sha"] = str(report.get("git_sha", ""))
    evidence["curated_git_dirty"] = bool(report.get("git_dirty", False))

    diversity = report.get("diversity_loss", {})
    if not isinstance(diversity, Mapping):
        diversity = {}
    diversity_evidence: dict[str, object] = {}
    for dimension in cfg.diversity_dimensions:
        dimension_report = diversity.get(dimension, {})
        if not isinstance(dimension_report, Mapping):
            blockers.append(f"missing diversity_loss for {dimension}")
            continue
        lost_groups = _as_int(dimension_report.get("groups_without_retained"))
        total_groups = _as_int(dimension_report.get("total_groups"))
        diversity_evidence[dimension] = {
            "total_groups": total_groups,
            "groups_without_retained": lost_groups,
        }
        if lost_groups > 0:
            blockers.append(f"{dimension} has {lost_groups} groups without retained clips")
    evidence["diversity_loss"] = diversity_evidence


def _audit_threshold_reports(
    reports: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
    cfg: CurationPolicyAuditConfig,
    blockers: list[str],
    warnings: list[str],
    evidence: dict[str, object],
) -> None:
    if not reports:
        blockers.append("no threshold proposal files were provided")
        evidence["threshold_reports"] = []
        return
    policy_accepted = _audit_threshold_policy(policy, cfg, blockers, warnings, evidence)
    if not cfg.thresholds_accepted and not policy_accepted:
        blockers.append("threshold proposals have not been explicitly accepted as a policy")

    threshold_evidence = []
    for index, report in enumerate(reports):
        sample_count = _as_int(report.get("sample_count"))
        proposals = report.get("proposals", [])
        group_by = report.get("group_by", [])
        grouped_rows = report.get("grouped_rows", {})
        if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
            proposals = []
        if not isinstance(group_by, Sequence) or isinstance(group_by, (str, bytes)):
            group_by = []
        if not isinstance(grouped_rows, Mapping):
            grouped_rows = {}
        missing_groups = [field for field in cfg.required_group_by if field not in group_by]
        empty_groups = [
            field
            for field in cfg.required_group_by
            if field in group_by and _as_int(grouped_rows.get(field)) <= 0
        ]
        if sample_count <= 0:
            blockers.append(f"threshold report {index} has no samples")
        if not proposals:
            blockers.append(f"threshold report {index} has no global proposals")
        for field in missing_groups:
            blockers.append(f"threshold report {index} is not grouped by {field}")
        for field in empty_groups:
            blockers.append(f"threshold report {index} has no grouped rows for {field}")
        if sample_count > 0 and sample_count < 1000:
            warnings.append(f"threshold report {index} is based on {sample_count} samples")
        threshold_evidence.append(
            {
                "sample_count": sample_count,
                "proposal_count": len(proposals),
                "group_by": list(group_by),
                "grouped_rows": dict(grouped_rows),
                "lower_metrics": list(report.get("lower_metrics", []))
                if isinstance(report.get("lower_metrics", []), Sequence)
                else [],
            }
        )
    evidence["threshold_reports"] = threshold_evidence


def _audit_threshold_policy(
    policy: Mapping[str, object],
    cfg: CurationPolicyAuditConfig,
    blockers: list[str],
    warnings: list[str],
    evidence: dict[str, object],
) -> bool:
    if not policy:
        evidence["threshold_policy"] = {}
        return False
    status = str(policy.get("status", "")).strip()
    policy_id = str(policy.get("policy_id", "")).strip()
    accepted_by = str(policy.get("accepted_by", "")).strip()
    rationale = str(policy.get("rationale", "")).strip()
    summaries = policy.get("proposal_summaries", [])
    if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes)):
        summaries = []
    evidence["threshold_policy"] = {
        "policy_id": policy_id,
        "status": status,
        "accepted_by": accepted_by,
        "representative": bool(policy.get("representative", False)),
        "proposal_count": _as_int(policy.get("proposal_count")),
        "total_samples": _as_int(policy.get("total_samples")),
        "summary_count": len(summaries),
        "git_sha": str(policy.get("git_sha", "")),
        "git_dirty": bool(policy.get("git_dirty", False)),
    }
    if status != "accepted":
        blockers.append("threshold policy artifact is not accepted")
    if policy_id != cfg.policy_id:
        blockers.append(f"threshold policy ID mismatch: expected {cfg.policy_id}, found {policy_id}")
    if not accepted_by:
        blockers.append("threshold policy artifact is missing accepted_by")
    if not rationale:
        blockers.append("threshold policy artifact is missing rationale")
    if not summaries:
        blockers.append("threshold policy artifact has no proposal summaries")
    if cfg.require_clean_report_git and bool(policy.get("git_dirty", False)):
        blockers.append("threshold policy artifact was generated from a dirty git tree")
    if bool(policy.get("representative", False)) and not cfg.allow_representative:
        warnings.append("threshold policy artifact is marked representative")
    return status == "accepted" and policy_id == cfg.policy_id and bool(accepted_by) and bool(rationale)


def _audit_manual_review(
    report: Mapping[str, object],
    items: Sequence[Mapping[str, object]],
    decision_report: Mapping[str, object],
    cfg: CurationPolicyAuditConfig,
    blockers: list[str],
    warnings: list[str],
    evidence: dict[str, object],
) -> None:
    if not report:
        blockers.append("manual review report is missing")
        evidence["manual_review"] = {}
        return

    reviewed_rows = _as_int(report.get("reviewed_rows"))
    family_counts = report.get("family_counts", {})
    if not isinstance(family_counts, Mapping):
        family_counts = {}
    evidence["manual_review"] = {
        "reviewed_rows": reviewed_rows,
        "family_counts": dict(family_counts),
        "manifest_items": len(items),
        "decision_report_present": bool(decision_report),
    }
    if reviewed_rows <= 0:
        blockers.append("manual review report has no reviewed rows")
    if cfg.require_clean_report_git and bool(report.get("git_dirty", False)):
        blockers.append("manual review report was generated from a dirty git tree")

    if not cfg.require_review_decisions:
        if not items:
            warnings.append("manual review decisions were not verified because manifest is missing")
        return

    if not items:
        blockers.append("review manifest JSONL is required to verify manual decisions")
        return

    incomplete = []
    invalid_actions = []
    for item in items:
        fields = item.get("review_fields", {})
        if not isinstance(fields, Mapping):
            incomplete.append(str(item.get("review_id", "")))
            continue
        decision = str(fields.get("decision", "")).strip()
        recommended_action = str(fields.get("recommended_action", "")).strip()
        if not decision or not recommended_action:
            incomplete.append(str(item.get("review_id", "")))
        elif recommended_action not in RECOMMENDED_REVIEW_ACTIONS:
            invalid_actions.append(str(item.get("review_id", "")))
    evidence["manual_review"]["complete_decisions"] = len(items) - len(incomplete)
    evidence["manual_review"]["incomplete_decisions"] = len(incomplete)
    evidence["manual_review"]["invalid_recommended_actions"] = len(invalid_actions)
    if decision_report:
        evidence["manual_review"]["decision_report"] = {
            "decision_rows": _as_int(decision_report.get("decision_rows")),
            "matched_decisions": _as_int(decision_report.get("matched_decisions")),
            "complete_decisions": _as_int(decision_report.get("complete_decisions")),
            "incomplete_decisions": _as_int(decision_report.get("incomplete_decisions")),
            "git_sha": str(decision_report.get("git_sha", "")),
            "git_dirty": bool(decision_report.get("git_dirty", False)),
        }
        if cfg.require_clean_report_git and bool(decision_report.get("git_dirty", False)):
            blockers.append("manual review decision report was generated from a dirty git tree")
    if incomplete:
        sample = ", ".join(identifier for identifier in incomplete[:5] if identifier)
        suffix = f": {sample}" if sample else ""
        blockers.append(f"{len(incomplete)} manual review items lack decisions{suffix}")
    if invalid_actions:
        sample = ", ".join(identifier for identifier in invalid_actions[:5] if identifier)
        suffix = f": {sample}" if sample else ""
        blockers.append(f"{len(invalid_actions)} manual review items have invalid actions{suffix}")


def discover_threshold_proposals_for_run(curated_run_dir: Path) -> list[Path]:
    """Find threshold proposal files referenced by a curated run report."""

    curated_report_json = curated_run_dir.expanduser() / "curated_report.json"
    if not curated_report_json.exists():
        raise FileNotFoundError(f"missing curated report: {curated_report_json}")
    return discover_threshold_proposals_from_report(_read_json(curated_report_json))


def discover_threshold_proposals_from_report(curated_report: Mapping[str, object]) -> list[Path]:
    """Find proposal JSONs whose stats path matches a curated report stats path."""

    stats_paths = {
        str(curated_report.get("source_stats_jsonl", "")).strip(),
        str(curated_report.get("source_fk_stats_jsonl", "")).strip(),
        str(curated_report.get("g1_stats_jsonl", "")).strip(),
        str(curated_report.get("pair_stats_jsonl", "")).strip(),
    }
    stats_paths.discard("")
    proposals: list[Path] = []
    seen: set[Path] = set()
    for stats_path in sorted(stats_paths):
        stats = Path(stats_path)
        if not stats.parent.exists():
            continue
        for candidate in sorted(stats.parent.glob("*threshold*proposals*.json")):
            try:
                payload = _read_json(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if str(payload.get("stats_jsonl", "")).strip() != stats_path:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            proposals.append(candidate)
    return proposals


def _existing_path(path: Path | None) -> Path | None:
    if path and path.exists():
        return path
    return None


def _next_actions_for_audit(audit: CurationPolicyAuditResult) -> list[str]:
    actions: list[str] = []
    blocker_text = "\n".join(audit.blockers)
    if "full scan coverage is required" in blocker_text:
        actions.append(
            "Run full or explicitly accepted representative source/source-FK/G1/pair quality scans."
        )
    if "threshold proposals" in blocker_text or "threshold policy" in blocker_text:
        actions.append(
            "Accept reviewed threshold proposals into a threshold_policy.json for this policy ID."
        )
    if "manual review" in blocker_text or "review manifest" in blocker_text:
        actions.append(
            "Complete manual review decisions and merge them into a reviewed manifest/report."
        )
    if "diversity_loss" in blocker_text or "groups without retained clips" in blocker_text:
        actions.append(
            "Revise the curation policy to retain actor/skeleton/category/split diversity."
        )
    if "dirty git tree" in blocker_text:
        actions.append("Regenerate policy artifacts from a clean committed git state.")
    if not actions and not audit.promotable:
        actions.append("Inspect audit blockers and regenerate the missing policy evidence.")
    if audit.promotable:
        actions.append("Use this policy audit in formal M5 training config as data.quality_policy_audit.")
    return list(dict.fromkeys(actions))


def _read_json(path: Path | None) -> dict[str, object]:
    if not path:
        return {}
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _read_jsonl(path: Path | None) -> list[dict[str, object]]:
    if not path:
        return []
    rows = []
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
