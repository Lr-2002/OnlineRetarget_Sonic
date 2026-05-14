import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.policy_audit import (
    CurationPolicyAuditConfig,
    audit_curation_policy,
)
from online_retarget.data.review_manifest import merge_review_decisions


class CurationPolicyAuditTests(unittest.TestCase):
    def test_blocks_representative_unaccepted_policy_with_incomplete_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curated = _write_curated_report(root / "curated_report.json", row_count=10, scanned=3)
            thresholds = _write_thresholds(root / "thresholds.json", sample_count=3)
            review_report = _write_review_report(root / "review_report.json")
            manifest = _write_review_manifest(root / "review_manifest.jsonl", complete=False)

            result = audit_curation_policy(
                curated_report_json=curated,
                threshold_proposal_jsons=(thresholds,),
                review_report_json=review_report,
                review_manifest_jsonl=manifest,
                config=CurationPolicyAuditConfig(policy_id="candidate_v0"),
            )

        self.assertFalse(result.promotable)
        blockers = "\n".join(result.blockers)
        self.assertIn("full scan coverage is required", blockers)
        self.assertIn("threshold proposals have not been explicitly accepted", blockers)
        self.assertIn("manual review items lack decisions", blockers)
        self.assertEqual(result.evidence["scan_coverage"]["merged_g1_rows"]["count"], 3)

    def test_promotes_when_policy_evidence_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curated = _write_curated_report(root / "curated_report.json", row_count=10, scanned=10)
            thresholds = _write_thresholds(root / "thresholds.json", sample_count=10)
            review_report = _write_review_report(root / "review_report.json")
            manifest = _write_review_manifest(root / "review_manifest.jsonl", complete=True)
            output = root / "audit.json"

            result = audit_curation_policy(
                curated_report_json=curated,
                threshold_proposal_jsons=(thresholds,),
                review_report_json=review_report,
                review_manifest_jsonl=manifest,
                output_json=output,
                config=CurationPolicyAuditConfig(
                    policy_id="candidate_v1",
                    thresholds_accepted=True,
                ),
            )

            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(result.promotable)
        self.assertEqual(result.blockers, [])
        self.assertEqual(written["status"], "promotable")

    def test_representative_mode_warns_instead_of_blocking_scan_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curated = _write_curated_report(root / "curated_report.json", row_count=10, scanned=3)
            thresholds = _write_thresholds(root / "thresholds.json", sample_count=3)
            review_report = _write_review_report(root / "review_report.json")
            manifest = _write_review_manifest(root / "review_manifest.jsonl", complete=True)

            result = audit_curation_policy(
                curated_report_json=curated,
                threshold_proposal_jsons=(thresholds,),
                review_report_json=review_report,
                review_manifest_jsonl=manifest,
                config=CurationPolicyAuditConfig(
                    policy_id="candidate_representative",
                    allow_representative=True,
                    thresholds_accepted=True,
                ),
            )

        self.assertTrue(result.promotable)
        self.assertIn("representative scan mode is allowed", "\n".join(result.warnings))

    def test_audit_accepts_manifest_after_decision_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curated = _write_curated_report(root / "curated_report.json", row_count=10, scanned=10)
            thresholds = _write_thresholds(root / "thresholds.json", sample_count=10)
            review_report = _write_review_report(root / "review_report.json")
            manifest = _write_review_manifest(root / "review_manifest.jsonl", complete=False)
            decisions = root / "decisions.jsonl"
            decisions.write_text(
                json.dumps(
                    {
                        "review_id": "jump:1:fixture",
                        "decision": "confirmed",
                        "reviewer": "unit-test",
                        "confirmed_issue": "yes",
                        "recommended_action": "quarantine",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            reviewed = merge_review_decisions(
                review_manifest_jsonl=manifest,
                decisions_file=decisions,
                output_jsonl=root / "review_manifest.reviewed.jsonl",
            )

            result = audit_curation_policy(
                curated_report_json=curated,
                threshold_proposal_jsons=(thresholds,),
                review_report_json=review_report,
                review_manifest_jsonl=reviewed.output_jsonl,
                review_decision_report_json=reviewed.report_json,
                config=CurationPolicyAuditConfig(
                    policy_id="candidate_reviewed",
                    thresholds_accepted=True,
                    require_clean_report_git=False,
                ),
            )

        self.assertTrue(result.promotable)
        self.assertEqual(result.evidence["manual_review"]["incomplete_decisions"], 0)
        self.assertEqual(result.evidence["manual_review"]["decision_report"]["matched_decisions"], 1)

    def test_audit_blocks_invalid_review_recommended_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curated = _write_curated_report(root / "curated_report.json", row_count=10, scanned=10)
            thresholds = _write_thresholds(root / "thresholds.json", sample_count=10)
            review_report = _write_review_report(root / "review_report.json")
            manifest = root / "review_manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "review_id": "jump:1:fixture",
                        "review_fields": {
                            "decision": "confirmed",
                            "recommended_action": "maybe",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = audit_curation_policy(
                curated_report_json=curated,
                threshold_proposal_jsons=(thresholds,),
                review_report_json=review_report,
                review_manifest_jsonl=manifest,
                config=CurationPolicyAuditConfig(
                    policy_id="candidate_bad_review",
                    thresholds_accepted=True,
                ),
            )

        self.assertFalse(result.promotable)
        self.assertIn("invalid actions", "\n".join(result.blockers))


def _write_curated_report(path: Path, row_count: int, scanned: int) -> Path:
    payload = {
        "row_count": row_count,
        "merged_source_rows": scanned,
        "merged_source_fk_rows": scanned,
        "merged_g1_rows": scanned,
        "merged_pair_rows": scanned,
        "action_counts": {"keep": row_count - 1, "quarantine": 1},
        "diversity_loss": {
            "actor_uid": {"total_groups": 2, "groups_without_retained": 0},
            "source_skeleton": {"total_groups": 2, "groups_without_retained": 0},
            "category": {"total_groups": 2, "groups_without_retained": 0},
            "split": {"total_groups": 3, "groups_without_retained": 0},
        },
        "git_sha": "abc123",
        "git_dirty": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_thresholds(path: Path, sample_count: int) -> Path:
    payload = {
        "sample_count": sample_count,
        "proposals": [
            {
                "metric": "joint_jump_rate",
                "percentile": 0.95,
                "value": 0.1,
                "action": "quarantine",
                "tail": "upper",
                "comparison": ">",
                "rationale": "fixture",
            }
        ],
        "group_by": ["category", "split"],
        "grouped_rows": {"category": sample_count, "split": sample_count},
        "lower_metrics": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_review_report(path: Path) -> Path:
    payload = {
        "reviewed_rows": 1,
        "family_counts": {"jump": 1},
        "git_sha": "abc123",
        "git_dirty": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_review_manifest(path: Path, complete: bool) -> Path:
    decision = "accept" if complete else ""
    recommended_action = "keep" if complete else ""
    item = {
        "review_id": "jump:1:fixture",
        "review_fields": {
            "decision": decision,
            "recommended_action": recommended_action,
        },
    }
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
