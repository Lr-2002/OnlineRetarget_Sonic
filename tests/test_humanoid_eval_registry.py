import unittest

from online_retarget.humanoid_eval_registry import (
    MISSING,
    NOT_APPLICABLE,
    METHODS,
    PROTOCOLS,
    make_result,
    metric_spec,
    missing_result,
    not_applicable_result,
    registry_manifest,
)


class HumanoidEvalRegistryTests(unittest.TestCase):
    def test_registry_keeps_methods_and_harnesses_separate(self):
        manifest = registry_manifest()

        self.assertEqual({"gmr", "nmr", "phc"} <= set(METHODS), True)
        self.assertEqual(PROTOCOLS["gmr_lafan1_beyondmimic"]["harness"], "beyondmimic")
        self.assertNotIn("beyondmimic", METHODS)
        self.assertEqual(manifest["schema_version"], "humanoid_retarget_eval.v1")

    def test_aliases_resolve_without_collapsing_protocol_metrics(self):
        self.assertEqual(metric_spec("W-MPJPE").metric_id, "weighted_mpjpe")
        self.assertEqual(metric_spec("joint_pos_rmse_raw").metric_id, "g1_joint_pos_rmse_rad")

    def test_missing_and_not_applicable_are_not_numeric_zero(self):
        missing = missing_result(
            method_id="gmr",
            protocol_id="gmr_lafan1_beyondmimic",
            sequence_id="seq",
            metric_id="policy_success",
            source="paper_table",
            notes="expected but not reported",
        )
        not_applicable = not_applicable_result(
            method_id="online_retarget_a0",
            protocol_id="sonic_kin_soma_motionlib_training",
            sequence_id="seq",
            metric_id="policy_success",
            source="training_time",
            notes="no simulator policy rollout",
        )

        self.assertIsNone(missing.value)
        self.assertEqual(missing.status, MISSING)
        self.assertIsNone(not_applicable.value)
        self.assertEqual(not_applicable.status, NOT_APPLICABLE)

    def test_measured_result_uses_metric_unit_and_direction(self):
        result = make_result(
            method_id="online_retarget_a0",
            protocol_id="sonic_kin_soma_motionlib_training",
            sequence_id="step_1",
            metric_id="g1_joint_pos_rmse_rad",
            value=0.25,
            source="validation",
        )

        self.assertEqual(result.unit, "radian")
        self.assertEqual(result.direction, "lower_is_better")
        self.assertEqual(result.value, 0.25)


if __name__ == "__main__":
    unittest.main()
