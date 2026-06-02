import unittest

from online_retarget.metrics import compute_metric_bundle, metric_metadata


class HumanoidEvalRegistryTests(unittest.TestCase):
    def test_registry_exposes_formulas_method_coverage_and_not_applicable(self):
        metadata = metric_metadata(
            (
                "global_body_position_error",
                "root_relative_body_position_error",
                "joint_jump_count",
            )
        )

        self.assertEqual(
            metadata["global_body_position_error"]["formula"],
            "mean_l2_error(world_body_position)",
        )
        self.assertIn("gmr", metadata["global_body_position_error"]["method_coverage"])
        self.assertIn("phc", metadata["global_body_position_error"]["method_coverage"])
        self.assertIn("nmr", metadata["root_relative_body_position_error"]["method_coverage"])

        bundle = compute_metric_bundle({"method_id": "gmr"}, ("joint_jump_count",))

        self.assertEqual(bundle["joint_jump_count"].status, "not_applicable")
        self.assertIn("method_id=gmr", bundle["joint_jump_count"].reason)


if __name__ == "__main__":
    unittest.main()
