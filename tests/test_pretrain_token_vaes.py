import unittest

from online_retarget.data.schema import ObservationSpec

try:
    import scripts.pretrain_token_vaes as pretrain_token_vaes
except ModuleNotFoundError as exc:  # pragma: no cover - minimal env without torch.
    if exc.name != "torch":
        raise
    pretrain_token_vaes = None


class PretrainTokenVaeTests(unittest.TestCase):
    def test_component_tensors_follow_observation_spec_slices(self):
        if pretrain_token_vaes is None:
            self.skipTest("torch is not installed")
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        spec = ObservationSpec(history_frames=2, source_body_count=1, include_morphology=True)
        source_dim = spec.source_feature_dim()
        morphology_dim = spec.morphology_dim()
        observation = [float(value) for value in range(spec.flattened_dim())]
        samples = [{"observation": observation, "target_joints": [0.1, 0.2, 0.3]}]

        tensors = pretrain_token_vaes._component_tensors(
            torch,
            samples=samples,
            observation_spec=spec,
        )

        self.assertEqual(tuple(tensors["motion"].shape), (1, source_dim))
        self.assertEqual(tuple(tensors["skeleton"].shape), (1, morphology_dim))
        self.assertEqual(tuple(tensors["action"].shape), (1, 3))
        self.assertEqual(tensors["motion"][0, 0].item(), 0.0)
        self.assertEqual(tensors["skeleton"][0, 0].item(), float(source_dim))
        self.assertAlmostEqual(tensors["action"][0, 2].item(), 0.3, places=6)

    def test_component_config_parses_string_and_list(self):
        if pretrain_token_vaes is None:
            self.skipTest("torch is not installed")
        self.assertEqual(
            pretrain_token_vaes._components_from_config({"pretrain": {"components": "motion"}}),
            ("motion",),
        )
        self.assertEqual(
            pretrain_token_vaes._components_from_config(
                {"pretrain": {"components": ["skeleton", "action"]}}
            ),
            ("skeleton", "action"),
        )

    def test_wandb_mode_override_does_not_mutate_input(self):
        if pretrain_token_vaes is None:
            self.skipTest("torch is not installed")
        config = {"tracking": {"wandb_mode": "disabled"}}

        updated = pretrain_token_vaes._apply_wandb_mode_override(config, "online")

        self.assertEqual(updated["tracking"]["wandb_mode"], "online")
        self.assertEqual(config["tracking"]["wandb_mode"], "disabled")


if __name__ == "__main__":
    unittest.main()
