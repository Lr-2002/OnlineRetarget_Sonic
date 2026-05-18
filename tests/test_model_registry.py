import unittest

from online_retarget.data.schema import ObservationSpec
from online_retarget.models.registry import build_model


class ModelRegistryTests(unittest.TestCase):
    def test_builds_configured_model_families(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        spec = ObservationSpec(history_frames=2, source_body_count=1, include_morphology=False)
        input_dim = spec.flattened_dim()
        output_dim = 3
        configs = [
            {"model": {"family": "mlp", "hidden_dims": [8]}},
            {"model": {"family": "tf", "d_model": 8, "nhead": 2, "num_layers": 1}},
            {
                "model": {
                    "family": "fm",
                    "hidden_dims": [8],
                    "time_embed_dim": 4,
                    "inference_steps": 2,
                }
            },
        ]
        x = torch.zeros(4, input_dim)
        for config in configs:
            built = build_model(
                config,
                input_dim=input_dim,
                output_dim=output_dim,
                observation_spec=spec,
            )
            if built.family == "flow_matching":
                y = built.model.sample(x, steps=1)
            else:
                y = built.model(x)
            self.assertEqual(tuple(y.shape), (4, output_dim))


if __name__ == "__main__":
    unittest.main()
