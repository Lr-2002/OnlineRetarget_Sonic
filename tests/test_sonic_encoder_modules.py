import unittest
import importlib.util

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from online_retarget.sonic_encoder_modules import (
        AdapterSomaEncoderModule,
        ConcatSomaEncoderModule,
        ExpertSomaEncoderModule,
        FilmSomaEncoderModule,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for Sonic encoder module tests")
class SonicEncoderModuleTests(unittest.TestCase):
    def test_concat_encoder_preserves_sonic_temporal_output_shape(self):
        module = ConcatSomaEncoderModule(
            input_dim=5,
            output_dim=4,
            hidden_dims=(8,),
            num_input_temporal_dims=2,
            num_output_temporal_dims=2,
        )
        output = module(torch.randn(3, 2, 5))

        self.assertEqual(tuple(output.shape), (3, 2, 4))

    def test_film_encoder_splits_conditioning_per_frame_before_flatten(self):
        module = FilmSomaEncoderModule(
            input_dim=5,
            output_dim=4,
            conditioning_dim=2,
            hidden_dim=8,
            num_layers=2,
            num_input_temporal_dims=2,
            num_output_temporal_dims=2,
        )
        x = torch.zeros(3, 2, 5)
        x[..., :3] = 1.0
        x[..., 3:] = 0.5

        motion, conditioning = module._split_motion_conditioning(x, 2)
        output = module(x)

        self.assertEqual(tuple(motion.shape), (3, 6))
        self.assertEqual(tuple(conditioning.shape), (3, 4))
        self.assertTrue(torch.all(conditioning == 0.5))
        self.assertEqual(tuple(output.shape), (3, 2, 4))

    def test_adapter_routes_from_morphology_cluster_scalar(self):
        module = AdapterSomaEncoderModule(
            input_dim=5,
            output_dim=4,
            conditioning_dim=2,
            hidden_dim=8,
            adapter_dim=4,
            num_adapters=4,
            num_input_temporal_dims=2,
            num_output_temporal_dims=2,
        )
        x = torch.zeros(3, 2, 5)
        x[:, :, :3] = 1.0
        x[0, :, 4] = 0.0
        x[1, :, 4] = 1.0 / 3.0
        x[2, :, 4] = 1.0

        output = module(x)

        self.assertEqual(tuple(output.shape), (3, 2, 4))
        self.assertEqual(module.last_routes.tolist(), [0, 1, 3])
        self.assertEqual(module.route_summary()["counts"], [1, 1, 0, 1])

    def test_expert_routes_from_morphology_cluster_scalar(self):
        module = ExpertSomaEncoderModule(
            input_dim=5,
            output_dim=4,
            conditioning_dim=2,
            hidden_dim=8,
            num_experts=4,
            num_input_temporal_dims=2,
            num_output_temporal_dims=2,
        )
        x = torch.zeros(2, 2, 5)
        x[:, :, :3] = 1.0
        x[0, :, 4] = 2.0 / 3.0
        x[1, :, 4] = 1.0

        output = module(x)

        self.assertEqual(tuple(output.shape), (2, 2, 4))
        self.assertEqual(module.last_routes.tolist(), [2, 3])
        self.assertEqual(module.route_summary()["counts"], [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
