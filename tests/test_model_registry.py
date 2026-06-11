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
                    "family": "token_transformer",
                    "latent_dim": 8,
                    "nhead": 2,
                    "num_encoder_layers": 1,
                    "num_decoder_layers": 1,
                    "dim_feedforward": 16,
                }
            },
            {
                "model": {
                    "family": "fm",
                    "hidden_dims": [8],
                    "time_embed_dim": 4,
                    "inference_steps": 2,
                }
            },
            {
                "model": {
                    "family": "dp",
                    "hidden_dims": [8],
                    "time_embed_dim": 4,
                    "diffusion_steps": 4,
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
            elif built.family == "diffusion_policy":
                y = built.model.sample(x, steps=1, start="zeros")
                loss = built.model.diffusion_loss(
                    x,
                    torch.zeros(4, output_dim),
                    noise=torch.zeros(4, output_dim),
                    timesteps=torch.zeros(4, dtype=torch.long),
                )
                self.assertEqual(tuple(loss.shape), ())
            else:
                y = built.model(x)
            self.assertEqual(tuple(y.shape), (4, output_dim))

    def test_token_vae_forward_and_loss(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from online_retarget.models.token_vae import MLPTokenVAE, vae_loss

        model = MLPTokenVAE(input_dim=5, latent_dim=3, hidden_dims=(7,))
        x = torch.zeros(4, 5)

        reconstruction, mu, logvar, latent = model(x, sample=False)
        loss, reconstruction_mse, kl = vae_loss(reconstruction, x, mu, logvar, beta=1.0e-4)

        self.assertEqual(tuple(reconstruction.shape), (4, 5))
        self.assertEqual(tuple(mu.shape), (4, 3))
        self.assertEqual(tuple(logvar.shape), (4, 3))
        self.assertEqual(tuple(latent.shape), (4, 3))
        self.assertEqual(tuple(loss.shape), ())
        self.assertEqual(tuple(reconstruction_mse.shape), ())
        self.assertEqual(tuple(kl.shape), ())


if __name__ == "__main__":
    unittest.main()
