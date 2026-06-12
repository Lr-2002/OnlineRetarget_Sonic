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

    def test_builds_temporal_diffusion_policy_preserves_action_horizon(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        spec = ObservationSpec(history_frames=2, source_body_count=2)
        built = build_model(
            {
                "model": {
                    "family": "temporal_diffusion_policy",
                    "d_model": 8,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 16,
                    "time_embed_dim": 4,
                    "diffusion_steps": 4,
                    "inference_steps": 1,
                    "action_dim": 2,
                    "source_body_token_dim": 15,
                    "source_skeleton_dim": 8,
                    "morphology_dim": 2,
                    "robot_state_dim": 3,
                    "max_horizon": 4,
                    "output_mode": "residual_prev_action",
                }
            },
            input_dim=spec.flattened_dim(),
            output_dim=2,
            observation_spec=spec,
        )
        source_body_tokens = torch.zeros(3, 2, 2, 15)
        source_skeleton = torch.zeros(3, 8)
        morphology = torch.zeros(3, 2)
        robot_state = torch.zeros(3, 3)
        prev_action = torch.zeros(3, 2)
        target = torch.zeros(3, 2, 2)

        loss = built.model.diffusion_loss(
            source_body_tokens,
            target,
            source_skeleton=source_skeleton,
            morphology=morphology,
            robot_state=robot_state,
            prev_action=prev_action,
            noise=torch.zeros_like(target),
            timesteps=torch.zeros(3, dtype=torch.long),
            loss_config={
                "denoise": 1.0,
                "x0_reconstruction": 0.1,
                "velocity": 0.1,
                "acceleration": 0.1,
                "joint_jump": 0.1,
            },
        )
        prediction = built.model.sample(
            source_body_tokens,
            source_skeleton=source_skeleton,
            morphology=morphology,
            robot_state=robot_state,
            prev_action=prev_action,
            steps=1,
            start="zeros",
        )

        self.assertEqual(built.family, "temporal_diffusion_policy")
        self.assertEqual(built.model.output_mode, "residual_prev_action")
        self.assertEqual(tuple(loss.shape), ())
        self.assertEqual(tuple(prediction.shape), (3, 2, 2))
        target = torch.tensor([[[1.0, 2.0], [1.5, 1.0]]])
        prev_action = torch.tensor([[0.5, 1.5]])
        residual = built.model._to_model_action(target, prev_action)
        reconstructed = built.model._from_model_action(residual, prev_action)
        self.assertTrue(torch.equal(residual, torch.tensor([[[0.5, 0.5], [1.0, -0.5]]])))
        self.assertTrue(torch.equal(reconstructed, target))

    def test_builds_diffusion_policy_unet_small(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        spec = ObservationSpec(history_frames=5, source_body_count=2)
        built = build_model(
            {
                "model": {
                    "family": "diffusion_policy_unet_small",
                    "action_dim": 2,
                    "reference_body_token_dim": 3,
                    "reference_history_frames": 5,
                    "reference_body_count": 2,
                    "robot_state_dim": 4,
                    "down_dims": [8, 16],
                    "condition_dim": 16,
                    "diffusion_step_embed_dim": 8,
                    "kernel_size": 3,
                    "n_groups": 4,
                    "diffusion_steps": 4,
                    "inference_steps": 4,
                    "max_action_horizon": 2,
                    "output_mode": "residual_prev_action",
                }
            },
            input_dim=spec.flattened_dim(),
            output_dim=2,
            observation_spec=spec,
        )
        reference_history_tokens = torch.zeros(3, 5, 2, 3)
        robot_state = torch.zeros(3, 4)
        prev_action = torch.zeros(3, 2)
        target = torch.zeros(3, 2, 2)

        loss = built.model.diffusion_loss(
            reference_history_tokens,
            target,
            robot_state=robot_state,
            prev_action=prev_action,
            noise=torch.zeros_like(target),
            timesteps=torch.zeros(3, dtype=torch.long),
        )
        prediction = built.model.sample(
            reference_history_tokens,
            robot_state=robot_state,
            prev_action=prev_action,
            action_horizon=2,
            steps=4,
            start="zeros",
        )

        self.assertEqual(built.family, "diffusion_policy_unet_small")
        self.assertEqual(tuple(loss.shape), ())
        self.assertEqual(tuple(prediction.shape), (3, 2, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_temporal_jump_loss_uses_velocity_threshold_units(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        model = _temporal_policy(torch, action_dim=1)
        prediction = torch.tensor([[[0.0], [0.5]]])
        target = torch.zeros_like(prediction)

        loss = model._stability_loss(
            prediction,
            target,
            {"joint_jump": 1.0, "joint_jump_velocity": 20.0},
            fps=torch.tensor([50.0]),
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_temporal_delta_smoothness_is_distinct_from_velocity_loss(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        model = _temporal_policy(torch, action_dim=1)
        prediction = torch.tensor([[[0.0], [0.2], [0.4]]])
        target = torch.tensor([[[1.0], [1.2], [1.4]]])
        prev_action = torch.zeros(1, 1)

        velocity_loss = model._stability_loss(
            prediction,
            target,
            {"velocity": 1.0},
            prev_action=prev_action,
        )
        delta_loss = model._stability_loss(
            prediction,
            target,
            {"delta_smoothness": 1.0},
            prev_action=prev_action,
        )

        self.assertAlmostEqual(float(velocity_loss.item()), 0.0, places=6)
        self.assertGreater(float(delta_loss.item()), 0.0)

def _temporal_policy(torch, *, action_dim: int):
    from online_retarget.models.temporal import TemporalDiffusionPolicyRetargeter

    return TemporalDiffusionPolicyRetargeter(
        action_dim=action_dim,
        source_body_token_dim=3,
        source_skeleton_dim=0,
        morphology_dim=0,
        robot_state_dim=0,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        time_embed_dim=4,
        diffusion_steps=4,
        inference_steps=1,
        max_horizon=4,
    )


if __name__ == "__main__":
    unittest.main()
