import unittest


class DiffusionPolicyUNetSmallTests(unittest.TestCase):
    def test_forward_loss_and_sample_are_finite(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from online_retarget.models.diffusion_policy_unet import DiffusionPolicyUNetSmall

        torch.manual_seed(7)
        model = DiffusionPolicyUNetSmall(
            action_dim=3,
            reference_body_token_dim=4,
            reference_history_frames=5,
            reference_body_count=2,
            robot_state_dim=6,
            down_dims=(8, 16),
            condition_dim=16,
            diffusion_step_embed_dim=8,
            kernel_size=3,
            groups=4,
            diffusion_steps=4,
            inference_steps=4,
            max_action_horizon=4,
            output_mode="residual_prev_action",
        )
        reference_history_tokens = torch.randn(2, 5, 2, 4)
        noisy_action = torch.randn(2, 4, 3)
        target_action = torch.randn(2, 4, 3)
        robot_state = torch.randn(2, 6)
        prev_action = torch.randn(2, 3)
        timesteps = torch.tensor([0, 3], dtype=torch.long)

        pred_noise = model(
            reference_history_tokens,
            noisy_action,
            timesteps,
            robot_state=robot_state,
            prev_action=prev_action,
        )
        loss = model.diffusion_loss(
            reference_history_tokens,
            target_action,
            robot_state=robot_state,
            prev_action=prev_action,
            noise=torch.zeros_like(target_action),
            timesteps=torch.zeros(2, dtype=torch.long),
        )
        sample = model.sample(
            reference_history_tokens,
            robot_state=robot_state,
            prev_action=prev_action,
            action_horizon=4,
            steps=4,
            start="zeros",
        )

        self.assertEqual(tuple(pred_noise.shape), (2, 4, 3))
        self.assertEqual(tuple(loss.shape), ())
        self.assertEqual(tuple(sample.shape), (2, 4, 3))
        self.assertTrue(torch.isfinite(pred_noise).all())
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(sample).all())

    def test_requires_causal_reference_history_shape(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from online_retarget.models.diffusion_policy_unet import DiffusionPolicyUNetSmall

        model = DiffusionPolicyUNetSmall(
            action_dim=2,
            reference_body_token_dim=3,
            reference_history_frames=5,
            reference_body_count=2,
            robot_state_dim=0,
            down_dims=(8,),
            condition_dim=8,
            diffusion_step_embed_dim=4,
            diffusion_steps=4,
            max_action_horizon=2,
            output_mode="absolute",
        )

        with self.assertRaisesRegex(ValueError, "reference_history_tokens"):
            model(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 2),
                torch.zeros(1, dtype=torch.long),
            )

    def test_rejects_skipped_timestep_sampling(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from online_retarget.models.diffusion_policy_unet import DiffusionPolicyUNetSmall

        model = DiffusionPolicyUNetSmall(
            action_dim=2,
            reference_body_token_dim=3,
            reference_history_frames=5,
            reference_body_count=2,
            robot_state_dim=0,
            down_dims=(8,),
            condition_dim=8,
            diffusion_step_embed_dim=4,
            diffusion_steps=4,
            inference_steps=4,
            max_action_horizon=2,
            output_mode="absolute",
        )

        with self.assertRaisesRegex(ValueError, "inference_steps == diffusion_steps"):
            model.sample(
                torch.zeros(1, 5, 2, 3),
                action_horizon=2,
                steps=2,
                start="zeros",
            )

    def test_requires_exact_robot_state_when_enabled(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from online_retarget.models.diffusion_policy_unet import DiffusionPolicyUNetSmall

        model = DiffusionPolicyUNetSmall(
            action_dim=2,
            reference_body_token_dim=3,
            reference_history_frames=5,
            reference_body_count=2,
            robot_state_dim=4,
            down_dims=(8,),
            condition_dim=8,
            diffusion_step_embed_dim=4,
            diffusion_steps=4,
            max_action_horizon=2,
            output_mode="absolute",
        )
        reference_history_tokens = torch.zeros(1, 5, 2, 3)
        noisy_action = torch.zeros(1, 2, 2)
        timesteps = torch.zeros(1, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "robot_state is required"):
            model(reference_history_tokens, noisy_action, timesteps)
        with self.assertRaisesRegex(ValueError, "robot_state width"):
            model(
                reference_history_tokens,
                noisy_action,
                timesteps,
                robot_state=torch.zeros(1, 3),
            )
