import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from online_retarget.sonic_losses import ActionSmoothnessLoss, G1DynamicsActionLoss


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for Sonic loss tests")
class SonicLossTests(unittest.TestCase):
    def test_g1_dynamics_loss_uses_g1_dyn_action_output(self):
        loss_fn = G1DynamicsActionLoss()
        loss_inputs = {
            "decoded_outputs": {
                "g1_dyn": {"action": torch.tensor([[1.0, 3.0]])},
                "g1_kin": {"action": torch.tensor([[100.0, 100.0]])},
            },
            "action_mean": torch.tensor([[50.0, 50.0]]),
            "tokenizer_obs": {"g1_target_action": torch.tensor([[1.0, 1.0]])},
        }

        loss = loss_fn(loss_inputs)

        self.assertAlmostEqual(float(loss), 2.0)

    def test_g1_dynamics_loss_accepts_body_action_decoder_output(self):
        loss_fn = G1DynamicsActionLoss()
        loss_inputs = {
            "decoded_outputs": {"g1_dyn": {"body_action": torch.tensor([[1.0, 3.0]])}},
            "action_mean": torch.tensor([[50.0, 50.0]]),
            "tokenizer_obs": {"g1_target_action": torch.tensor([[1.0, 1.0]])},
        }

        loss = loss_fn(loss_inputs)

        self.assertAlmostEqual(float(loss), 2.0)

    def test_g1_dynamics_loss_accepts_meta_action_decoder_output(self):
        loss_fn = G1DynamicsActionLoss()
        loss_inputs = {
            "decoded_outputs": {"g1_dyn": {"meta_action": torch.tensor([[1.0, 3.0]])}},
            "action_mean": torch.tensor([[50.0, 50.0]]),
            "tokenizer_obs": {"g1_target_action": torch.tensor([[1.0, 1.0]])},
        }

        loss = loss_fn(loss_inputs)

        self.assertAlmostEqual(float(loss), 2.0)

    def test_g1_dynamics_loss_aligns_single_step_target_to_temporal_prediction(self):
        loss_fn = G1DynamicsActionLoss()
        loss_inputs = {
            "decoded_outputs": {
                "g1_dyn": {
                    "action": torch.tensor(
                        [
                            [
                                [1.0, 2.0],
                                [1.0, 4.0],
                            ]
                        ]
                    )
                }
            },
            "action_mean": torch.tensor([[50.0, 50.0]]),
            "tokenizer_obs": {"g1_target_action": torch.tensor([[1.0, 1.0]])},
        }

        loss = loss_fn(loss_inputs)

        self.assertAlmostEqual(float(loss), 2.5)

    def test_g1_dynamics_loss_raises_when_g1_dyn_decoder_is_missing(self):
        loss_fn = G1DynamicsActionLoss()
        loss_inputs = {
            "decoded_outputs": {"g1_kin": {"action": torch.tensor([[1.0, 3.0]])}},
            "action_mean": torch.tensor([[1.0, 3.0]]),
            "tokenizer_obs": {"g1_target_action": torch.tensor([[1.0, 1.0]])},
        }

        with self.assertRaisesRegex(KeyError, "g1_dyn"):
            loss_fn(loss_inputs)

    def test_g1_dynamics_loss_raises_when_g1_dyn_action_is_missing(self):
        loss_fn = G1DynamicsActionLoss()
        loss_inputs = {
            "decoded_outputs": {"g1_dyn": {"latent": torch.tensor([[1.0, 3.0]])}},
            "action_mean": torch.tensor([[1.0, 3.0]]),
            "tokenizer_obs": {"g1_target_action": torch.tensor([[1.0, 1.0]])},
        }

        with self.assertRaisesRegex(KeyError, "available outputs: latent"):
            loss_fn(loss_inputs)

    def test_action_smoothness_loss_uses_g1_dyn_and_penalizes_temporal_jumps(self):
        loss_fn = ActionSmoothnessLoss()
        loss_inputs = {
            "decoded_outputs": {
                "g1_dyn": {
                    "action": torch.tensor(
                        [
                            [
                                [1.0, 1.0],
                                [2.0, 3.0],
                                [2.0, 5.0],
                            ]
                        ]
                    )
                }
            },
            "action_mean": torch.zeros(1, 3, 2),
            "tokenizer_obs": {},
        }

        loss = loss_fn(loss_inputs)

        self.assertAlmostEqual(float(loss), 2.25)


if __name__ == "__main__":
    unittest.main()
