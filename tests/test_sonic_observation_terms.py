from __future__ import annotations

import unittest

try:
    import torch

    from online_retarget.sonic_observation_terms import root_pos_w_mf, root_rot_w_mf
except ModuleNotFoundError:
    torch = None


class _CommandManager:
    def __init__(self, command: object) -> None:
        self._command = command

    def get_term(self, name: str) -> object:
        if name != "motion":
            raise KeyError(name)
        return self._command


class _Env:
    def __init__(self, command: object) -> None:
        self.command_manager = _CommandManager(command)


class _Command:
    num_envs = 2
    num_future_frames = 2
    smpl_num_future_frames = 2
    device = torch.device("cpu") if torch is not None else None


@unittest.skipIf(torch is None, "torch is not installed")
class SonicObservationTermTests(unittest.TestCase):
    def test_root_pos_handles_flat_multi_future_body_tensor(self) -> None:
        command = _Command()
        body_pos = torch.tensor(
            [
                [
                    [[1.0, 2.0, 0.5], [10.0, 20.0, 1.0], [100.0, 200.0, 2.0]],
                    [[2.0, 4.0, 0.6], [13.0, 25.0, 1.5], [101.0, 201.0, 2.5]],
                ],
                [
                    [[3.0, 6.0, 0.7], [30.0, 40.0, 1.2], [102.0, 202.0, 2.2]],
                    [[4.0, 8.0, 0.8], [34.0, 47.0, 1.7], [103.0, 203.0, 2.7]],
                ],
            ]
        )
        command.body_pos_w_multi_future = body_pos.reshape(command.num_envs, -1)

        actual = root_pos_w_mf(_Env(command), "motion", root_body_index=1)

        expected = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [3.0, 5.0, 1.5]],
                [[0.0, 0.0, 1.2], [4.0, 7.0, 1.7]],
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_root_rot_handles_flat_multi_future_body_quat_tensor(self) -> None:
        command = _Command()
        quats = torch.zeros(command.num_envs, command.num_future_frames, 3, 4)
        quats[:, :, :, 0] = 1.0
        command.body_quat_w_multi_future = quats.reshape(command.num_envs, -1)

        actual = root_rot_w_mf(_Env(command), "motion", root_body_index=1)

        expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand(
            command.num_envs,
            command.num_future_frames,
            6,
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
