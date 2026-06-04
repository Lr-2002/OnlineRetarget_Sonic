from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.isaac_src_metrics import (
    METRIC_SCHEMA_VERSION,
    aggregate_packet_metrics,
)


class IsaacSrcMetricAggregationTests(unittest.TestCase):
    def test_aggregate_packet_metrics_summarizes_safe_families_and_blocked_nulls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets_path = root / "isaac_src_packets.jsonl"
            output_dir = root / "metrics"
            packets = [
                _packet(
                    frame_idx=0,
                    pred=_state(
                        foot_in_contact=[True, False],
                        foot_contact_force_n=[24.0, 0.0],
                        support_pair_count=1,
                        support_margin_m=0.01,
                        floating_guard=False,
                        foot_slide_speed_mps=[None, None],
                        foot_slide_flags=[None, None],
                        foot_skate_distance_m=[0.0, None],
                        foot_skate_flags=[False, None],
                        foot_float_clearance_m=[0.01, None],
                        foot_float_flags=[False, None],
                    ),
                    target=_state(
                        foot_in_contact=[False, False],
                        foot_contact_force_n=[0.0, 0.0],
                        support_pair_count=0,
                        support_margin_m=0.09,
                        floating_guard=True,
                        foot_slide_speed_mps=[None, None],
                        foot_slide_flags=[None, None],
                        foot_skate_distance_m=[None, None],
                        foot_skate_flags=[None, None],
                        foot_float_clearance_m=[None, None],
                        foot_float_flags=[None, None],
                    ),
                ),
                _packet(
                    frame_idx=1,
                    pred=_state(
                        foot_in_contact=[True, False],
                        foot_contact_force_n=[30.0, 0.0],
                        support_pair_count=1,
                        support_margin_m=0.02,
                        floating_guard=False,
                        foot_slide_speed_mps=[0.5, None],
                        foot_slide_flags=[True, None],
                        foot_skate_distance_m=[0.03, None],
                        foot_skate_flags=[True, None],
                        foot_float_clearance_m=[0.06, None],
                        foot_float_flags=[True, None],
                    ),
                    target=_state(
                        foot_in_contact=[False, False],
                        foot_contact_force_n=[0.0, 0.0],
                        support_pair_count=0,
                        support_margin_m=0.08,
                        floating_guard=True,
                        foot_slide_speed_mps=[None, None],
                        foot_slide_flags=[None, None],
                        foot_skate_distance_m=[None, None],
                        foot_skate_flags=[None, None],
                        foot_float_clearance_m=[None, None],
                        foot_float_flags=[None, None],
                    ),
                ),
            ]
            packets_path.write_text(
                "".join(json.dumps(packet, sort_keys=True) + "\n" for packet in packets),
                encoding="utf-8",
            )

            summary = aggregate_packet_metrics(input_jsonl=packets_path, output_dir=output_dir)
            side_rows = _read_csv(output_dir / "per_frame_side_metrics.csv")
            foot_rows = _read_csv(output_dir / "per_frame_side_foot_metrics.csv")

        self.assertEqual(summary["schema_version"], METRIC_SCHEMA_VERSION)
        self.assertEqual(summary["packets_read"], 2)
        self.assertEqual(summary["side_packets_read"], 4)
        self.assertEqual(summary["frame_index"]["consecutive"], True)
        pred = summary["by_side"]["pred"]
        self.assertEqual(pred["foot_ground_contact_status_counts"], {"available": 2})
        self.assertEqual(pred["foot_ground_available_rate"], 1.0)
        self.assertEqual(pred["support_pair_count"]["mean"], 1.0)
        left_foot = pred["feet"][0]
        self.assertEqual(left_foot["contact_rate"], 1.0)
        self.assertEqual(left_foot["force_n"]["max"], 30.0)
        self.assertEqual(left_foot["slide_flags"]["true_count"], 1)
        self.assertEqual(left_foot["slide_flags"]["slide_violation_rate"], 1.0)
        self.assertEqual(left_foot["skate_distance_m"]["max"], 0.03)
        self.assertEqual(left_foot["float_flags"]["true_count"], 1)
        self.assertEqual(pred["blocked_null_accounting"]["self_collision_count_null_count"], 2)
        self.assertEqual(pred["blocked_null_accounting"]["self_collision_count_non_null_count"], 0)
        self.assertEqual(pred["self_collision_status_counts"], {"blocked": 2})
        self.assertEqual(summary["body_position_metrics"]["mpjpe"]["status"], "available")
        self.assertEqual(summary["body_position_metrics"]["mpjpe"]["available_frame_count"], 2)
        self.assertEqual(summary["body_position_metrics"]["mpjpe"]["mean_m"], 0.0)
        self.assertEqual(
            summary["body_position_metrics"]["w_mpjpe"]["status"],
            "unavailable",
        )
        target = summary["by_side"]["target"]
        self.assertEqual(target["floating_guard"]["true_count"], 2)
        self.assertEqual(target["floating_guard"]["floating_guard_violation_rate"], 1.0)
        self.assertEqual(target["floating_guard"]["floating_guard_pass_rate"], 0.0)
        self.assertEqual(len(side_rows), 4)
        self.assertEqual(len(foot_rows), 8)
        self.assertEqual(side_rows[0]["self_collision_count_is_null"], "True")

    def test_body_position_mpjpe_uses_real_pred_target_body_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets_path = root / "isaac_src_packets.jsonl"
            output_dir = root / "metrics"
            pred_state_0 = _state(
                foot_in_contact=[False, False],
                foot_contact_force_n=[0.0, 0.0],
                support_pair_count=0,
                support_margin_m=0.0,
                floating_guard=False,
                foot_slide_speed_mps=[None, None],
                foot_slide_flags=[None, None],
                foot_skate_distance_m=[None, None],
                foot_skate_flags=[None, None],
                foot_float_clearance_m=[None, None],
                foot_float_flags=[None, None],
            )
            target_state_0 = _state(
                foot_in_contact=[False, False],
                foot_contact_force_n=[0.0, 0.0],
                support_pair_count=0,
                support_margin_m=0.0,
                floating_guard=False,
                foot_slide_speed_mps=[None, None],
                foot_slide_flags=[None, None],
                foot_skate_distance_m=[None, None],
                foot_skate_flags=[None, None],
                foot_float_clearance_m=[None, None],
                foot_float_flags=[None, None],
            )
            pred_state_0["body_names"] = ["pelvis", "torso_link"]
            target_state_0["body_names"] = ["pelvis", "torso_link"]
            pred_state_0["body_pos_world_m"] = [[0.0, 0.0, 0.8], [0.0, 0.0, 1.0]]
            target_state_0["body_pos_world_m"] = [[0.0, 0.0, 0.9], [0.0, 0.2, 1.0]]

            pred_state_1 = dict(pred_state_0)
            target_state_1 = dict(target_state_0)
            pred_state_1["body_pos_world_m"] = [[1.0, 0.0, 0.8], [1.0, 0.0, 1.0]]
            target_state_1["body_pos_world_m"] = [[1.0, 0.0, 0.8], [1.3, 0.4, 1.0]]
            packets_path.write_text(
                "".join(
                    json.dumps(packet, sort_keys=True) + "\n"
                    for packet in [
                        _packet(frame_idx=0, pred=pred_state_0, target=target_state_0),
                        _packet(
                            frame_idx=1,
                            pred=pred_state_1,
                            target=target_state_1,
                            body_position_weights=[1.0, 3.0],
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = aggregate_packet_metrics(input_jsonl=packets_path, output_dir=output_dir)
            pair_rows = _read_csv(output_dir / "per_frame_pair_body_metrics.csv")

        mpjpe = summary["body_position_metrics"]["mpjpe"]
        self.assertEqual(mpjpe["status"], "available")
        self.assertEqual(mpjpe["available_frame_count"], 2)
        self.assertEqual(mpjpe["body_sample_count"], 4)
        self.assertAlmostEqual(mpjpe["mean_m"], 0.2)
        self.assertAlmostEqual(mpjpe["frame_m"]["mean"], 0.2)
        w_mpjpe = summary["body_position_metrics"]["w_mpjpe"]
        self.assertEqual(w_mpjpe["status"], "available")
        self.assertEqual(w_mpjpe["available_frame_count"], 1)
        self.assertEqual(w_mpjpe["unavailable_frame_count"], 1)
        self.assertEqual(
            w_mpjpe["unavailable_reason_counts"],
            {"body_position_weights missing; w_mpjpe unavailable": 1},
        )
        self.assertAlmostEqual(w_mpjpe["mean_m"], 0.375)
        self.assertEqual(len(pair_rows), 2)
        self.assertEqual(pair_rows[0]["mpjpe_status"], "available")
        self.assertEqual(pair_rows[0]["mpjpe_body_sample_count"], "2")
        self.assertEqual(pair_rows[1]["w_mpjpe_status"], "available")

    def test_body_position_mpjpe_stays_unavailable_when_positions_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets_path = root / "isaac_src_packets.jsonl"
            output_dir = root / "metrics"
            pred_state = _state(
                foot_in_contact=[False, False],
                foot_contact_force_n=[0.0, 0.0],
                support_pair_count=0,
                support_margin_m=0.0,
                floating_guard=False,
                foot_slide_speed_mps=[None, None],
                foot_slide_flags=[None, None],
                foot_skate_distance_m=[None, None],
                foot_skate_flags=[None, None],
                foot_float_clearance_m=[None, None],
                foot_float_flags=[None, None],
            )
            target_state = dict(pred_state)
            pred_state.pop("body_pos_world_m")
            packets_path.write_text(
                json.dumps(_packet(frame_idx=0, pred=pred_state, target=target_state)) + "\n",
                encoding="utf-8",
            )

            summary = aggregate_packet_metrics(input_jsonl=packets_path, output_dir=output_dir)
            pair_rows = _read_csv(output_dir / "per_frame_pair_body_metrics.csv")

        mpjpe = summary["body_position_metrics"]["mpjpe"]
        self.assertEqual(mpjpe["status"], "unavailable")
        self.assertIsNone(mpjpe["mean_m"])
        self.assertEqual(mpjpe["available_frame_count"], 0)
        self.assertEqual(mpjpe["unavailable_frame_count"], 1)
        self.assertEqual(
            mpjpe["unavailable_reason_counts"],
            {"pred.body_pos_world_m is missing or not a list": 1},
        )
        self.assertEqual(pair_rows[0]["mpjpe_status"], "unavailable")
        self.assertEqual(
            pair_rows[0]["mpjpe_reason"],
            "pred.body_pos_world_m is missing or not a list",
        )

    def test_blocked_foot_ground_does_not_fabricate_contact_or_force_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets_path = root / "isaac_src_packets.jsonl"
            output_dir = root / "metrics"
            blocked_state = _state(
                foot_ground_contact_status="blocked",
                foot_ground_contact_reason="force_matrix_w missing",
                foot_artifact_status="blocked",
                foot_artifact_reason="force_matrix_w missing",
                floating_guard_status="blocked",
                foot_in_contact=[False, False],
                foot_contact_force_n=[None, None],
                support_pair_count=None,
                support_margin_m=0.12,
                floating_guard=None,
                foot_slide_speed_mps=[None, None],
                foot_slide_flags=[None, None],
                foot_skate_distance_m=[None, None],
                foot_skate_flags=[None, None],
                foot_float_clearance_m=[None, None],
                foot_float_flags=[None, None],
            )
            packets_path.write_text(
                json.dumps(_packet(frame_idx=0, pred=blocked_state, target=blocked_state)) + "\n",
                encoding="utf-8",
            )

            summary = aggregate_packet_metrics(input_jsonl=packets_path, output_dir=output_dir)

        pred = summary["by_side"]["pred"]
        self.assertEqual(pred["foot_ground_contact_status_counts"], {"blocked": 1})
        self.assertEqual(pred["foot_ground_available_rate"], 0.0)
        self.assertEqual(pred["support_pair_count"]["count"], 0)
        self.assertEqual(pred["support_pair_count"]["null_count"], 1)
        self.assertEqual(pred["feet"][0]["contact_observed_count"], 0)
        self.assertEqual(pred["feet"][0]["contact_null_count"], 1)
        self.assertEqual(pred["feet"][0]["force_n"]["count"], 0)
        self.assertEqual(pred["feet"][0]["force_n"]["null_count"], 1)


def _packet(
    *,
    frame_idx: int,
    pred: dict[str, object],
    target: dict[str, object],
    body_position_weights: list[float] | None = None,
) -> dict[str, object]:
    packet: dict[str, object] = {
        "schema_version": "lr239.isaac_src_contact_packets.v1",
        "variant": "soma_uniform",
        "frame_idx": frame_idx,
        "t": frame_idx / 50.0,
        "dt": 1.0 / 50.0,
        "pred": pred,
        "target": target,
        "contract": "test-contract",
    }
    if body_position_weights is not None:
        packet["body_position_weights"] = body_position_weights
    return packet


def _state(
    *,
    foot_in_contact: list[bool],
    foot_contact_force_n: list[float | None],
    support_pair_count: int | None,
    support_margin_m: float,
    floating_guard: bool | None,
    foot_slide_speed_mps: list[float | None],
    foot_slide_flags: list[bool | None],
    foot_skate_distance_m: list[float | None],
    foot_skate_flags: list[bool | None],
    foot_float_clearance_m: list[float | None],
    foot_float_flags: list[bool | None],
    foot_ground_contact_status: str = "available",
    foot_ground_contact_reason: str = "",
    foot_artifact_status: str = "available",
    foot_artifact_reason: str = "",
    floating_guard_status: str = "available",
) -> dict[str, object]:
    pairs = (
        [
            {
                "body_a": "left_ankle_roll_link",
                "body_b": "/World/Ground",
                "force_n": 24.0,
                "position_world_m": [0.0, 0.0, 0.0],
                "source": "test",
            }
            for _index in range(support_pair_count)
        ]
        if support_pair_count is not None
        else []
    )
    return {
        "root_pos_world_m": [0.0, 0.0, 0.8],
        "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "joint_q_rad": [0.0] * 29,
        "body_names": ["pelvis"],
        "body_pos_world_m": [[0.0, 0.0, 0.8]],
        "foot_contact_force_n": foot_contact_force_n,
        "foot_in_contact": foot_in_contact,
        "foot_ground_contact_status": foot_ground_contact_status,
        "foot_ground_contact_reason": foot_ground_contact_reason,
        "support_margin_m": support_margin_m,
        "floating_guard": floating_guard,
        "floating_guard_status": floating_guard_status,
        "floating_guard_reason": "",
        "foot_slide_speed_mps": foot_slide_speed_mps,
        "foot_slide_flags": foot_slide_flags,
        "foot_skate_distance_m": foot_skate_distance_m,
        "foot_skate_flags": foot_skate_flags,
        "foot_float_clearance_m": foot_float_clearance_m,
        "foot_float_flags": foot_float_flags,
        "foot_artifact_status": foot_artifact_status,
        "foot_artifact_reason": foot_artifact_reason,
        "foot_ground_contact_pairs": pairs,
        "contact_pairs": pairs,
        "body_pair_contacts": None,
        "body_pair_contact_status": "blocked",
        "body_pair_contact_reason": "not bound",
        "self_collision_count": None,
        "self_collision_status": "blocked",
        "self_collision_reason": "not bound",
        "cross_ratio": None,
        "cross_ratio_guard": None,
        "cross_ratio_status": "blocked",
        "cross_ratio_reason": "not bound",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
