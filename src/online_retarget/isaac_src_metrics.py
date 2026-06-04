"""Metric aggregation for LR-239 Isaac/SRC replay packet JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from online_retarget.isaac_src_replay import DEFAULT_FOOT_LINKS


METRIC_SCHEMA_VERSION = "lr239.isaac_src_packet_metrics.v1"
SIDES: tuple[str, str] = ("pred", "target")


@dataclass(frozen=True)
class PacketMetricOutputs:
    summary_json: Path
    per_frame_pair_json: Path
    per_frame_pair_jsonl: Path
    per_frame_pair_csv: Path
    per_frame_side_json: Path
    per_frame_side_jsonl: Path
    per_frame_side_csv: Path
    per_frame_side_foot_csv: Path

    def to_payload(self) -> dict[str, str]:
        return {
            "summary_json": str(self.summary_json),
            "per_frame_pair_json": str(self.per_frame_pair_json),
            "per_frame_pair_jsonl": str(self.per_frame_pair_jsonl),
            "per_frame_pair_csv": str(self.per_frame_pair_csv),
            "per_frame_side_json": str(self.per_frame_side_json),
            "per_frame_side_jsonl": str(self.per_frame_side_jsonl),
            "per_frame_side_csv": str(self.per_frame_side_csv),
            "per_frame_side_foot_csv": str(self.per_frame_side_foot_csv),
        }


class NumericStats:
    def __init__(self) -> None:
        self.count = 0
        self.null_count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def add(self, value: object) -> None:
        number = _number_or_none(value)
        if number is None:
            self.null_count += 1
            return
        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def payload(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "null_count": self.null_count,
            "mean": None if self.count == 0 else self.total / self.count,
            "min": self.minimum,
            "max": self.maximum,
        }


class BoolStats:
    def __init__(self) -> None:
        self.true_count = 0
        self.false_count = 0
        self.null_count = 0

    def add(self, value: object) -> None:
        if isinstance(value, bool):
            if value:
                self.true_count += 1
            else:
                self.false_count += 1
            return
        self.null_count += 1

    @property
    def observed_count(self) -> int:
        return self.true_count + self.false_count

    def payload(self, *, true_rate_name: str = "true_rate") -> dict[str, float | int | None]:
        return {
            "true_count": self.true_count,
            "false_count": self.false_count,
            "null_count": self.null_count,
            "observed_count": self.observed_count,
            true_rate_name: _rate(self.true_count, self.observed_count),
        }


class FootStats:
    def __init__(self, foot_name: str) -> None:
        self.foot_name = foot_name
        self.contact_observed_count = 0
        self.contact_true_count = 0
        self.contact_false_count = 0
        self.contact_null_count = 0
        self.force_n = NumericStats()
        self.slide_speed_mps = NumericStats()
        self.slide_flags = BoolStats()
        self.skate_distance_m = NumericStats()
        self.skate_flags = BoolStats()
        self.float_clearance_m = NumericStats()
        self.float_flags = BoolStats()

    def add(
        self,
        *,
        foot_ground_available: bool,
        foot_in_contact: object,
        force_n: object,
        slide_speed_mps: object,
        slide_flag: object,
        skate_distance_m: object,
        skate_flag: object,
        float_clearance_m: object,
        float_flag: object,
    ) -> None:
        if foot_ground_available and isinstance(foot_in_contact, bool):
            self.contact_observed_count += 1
            if foot_in_contact:
                self.contact_true_count += 1
            else:
                self.contact_false_count += 1
        else:
            self.contact_null_count += 1
        self.force_n.add(force_n)
        self.slide_speed_mps.add(slide_speed_mps)
        self.slide_flags.add(slide_flag)
        self.skate_distance_m.add(skate_distance_m)
        self.skate_flags.add(skate_flag)
        self.float_clearance_m.add(float_clearance_m)
        self.float_flags.add(float_flag)

    def payload(self) -> dict[str, object]:
        return {
            "foot_name": self.foot_name,
            "contact_observed_count": self.contact_observed_count,
            "contact_true_count": self.contact_true_count,
            "contact_false_count": self.contact_false_count,
            "contact_null_count": self.contact_null_count,
            "contact_rate": _rate(self.contact_true_count, self.contact_observed_count),
            "force_n": self.force_n.payload(),
            "slide_speed_mps": self.slide_speed_mps.payload(),
            "slide_flags": self.slide_flags.payload(true_rate_name="slide_violation_rate"),
            "skate_distance_m": self.skate_distance_m.payload(),
            "skate_flags": self.skate_flags.payload(true_rate_name="skate_violation_rate"),
            "float_clearance_m": self.float_clearance_m.payload(),
            "float_flags": self.float_flags.payload(true_rate_name="float_violation_rate"),
        }


class BodyPositionMetricStats:
    def __init__(self) -> None:
        self.frame_count = 0
        self.mpjpe_available_frame_count = 0
        self.mpjpe_unavailable_frame_count = 0
        self.mpjpe_error_sum_m = 0.0
        self.mpjpe_body_sample_count = 0
        self.mpjpe_frame_m = NumericStats()
        self.mpjpe_reason_counts: Counter[str] = Counter()
        self.w_mpjpe_available_frame_count = 0
        self.w_mpjpe_unavailable_frame_count = 0
        self.w_mpjpe_weighted_error_sum_m = 0.0
        self.w_mpjpe_weight_sum = 0.0
        self.w_mpjpe_frame_m = NumericStats()
        self.w_mpjpe_reason_counts: Counter[str] = Counter()

    def add(self, row: Mapping[str, object]) -> None:
        self.frame_count += 1
        if row.get("mpjpe_status") == "available":
            error_sum = _number_or_none(row.get("mpjpe_error_sum_m"))
            sample_count = _int_or_none(row.get("mpjpe_body_sample_count"))
            if error_sum is None or sample_count is None or sample_count <= 0:
                self.mpjpe_unavailable_frame_count += 1
                self.mpjpe_reason_counts["mpjpe accounting fields are invalid"] += 1
            else:
                self.mpjpe_available_frame_count += 1
                self.mpjpe_error_sum_m += error_sum
                self.mpjpe_body_sample_count += sample_count
                self.mpjpe_frame_m.add(row.get("mpjpe_m"))
        else:
            self.mpjpe_unavailable_frame_count += 1
            self.mpjpe_reason_counts[_reason(row.get("mpjpe_reason"))] += 1

        if row.get("w_mpjpe_status") == "available":
            weighted_sum = _number_or_none(row.get("w_mpjpe_weighted_error_sum_m"))
            weight_sum = _number_or_none(row.get("w_mpjpe_weight_sum"))
            if weighted_sum is None or weight_sum is None or weight_sum <= 0:
                self.w_mpjpe_unavailable_frame_count += 1
                self.w_mpjpe_reason_counts["w_mpjpe accounting fields are invalid"] += 1
            else:
                self.w_mpjpe_available_frame_count += 1
                self.w_mpjpe_weighted_error_sum_m += weighted_sum
                self.w_mpjpe_weight_sum += weight_sum
                self.w_mpjpe_frame_m.add(row.get("w_mpjpe_m"))
        else:
            self.w_mpjpe_unavailable_frame_count += 1
            self.w_mpjpe_reason_counts[_reason(row.get("w_mpjpe_reason"))] += 1

    def payload(self) -> dict[str, object]:
        return {
            "mpjpe": {
                "status": (
                    "available"
                    if self.mpjpe_body_sample_count > 0
                    else "unavailable"
                ),
                "mean_m": (
                    None
                    if self.mpjpe_body_sample_count == 0
                    else self.mpjpe_error_sum_m / self.mpjpe_body_sample_count
                ),
                "frame_m": self.mpjpe_frame_m.payload(),
                "available_frame_count": self.mpjpe_available_frame_count,
                "unavailable_frame_count": self.mpjpe_unavailable_frame_count,
                "body_sample_count": self.mpjpe_body_sample_count,
                "unavailable_reason_counts": _counter_payload(
                    self.mpjpe_reason_counts
                ),
            },
            "w_mpjpe": {
                "status": (
                    "available"
                    if self.w_mpjpe_weight_sum > 0
                    else "unavailable"
                ),
                "mean_m": (
                    None
                    if self.w_mpjpe_weight_sum == 0
                    else self.w_mpjpe_weighted_error_sum_m / self.w_mpjpe_weight_sum
                ),
                "frame_m": self.w_mpjpe_frame_m.payload(),
                "available_frame_count": self.w_mpjpe_available_frame_count,
                "unavailable_frame_count": self.w_mpjpe_unavailable_frame_count,
                "weight_sum": (
                    None if self.w_mpjpe_weight_sum == 0 else self.w_mpjpe_weight_sum
                ),
                "unavailable_reason_counts": _counter_payload(
                    self.w_mpjpe_reason_counts
                ),
            },
        }


class SideStats:
    def __init__(self, side: str, foot_links: Sequence[str]) -> None:
        self.side = side
        self.frame_count = 0
        self.foot_links = tuple(foot_links)
        self.foot_ground_contact_status_counts: Counter[str] = Counter()
        self.foot_artifact_status_counts: Counter[str] = Counter()
        self.floating_guard_status_counts: Counter[str] = Counter()
        self.body_pair_contact_status_counts: Counter[str] = Counter()
        self.self_collision_status_counts: Counter[str] = Counter()
        self.cross_ratio_status_counts: Counter[str] = Counter()
        self.support_pair_count = NumericStats()
        self.support_margin_m = NumericStats()
        self.floating_guard = BoolStats()
        self.body_pair_contacts_null_count = 0
        self.body_pair_contacts_non_null_count = 0
        self.self_collision_count_null_count = 0
        self.self_collision_count_non_null_count = 0
        self.cross_ratio_null_count = 0
        self.cross_ratio_non_null_count = 0
        self.cross_ratio_guard_null_count = 0
        self.cross_ratio_guard_non_null_count = 0
        self.foot_stats = [FootStats(foot_name) for foot_name in self.foot_links]

    def add(self, packet: Mapping[str, object]) -> None:
        self.frame_count += 1
        foot_ground_status = _status(packet.get("foot_ground_contact_status"))
        foot_artifact_status = _status(packet.get("foot_artifact_status"))
        floating_status = _status(packet.get("floating_guard_status"))
        body_pair_status = _status(packet.get("body_pair_contact_status"))
        self_status = _status(packet.get("self_collision_status"))
        cross_status = _status(packet.get("cross_ratio_status"))
        self.foot_ground_contact_status_counts[foot_ground_status] += 1
        self.foot_artifact_status_counts[foot_artifact_status] += 1
        self.floating_guard_status_counts[floating_status] += 1
        self.body_pair_contact_status_counts[body_pair_status] += 1
        self.self_collision_status_counts[self_status] += 1
        self.cross_ratio_status_counts[cross_status] += 1

        foot_ground_available = foot_ground_status == "available"
        support_pairs = packet.get("foot_ground_contact_pairs")
        if foot_ground_available and isinstance(support_pairs, list):
            self.support_pair_count.add(len(support_pairs))
        else:
            self.support_pair_count.add(None)
        self.support_margin_m.add(packet.get("support_margin_m"))
        self.floating_guard.add(
            packet.get("floating_guard") if floating_status == "available" else None
        )
        self._add_null_accounting(packet)

        foot_in_contact = _sequence(packet.get("foot_in_contact"))
        forces = _sequence(packet.get("foot_contact_force_n"))
        slide_speeds = _sequence(packet.get("foot_slide_speed_mps"))
        slide_flags = _sequence(packet.get("foot_slide_flags"))
        skate_distances = _sequence(packet.get("foot_skate_distance_m"))
        skate_flags = _sequence(packet.get("foot_skate_flags"))
        float_clearances = _sequence(packet.get("foot_float_clearance_m"))
        float_flags = _sequence(packet.get("foot_float_flags"))
        for index, stats in enumerate(self.foot_stats):
            stats.add(
                foot_ground_available=foot_ground_available,
                foot_in_contact=_at(foot_in_contact, index),
                force_n=_at(forces, index),
                slide_speed_mps=_at(slide_speeds, index),
                slide_flag=_at(slide_flags, index),
                skate_distance_m=_at(skate_distances, index),
                skate_flag=_at(skate_flags, index),
                float_clearance_m=_at(float_clearances, index),
                float_flag=_at(float_flags, index),
            )

    def _add_null_accounting(self, packet: Mapping[str, object]) -> None:
        if packet.get("body_pair_contacts") is None:
            self.body_pair_contacts_null_count += 1
        else:
            self.body_pair_contacts_non_null_count += 1
        if packet.get("self_collision_count") is None:
            self.self_collision_count_null_count += 1
        else:
            self.self_collision_count_non_null_count += 1
        if packet.get("cross_ratio") is None:
            self.cross_ratio_null_count += 1
        else:
            self.cross_ratio_non_null_count += 1
        if packet.get("cross_ratio_guard") is None:
            self.cross_ratio_guard_null_count += 1
        else:
            self.cross_ratio_guard_non_null_count += 1

    def payload(self) -> dict[str, object]:
        floating_payload = self.floating_guard.payload(
            true_rate_name="floating_guard_violation_rate"
        )
        floating_payload["floating_guard_pass_rate"] = _rate(
            self.floating_guard.false_count,
            self.floating_guard.observed_count,
        )
        return {
            "side": self.side,
            "frame_count": self.frame_count,
            "foot_ground_contact_status_counts": _counter_payload(
                self.foot_ground_contact_status_counts
            ),
            "foot_ground_available_rate": _rate(
                self.foot_ground_contact_status_counts.get("available", 0),
                self.frame_count,
            ),
            "foot_artifact_status_counts": _counter_payload(self.foot_artifact_status_counts),
            "foot_artifact_available_rate": _rate(
                self.foot_artifact_status_counts.get("available", 0),
                self.frame_count,
            ),
            "floating_guard_status_counts": _counter_payload(
                self.floating_guard_status_counts
            ),
            "floating_guard_available_rate": _rate(
                self.floating_guard_status_counts.get("available", 0),
                self.frame_count,
            ),
            "floating_guard": floating_payload,
            "support_margin_m": self.support_margin_m.payload(),
            "support_pair_count": self.support_pair_count.payload(),
            "feet": [stats.payload() for stats in self.foot_stats],
            "body_pair_contact_status_counts": _counter_payload(
                self.body_pair_contact_status_counts
            ),
            "self_collision_status_counts": _counter_payload(
                self.self_collision_status_counts
            ),
            "cross_ratio_status_counts": _counter_payload(self.cross_ratio_status_counts),
            "blocked_null_accounting": {
                "body_pair_contacts_null_count": self.body_pair_contacts_null_count,
                "body_pair_contacts_non_null_count": self.body_pair_contacts_non_null_count,
                "self_collision_count_null_count": self.self_collision_count_null_count,
                "self_collision_count_non_null_count": self.self_collision_count_non_null_count,
                "cross_ratio_null_count": self.cross_ratio_null_count,
                "cross_ratio_non_null_count": self.cross_ratio_non_null_count,
                "cross_ratio_guard_null_count": self.cross_ratio_guard_null_count,
                "cross_ratio_guard_non_null_count": self.cross_ratio_guard_non_null_count,
            },
        }


def aggregate_packet_metrics(
    *,
    input_jsonl: Path,
    output_dir: Path,
    foot_links: Sequence[str] = DEFAULT_FOOT_LINKS,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = PacketMetricOutputs(
        summary_json=output_dir / "packet_metric_summary.json",
        per_frame_pair_json=output_dir / "per_frame_pair_body_metrics.json",
        per_frame_pair_jsonl=output_dir / "per_frame_pair_body_metrics.jsonl",
        per_frame_pair_csv=output_dir / "per_frame_pair_body_metrics.csv",
        per_frame_side_json=output_dir / "per_frame_side_metrics.json",
        per_frame_side_jsonl=output_dir / "per_frame_side_metrics.jsonl",
        per_frame_side_csv=output_dir / "per_frame_side_metrics.csv",
        per_frame_side_foot_csv=output_dir / "per_frame_side_foot_metrics.csv",
    )
    stats = {side: SideStats(side, foot_links) for side in SIDES}
    body_position_stats = BodyPositionMetricStats()
    variant_counts: Counter[str] = Counter()
    frame_indices: list[int] = []
    pair_rows: list[dict[str, object]] = []
    side_rows: list[dict[str, object]] = []
    foot_rows: list[dict[str, object]] = []

    for packet in _iter_jsonl(input_jsonl):
        frame_idx = _int_or_none(packet.get("frame_idx"))
        if frame_idx is not None:
            frame_indices.append(frame_idx)
        variant = str(packet.get("variant", ""))
        variant_counts[variant] += 1
        pair_row = _body_position_metric_row(packet)
        body_position_stats.add(pair_row)
        pair_rows.append(pair_row)
        for side in SIDES:
            state_packet = packet.get(side)
            if not isinstance(state_packet, Mapping):
                continue
            stats[side].add(state_packet)
            side_row = _side_row(packet, side, state_packet)
            side_rows.append(side_row)
            foot_rows.extend(_foot_rows(side_row, state_packet, foot_links))

    _write_json(outputs.per_frame_pair_json, {"rows": pair_rows})
    _write_jsonl(outputs.per_frame_pair_jsonl, pair_rows)
    _write_csv(outputs.per_frame_pair_csv, pair_rows)
    _write_json(outputs.per_frame_side_json, {"rows": side_rows})
    _write_jsonl(outputs.per_frame_side_jsonl, side_rows)
    _write_csv(outputs.per_frame_side_csv, side_rows)
    _write_csv(outputs.per_frame_side_foot_csv, foot_rows)
    summary: dict[str, object] = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "outputs": outputs.to_payload(),
        "packets_read": len(frame_indices),
        "side_packets_read": len(side_rows),
        "variant_counts": _counter_payload(variant_counts),
        "frame_index": {
            "first": min(frame_indices) if frame_indices else None,
            "last": max(frame_indices) if frame_indices else None,
            "unique_count": len(set(frame_indices)),
            "consecutive": _consecutive(frame_indices),
        },
        "body_position_metrics": body_position_stats.payload(),
        "by_side": {side: stats[side].payload() for side in SIDES},
        "metric_rules": {
            "body_position_mpjpe_policy": (
                "mpjpe is computed only from paired pred/target body_pos_world_m "
                "arrays in the same body order; missing, malformed, or misaligned "
                "body-position inputs are reported as unavailable and are not "
                "converted to zero or inferred from joints"
            ),
            "weighted_body_position_mpjpe_policy": (
                "w_mpjpe requires explicit body_position_weights matching "
                "body_pos_world_m; weights are not inferred"
            ),
            "body_self_src_numeric_policy": (
                "body_pair_contacts, self_collision_count, cross_ratio, and "
                "cross_ratio_guard are summarized only as status/null availability; "
                "null or blocked values are not converted to zero"
            ),
            "floating_guard_true_semantics": (
                "floating_guard=true is counted as a floating violation; "
                "floating_guard=false is counted as a pass"
            ),
        },
    }
    _write_json(outputs.summary_json, summary)
    return summary


def _body_position_metric_row(frame_packet: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": frame_packet.get("schema_version"),
        "variant": frame_packet.get("variant"),
        "frame_idx": frame_packet.get("frame_idx"),
        "t": frame_packet.get("t"),
        "dt": frame_packet.get("dt"),
        "mpjpe_status": "unavailable",
        "mpjpe_reason": "",
        "mpjpe_m": None,
        "mpjpe_error_sum_m": None,
        "mpjpe_body_sample_count": 0,
        "body_count": None,
        "w_mpjpe_status": "unavailable",
        "w_mpjpe_reason": "",
        "w_mpjpe_m": None,
        "w_mpjpe_weighted_error_sum_m": None,
        "w_mpjpe_weight_sum": None,
    }
    pred_state = frame_packet.get("pred")
    target_state = frame_packet.get("target")
    if not isinstance(pred_state, Mapping) or not isinstance(target_state, Mapping):
        reason = "pred and target state packets must both be present"
        row["mpjpe_reason"] = reason
        row["w_mpjpe_reason"] = f"mpjpe unavailable: {reason}"
        return row

    pred_positions, pred_reason = _body_positions_or_reason(pred_state, "pred")
    if pred_positions is None:
        row["mpjpe_reason"] = pred_reason
        row["w_mpjpe_reason"] = f"mpjpe unavailable: {pred_reason}"
        return row
    target_positions, target_reason = _body_positions_or_reason(target_state, "target")
    if target_positions is None:
        row["mpjpe_reason"] = target_reason
        row["w_mpjpe_reason"] = f"mpjpe unavailable: {target_reason}"
        return row
    if len(pred_positions) != len(target_positions):
        reason = (
            "body_pos_world_m body count mismatch: "
            f"pred {len(pred_positions)} != target {len(target_positions)}"
        )
        row["mpjpe_reason"] = reason
        row["w_mpjpe_reason"] = f"mpjpe unavailable: {reason}"
        return row
    names_reason = _body_names_alignment_reason(
        pred_state,
        target_state,
        len(pred_positions),
        len(target_positions),
    )
    if names_reason:
        row["mpjpe_reason"] = names_reason
        row["w_mpjpe_reason"] = f"mpjpe unavailable: {names_reason}"
        return row

    errors = [
        math.dist(pred_position, target_position)
        for pred_position, target_position in zip(
            pred_positions,
            target_positions,
            strict=True,
        )
    ]
    error_sum = sum(errors)
    body_count = len(errors)
    row.update(
        {
            "mpjpe_status": "available",
            "mpjpe_reason": "",
            "mpjpe_m": error_sum / body_count,
            "mpjpe_error_sum_m": error_sum,
            "mpjpe_body_sample_count": body_count,
            "body_count": body_count,
        }
    )

    weights, weights_reason = _body_position_weights_or_reason(
        frame_packet,
        pred_state,
        target_state,
        body_count,
    )
    if weights is None:
        row["w_mpjpe_reason"] = weights_reason
        return row

    weighted_error_sum = sum(error * weight for error, weight in zip(errors, weights, strict=True))
    weight_sum = sum(weights)
    row.update(
        {
            "w_mpjpe_status": "available",
            "w_mpjpe_reason": "",
            "w_mpjpe_m": weighted_error_sum / weight_sum,
            "w_mpjpe_weighted_error_sum_m": weighted_error_sum,
            "w_mpjpe_weight_sum": weight_sum,
        }
    )
    return row


def _body_positions_or_reason(
    state_packet: Mapping[str, object],
    side: str,
) -> tuple[list[tuple[float, float, float]] | None, str]:
    raw_positions = state_packet.get("body_pos_world_m")
    if not isinstance(raw_positions, list):
        return None, f"{side}.body_pos_world_m is missing or not a list"
    if not raw_positions:
        return None, f"{side}.body_pos_world_m is empty"
    positions: list[tuple[float, float, float]] = []
    for body_index, raw_position in enumerate(raw_positions):
        if not isinstance(raw_position, list) or len(raw_position) != 3:
            return (
                None,
                f"{side}.body_pos_world_m[{body_index}] is not a 3D vector",
            )
        coords: list[float] = []
        for axis_index, raw_value in enumerate(raw_position):
            value = _finite_number_or_none(raw_value)
            if value is None:
                return (
                    None,
                    (
                        f"{side}.body_pos_world_m[{body_index}][{axis_index}] "
                        "is not a finite number"
                    ),
                )
            coords.append(value)
        positions.append((coords[0], coords[1], coords[2]))
    return positions, ""


def _body_names_alignment_reason(
    pred_state: Mapping[str, object],
    target_state: Mapping[str, object],
    pred_body_count: int,
    target_body_count: int,
) -> str:
    pred_names, pred_reason = _body_names_or_reason(
        pred_state,
        "pred",
        pred_body_count,
    )
    if pred_reason:
        return pred_reason
    target_names, target_reason = _body_names_or_reason(
        target_state,
        "target",
        target_body_count,
    )
    if target_reason:
        return target_reason
    if pred_names is not None and target_names is not None and pred_names != target_names:
        return "pred.body_names and target.body_names do not match"
    return ""


def _body_names_or_reason(
    state_packet: Mapping[str, object],
    side: str,
    expected_count: int,
) -> tuple[tuple[str, ...] | None, str]:
    if "body_names" not in state_packet:
        return None, ""
    raw_names = state_packet.get("body_names")
    if not isinstance(raw_names, list):
        return None, f"{side}.body_names is not a list"
    names = tuple(str(name) for name in raw_names)
    if len(names) != expected_count:
        return (
            None,
            (
                f"{side}.body_names length {len(names)} does not match "
                f"body_pos_world_m length {expected_count}"
            ),
        )
    return names, ""


def _body_position_weights_or_reason(
    frame_packet: Mapping[str, object],
    pred_state: Mapping[str, object],
    target_state: Mapping[str, object],
    body_count: int,
) -> tuple[list[float] | None, str]:
    for source_name, source in (
        ("frame", frame_packet),
        ("pred", pred_state),
        ("target", target_state),
    ):
        if "body_position_weights" in source:
            return _weights_or_reason(
                source.get("body_position_weights"),
                f"{source_name}.body_position_weights",
                body_count,
            )
    return None, "body_position_weights missing; w_mpjpe unavailable"


def _weights_or_reason(
    raw_weights: object,
    source_name: str,
    body_count: int,
) -> tuple[list[float] | None, str]:
    if not isinstance(raw_weights, list):
        return None, f"{source_name} is not a list"
    if len(raw_weights) != body_count:
        return (
            None,
            f"{source_name} length {len(raw_weights)} does not match body_count {body_count}",
        )
    weights: list[float] = []
    for index, raw_weight in enumerate(raw_weights):
        weight = _finite_number_or_none(raw_weight)
        if weight is None or weight < 0:
            return None, f"{source_name}[{index}] is not a non-negative finite number"
        weights.append(weight)
    if sum(weights) <= 0:
        return None, f"{source_name} sum must be positive"
    return weights, ""


def _side_row(
    frame_packet: Mapping[str, object],
    side: str,
    state_packet: Mapping[str, object],
) -> dict[str, object]:
    support_pairs = state_packet.get("foot_ground_contact_pairs")
    support_pair_count = len(support_pairs) if isinstance(support_pairs, list) else None
    return {
        "schema_version": frame_packet.get("schema_version"),
        "variant": frame_packet.get("variant"),
        "frame_idx": frame_packet.get("frame_idx"),
        "t": frame_packet.get("t"),
        "dt": frame_packet.get("dt"),
        "side": side,
        "foot_ground_contact_status": state_packet.get("foot_ground_contact_status"),
        "foot_ground_contact_reason": state_packet.get("foot_ground_contact_reason"),
        "foot_artifact_status": state_packet.get("foot_artifact_status"),
        "foot_artifact_reason": state_packet.get("foot_artifact_reason"),
        "floating_guard_status": state_packet.get("floating_guard_status"),
        "floating_guard_reason": state_packet.get("floating_guard_reason"),
        "floating_guard": state_packet.get("floating_guard"),
        "support_margin_m": state_packet.get("support_margin_m"),
        "support_pair_count": support_pair_count,
        "foot_contact_force_n": state_packet.get("foot_contact_force_n"),
        "foot_in_contact": state_packet.get("foot_in_contact"),
        "foot_slide_speed_mps": state_packet.get("foot_slide_speed_mps"),
        "foot_slide_flags": state_packet.get("foot_slide_flags"),
        "foot_skate_distance_m": state_packet.get("foot_skate_distance_m"),
        "foot_skate_flags": state_packet.get("foot_skate_flags"),
        "foot_float_clearance_m": state_packet.get("foot_float_clearance_m"),
        "foot_float_flags": state_packet.get("foot_float_flags"),
        "body_pair_contact_status": state_packet.get("body_pair_contact_status"),
        "body_pair_contact_reason": state_packet.get("body_pair_contact_reason"),
        "body_pair_contacts_is_null": state_packet.get("body_pair_contacts") is None,
        "self_collision_status": state_packet.get("self_collision_status"),
        "self_collision_reason": state_packet.get("self_collision_reason"),
        "self_collision_count_is_null": state_packet.get("self_collision_count") is None,
        "cross_ratio_status": state_packet.get("cross_ratio_status"),
        "cross_ratio_reason": state_packet.get("cross_ratio_reason"),
        "cross_ratio_is_null": state_packet.get("cross_ratio") is None,
        "cross_ratio_guard_is_null": state_packet.get("cross_ratio_guard") is None,
    }


def _foot_rows(
    side_row: Mapping[str, object],
    state_packet: Mapping[str, object],
    foot_links: Sequence[str],
) -> list[dict[str, object]]:
    forces = _sequence(state_packet.get("foot_contact_force_n"))
    contacts = _sequence(state_packet.get("foot_in_contact"))
    slide_speeds = _sequence(state_packet.get("foot_slide_speed_mps"))
    slide_flags = _sequence(state_packet.get("foot_slide_flags"))
    skate_distances = _sequence(state_packet.get("foot_skate_distance_m"))
    skate_flags = _sequence(state_packet.get("foot_skate_flags"))
    float_clearances = _sequence(state_packet.get("foot_float_clearance_m"))
    float_flags = _sequence(state_packet.get("foot_float_flags"))
    rows = []
    for index, foot_name in enumerate(foot_links):
        rows.append(
            {
                "variant": side_row.get("variant"),
                "frame_idx": side_row.get("frame_idx"),
                "t": side_row.get("t"),
                "dt": side_row.get("dt"),
                "side": side_row.get("side"),
                "foot_index": index,
                "foot_name": foot_name,
                "foot_ground_contact_status": side_row.get("foot_ground_contact_status"),
                "foot_artifact_status": side_row.get("foot_artifact_status"),
                "floating_guard_status": side_row.get("floating_guard_status"),
                "support_margin_m": side_row.get("support_margin_m"),
                "floating_guard": side_row.get("floating_guard"),
                "support_pair_count": side_row.get("support_pair_count"),
                "foot_contact_force_n": _at(forces, index),
                "foot_in_contact": _at(contacts, index),
                "foot_slide_speed_mps": _at(slide_speeds, index),
                "foot_slide_flag": _at(slide_flags, index),
                "foot_skate_distance_m": _at(skate_distances, index),
                "foot_skate_flag": _at(skate_flags, index),
                "foot_float_clearance_m": _at(float_clearances, index),
                "foot_float_flag": _at(float_flags, index),
                "body_pair_contact_status": side_row.get("body_pair_contact_status"),
                "body_pair_contacts_is_null": side_row.get("body_pair_contacts_is_null"),
                "self_collision_status": side_row.get("self_collision_status"),
                "self_collision_count_is_null": side_row.get("self_collision_count_is_null"),
                "cross_ratio_status": side_row.get("cross_ratio_status"),
                "cross_ratio_is_null": side_row.get("cross_ratio_is_null"),
                "cross_ratio_guard_is_null": side_row.get("cross_ratio_guard_is_null"),
            }
        )
    return rows


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            yield payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _finite_number_or_none(value: object) -> float | None:
    number = _number_or_none(value)
    if number is None or not math.isfinite(number):
        return None
    return number


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()


def _at(values: Sequence[object], index: int) -> object:
    return values[index] if index < len(values) else None


def _status(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    return "missing"


def _reason(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    return "missing reason"


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _consecutive(frame_indices: Sequence[int]) -> bool:
    if not frame_indices:
        return False
    ordered = sorted(set(frame_indices))
    return ordered == list(range(ordered[0], ordered[-1] + 1))


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate LR-239 Isaac/SRC packet metrics.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--foot-link",
        action="append",
        default=[],
        help="Foot link name in packet order. Repeat for multiple feet.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    foot_links = tuple(args.foot_link) if args.foot_link else DEFAULT_FOOT_LINKS
    summary = aggregate_packet_metrics(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        foot_links=foot_links,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
