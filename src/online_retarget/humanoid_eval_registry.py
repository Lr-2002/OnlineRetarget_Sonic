"""Structured metric registry for humanoid retarget evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "humanoid_retarget_eval.v1"
MEASURED = "measured"
MISSING = "missing"
NOT_APPLICABLE = "not_applicable"
VALID_STATUSES = {MEASURED, MISSING, NOT_APPLICABLE}


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    unit: str
    direction: str
    description: str
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    method_id: str
    protocol_id: str
    sequence_id: str
    metric_id: str
    value: float | None
    unit: str
    direction: str
    source: str
    notes: str = ""
    status: str = MEASURED

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"unsupported evaluation status: {self.status}")
        if self.status == MEASURED and self.value is None:
            raise ValueError("measured evaluation results require a numeric value")
        if self.status != MEASURED and self.value is not None:
            raise ValueError("missing/not_applicable evaluation results must not carry a value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


METHODS: dict[str, dict[str, str]] = {
    "gmr": {
        "name": "GMR",
        "type": "retargeting_method",
    },
    "nmr": {
        "name": "NMR",
        "type": "retargeting_method",
    },
    "phc": {
        "name": "PHC",
        "type": "retargeting_or_tracking_baseline",
    },
    "online_retarget_a0": {
        "name": "OnlineRetarget A0 kin-only supervised trainer",
        "type": "training_lane",
    },
}

PROTOCOLS: dict[str, dict[str, str]] = {
    "gmr_lafan1_beyondmimic": {
        "name": "GMR LAFAN1 via BeyondMimic evaluation harness",
        "harness": "beyondmimic",
    },
    "nmr_amass_g1": {
        "name": "NMR AMASS-to-G1 evaluation",
        "harness": "nmr_eval",
    },
    "sonic_kin_soma_motionlib_training": {
        "name": "SONIC kin-only SOMA motionlib training/eval",
        "harness": "training_time_logging",
    },
}

METRICS: dict[str, MetricSpec] = {
    "policy_success": MetricSpec(
        "policy_success",
        "ratio",
        "higher_is_better",
        "Downstream policy or tracking success rate.",
    ),
    "global_body_position_error": MetricSpec(
        "global_body_position_error",
        "meter",
        "lower_is_better",
        "Mean global body-position error.",
    ),
    "root_relative_body_position_error": MetricSpec(
        "root_relative_body_position_error",
        "meter",
        "lower_is_better",
        "Mean body-position error after root-relative alignment.",
    ),
    "weighted_mpjpe": MetricSpec(
        "weighted_mpjpe",
        "meter",
        "lower_is_better",
        "Weighted mean per-joint/body position error.",
        aliases=("W-MPJPE", "w_mpjpe"),
    ),
    "joint_rotation_error": MetricSpec(
        "joint_rotation_error",
        "radian",
        "lower_is_better",
        "Joint rotation error.",
    ),
    "joint_jump_count": MetricSpec(
        "joint_jump_count",
        "count",
        "lower_is_better",
        "Count of abrupt joint-motion events.",
    ),
    "self_collision_count": MetricSpec(
        "self_collision_count",
        "count",
        "lower_is_better",
        "Count of self-collision events.",
    ),
    "joint_limit_proximity_count": MetricSpec(
        "joint_limit_proximity_count",
        "count",
        "lower_is_better",
        "Count of samples near or beyond joint limits.",
    ),
    "source_faithfulness_user_study": MetricSpec(
        "source_faithfulness_user_study",
        "score",
        "higher_is_better",
        "Human-rated source-faithfulness score.",
    ),
    "physical_failure_guard": MetricSpec(
        "physical_failure_guard",
        "count",
        "lower_is_better",
        "Physical-failure guard count from floating/cross-ratio/PHC-style thresholds.",
    ),
    "runtime_hz": MetricSpec(
        "runtime_hz",
        "hertz",
        "higher_is_better",
        "Runtime throughput.",
    ),
    "g1_joint_pos_rmse_rad": MetricSpec(
        "g1_joint_pos_rmse_rad",
        "radian",
        "lower_is_better",
        "G1 joint-angle command RMSE over the predicted future-window joint-position targets.",
        aliases=("joint_pos_rmse_raw",),
    ),
    "body_position_mpjpe": MetricSpec(
        "body_position_mpjpe",
        "meter",
        "lower_is_better",
        "Body-position MPJPE from a supplemental FK/body-position evaluator.",
    ),
}

_ALIASES = {
    alias: spec.metric_id
    for spec in METRICS.values()
    for alias in spec.aliases
}


def metric_spec(metric_id: str) -> MetricSpec:
    canonical = _ALIASES.get(metric_id, metric_id)
    try:
        return METRICS[canonical]
    except KeyError as exc:
        raise KeyError(f"unknown metric_id: {metric_id}") from exc


def registry_manifest() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "methods": METHODS,
        "protocols": PROTOCOLS,
        "metrics": {metric_id: spec.to_dict() for metric_id, spec in METRICS.items()},
        "status_values": sorted(VALID_STATUSES),
        "dash_table_semantics": (
            "paper-table dashes are encoded as missing or not_applicable, never numeric zero"
        ),
    }


def make_result(
    *,
    method_id: str,
    protocol_id: str,
    sequence_id: str,
    metric_id: str,
    value: float | None,
    source: str,
    notes: str = "",
    status: str = MEASURED,
) -> EvaluationResult:
    spec = metric_spec(metric_id)
    return EvaluationResult(
        method_id=method_id,
        protocol_id=protocol_id,
        sequence_id=sequence_id,
        metric_id=spec.metric_id,
        value=float(value) if value is not None else None,
        unit=spec.unit,
        direction=spec.direction,
        source=source,
        notes=notes,
        status=status,
    )


def missing_result(
    *,
    method_id: str,
    protocol_id: str,
    sequence_id: str,
    metric_id: str,
    source: str,
    notes: str,
) -> EvaluationResult:
    return make_result(
        method_id=method_id,
        protocol_id=protocol_id,
        sequence_id=sequence_id,
        metric_id=metric_id,
        value=None,
        source=source,
        notes=notes,
        status=MISSING,
    )


def not_applicable_result(
    *,
    method_id: str,
    protocol_id: str,
    sequence_id: str,
    metric_id: str,
    source: str,
    notes: str,
) -> EvaluationResult:
    return make_result(
        method_id=method_id,
        protocol_id=protocol_id,
        sequence_id=sequence_id,
        metric_id=metric_id,
        value=None,
        source=source,
        notes=notes,
        status=NOT_APPLICABLE,
    )


def results_to_dicts(results: Any) -> list[dict[str, Any]]:
    return [result.to_dict() if hasattr(result, "to_dict") else dict(result) for result in results]


def result_dicts(results: Any) -> list[dict[str, Any]]:
    return results_to_dicts(results)


def status_counts(results: Any) -> dict[str, int]:
    counts = {status: 0 for status in sorted(VALID_STATUSES)}
    for result in results:
        status = result.status if hasattr(result, "status") else str(result.get("status", ""))
        if status in counts:
            counts[status] += 1
    return counts


def assert_known_method_protocol(method_id: str, protocol_id: str) -> None:
    if method_id not in METHODS:
        raise KeyError(f"unknown method_id: {method_id}")
    if protocol_id not in PROTOCOLS:
        raise KeyError(f"unknown protocol_id: {protocol_id}")
