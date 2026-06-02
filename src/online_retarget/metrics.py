"""Canonical metric definitions for online and offline retargeting checks.

The public ``joint_rmse`` / ``mpjpe`` style functions are kept as thin
compatibility wrappers. New online and offline callers should use
``compute_metric_bundle`` so they share the same definitions, reducers,
availability checks, and metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping, Sequence


Vector = Sequence[float]
JointFrame = Sequence[Vector]
Motion3D = Sequence[JointFrame]
Motion1D = Sequence[Sequence[float]]

MetricFields = Mapping[str, Any]

METRIC_AVAILABLE = "available"
METRIC_UNAVAILABLE = "unavailable"
METRIC_BLOCKED = "blocked"
METRIC_NOT_APPLICABLE = "not_applicable"

LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"
INFORMATIONAL = "informational"
PASS_REQUIRED = "pass_required"

FRAME_MASK = "optional frame_mask drops whole frames before the reducer"
JOINT_MASK = "optional frame_mask and joint_mask drop frames or joint dimensions"
BODY_MASK = "optional frame_mask and body_mask drop frames or body/link positions"
NO_MASK = "not maskable; computed from scalar artifact fields"


@dataclass(frozen=True)
class MetricMetadata:
    name: str
    unit: str
    direction: str
    required_fields: tuple[str, ...]
    mask_semantics: str
    reducer: str
    description: str
    paper_labels: tuple[str, ...] = ()
    formula: str = ""
    method_coverage: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["required_fields"] = list(self.required_fields)
        payload["paper_labels"] = list(self.paper_labels)
        payload["method_coverage"] = dict(self.method_coverage or {})
        return payload


@dataclass(frozen=True)
class MetricValue:
    metadata: MetricMetadata
    status: str
    value: float | None = None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.status == METRIC_AVAILABLE and self.value is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.metadata.name,
            "status": self.status,
            "value": self.value,
            "reason": self.reason,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class MetricDefinition:
    metadata: MetricMetadata
    compute: Callable[[MetricFields], MetricValue]


DEFAULT_EVALUATION_METRIC_NAMES: tuple[str, ...] = (
    "joint_mae",
    "joint_mse",
    "joint_rmse",
    "g1_joint_pos_rmse_rad",
    "max_joint_abs_error",
    "joint_velocity_rmse",
    "action_similarity",
    "predicted_joint_jump_rate",
    "target_joint_jump_rate",
    "predicted_minus_target_joint_jump_rate",
    "root_position_rmse",
    "root_rot6d_rmse",
    "global_body_position_error",
    "root_relative_body_position_error",
    "joint_rotation_error",
    "joint_velocity_error",
    "joint_acceleration_error",
    "joint_jump_count",
    "joint_limit_proximity_count",
    "self_collision_count",
    "floating_guard",
    "cross_ratio_guard",
    "phc_failure_guard",
    "mpjpe",
    "w_mpjpe",
    "predicted_frame_count",
    "target_frame_count",
    "frame_count_delta",
    "video_frame_count",
    "video_status_ok",
)

DEFAULT_ONLINE_METRIC_NAMES: tuple[str, ...] = (
    "loss",
    "g1_joint_pos_rmse_rad",
    "joint_velocity_rmse",
    "root_position_rmse",
    "root_rot6d_rmse",
    "mpjpe",
    "w_mpjpe",
)


def compute_metric_bundle(
    fields: MetricFields,
    metric_names: Sequence[str] | None = None,
    *,
    fps: float | None = None,
    joint_jump_velocity: float | None = None,
) -> dict[str, MetricValue]:
    """Compute requested metric results from one artifact or online batch.

    ``fields`` may hold Python sequences, NumPy arrays, or torch tensors. Tensor
    values are detached and copied to CPU before pure-Python reducers run.
    """

    payload = dict(fields)
    if fps is not None and "fps" not in payload:
        payload["fps"] = fps
    if joint_jump_velocity is not None and "joint_jump_velocity" not in payload:
        payload["joint_jump_velocity"] = joint_jump_velocity
    names = tuple(metric_names or DEFAULT_EVALUATION_METRIC_NAMES)
    results: dict[str, MetricValue] = {}
    for name in names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"unknown metric: {name}")
        definition = METRIC_REGISTRY[name]
        not_applicable = _method_not_applicable(definition.metadata, payload)
        if not_applicable is not None:
            results[name] = not_applicable
        else:
            results[name] = definition.compute(payload)
    return results


def compute_online_metrics(
    fields: MetricFields,
    metric_names: Sequence[str] | None = None,
    *,
    prefix: str = "",
    include_availability: bool = False,
) -> dict[str, float]:
    """Return W&B-ready scalar metrics for a training/eval batch."""

    bundle = compute_metric_bundle(fields, metric_names or DEFAULT_ONLINE_METRIC_NAMES)
    scalars: dict[str, float] = {}
    for name, result in bundle.items():
        key = f"{prefix}{name}"
        if result.available:
            scalars[key] = float(result.value)
        if include_availability:
            scalars[f"{key}_available"] = 1.0 if result.available else 0.0
    return scalars


def metric_metadata(metric_names: Sequence[str] | None = None) -> dict[str, dict[str, object]]:
    """Return registry metadata for docs, reports, and validation gates."""

    names = tuple(metric_names or DEFAULT_EVALUATION_METRIC_NAMES)
    metadata: dict[str, dict[str, object]] = {}
    for name in names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"unknown metric: {name}")
        metadata[name] = METRIC_REGISTRY[name].metadata.to_dict()
    return metadata


def flatten_metric_bundle(bundle: Mapping[str, MetricValue]) -> dict[str, object]:
    """Flatten metric results into CSV-friendly scalar/status/reason columns."""

    row: dict[str, object] = {}
    for name, result in bundle.items():
        if result.available:
            row[name] = float(result.value)
        row[f"{name}_status"] = result.status
        if result.reason:
            row[f"{name}_reason"] = result.reason
    return row


def mpjpe(predicted: Motion3D, target: Motion3D) -> float:
    """Mean per-joint position error for motions shaped T x J x 3."""

    return _mpjpe_value(predicted, target)


def weighted_mpjpe(predicted: Motion3D, target: Motion3D, weights: Sequence[float]) -> float:
    """Weighted mean per-joint position error for motions shaped T x J x 3."""

    return _weighted_mpjpe_value(predicted, target, weights)


def joint_mae(predicted: Motion1D, target: Motion1D) -> float:
    """Mean absolute error for joint vectors shaped T x D."""

    total, count = _joint_abs_error_total(predicted, target)
    return _safe_mean(total, count)


def joint_mse(predicted: Motion1D, target: Motion1D) -> float:
    """Mean squared error for joint vectors shaped T x D."""

    total, count = _joint_squared_error_total(predicted, target)
    return _safe_mean(total, count)


def joint_rmse(predicted: Motion1D, target: Motion1D) -> float:
    """Root mean squared error for joint vectors shaped T x D."""

    return math.sqrt(joint_mse(predicted, target))


def max_joint_abs_error(predicted: Motion1D, target: Motion1D) -> float:
    """Maximum absolute joint error over a motion shaped T x D."""

    max_error = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_value, target_value in _zip_equal(pred_frame, target_frame, "joint dimensions"):
            max_error = max(max_error, abs(pred_value - target_value))
            count += 1
    if count == 0:
        raise ValueError("metric received no samples")
    return max_error


def joint_velocity_rmse(predicted: Motion1D, target: Motion1D, fps: float) -> float:
    """RMSE between predicted and target joint velocities from finite differences."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(predicted) != len(target):
        raise ValueError(f"mismatched frames: {len(predicted)} != {len(target)}")
    if len(predicted) < 2:
        return 0.0
    return _joint_velocity_rmse_from_positions(predicted, target, fps=fps)


def action_similarity(predicted: Motion1D, target: Motion1D) -> float:
    """Average cosine similarity over action vectors shaped T x D."""

    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        total += _cosine(pred_frame, target_frame)
        count += 1
    return _safe_mean(total, count)


def joint_jump_rate(joint_positions: Motion1D, fps: float, max_velocity: float) -> float:
    """Fraction of joint velocity samples exceeding a hardware or data threshold."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_velocity <= 0:
        raise ValueError("max_velocity must be positive")
    if len(joint_positions) < 2:
        return 0.0
    jumps = 0
    total = 0
    for prev, cur in zip(joint_positions, joint_positions[1:]):
        for prev_value, cur_value in _zip_equal(prev, cur, "joint dimensions"):
            jumps += abs(cur_value - prev_value) * fps > max_velocity
            total += 1
    return _safe_mean(float(jumps), total)


def joint_limit_violation_rate(
    joint_positions: Motion1D,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> float:
    """Fraction of joint samples outside closed position limits."""

    if len(lower_limits) != len(upper_limits):
        raise ValueError("lower_limits and upper_limits must have the same length")
    violations = 0
    total = 0
    for frame in joint_positions:
        if len(frame) != len(lower_limits):
            raise ValueError("joint frame width must match limit width")
        for value, lower, upper in zip(frame, lower_limits, upper_limits):
            violations += value < lower or value > upper
            total += 1
    return _safe_mean(float(violations), total)


def contact_artifact_metrics(
    body_positions: Motion3D,
    *,
    fps: float,
    foot_indices: Sequence[int],
    ground_height: float = 0.0,
    up_axis: int | str = 2,
    contact_height_threshold: float = 0.04,
    max_contact_slide_speed: float = 0.25,
    max_contact_skate_distance: float = 0.02,
    contact_reference: Motion3D | None = None,
) -> dict[str, float]:
    """Foot contact artifacts for motions shaped T x J x 3.

    Contact is inferred from ``contact_reference`` when available; otherwise it
    is inferred from ``body_positions``. This lets evaluation measure whether a
    prediction floats or skates during target-contact frames.
    """

    _validate_contact_inputs(
        body_positions=body_positions,
        fps=fps,
        foot_indices=foot_indices,
        ground_height=ground_height,
        contact_height_threshold=contact_height_threshold,
        max_contact_slide_speed=max_contact_slide_speed,
        max_contact_skate_distance=max_contact_skate_distance,
        contact_reference=contact_reference,
    )
    axis = _axis_index(up_axis)
    horizontal_axes = tuple(index for index in range(3) if index != axis)
    reference = contact_reference if contact_reference is not None else body_positions

    contact_samples = 0
    foot_float_samples = 0
    total_foot_samples = 0
    foot_clearance_total = 0.0
    contact_clearance_total = 0.0
    penetration_frames = 0
    penetration_depth = 0.0

    for frame, ref_frame in _zip_equal(body_positions, reference, "frames"):
        frame_clearances = [_point_clearance(point, axis, ground_height) for point in frame]
        min_clearance = min(frame_clearances)
        if min_clearance < 0.0:
            penetration_frames += 1
            penetration_depth = max(penetration_depth, -min_clearance)
        for foot_index in foot_indices:
            foot_clearance = frame_clearances[foot_index]
            ref_clearance = _point_clearance(ref_frame[foot_index], axis, ground_height)
            expected_contact = ref_clearance <= contact_height_threshold
            total_foot_samples += 1
            foot_clearance_total += foot_clearance
            if expected_contact:
                contact_samples += 1
                contact_clearance_total += foot_clearance
                if foot_clearance > contact_height_threshold:
                    foot_float_samples += 1

    slide_samples = 0
    slide_violations = 0
    max_slide_speed = 0.0
    skate_distances: list[float] = []
    active_contact_start: dict[int, Vector] = {}
    active_contact_last: dict[int, Vector] = {}
    for frame_index, (prev_frame, cur_frame, prev_ref_frame, cur_ref_frame) in enumerate(
        zip(
            body_positions,
            body_positions[1:],
            reference,
            reference[1:],
        )
    ):
        for foot_index in foot_indices:
            prev_contact = (
                _point_clearance(prev_ref_frame[foot_index], axis, ground_height)
                <= contact_height_threshold
            )
            cur_contact = (
                _point_clearance(cur_ref_frame[foot_index], axis, ground_height)
                <= contact_height_threshold
            )
            if not (prev_contact and cur_contact):
                if prev_contact and foot_index in active_contact_start:
                    skate_distances.append(
                        _horizontal_distance(
                            active_contact_start.pop(foot_index),
                            active_contact_last.pop(foot_index, prev_frame[foot_index]),
                            horizontal_axes,
                        )
                    )
                if cur_contact:
                    active_contact_start[foot_index] = cur_frame[foot_index]
                    active_contact_last[foot_index] = cur_frame[foot_index]
                continue
            if foot_index not in active_contact_start:
                active_contact_start[foot_index] = prev_frame[foot_index]
            active_contact_last[foot_index] = cur_frame[foot_index]
            speed = _horizontal_speed(
                prev_frame[foot_index],
                cur_frame[foot_index],
                horizontal_axes,
                fps,
            )
            max_slide_speed = max(max_slide_speed, speed)
            slide_violations += speed > max_contact_slide_speed
            slide_samples += 1
            if frame_index == len(body_positions) - 2:
                skate_distances.append(
                    _horizontal_distance(
                        active_contact_start.pop(foot_index),
                        cur_frame[foot_index],
                        horizontal_axes,
                    )
                )

    for foot_index, start_position in active_contact_start.items():
        end_position = active_contact_last.get(foot_index, body_positions[-1][foot_index])
        skate_distances.append(_horizontal_distance(start_position, end_position, horizontal_axes))

    skate_violations = sum(distance > max_contact_skate_distance for distance in skate_distances)

    return {
        "contact_frame_ratio": _zero_mean(float(contact_samples), total_foot_samples),
        "foot_float_rate": _zero_mean(float(foot_float_samples), contact_samples),
        "contact_slide_rate": _zero_mean(float(slide_violations), slide_samples),
        "max_contact_slide_speed": max_slide_speed,
        "contact_skate_rate": _zero_mean(float(skate_violations), len(skate_distances)),
        "max_contact_skate_distance": max(skate_distances) if skate_distances else 0.0,
        "mean_foot_clearance": _zero_mean(foot_clearance_total, total_foot_samples),
        "mean_contact_foot_clearance": _zero_mean(contact_clearance_total, contact_samples),
        "ground_penetration_rate": _zero_mean(float(penetration_frames), len(body_positions)),
        "penetration_depth": penetration_depth,
    }


def _metadata(
    name: str,
    unit: str,
    direction: str,
    required_fields: tuple[str, ...],
    mask_semantics: str,
    reducer: str,
    description: str,
    *,
    paper_labels: tuple[str, ...] = (),
    formula: str = "",
    method_coverage: Mapping[str, object] | None = None,
) -> MetricMetadata:
    return MetricMetadata(
        name=name,
        unit=unit,
        direction=direction,
        required_fields=required_fields,
        mask_semantics=mask_semantics,
        reducer=reducer,
        description=description,
        paper_labels=paper_labels,
        formula=formula,
        method_coverage=method_coverage,
    )


def _ok(metadata: MetricMetadata, value: float) -> MetricValue:
    if not math.isfinite(float(value)):
        return _unavailable(metadata, "metric value is not finite")
    return MetricValue(metadata=metadata, status=METRIC_AVAILABLE, value=float(value))


def _unavailable(metadata: MetricMetadata, reason: str) -> MetricValue:
    return MetricValue(metadata=metadata, status=METRIC_UNAVAILABLE, reason=reason)


def _blocked(metadata: MetricMetadata, reason: str) -> MetricValue:
    return MetricValue(metadata=metadata, status=METRIC_BLOCKED, reason=reason)


def _not_applicable(metadata: MetricMetadata, reason: str) -> MetricValue:
    return MetricValue(metadata=metadata, status=METRIC_NOT_APPLICABLE, reason=reason)


def _method_not_applicable(metadata: MetricMetadata, fields: MetricFields) -> MetricValue | None:
    coverage = metadata.method_coverage or {}
    if not coverage:
        return None
    method = _first_present(fields, ("method_id", "method", "method_name"))
    if method is None:
        return None
    method_id = str(method).lower()
    if method_id in coverage:
        return None
    return _not_applicable(
        metadata,
        f"metric is not defined for method_id={method_id}",
    )


def _pair_metric(
    metadata: MetricMetadata,
    fields: MetricFields,
    *,
    predicted_aliases: tuple[str, ...],
    target_aliases: tuple[str, ...],
    value_fn: Callable[[Motion1D, Motion1D], float],
    require_two_frames: bool = False,
) -> MetricValue:
    predicted = _first_present(fields, predicted_aliases)
    target = _first_present(fields, target_aliases)
    if predicted is None or target is None:
        return _unavailable(
            metadata,
            _missing_reason(predicted_aliases, target_aliases, predicted, target),
        )
    pred_motion = _as_motion1d(predicted)
    target_motion = _as_motion1d(target)
    if require_two_frames and (len(pred_motion) < 2 or len(target_motion) < 2):
        return _unavailable(metadata, "requires at least two frames or explicit velocity fields")
    frame_mask = _optional_mask(fields, "frame_mask")
    joint_mask = _optional_mask(fields, "joint_mask")
    if frame_mask is not None or joint_mask is not None:
        pred_motion, target_motion = _mask_motion1d_pair(
            pred_motion,
            target_motion,
            frame_mask=frame_mask,
            joint_mask=joint_mask,
        )
    try:
        return _ok(metadata, value_fn(pred_motion, target_motion))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _body_metric(
    metadata: MetricMetadata,
    fields: MetricFields,
    *,
    weighted: bool,
) -> MetricValue:
    predicted = _first_present(fields, _PREDICTED_BODY_ALIASES)
    target = _first_present(fields, _TARGET_BODY_ALIASES)
    if predicted is None or target is None:
        return _unavailable(
            metadata,
            "requires paired predicted and target G1 body/link position arrays",
        )
    contract = _body_position_contract(fields)
    if not _contract_is_pinned(contract):
        return _blocked(
            metadata,
            "requires pinned FK/link/root-alignment contract before body-position "
            "metrics are valid",
        )
    pred_motion = _as_motion3d(predicted)
    target_motion = _as_motion3d(target)
    frame_mask = _optional_mask(fields, "frame_mask")
    body_mask = _optional_mask(fields, "body_mask")
    if frame_mask is not None or body_mask is not None:
        pred_motion, target_motion = _mask_motion3d_pair(
            pred_motion,
            target_motion,
            frame_mask=frame_mask,
            body_mask=body_mask,
        )
    try:
        if weighted:
            weights = _first_present(
                fields,
                ("body_position_weights", "mpjpe_weights", "joint_weights", "body_weights"),
            )
            if weights is None:
                return _unavailable(metadata, "requires body_position_weights or mpjpe_weights")
            value = _weighted_mpjpe_value(pred_motion, target_motion, _as_float_vector(weights))
        else:
            value = _mpjpe_value(pred_motion, target_motion)
    except ValueError as exc:
        return _unavailable(metadata, str(exc))
    return _ok(metadata, value)


def _root_relative_body_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    predicted = _first_present(fields, _PREDICTED_BODY_ALIASES)
    target = _first_present(fields, _TARGET_BODY_ALIASES)
    if predicted is None or target is None:
        return _unavailable(
            metadata,
            "requires paired predicted and target G1 body/link position arrays",
        )
    contract = _body_position_contract(fields)
    if not _contract_is_pinned(contract):
        return _blocked(
            metadata,
            "requires pinned FK/link/root-alignment contract before body-root subtraction "
            "is valid",
        )
    try:
        pred_motion = _as_motion3d(predicted)
        target_motion = _as_motion3d(target)
        pred_roots, target_roots = _root_position_motions(fields, pred_motion, target_motion)
        relative_pred = _subtract_root_positions(pred_motion, pred_roots)
        relative_target = _subtract_root_positions(target_motion, target_roots)
        frame_mask = _optional_mask(fields, "frame_mask")
        body_mask = _optional_mask(fields, "body_mask")
        if frame_mask is not None or body_mask is not None:
            relative_pred, relative_target = _mask_motion3d_pair(
                relative_pred,
                relative_target,
                frame_mask=frame_mask,
                body_mask=body_mask,
            )
        return _ok(metadata, _mpjpe_value(relative_pred, relative_target))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _loss_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    loss = _first_present(fields, ("loss", "train_loss", "eval_loss"))
    if loss is None:
        return _unavailable(metadata, "requires loss, train_loss, or eval_loss")
    try:
        return _ok(metadata, _as_float(loss))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _joint_velocity_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    predicted_velocity = _first_present(fields, _PREDICTED_JOINT_VELOCITY_ALIASES)
    target_velocity = _first_present(fields, _TARGET_JOINT_VELOCITY_ALIASES)
    if predicted_velocity is not None or target_velocity is not None:
        return _pair_metric(
            metadata,
            fields,
            predicted_aliases=_PREDICTED_JOINT_VELOCITY_ALIASES,
            target_aliases=_TARGET_JOINT_VELOCITY_ALIASES,
            value_fn=lambda pred, target: math.sqrt(
                _safe_mean(*_joint_squared_error_total(pred, target))
            ),
        )
    predicted = _first_present(fields, _PREDICTED_JOINT_ALIASES)
    target = _first_present(fields, _TARGET_JOINT_ALIASES)
    if predicted is None or target is None:
        return _unavailable(metadata, "requires joint positions or explicit joint velocities")
    if fields.get("independent_batch") is True:
        return _unavailable(
            metadata,
            "independent online batches need explicit joint velocity fields for velocity RMSE",
        )
    fps = _fps(fields)
    if fps <= 0:
        return _unavailable(metadata, "fps must be positive")
    pred_motion = _as_motion1d(predicted)
    target_motion = _as_motion1d(target)
    if len(pred_motion) < 2 or len(target_motion) < 2:
        return _unavailable(metadata, "requires at least two frames when velocities are absent")
    frame_mask = _optional_mask(fields, "frame_mask")
    joint_mask = _optional_mask(fields, "joint_mask")
    if frame_mask is not None or joint_mask is not None:
        pred_motion, target_motion = _mask_motion1d_pair(
            pred_motion,
            target_motion,
            frame_mask=frame_mask,
            joint_mask=joint_mask,
        )
    try:
        return _ok(
            metadata,
            _joint_velocity_rmse_from_positions(pred_motion, target_motion, fps=fps),
        )
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _joint_acceleration_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    predicted_acceleration = _first_present(fields, _PREDICTED_JOINT_ACCELERATION_ALIASES)
    target_acceleration = _first_present(fields, _TARGET_JOINT_ACCELERATION_ALIASES)
    if predicted_acceleration is not None or target_acceleration is not None:
        return _pair_metric(
            metadata,
            fields,
            predicted_aliases=_PREDICTED_JOINT_ACCELERATION_ALIASES,
            target_aliases=_TARGET_JOINT_ACCELERATION_ALIASES,
            value_fn=lambda pred, target: math.sqrt(
                _safe_mean(*_joint_squared_error_total(pred, target))
            ),
        )
    if fields.get("independent_batch") is True:
        return _unavailable(
            metadata,
            "independent online batches need explicit joint acceleration fields",
        )
    fps = _fps(fields)
    if fps <= 0:
        return _unavailable(metadata, "fps must be positive")
    predicted_velocity = _first_present(fields, _PREDICTED_JOINT_VELOCITY_ALIASES)
    target_velocity = _first_present(fields, _TARGET_JOINT_VELOCITY_ALIASES)
    try:
        if predicted_velocity is not None or target_velocity is not None:
            if predicted_velocity is None or target_velocity is None:
                return _unavailable(
                    metadata,
                    "requires paired predicted and target joint velocity fields",
                )
            pred_motion = _as_motion1d(predicted_velocity)
            target_motion = _as_motion1d(target_velocity)
            frame_mask = _optional_mask(fields, "frame_mask")
            joint_mask = _optional_mask(fields, "joint_mask")
            if frame_mask is not None or joint_mask is not None:
                pred_motion, target_motion = _mask_motion1d_pair(
                    pred_motion,
                    target_motion,
                    frame_mask=frame_mask,
                    joint_mask=joint_mask,
                )
            if len(pred_motion) < 2 or len(target_motion) < 2:
                return _unavailable(
                    metadata,
                    "requires at least two velocity frames when accelerations are absent",
                )
            return _ok(
                metadata,
                _joint_acceleration_rmse_from_velocities(
                    pred_motion,
                    target_motion,
                    fps=fps,
                ),
            )
        predicted = _first_present(fields, _PREDICTED_JOINT_ALIASES)
        target = _first_present(fields, _TARGET_JOINT_ALIASES)
        if predicted is None or target is None:
            return _unavailable(
                metadata,
                "requires joint accelerations, velocities, or positions with fps",
            )
        pred_motion = _as_motion1d(predicted)
        target_motion = _as_motion1d(target)
        frame_mask = _optional_mask(fields, "frame_mask")
        joint_mask = _optional_mask(fields, "joint_mask")
        if frame_mask is not None or joint_mask is not None:
            pred_motion, target_motion = _mask_motion1d_pair(
                pred_motion,
                target_motion,
                frame_mask=frame_mask,
                joint_mask=joint_mask,
            )
        if len(pred_motion) < 3 or len(target_motion) < 3:
            return _unavailable(
                metadata,
                "requires at least three position frames when accelerations are absent",
            )
        return _ok(
            metadata,
            _joint_acceleration_rmse_from_positions(pred_motion, target_motion, fps=fps),
        )
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _joint_rotation_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    predicted = _first_present(fields, _PREDICTED_JOINT_ROTATION_ALIASES)
    target = _first_present(fields, _TARGET_JOINT_ROTATION_ALIASES)
    if predicted is None or target is None:
        return _unavailable(
            metadata,
            _missing_reason(_PREDICTED_JOINT_ROTATION_ALIASES, _TARGET_JOINT_ROTATION_ALIASES, predicted, target),
        )
    try:
        pred_motion = _as_rotation_motion(predicted)
        target_motion = _as_rotation_motion(target)
        frame_mask = _optional_mask(fields, "frame_mask")
        joint_mask = _optional_mask(fields, "joint_mask")
        if frame_mask is not None or joint_mask is not None:
            pred_motion, target_motion = _mask_rotation_motion_pair(
                pred_motion,
                target_motion,
                frame_mask=frame_mask,
                joint_mask=joint_mask,
            )
        return _ok(metadata, _mean_geodesic_rotation_error(pred_motion, target_motion))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _joint_jump_count_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    positions = _first_present(fields, _JOINT_COUNT_POSITION_ALIASES)
    if positions is None:
        return _unavailable(
            metadata,
            f"requires one of {', '.join(_JOINT_COUNT_POSITION_ALIASES)}",
        )
    try:
        motion = _as_motion1d(positions)
        frame_mask = _optional_mask(fields, "frame_mask")
        joint_mask = _optional_mask(fields, "joint_mask")
        if frame_mask is not None or joint_mask is not None:
            motion = _mask_motion1d(motion, frame_mask=frame_mask, joint_mask=joint_mask)
        if len(motion) < 2:
            return _unavailable(metadata, "requires at least two frames")
        threshold = _as_float(fields.get("joint_jump_threshold_rad", 0.5))
        if threshold <= 0:
            return _unavailable(metadata, "joint_jump_threshold_rad must be positive")
        return _ok(metadata, float(_joint_jump_count(motion, threshold_rad=threshold)))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _joint_limit_proximity_count_metric(
    metadata: MetricMetadata,
    fields: MetricFields,
) -> MetricValue:
    positions = _first_present(fields, _JOINT_COUNT_POSITION_ALIASES)
    lower_limits = _first_present(fields, _JOINT_LOWER_LIMIT_ALIASES)
    upper_limits = _first_present(fields, _JOINT_UPPER_LIMIT_ALIASES)
    if positions is None or lower_limits is None or upper_limits is None:
        return _unavailable(
            metadata,
            "requires joint positions plus paired lower/upper joint limits",
        )
    try:
        motion = _as_motion1d(positions)
        lower = _as_float_vector(lower_limits)
        upper = _as_float_vector(upper_limits)
        if len(lower) != len(upper):
            return _unavailable(metadata, "lower and upper joint limits must have same width")
        frame_mask = _optional_mask(fields, "frame_mask")
        joint_mask = _optional_mask(fields, "joint_mask")
        if frame_mask is not None and len(frame_mask) != len(motion):
            return _unavailable(metadata, "frame_mask width must match frame count")
        if joint_mask is not None and len(joint_mask) != len(lower):
            return _unavailable(metadata, "joint_mask width must match joint limit width")
        threshold = _as_float(fields.get("joint_limit_proximity_threshold_rad", 0.05))
        if threshold < 0:
            return _unavailable(metadata, "joint_limit_proximity_threshold_rad must be non-negative")
        count = 0
        for frame_index, frame in enumerate(motion):
            if len(frame) != len(lower):
                return _unavailable(metadata, "joint frame width must match joint limit width")
            if frame_mask is not None and not frame_mask[frame_index]:
                continue
            for joint_index, value in enumerate(frame):
                if joint_mask is not None and not joint_mask[joint_index]:
                    continue
                distance_to_limit = min(value - lower[joint_index], upper[joint_index] - value)
                count += distance_to_limit <= threshold
        return _ok(metadata, float(count))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _self_collision_count_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    explicit = _first_present(fields, _SELF_COLLISION_COUNT_ALIASES)
    if explicit is not None:
        try:
            return _ok(metadata, _as_float(explicit))
        except ValueError as exc:
            return _unavailable(metadata, str(exc))
    frame_flags = _first_present(fields, _SELF_COLLISION_FRAME_ALIASES)
    if frame_flags is not None:
        try:
            return _ok(metadata, float(_count_truthy_frames(frame_flags)))
        except ValueError as exc:
            return _unavailable(metadata, str(exc))
    contacts = _first_present(fields, _MUJOCO_SELF_CONTACT_ALIASES)
    if contacts is not None:
        try:
            allowed = _allowed_contact_pairs(fields)
            return _ok(metadata, float(_count_non_allowed_contact_frames(contacts, allowed)))
        except ValueError as exc:
            return _unavailable(metadata, str(exc))
    return _unavailable(
        metadata,
        "requires MuJoCo self-contact count, frame flags, or per-frame contact pairs",
    )


def _floating_guard_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    try:
        threshold = _as_float(fields.get("floating_mean_lowest_foot_threshold_m", 0.10))
        mean_lowest_foot_height = _mean_lowest_foot_height(fields)
        if mean_lowest_foot_height is None:
            return _unavailable(
                metadata,
                "requires mean lowest-foot height or body positions with foot indices",
            )
        return _ok(metadata, 1.0 if mean_lowest_foot_height <= threshold else 0.0)
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _cross_ratio_guard_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    try:
        threshold = _as_float(fields.get("cross_ratio_threshold", 0.05))
        ratio_value = _first_present(fields, ("cross_ratio", "self_intersection_ratio"))
        if ratio_value is None:
            frames = _first_present(fields, ("self_intersection_frames", "crossing_frames"))
            if frames is None:
                return _unavailable(
                    metadata,
                    "requires cross_ratio/self_intersection_ratio or self_intersection_frames",
                )
            flags = _as_bool_sequence(frames)
            ratio = _safe_mean(float(sum(flags)), len(flags))
        else:
            ratio = _as_float(ratio_value)
        return _ok(metadata, 1.0 if ratio <= threshold else 0.0)
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _phc_failure_guard_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    value = _first_present(
        fields,
        (
            "phc_avg_body_joint_distance",
            "avg_body_joint_distance",
            "average_body_joint_distance",
        ),
    )
    if value is None:
        return _unavailable(metadata, "requires PHC avg_body_joint_distance")
    try:
        threshold = _as_float(fields.get("phc_avg_body_joint_distance_threshold_m", 0.5))
        distance = _as_float(value)
        return _ok(metadata, 1.0 if distance <= threshold else 0.0)
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _joint_jump_metric(
    metadata: MetricMetadata,
    fields: MetricFields,
    aliases: tuple[str, ...],
) -> MetricValue:
    positions = _first_present(fields, aliases)
    if positions is None:
        return _unavailable(metadata, f"requires one of {', '.join(aliases)}")
    fps = _fps(fields)
    max_velocity = _as_float(fields.get("joint_jump_velocity", 20.0))
    motion = _as_motion1d(positions)
    if len(motion) < 2:
        return _unavailable(metadata, "requires at least two frames")
    try:
        return _ok(metadata, joint_jump_rate(motion, fps=fps, max_velocity=max_velocity))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _jump_delta_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    predicted = _joint_jump_metric(metadata, fields, _PREDICTED_JOINT_ALIASES)
    target = _joint_jump_metric(metadata, fields, _TARGET_JOINT_ALIASES)
    if not predicted.available or not target.available:
        return _unavailable(metadata, "requires available predicted and target joint jump rates")
    return _ok(metadata, float(predicted.value) - float(target.value))


def _frame_count_metric(
    metadata: MetricMetadata,
    fields: MetricFields,
    *,
    explicit_aliases: tuple[str, ...],
    fallback_aliases: tuple[str, ...],
) -> MetricValue:
    explicit = _first_present(fields, explicit_aliases)
    if explicit is not None:
        try:
            return _ok(metadata, _as_float(explicit))
        except ValueError as exc:
            return _unavailable(metadata, str(exc))
    value = _first_present(fields, fallback_aliases)
    if value is None:
        reason = f"requires one of {', '.join(explicit_aliases + fallback_aliases)}"
        return _unavailable(metadata, reason)
    try:
        return _ok(metadata, float(len(_as_sequence(value))))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _frame_count_delta_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    predicted = _frame_count_metric(
        metadata,
        fields,
        explicit_aliases=("predicted_frame_count", "prediction_frame_count"),
        fallback_aliases=_PREDICTED_JOINT_ALIASES + _PREDICTED_BODY_ALIASES,
    )
    target = _frame_count_metric(
        metadata,
        fields,
        explicit_aliases=("target_frame_count",),
        fallback_aliases=_TARGET_JOINT_ALIASES + _TARGET_BODY_ALIASES,
    )
    if not predicted.available or not target.available:
        return _unavailable(metadata, "requires predicted and target frame counts")
    return _ok(metadata, abs(float(predicted.value) - float(target.value)))


def _video_frame_count_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    value = _first_present(
        fields,
        (
            "video_frame_count",
            "rendered_frame_count",
            "frames_written",
            "raw_trajectory_frames",
        ),
    )
    if value is None:
        return _unavailable(metadata, "requires video_frame_count or rendered_frame_count")
    try:
        return _ok(metadata, _as_float(value))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


def _video_status_metric(metadata: MetricMetadata, fields: MetricFields) -> MetricValue:
    status = _first_present(fields, ("video_status", "render_status", "status"))
    if status is None:
        return _unavailable(metadata, "requires video_status, render_status, or status")
    if isinstance(status, bool):
        return _ok(metadata, 1.0 if status else 0.0)
    return _ok(metadata, 1.0 if str(status).lower() in {"ok", "success", "passed", "true"} else 0.0)


def _root_metric(
    metadata: MetricMetadata,
    fields: MetricFields,
    predicted_aliases: tuple[str, ...],
    target_aliases: tuple[str, ...],
) -> MetricValue:
    return _pair_metric(
        metadata,
        fields,
        predicted_aliases=predicted_aliases,
        target_aliases=target_aliases,
        value_fn=lambda pred, target: math.sqrt(
            _safe_mean(*_joint_squared_error_total(pred, target))
        ),
    )


def _first_present(fields: MetricFields, aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in fields and fields[alias] is not None:
            return fields[alias]
    return None


def _missing_reason(
    predicted_aliases: tuple[str, ...],
    target_aliases: tuple[str, ...],
    predicted: Any,
    target: Any,
) -> str:
    missing: list[str] = []
    if predicted is None:
        missing.append("predicted field")
    if target is None:
        missing.append("target field")
    return f"requires {', '.join(missing)} from aliases {predicted_aliases} / {target_aliases}"


def _as_sequence(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _as_float(value: Any) -> float:
    value = _as_sequence(value)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("scalar metric field must contain exactly one value")
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric field is not numeric") from exc


def _as_float_vector(value: Any) -> list[float]:
    value = _as_sequence(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("weights must be a numeric sequence")
    return [float(item) for item in value]


def _as_motion1d(value: Any) -> list[list[float]]:
    value = _as_sequence(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("motion must be a sequence")
    if not value:
        raise ValueError("motion must not be empty")
    first = value[0]
    if _is_number(first):
        return [[float(item) for item in value]]
    motion: list[list[float]] = []
    for frame in value:
        if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
            raise ValueError("motion frames must be numeric sequences")
        motion.append([float(item) for item in frame])
    return motion


def _as_motion3d(value: Any) -> list[list[list[float]]]:
    value = _as_sequence(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("3D motion must be a sequence")
    if not value:
        raise ValueError("3D motion must not be empty")
    motion: list[list[list[float]]] = []
    for frame in value:
        if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
            raise ValueError("3D motion frames must be sequences")
        joints: list[list[float]] = []
        for point in frame:
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
                raise ValueError("3D points must be sequences")
            vector = [float(item) for item in point]
            if len(vector) != 3:
                raise ValueError("3D body/link positions must end in 3 values")
            joints.append(vector)
        motion.append(joints)
    return motion


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _optional_mask(fields: MetricFields, key: str) -> list[bool] | None:
    value = fields.get(key)
    if value is None:
        return None
    value = _as_sequence(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")
    return [bool(item) for item in value]


def _mask_motion1d_pair(
    predicted: Motion1D,
    target: Motion1D,
    *,
    frame_mask: Sequence[bool] | None,
    joint_mask: Sequence[bool] | None,
) -> tuple[list[list[float]], list[list[float]]]:
    masked_predicted: list[list[float]] = []
    masked_target: list[list[float]] = []
    if frame_mask is not None and len(frame_mask) != len(predicted):
        raise ValueError("frame_mask width must match frame count")
    for frame_index, (pred_frame, target_frame) in enumerate(
        _zip_equal(predicted, target, "frames")
    ):
        if frame_mask is not None and not frame_mask[frame_index]:
            continue
        if joint_mask is not None and len(joint_mask) != len(pred_frame):
            raise ValueError("joint_mask width must match joint dimension")
        pred_values: list[float] = []
        target_values: list[float] = []
        for value_index, (pred_value, target_value) in enumerate(
            _zip_equal(pred_frame, target_frame, "joint dimensions")
        ):
            if joint_mask is not None and not joint_mask[value_index]:
                continue
            pred_values.append(float(pred_value))
            target_values.append(float(target_value))
        masked_predicted.append(pred_values)
        masked_target.append(target_values)
    return masked_predicted, masked_target


def _mask_motion1d(
    motion: Motion1D,
    *,
    frame_mask: Sequence[bool] | None,
    joint_mask: Sequence[bool] | None,
) -> list[list[float]]:
    if frame_mask is not None and len(frame_mask) != len(motion):
        raise ValueError("frame_mask width must match frame count")
    masked_motion: list[list[float]] = []
    for frame_index, frame in enumerate(motion):
        if frame_mask is not None and not frame_mask[frame_index]:
            continue
        if joint_mask is not None and len(joint_mask) != len(frame):
            raise ValueError("joint_mask width must match joint dimension")
        values: list[float] = []
        for joint_index, value in enumerate(frame):
            if joint_mask is not None and not joint_mask[joint_index]:
                continue
            values.append(float(value))
        masked_motion.append(values)
    return masked_motion


def _mask_motion3d_pair(
    predicted: Motion3D,
    target: Motion3D,
    *,
    frame_mask: Sequence[bool] | None,
    body_mask: Sequence[bool] | None,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    masked_predicted: list[list[list[float]]] = []
    masked_target: list[list[list[float]]] = []
    if frame_mask is not None and len(frame_mask) != len(predicted):
        raise ValueError("frame_mask width must match frame count")
    for frame_index, (pred_frame, target_frame) in enumerate(
        _zip_equal(predicted, target, "frames")
    ):
        if frame_mask is not None and not frame_mask[frame_index]:
            continue
        if body_mask is not None and len(body_mask) != len(pred_frame):
            raise ValueError("body_mask width must match body/link count")
        pred_points: list[list[float]] = []
        target_points: list[list[float]] = []
        for body_index, (pred_point, target_point) in enumerate(
            _zip_equal(pred_frame, target_frame, "body/link positions")
        ):
            if body_mask is not None and not body_mask[body_index]:
                continue
            pred_points.append([float(item) for item in pred_point])
            target_points.append([float(item) for item in target_point])
        masked_predicted.append(pred_points)
        masked_target.append(target_points)
    return masked_predicted, masked_target


def _as_rotation_motion(value: Any) -> list[list[list[float]]]:
    value = _as_sequence(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("joint rotations must be a sequence")
    if not value:
        raise ValueError("joint rotations must not be empty")
    if _is_rotation_leaf(value):
        return [[_flatten_rotation(value)]]
    first = value[0]
    if _is_rotation_leaf(first):
        return [[_flatten_rotation(frame)] for frame in value]
    motion: list[list[list[float]]] = []
    for frame in value:
        if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
            raise ValueError("rotation frames must be sequences")
        if not frame:
            raise ValueError("rotation frames must not be empty")
        joints: list[list[float]] = []
        for rotation in frame:
            joints.append(_flatten_rotation(rotation))
        motion.append(joints)
    return motion


def _mask_rotation_motion_pair(
    predicted: Sequence[Sequence[Sequence[float]]],
    target: Sequence[Sequence[Sequence[float]]],
    *,
    frame_mask: Sequence[bool] | None,
    joint_mask: Sequence[bool] | None,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    if frame_mask is not None and len(frame_mask) != len(predicted):
        raise ValueError("frame_mask width must match frame count")
    masked_predicted: list[list[list[float]]] = []
    masked_target: list[list[list[float]]] = []
    for frame_index, (pred_frame, target_frame) in enumerate(
        _zip_equal(predicted, target, "frames")
    ):
        if frame_mask is not None and not frame_mask[frame_index]:
            continue
        if joint_mask is not None and len(joint_mask) != len(pred_frame):
            raise ValueError("joint_mask width must match joint rotation count")
        pred_rotations: list[list[float]] = []
        target_rotations: list[list[float]] = []
        for joint_index, (pred_rotation, target_rotation) in enumerate(
            _zip_equal(pred_frame, target_frame, "joint rotations")
        ):
            if joint_mask is not None and not joint_mask[joint_index]:
                continue
            pred_rotations.append(list(pred_rotation))
            target_rotations.append(list(target_rotation))
        masked_predicted.append(pred_rotations)
        masked_target.append(target_rotations)
    return masked_predicted, masked_target


def _flatten_rotation(rotation: Any) -> list[float]:
    rotation = _as_sequence(rotation)
    if _is_flat_numeric_sequence(rotation):
        values = [float(item) for item in rotation]
        if len(values) in {4, 9}:
            return values
    if _is_matrix3x3(rotation):
        return [float(item) for row in rotation for item in row]
    raise ValueError("joint rotations must be quaternions or 3x3 rotation matrices")


def _is_rotation_leaf(value: Any) -> bool:
    return _is_flat_numeric_sequence(value) and len(value) in {4, 9} or _is_matrix3x3(value)


def _is_flat_numeric_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
        and all(_is_number(item) for item in value)
    )


def _is_matrix3x3(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
        and all(_is_flat_numeric_sequence(row) and len(row) == 3 for row in value)
    )


def _mean_geodesic_rotation_error(
    predicted: Sequence[Sequence[Sequence[float]]],
    target: Sequence[Sequence[Sequence[float]]],
) -> float:
    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_rotation, target_rotation in _zip_equal(
            pred_frame,
            target_frame,
            "joint rotations",
        ):
            total += _geodesic_rotation_error(pred_rotation, target_rotation)
            count += 1
    return _safe_mean(total, count)


def _geodesic_rotation_error(predicted: Sequence[float], target: Sequence[float]) -> float:
    if len(predicted) != len(target):
        raise ValueError("predicted and target joint rotations must use the same representation")
    if len(predicted) == 4:
        pred_quat = _normalized_quaternion(predicted)
        target_quat = _normalized_quaternion(target)
        dot = abs(sum(a * b for a, b in zip(pred_quat, target_quat)))
        return 2.0 * math.acos(_clamp(dot, -1.0, 1.0))
    if len(predicted) == 9:
        trace = sum(a * b for a, b in zip(predicted, target))
        return math.acos(_clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    raise ValueError("joint rotations must be quaternions or 3x3 rotation matrices")


def _normalized_quaternion(quaternion: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        raise ValueError("quaternion norm must be non-zero")
    return [float(value) / norm for value in quaternion]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _joint_abs_error_total(
    predicted: Motion1D,
    target: Motion1D,
) -> tuple[float, int]:
    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_value, target_value in _zip_equal(pred_frame, target_frame, "joint dimensions"):
            total += abs(pred_value - target_value)
            count += 1
    return total, count


def _joint_squared_error_total(
    predicted: Motion1D,
    target: Motion1D,
) -> tuple[float, int]:
    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_value, target_value in _zip_equal(pred_frame, target_frame, "joint dimensions"):
            total += (pred_value - target_value) ** 2
            count += 1
    return total, count


def _joint_velocity_rmse_from_positions(
    predicted: Motion1D,
    target: Motion1D,
    *,
    fps: float,
) -> float:
    total = 0.0
    count = 0
    for pred_prev, pred_cur, target_prev, target_cur in zip(
        predicted,
        predicted[1:],
        target,
        target[1:],
    ):
        for pred_prev_value, pred_cur_value, target_prev_value, target_cur_value in zip(
            pred_prev,
            pred_cur,
            target_prev,
            target_cur,
            strict=True,
        ):
            pred_velocity = (pred_cur_value - pred_prev_value) * fps
            target_velocity = (target_cur_value - target_prev_value) * fps
            total += (pred_velocity - target_velocity) ** 2
            count += 1
    return math.sqrt(_safe_mean(total, count))


def _joint_acceleration_rmse_from_velocities(
    predicted: Motion1D,
    target: Motion1D,
    *,
    fps: float,
) -> float:
    if len(predicted) != len(target):
        raise ValueError(f"mismatched frames: {len(predicted)} != {len(target)}")
    total = 0.0
    count = 0
    for frame_index in range(1, len(predicted)):
        pred_prev = predicted[frame_index - 1]
        pred_cur = predicted[frame_index]
        target_prev = target[frame_index - 1]
        target_cur = target[frame_index]
        for pred_prev_value, pred_cur_value, target_prev_value, target_cur_value in zip(
            pred_prev,
            pred_cur,
            target_prev,
            target_cur,
            strict=True,
        ):
            pred_acceleration = (pred_cur_value - pred_prev_value) * fps
            target_acceleration = (target_cur_value - target_prev_value) * fps
            total += (pred_acceleration - target_acceleration) ** 2
            count += 1
    return math.sqrt(_safe_mean(total, count))


def _joint_acceleration_rmse_from_positions(
    predicted: Motion1D,
    target: Motion1D,
    *,
    fps: float,
) -> float:
    if len(predicted) != len(target):
        raise ValueError(f"mismatched frames: {len(predicted)} != {len(target)}")
    total = 0.0
    count = 0
    scale = fps * fps
    for frame_index in range(2, len(predicted)):
        pred_prev2 = predicted[frame_index - 2]
        pred_prev = predicted[frame_index - 1]
        pred_cur = predicted[frame_index]
        target_prev2 = target[frame_index - 2]
        target_prev = target[frame_index - 1]
        target_cur = target[frame_index]
        for (
            pred_prev2_value,
            pred_prev_value,
            pred_cur_value,
            target_prev2_value,
            target_prev_value,
            target_cur_value,
        ) in zip(
            pred_prev2,
            pred_prev,
            pred_cur,
            target_prev2,
            target_prev,
            target_cur,
            strict=True,
        ):
            pred_acceleration = (pred_cur_value - 2.0 * pred_prev_value + pred_prev2_value) * scale
            target_acceleration = (
                target_cur_value - 2.0 * target_prev_value + target_prev2_value
            ) * scale
            total += (pred_acceleration - target_acceleration) ** 2
            count += 1
    return math.sqrt(_safe_mean(total, count))


def _mpjpe_value(predicted: Motion3D, target: Motion3D) -> float:
    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_joint, target_joint in _zip_equal(pred_frame, target_frame, "joints"):
            if len(pred_joint) != 3 or len(target_joint) != 3:
                raise ValueError("mpjpe expects 3D joint vectors")
            total += math.dist(pred_joint, target_joint)
            count += 1
    return _safe_mean(total, count)


def _weighted_mpjpe_value(predicted: Motion3D, target: Motion3D, weights: Sequence[float]) -> float:
    if not weights:
        raise ValueError("W-MPJPE requires at least one weight")
    total = 0.0
    weight_total = 0.0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        if len(pred_frame) != len(weights):
            raise ValueError("W-MPJPE weight width must match body/link count")
        for body_index, (pred_joint, target_joint) in enumerate(
            _zip_equal(pred_frame, target_frame, "joints")
        ):
            if len(pred_joint) != 3 or len(target_joint) != 3:
                raise ValueError("W-MPJPE expects 3D joint vectors")
            weight = float(weights[body_index])
            if weight < 0.0:
                raise ValueError("W-MPJPE weights must be non-negative")
            total += weight * math.dist(pred_joint, target_joint)
            weight_total += weight
    if weight_total == 0.0:
        raise ValueError("W-MPJPE weights sum to zero")
    return total / weight_total


def _body_position_contract(fields: MetricFields) -> Any:
    return _first_present(
        fields,
        (
            "body_position_mpjpe_contract",
            "g1_body_position_contract",
            "fk_link_root_alignment_contract",
            "mpjpe_contract",
        ),
    )


def _root_position_motions(
    fields: MetricFields,
    pred_motion: Motion3D,
    target_motion: Motion3D,
) -> tuple[list[list[float]], list[list[float]]]:
    pred_root = _first_present(fields, _PREDICTED_ROOT_POS_ALIASES)
    target_root = _first_present(fields, _TARGET_ROOT_POS_ALIASES)
    if pred_root is not None or target_root is not None:
        if pred_root is None or target_root is None:
            raise ValueError("requires paired predicted and target root position arrays")
        pred_roots = _as_root_motion(pred_root)
        target_roots = _as_root_motion(target_root)
        _validate_root_frame_count(pred_roots, pred_motion, "predicted root positions")
        _validate_root_frame_count(target_roots, target_motion, "target root positions")
        return pred_roots, target_roots
    root_index = _root_body_index(fields)
    if root_index is None:
        raise ValueError("requires root position arrays or explicit root_body_index/root_link_index")
    try:
        return (
            [[float(item) for item in frame[root_index]] for frame in pred_motion],
            [[float(item) for item in frame[root_index]] for frame in target_motion],
        )
    except IndexError as exc:
        raise ValueError("root body index out of range") from exc


def _as_root_motion(value: Any) -> list[list[float]]:
    motion = _as_motion1d(value)
    for frame in motion:
        if len(frame) != 3:
            raise ValueError("root position arrays must contain 3D vectors")
    return motion


def _validate_root_frame_count(
    roots: Sequence[Sequence[float]],
    motion: Motion3D,
    label: str,
) -> None:
    if len(roots) != len(motion):
        raise ValueError(f"{label} frame count must match body-position frame count")


def _subtract_root_positions(
    motion: Motion3D,
    roots: Sequence[Sequence[float]],
) -> list[list[list[float]]]:
    relative: list[list[list[float]]] = []
    for frame, root in _zip_equal(motion, roots, "frames"):
        if len(root) != 3:
            raise ValueError("root position arrays must contain 3D vectors")
        relative.append(
            [
                [
                    float(point[0]) - float(root[0]),
                    float(point[1]) - float(root[1]),
                    float(point[2]) - float(root[2]),
                ]
                for point in frame
            ]
        )
    return relative


def _root_body_index(fields: MetricFields) -> int | None:
    explicit = _first_present(
        fields,
        ("root_body_index", "body_root_index", "root_link_index", "root_index"),
    )
    if explicit is not None:
        return int(_as_float(explicit))
    root_name = _first_present(fields, ("root_body_name", "root_link_name", "root_name"))
    if root_name is None:
        return None
    names = _first_present(fields, ("body_names", "link_order", "link_names"))
    if names is None:
        contract = _body_position_contract(fields)
        if isinstance(contract, Mapping):
            names = _first_present(contract, ("body_names", "link_order", "link_names"))
    if names is None:
        return None
    body_names = [str(name) for name in _as_sequence(names)]
    try:
        return body_names.index(str(root_name))
    except ValueError:
        return None


def _contract_is_pinned(contract: Any) -> bool:
    if contract is None:
        return False
    if isinstance(contract, str):
        return contract.lower() in {"pinned", "accepted", "locked"}
    if not isinstance(contract, Mapping):
        return False
    status = str(contract.get("status", "")).lower()
    required = (
        contract.get("pinned") is True
        or status in {"pinned", "accepted", "locked"}
    )
    if not required:
        return False
    has_link_order = bool(contract.get("link_order") or contract.get("body_names"))
    has_units = str(contract.get("units", contract.get("position_units", ""))).lower() in {
        "m",
        "meter",
        "meters",
    }
    has_alignment = bool(
        contract.get("root_alignment")
        or contract.get("alignment")
        or contract.get("root_alignment_contract")
    )
    return has_link_order and has_units and has_alignment


def _joint_jump_count(motion: Motion1D, *, threshold_rad: float) -> int:
    jumps = 0
    for prev, cur in zip(motion, motion[1:]):
        for prev_value, cur_value in _zip_equal(prev, cur, "joint dimensions"):
            jumps += abs(_wrap_angle_delta(cur_value - prev_value)) > threshold_rad
    return jumps


def _wrap_angle_delta(delta: float) -> float:
    return (float(delta) + math.pi) % (2.0 * math.pi) - math.pi


def _count_truthy_frames(value: Any) -> int:
    return sum(_as_bool_sequence(value))


def _as_bool_sequence(value: Any) -> list[bool]:
    value = _as_sequence(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("frame flags must be a sequence")
    return [bool(item) for item in value]


def _allowed_contact_pairs(fields: MetricFields) -> set[tuple[str, str]]:
    raw_allowed = _first_present(
        fields,
        (
            "allowed_self_contact_pairs",
            "allowed_contact_pairs",
            "mujoco_allowed_self_contact_pairs",
        ),
    )
    if raw_allowed is None:
        return set()
    allowed_pairs = _as_sequence(raw_allowed)
    if not isinstance(allowed_pairs, Sequence) or isinstance(allowed_pairs, (str, bytes)):
        raise ValueError("allowed contact pairs must be a sequence")
    return {_contact_pair_key(pair) for pair in allowed_pairs}


def _count_non_allowed_contact_frames(
    contacts: Any,
    allowed: set[tuple[str, str]],
) -> int:
    frames = _as_sequence(contacts)
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("MuJoCo contacts must be a per-frame sequence")
    if not frames:
        return 0
    if _looks_like_contact(frames):
        frames = [frames]
    return sum(_frame_has_non_allowed_contact(frame, allowed) for frame in frames)


def _frame_has_non_allowed_contact(frame: Any, allowed: set[tuple[str, str]]) -> bool:
    frame = _as_sequence(frame)
    if isinstance(frame, bool):
        return frame
    if isinstance(frame, Mapping):
        if "has_self_collision" in frame:
            return bool(frame["has_self_collision"])
        if "has_self_contact" in frame:
            return bool(frame["has_self_contact"])
        frame = _first_present(frame, ("self_contacts", "contacts", "mujoco_contacts"))
        if frame is None:
            return False
    if _looks_like_contact(frame):
        frame = [frame]
    if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
        raise ValueError("MuJoCo contact frames must contain contact pairs")
    for contact in frame:
        if _contact_pair_key(contact) not in allowed:
            return True
    return False


def _looks_like_contact(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in value
            for key in (
                "body1",
                "body2",
                "body_a",
                "body_b",
                "geom1",
                "geom2",
                "geom_a",
                "geom_b",
                "pair",
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return not isinstance(value[0], (Mapping, list, tuple))
    return False


def _contact_pair_key(contact: Any) -> tuple[str, str]:
    contact = _as_sequence(contact)
    left: Any | None = None
    right: Any | None = None
    if isinstance(contact, Mapping):
        if "pair" in contact:
            return _contact_pair_key(contact["pair"])
        for left_key, right_key in (
            ("body1", "body2"),
            ("body_a", "body_b"),
            ("geom1", "geom2"),
            ("geom_a", "geom_b"),
            ("name1", "name2"),
        ):
            if left_key in contact and right_key in contact:
                left = contact[left_key]
                right = contact[right_key]
                break
    elif isinstance(contact, Sequence) and not isinstance(contact, (str, bytes)) and len(contact) >= 2:
        left = contact[0]
        right = contact[1]
    if left is None or right is None:
        raise ValueError("MuJoCo contact pairs need two body/geom names")
    a = str(left)
    b = str(right)
    return (a, b) if a <= b else (b, a)


def _mean_lowest_foot_height(fields: MetricFields) -> float | None:
    explicit_mean = _first_present(
        fields,
        (
            "mean_lowest_foot_height",
            "mean_lowest_foot_clearance",
            "lowest_foot_height_mean",
            "floating_mean_lowest_foot_m",
        ),
    )
    if explicit_mean is not None:
        return _mean_scalar_or_sequence(explicit_mean)
    heights = _first_present(
        fields,
        (
            "lowest_foot_heights",
            "lowest_foot_clearances",
            "min_foot_heights",
            "min_foot_clearances",
        ),
    )
    if heights is not None:
        return _mean_scalar_or_sequence(heights)
    body_positions = _first_present(fields, _PREDICTED_BODY_ALIASES + ("body_positions",))
    foot_indices = _metric_foot_indices(fields)
    if body_positions is None or not foot_indices:
        return None
    motion = _as_motion3d(body_positions)
    ground_height = _as_float(fields.get("ground_height", 0.0))
    axis = _axis_index(fields.get("up_axis", 2))
    frame_lows: list[float] = []
    for frame in motion:
        clearances = [
            _point_clearance(frame[index], axis, ground_height)
            for index in foot_indices
            if index < len(frame)
        ]
        if not clearances:
            raise ValueError("foot index out of range")
        frame_lows.append(min(clearances))
    return _safe_mean(sum(frame_lows), len(frame_lows))


def _mean_scalar_or_sequence(value: Any) -> float:
    value = _as_sequence(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("mean input sequence must not be empty")
        if all(_is_number(item) for item in value):
            return _safe_mean(sum(float(item) for item in value), len(value))
    return _as_float(value)


def _metric_foot_indices(fields: MetricFields) -> tuple[int, ...]:
    explicit = fields.get("foot_indices")
    if explicit is not None:
        values = _as_sequence(explicit)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("foot_indices must be a sequence")
        return tuple(int(_as_float(value)) for value in values)
    body_names = _first_present(fields, ("body_names", "link_order", "link_names"))
    foot_names = _first_present(fields, ("foot_body_names", "foot_names", "foot_link_names"))
    if body_names is None or foot_names is None:
        return ()
    body_name_to_index = {str(name): index for index, name in enumerate(_as_sequence(body_names))}
    return tuple(
        body_name_to_index[str(name)]
        for name in _as_sequence(foot_names)
        if str(name) in body_name_to_index
    )


def _fps(fields: MetricFields) -> float:
    return _as_float(fields.get("fps", 30.0))


def _cosine(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same width")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _validate_contact_inputs(
    *,
    body_positions: Motion3D,
    fps: float,
    foot_indices: Sequence[int],
    ground_height: float,
    contact_height_threshold: float,
    max_contact_slide_speed: float,
    max_contact_skate_distance: float,
    contact_reference: Motion3D | None,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not math.isfinite(ground_height):
        raise ValueError("ground_height must be finite")
    if contact_height_threshold < 0:
        raise ValueError("contact_height_threshold must be non-negative")
    if max_contact_slide_speed <= 0:
        raise ValueError("max_contact_slide_speed must be positive")
    if max_contact_skate_distance <= 0:
        raise ValueError("max_contact_skate_distance must be positive")
    if not foot_indices:
        raise ValueError("foot_indices must not be empty")
    if not body_positions:
        raise ValueError("body_positions must not be empty")
    reference = contact_reference if contact_reference is not None else body_positions
    for frame, ref_frame in _zip_equal(body_positions, reference, "frames"):
        if len(frame) != len(ref_frame):
            raise ValueError("contact reference body count must match body_positions")
        for point in (*frame, *ref_frame):
            if len(point) != 3:
                raise ValueError("contact metrics expect 3D body positions")
    body_count = len(body_positions[0])
    for foot_index in foot_indices:
        if foot_index < 0 or foot_index >= body_count:
            raise ValueError("foot index out of range")


def _axis_index(axis: int | str) -> int:
    if isinstance(axis, str):
        axis = axis.lower()
        if axis == "x":
            return 0
        if axis == "y":
            return 1
        if axis == "z":
            return 2
        try:
            axis = int(axis)
        except ValueError as exc:
            raise ValueError("up_axis must be x, y, z, 0, 1, or 2") from exc
    if axis in (0, 1, 2):
        return axis
    raise ValueError("up_axis must be x, y, z, 0, 1, or 2")


def _point_clearance(point: Vector, axis: int, ground_height: float) -> float:
    return point[axis] - ground_height


def _horizontal_speed(
    left: Vector,
    right: Vector,
    horizontal_axes: Sequence[int],
    fps: float,
) -> float:
    return _horizontal_distance(left, right, horizontal_axes) * fps


def _horizontal_distance(left: Vector, right: Vector, horizontal_axes: Sequence[int]) -> float:
    squared = sum((right[index] - left[index]) ** 2 for index in horizontal_axes)
    return math.sqrt(squared)


def _safe_mean(total: float, count: int) -> float:
    if count == 0:
        raise ValueError("metric received no samples")
    return total / count


def _zero_mean(total: float, count: int) -> float:
    if count == 0:
        return 0.0
    return total / count


def _zip_equal(left: Sequence, right: Sequence, label: str):
    if len(left) != len(right):
        raise ValueError(f"mismatched {label}: {len(left)} != {len(right)}")
    return zip(left, right)


_PREDICTED_JOINT_ALIASES = (
    "predicted_joints",
    "predicted_joint_positions",
    "pred_joint_positions",
    "pred_joint_pos",
)
_TARGET_JOINT_ALIASES = (
    "target_joints",
    "target_joint_positions",
    "target_joint_pos",
)
_PREDICTED_JOINT_VELOCITY_ALIASES = (
    "predicted_joint_velocities",
    "predicted_joint_velocity",
    "pred_joint_velocities",
    "pred_joint_velocity",
)
_TARGET_JOINT_VELOCITY_ALIASES = (
    "target_joint_velocities",
    "target_joint_velocity",
    "tgt_joint_velocities",
)
_PREDICTED_JOINT_ACCELERATION_ALIASES = (
    "predicted_joint_accelerations",
    "predicted_joint_acceleration",
    "pred_joint_accelerations",
    "pred_joint_acceleration",
)
_TARGET_JOINT_ACCELERATION_ALIASES = (
    "target_joint_accelerations",
    "target_joint_acceleration",
    "tgt_joint_accelerations",
)
_PREDICTED_JOINT_ROTATION_ALIASES = (
    "predicted_joint_rotations",
    "predicted_joint_rotation",
    "predicted_joint_quaternions",
    "predicted_joint_rotation_matrices",
    "pred_joint_rotations",
    "pred_joint_quaternions",
)
_TARGET_JOINT_ROTATION_ALIASES = (
    "target_joint_rotations",
    "target_joint_rotation",
    "target_joint_quaternions",
    "target_joint_rotation_matrices",
)
_JOINT_COUNT_POSITION_ALIASES = (
    "joint_positions",
    "qpos",
    "q",
) + _PREDICTED_JOINT_ALIASES
_JOINT_LOWER_LIMIT_ALIASES = (
    "joint_lower_limits",
    "lower_joint_limits",
    "joint_limit_lower",
    "lower_limits",
)
_JOINT_UPPER_LIMIT_ALIASES = (
    "joint_upper_limits",
    "upper_joint_limits",
    "joint_limit_upper",
    "upper_limits",
)
_SELF_COLLISION_COUNT_ALIASES = (
    "self_collision_count",
    "mujoco_self_collision_count",
    "mujoco_self_contact_count",
)
_SELF_COLLISION_FRAME_ALIASES = (
    "self_collision_frames",
    "mujoco_self_collision_frames",
    "mujoco_self_contact_frames",
)
_MUJOCO_SELF_CONTACT_ALIASES = (
    "mujoco_self_contacts",
    "mujoco_self_contact_pairs",
    "mujoco_contacts",
)
_PREDICTED_ROOT_POS_ALIASES = (
    "predicted_root_positions",
    "predicted_root_pos",
    "pred_root_positions",
    "pred_root_pos",
    "pred_root_pos_w",
    "predicted_root_pos_w",
)
_TARGET_ROOT_POS_ALIASES = (
    "target_root_positions",
    "target_root_pos",
    "target_root_pos_w",
)
_PREDICTED_ROOT_ROT6D_ALIASES = (
    "predicted_root_rot6d",
    "pred_root_rot6d",
    "predicted_root_rot_6d",
    "pred_root_rot_6d",
)
_TARGET_ROOT_ROT6D_ALIASES = (
    "target_root_rot6d",
    "target_root_rot_6d",
)
_PREDICTED_BODY_ALIASES = (
    "predicted_g1_body_pos",
    "predicted_body_pos",
    "predicted_link_positions",
    "predicted_body_positions",
    "pred_body_pos",
)
_TARGET_BODY_ALIASES = (
    "target_g1_body_pos",
    "target_body_pos",
    "target_link_positions",
    "target_body_positions",
)


_JOINT_MAE = _metadata(
    "joint_mae",
    "rad",
    LOWER_IS_BETTER,
    ("predicted_joints", "target_joints"),
    JOINT_MASK,
    "mean absolute error over all retained frames and joint dimensions",
    "Mean absolute G1 joint-position error.",
)
_JOINT_MSE = _metadata(
    "joint_mse",
    "rad^2",
    LOWER_IS_BETTER,
    ("predicted_joints", "target_joints"),
    JOINT_MASK,
    "mean squared error over all retained frames and joint dimensions",
    "Mean squared G1 joint-position error.",
)
_JOINT_RMSE = _metadata(
    "joint_rmse",
    "rad",
    LOWER_IS_BETTER,
    ("predicted_joints", "target_joints"),
    JOINT_MASK,
    "sqrt(mean squared error) over all retained frames and joint dimensions",
    "Root mean squared G1 joint-position error.",
)
_G1_JOINT_POS_RMSE = _metadata(
    "g1_joint_pos_rmse_rad",
    "rad",
    LOWER_IS_BETTER,
    ("predicted_joints", "target_joints"),
    JOINT_MASK,
    "sqrt(mean squared error) over all retained frames and joint dimensions",
    "A0 G1 joint-angle command RMSE.",
)
_MAX_JOINT_ABS = _metadata(
    "max_joint_abs_error",
    "rad",
    LOWER_IS_BETTER,
    ("predicted_joints", "target_joints"),
    JOINT_MASK,
    "maximum absolute error over retained frames and joint dimensions",
    "Worst absolute G1 joint-position error.",
)
_JOINT_VEL_RMSE = _metadata(
    "joint_velocity_rmse",
    "rad/s",
    LOWER_IS_BETTER,
    (
        "predicted_joint_velocities or predicted_joints",
        "target_joint_velocities or target_joints",
        "fps",
    ),
    JOINT_MASK,
    "sqrt(mean squared velocity error); explicit velocities preferred, "
    "finite differences need >=2 frames",
    "G1 joint-velocity RMSE when velocity data is available.",
)
_ROOT_POS_RMSE = _metadata(
    "root_position_rmse",
    "m",
    LOWER_IS_BETTER,
    ("pred_root_pos_w", "target_root_pos_w"),
    FRAME_MASK,
    "sqrt(mean squared error) over retained root position components",
    "Root-position RMSE in meters.",
)
_ROOT_ROT6D_RMSE = _metadata(
    "root_rot6d_rmse",
    "unitless",
    LOWER_IS_BETTER,
    ("predicted_root_rot6d", "target_root_rot6d"),
    FRAME_MASK,
    "sqrt(mean squared error) over retained 6D rotation components",
    "Root rotation 6D RMSE.",
)
_GLOBAL_BODY_POSITION_ERROR = _metadata(
    "global_body_position_error",
    "m",
    LOWER_IS_BETTER,
    ("predicted_g1_body_pos", "target_g1_body_pos", "pinned FK/link/root-alignment contract"),
    BODY_MASK,
    "mean L2 world body/link position error over retained frames and body/link positions",
    "Global body-position error using world body/link positions.",
    paper_labels=("E_g_mpbpe", "global MPJPE"),
    formula="mean_l2_error(world_body_position)",
    method_coverage={"gmr": True, "phc": True},
)
_ROOT_RELATIVE_BODY_POSITION_ERROR = _metadata(
    "root_relative_body_position_error",
    "m",
    LOWER_IS_BETTER,
    (
        "predicted_g1_body_pos",
        "target_g1_body_pos",
        "predicted/target root positions or root_body_index",
        "pinned FK/link/root-alignment contract",
    ),
    BODY_MASK,
    "mean L2 error after subtracting each frame root position from each body/link position",
    "Root-relative body-position error with explicit body-root subtraction.",
    paper_labels=("E_mpbpe", "MPJPE", "W-MPJPE", "root_relative_MPJPE"),
    formula="mean_l2_error(body_position_minus_root_position)",
    method_coverage={
        "gmr": {"label": "E_mpbpe"},
        "nmr": {"labels": ["MPJPE", "W-MPJPE"]},
        "phc": {"label": "root_relative_MPJPE"},
    },
)
_JOINT_ROTATION_ERROR = _metadata(
    "joint_rotation_error",
    "rad",
    LOWER_IS_BETTER,
    ("predicted_joint_rotations", "target_joint_rotations"),
    JOINT_MASK,
    "mean geodesic angle between predicted and reference full-joint rotations",
    "Full-joint rotation geodesic-angle error.",
    paper_labels=("E_mpjpe",),
    formula="mean_geodesic_angle(predicted_joint_rotation, reference_joint_rotation)",
    method_coverage={
        "gmr": True,
        "phc": {"extra_terms": ["acceleration_error", "velocity_error"]},
    },
)
_JOINT_VELOCITY_ERROR = _metadata(
    "joint_velocity_error",
    "rad/s",
    LOWER_IS_BETTER,
    (
        "predicted_joint_velocities or predicted_joints",
        "target_joint_velocities or target_joints",
        "fps",
    ),
    JOINT_MASK,
    "sqrt(mean squared velocity error); explicit velocities preferred",
    "PHC-style joint velocity error term.",
    formula="rmse(dq/dt)",
    method_coverage={"phc": {"extra_term": "velocity_error"}},
)
_JOINT_ACCELERATION_ERROR = _metadata(
    "joint_acceleration_error",
    "rad/s^2",
    LOWER_IS_BETTER,
    (
        "predicted_joint_accelerations or velocities or positions",
        "target_joint_accelerations or velocities or positions",
        "fps",
    ),
    JOINT_MASK,
    "sqrt(mean squared acceleration error); explicit accelerations preferred",
    "PHC-style joint acceleration error term.",
    formula="rmse(d2q/dt2)",
    method_coverage={"phc": {"extra_term": "acceleration_error"}},
)
_LOSS = _metadata(
    "loss",
    "loss",
    LOWER_IS_BETTER,
    ("loss",),
    NO_MASK,
    "passthrough scalar",
    "Training or validation loss passthrough.",
)
_ACTION_SIM = _metadata(
    "action_similarity",
    "cosine",
    HIGHER_IS_BETTER,
    ("predicted_joints", "target_joints"),
    FRAME_MASK,
    "mean cosine similarity over retained frames",
    "Cosine similarity of action vectors.",
)
_PRED_JUMP = _metadata(
    "predicted_joint_jump_rate",
    "rate",
    LOWER_IS_BETTER,
    ("predicted_joints", "fps", "joint_jump_velocity"),
    FRAME_MASK,
    "fraction of finite-difference joint-velocity samples over threshold",
    "Predicted joint jump rate.",
)
_TARGET_JUMP = _metadata(
    "target_joint_jump_rate",
    "rate",
    LOWER_IS_BETTER,
    ("target_joints", "fps", "joint_jump_velocity"),
    FRAME_MASK,
    "fraction of finite-difference joint-velocity samples over threshold",
    "Target joint jump rate.",
)
_JUMP_DELTA = _metadata(
    "predicted_minus_target_joint_jump_rate",
    "rate",
    LOWER_IS_BETTER,
    ("predicted_joints", "target_joints", "fps", "joint_jump_velocity"),
    FRAME_MASK,
    "predicted joint jump rate minus target joint jump rate",
    "Delta between predicted and target joint jump rates.",
)
_JOINT_JUMP_COUNT = _metadata(
    "joint_jump_count",
    "joint_frames",
    LOWER_IS_BETTER,
    ("joint_positions or qpos", "joint_jump_threshold_rad"),
    JOINT_MASK,
    "count of wrapped absolute adjacent-frame joint deltas over threshold",
    "NMR-style joint jump count using abs(delta_q) > 0.5 rad by default.",
    formula="sum_{t,j} 1[abs(wrap(q[t,j]-q[t-1,j])) > 0.5]",
    method_coverage={"nmr": True},
)
_JOINT_LIMIT_PROXIMITY_COUNT = _metadata(
    "joint_limit_proximity_count",
    "joint_frames",
    LOWER_IS_BETTER,
    ("joint_positions or qpos", "joint_lower_limits", "joint_upper_limits"),
    JOINT_MASK,
    "count of joint samples within threshold distance of a lower/upper limit",
    "NMR-style joint-limit proximity count.",
    formula="sum_{t,j} 1[min(q-lower, upper-q) <= 0.05]",
    method_coverage={"nmr": True},
)
_SELF_COLLISION_COUNT = _metadata(
    "self_collision_count",
    "frames",
    LOWER_IS_BETTER,
    ("MuJoCo self-contact count, frame flags, or per-frame contact pairs",),
    NO_MASK,
    "count of frames with non-allowed MuJoCo self-contact",
    "Self-collision count from explicit MuJoCo contact evidence.",
    formula="sum_t 1[exists non-allowed MuJoCo self-contact at t]",
    method_coverage={"nmr": True},
)
_FLOATING_GUARD = _metadata(
    "floating_guard",
    "0/1",
    PASS_REQUIRED,
    ("mean lowest-foot height or body positions with foot indices",),
    NO_MASK,
    "1 if mean lowest-foot height is <= 0.10 m, else 0",
    "NMR floating-lowest-foot physical failure guard.",
    formula="pass if mean_t min_foot_height[t] <= 0.10",
    method_coverage={"nmr": {"check": "floating_lowest_foot"}},
)
_CROSS_RATIO_GUARD = _metadata(
    "cross_ratio_guard",
    "0/1",
    PASS_REQUIRED,
    ("cross_ratio/self_intersection_ratio or self_intersection_frames",),
    NO_MASK,
    "1 if self-intersection cross ratio is <= 0.05, else 0",
    "NMR self-intersection cross-ratio physical failure guard.",
    formula="pass if (# self-intersection frames)/T <= 0.05",
    method_coverage={"nmr": {"check": "self_intersection_cross_ratio"}},
)
_PHC_FAILURE_GUARD = _metadata(
    "phc_failure_guard",
    "0/1",
    PASS_REQUIRED,
    ("phc_avg_body_joint_distance",),
    NO_MASK,
    "1 if avg body-joint distance is <= 0.5 m, else 0",
    "PHC avg body-joint distance failure guard.",
    formula="pass if avg_body_joint_distance <= 0.5",
    method_coverage={"phc": {"check": "avg_body_joint_distance_fail"}},
)
_MPJPE = _metadata(
    "mpjpe",
    "m",
    LOWER_IS_BETTER,
    ("predicted_g1_body_pos", "target_g1_body_pos", "pinned FK/link/root-alignment contract"),
    BODY_MASK,
    "mean Euclidean body/link error over retained frames and body/link positions",
    "True G1 body/link MPJPE gated by a pinned FK/link/root-alignment contract.",
)
_W_MPJPE = _metadata(
    "w_mpjpe",
    "m",
    LOWER_IS_BETTER,
    (
        "predicted_g1_body_pos",
        "target_g1_body_pos",
        "body_position_weights",
        "pinned FK/link/root-alignment contract",
    ),
    BODY_MASK,
    "weighted mean Euclidean body/link error over retained frames and body/link positions",
    "Weighted G1 body/link MPJPE gated by a pinned FK/link/root-alignment contract.",
)
_PRED_FRAME_COUNT = _metadata(
    "predicted_frame_count",
    "frames",
    INFORMATIONAL,
    ("predicted_frame_count or predicted artifact arrays",),
    NO_MASK,
    "count of predicted frames",
    "Predicted artifact frame count.",
)
_TARGET_FRAME_COUNT = _metadata(
    "target_frame_count",
    "frames",
    INFORMATIONAL,
    ("target_frame_count or target artifact arrays",),
    NO_MASK,
    "count of target frames",
    "Target artifact frame count.",
)
_FRAME_COUNT_DELTA = _metadata(
    "frame_count_delta",
    "frames",
    LOWER_IS_BETTER,
    ("predicted frame count", "target frame count"),
    NO_MASK,
    "absolute difference between predicted and target frame counts",
    "Prediction/target frame-count mismatch.",
)
_VIDEO_FRAME_COUNT = _metadata(
    "video_frame_count",
    "frames",
    INFORMATIONAL,
    ("video_frame_count or rendered_frame_count",),
    NO_MASK,
    "passthrough rendered video frame count",
    "Rendered video frame count when present.",
)
_VIDEO_STATUS_OK = _metadata(
    "video_status_ok",
    "0/1",
    HIGHER_IS_BETTER,
    ("video_status or render_status",),
    NO_MASK,
    "1 when status is ok/success/passed/true, else 0",
    "Video/render integrity status counter.",
)


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "joint_mae": MetricDefinition(
        _JOINT_MAE,
        lambda fields: _pair_metric(
            _JOINT_MAE,
            fields,
            predicted_aliases=_PREDICTED_JOINT_ALIASES,
            target_aliases=_TARGET_JOINT_ALIASES,
            value_fn=joint_mae,
        ),
    ),
    "joint_mse": MetricDefinition(
        _JOINT_MSE,
        lambda fields: _pair_metric(
            _JOINT_MSE,
            fields,
            predicted_aliases=_PREDICTED_JOINT_ALIASES,
            target_aliases=_TARGET_JOINT_ALIASES,
            value_fn=joint_mse,
        ),
    ),
    "joint_rmse": MetricDefinition(
        _JOINT_RMSE,
        lambda fields: _pair_metric(
            _JOINT_RMSE,
            fields,
            predicted_aliases=_PREDICTED_JOINT_ALIASES,
            target_aliases=_TARGET_JOINT_ALIASES,
            value_fn=joint_rmse,
        ),
    ),
    "g1_joint_pos_rmse_rad": MetricDefinition(
        _G1_JOINT_POS_RMSE,
        lambda fields: _pair_metric(
            _G1_JOINT_POS_RMSE,
            fields,
            predicted_aliases=_PREDICTED_JOINT_ALIASES,
            target_aliases=_TARGET_JOINT_ALIASES,
            value_fn=joint_rmse,
        ),
    ),
    "g1_kin_joint_rmse": MetricDefinition(
        _G1_JOINT_POS_RMSE,
        lambda fields: _pair_metric(
            _G1_JOINT_POS_RMSE,
            fields,
            predicted_aliases=_PREDICTED_JOINT_ALIASES,
            target_aliases=_TARGET_JOINT_ALIASES,
            value_fn=joint_rmse,
        ),
    ),
    "max_joint_abs_error": MetricDefinition(
        _MAX_JOINT_ABS,
        lambda fields: _pair_metric(
            _MAX_JOINT_ABS,
            fields,
            predicted_aliases=_PREDICTED_JOINT_ALIASES,
            target_aliases=_TARGET_JOINT_ALIASES,
            value_fn=max_joint_abs_error,
        ),
    ),
    "joint_velocity_rmse": MetricDefinition(
        _JOINT_VEL_RMSE,
        lambda fields: _joint_velocity_metric(_JOINT_VEL_RMSE, fields),
    ),
    "root_position_rmse": MetricDefinition(
        _ROOT_POS_RMSE,
        lambda fields: _root_metric(
            _ROOT_POS_RMSE,
            fields,
            _PREDICTED_ROOT_POS_ALIASES,
            _TARGET_ROOT_POS_ALIASES,
        ),
    ),
    "g1_kin_root_position_rmse": MetricDefinition(
        _ROOT_POS_RMSE,
        lambda fields: _root_metric(
            _ROOT_POS_RMSE,
            fields,
            _PREDICTED_ROOT_POS_ALIASES,
            _TARGET_ROOT_POS_ALIASES,
        ),
    ),
    "root_rot6d_rmse": MetricDefinition(
        _ROOT_ROT6D_RMSE,
        lambda fields: _root_metric(
            _ROOT_ROT6D_RMSE,
            fields,
            _PREDICTED_ROOT_ROT6D_ALIASES,
            _TARGET_ROOT_ROT6D_ALIASES,
        ),
    ),
    "g1_kin_root_orientation_rmse": MetricDefinition(
        _ROOT_ROT6D_RMSE,
        lambda fields: _root_metric(
            _ROOT_ROT6D_RMSE,
            fields,
            _PREDICTED_ROOT_ROT6D_ALIASES,
            _TARGET_ROOT_ROT6D_ALIASES,
        ),
    ),
    "global_body_position_error": MetricDefinition(
        _GLOBAL_BODY_POSITION_ERROR,
        lambda fields: _body_metric(_GLOBAL_BODY_POSITION_ERROR, fields, weighted=False),
    ),
    "E_g_mpbpe": MetricDefinition(
        _GLOBAL_BODY_POSITION_ERROR,
        lambda fields: _body_metric(_GLOBAL_BODY_POSITION_ERROR, fields, weighted=False),
    ),
    "global_mpjpe": MetricDefinition(
        _GLOBAL_BODY_POSITION_ERROR,
        lambda fields: _body_metric(_GLOBAL_BODY_POSITION_ERROR, fields, weighted=False),
    ),
    "root_relative_body_position_error": MetricDefinition(
        _ROOT_RELATIVE_BODY_POSITION_ERROR,
        lambda fields: _root_relative_body_metric(_ROOT_RELATIVE_BODY_POSITION_ERROR, fields),
    ),
    "E_mpbpe": MetricDefinition(
        _ROOT_RELATIVE_BODY_POSITION_ERROR,
        lambda fields: _root_relative_body_metric(_ROOT_RELATIVE_BODY_POSITION_ERROR, fields),
    ),
    "root_relative_MPJPE": MetricDefinition(
        _ROOT_RELATIVE_BODY_POSITION_ERROR,
        lambda fields: _root_relative_body_metric(_ROOT_RELATIVE_BODY_POSITION_ERROR, fields),
    ),
    "root_relative_mpjpe": MetricDefinition(
        _ROOT_RELATIVE_BODY_POSITION_ERROR,
        lambda fields: _root_relative_body_metric(_ROOT_RELATIVE_BODY_POSITION_ERROR, fields),
    ),
    "joint_rotation_error": MetricDefinition(
        _JOINT_ROTATION_ERROR,
        lambda fields: _joint_rotation_metric(_JOINT_ROTATION_ERROR, fields),
    ),
    "joint_velocity_error": MetricDefinition(
        _JOINT_VELOCITY_ERROR,
        lambda fields: _joint_velocity_metric(_JOINT_VELOCITY_ERROR, fields),
    ),
    "joint_acceleration_error": MetricDefinition(
        _JOINT_ACCELERATION_ERROR,
        lambda fields: _joint_acceleration_metric(_JOINT_ACCELERATION_ERROR, fields),
    ),
    "loss": MetricDefinition(_LOSS, lambda fields: _loss_metric(_LOSS, fields)),
    "train_loss": MetricDefinition(_LOSS, lambda fields: _loss_metric(_LOSS, fields)),
    "eval_loss": MetricDefinition(_LOSS, lambda fields: _loss_metric(_LOSS, fields)),
    "action_similarity": MetricDefinition(
        _ACTION_SIM,
        lambda fields: _pair_metric(
            _ACTION_SIM,
            fields,
            predicted_aliases=_PREDICTED_JOINT_ALIASES,
            target_aliases=_TARGET_JOINT_ALIASES,
            value_fn=action_similarity,
        ),
    ),
    "predicted_joint_jump_rate": MetricDefinition(
        _PRED_JUMP,
        lambda fields: _joint_jump_metric(_PRED_JUMP, fields, _PREDICTED_JOINT_ALIASES),
    ),
    "target_joint_jump_rate": MetricDefinition(
        _TARGET_JUMP,
        lambda fields: _joint_jump_metric(_TARGET_JUMP, fields, _TARGET_JOINT_ALIASES),
    ),
    "predicted_minus_target_joint_jump_rate": MetricDefinition(
        _JUMP_DELTA,
        lambda fields: _jump_delta_metric(_JUMP_DELTA, fields),
    ),
    "joint_jump_count": MetricDefinition(
        _JOINT_JUMP_COUNT,
        lambda fields: _joint_jump_count_metric(_JOINT_JUMP_COUNT, fields),
    ),
    "joint_limit_proximity_count": MetricDefinition(
        _JOINT_LIMIT_PROXIMITY_COUNT,
        lambda fields: _joint_limit_proximity_count_metric(_JOINT_LIMIT_PROXIMITY_COUNT, fields),
    ),
    "self_collision_count": MetricDefinition(
        _SELF_COLLISION_COUNT,
        lambda fields: _self_collision_count_metric(_SELF_COLLISION_COUNT, fields),
    ),
    "floating_guard": MetricDefinition(
        _FLOATING_GUARD,
        lambda fields: _floating_guard_metric(_FLOATING_GUARD, fields),
    ),
    "cross_ratio_guard": MetricDefinition(
        _CROSS_RATIO_GUARD,
        lambda fields: _cross_ratio_guard_metric(_CROSS_RATIO_GUARD, fields),
    ),
    "phc_failure_guard": MetricDefinition(
        _PHC_FAILURE_GUARD,
        lambda fields: _phc_failure_guard_metric(_PHC_FAILURE_GUARD, fields),
    ),
    "mpjpe": MetricDefinition(_MPJPE, lambda fields: _body_metric(_MPJPE, fields, weighted=False)),
    "body_position_mpjpe": MetricDefinition(
        _MPJPE,
        lambda fields: _body_metric(_MPJPE, fields, weighted=False),
    ),
    "w_mpjpe": MetricDefinition(
        _W_MPJPE,
        lambda fields: _body_metric(_W_MPJPE, fields, weighted=True),
    ),
    "weighted_mpjpe": MetricDefinition(
        _W_MPJPE,
        lambda fields: _body_metric(_W_MPJPE, fields, weighted=True),
    ),
    "predicted_frame_count": MetricDefinition(
        _PRED_FRAME_COUNT,
        lambda fields: _frame_count_metric(
            _PRED_FRAME_COUNT,
            fields,
            explicit_aliases=("predicted_frame_count", "prediction_frame_count"),
            fallback_aliases=_PREDICTED_JOINT_ALIASES + _PREDICTED_BODY_ALIASES,
        ),
    ),
    "target_frame_count": MetricDefinition(
        _TARGET_FRAME_COUNT,
        lambda fields: _frame_count_metric(
            _TARGET_FRAME_COUNT,
            fields,
            explicit_aliases=("target_frame_count",),
            fallback_aliases=_TARGET_JOINT_ALIASES + _TARGET_BODY_ALIASES,
        ),
    ),
    "frame_count_delta": MetricDefinition(
        _FRAME_COUNT_DELTA,
        lambda fields: _frame_count_delta_metric(_FRAME_COUNT_DELTA, fields),
    ),
    "video_frame_count": MetricDefinition(
        _VIDEO_FRAME_COUNT,
        lambda fields: _video_frame_count_metric(_VIDEO_FRAME_COUNT, fields),
    ),
    "video_status_ok": MetricDefinition(
        _VIDEO_STATUS_OK,
        lambda fields: _video_status_metric(_VIDEO_STATUS_OK, fields),
    ),
}
