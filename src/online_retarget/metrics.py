"""Evaluation metrics for offline retargeting checks.

The functions accept Python sequences so they can be tested before installing
NumPy/PyTorch. Training code can wrap them or replace the internals with tensor
implementations later while keeping the same metric definitions.
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

LOWER_IS_BETTER = "lower_is_better"
INFORMATIONAL = "informational"
BODY_MASK = "optional frame_mask and body_mask drop frames or body/link positions"


@dataclass(frozen=True)
class MetricMetadata:
    name: str
    unit: str
    direction: str
    required_fields: tuple[str, ...]
    mask_semantics: str
    reducer: str
    description: str
    source_ref: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["required_fields"] = list(self.required_fields)
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


DEFAULT_ONLINE_METRIC_NAMES: tuple[str, ...] = ("mpjpe", "w_mpjpe")


def mpjpe(predicted: Motion3D, target: Motion3D) -> float:
    """Mean per-joint position error for motions shaped T x J x 3."""

    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_joint, target_joint in _zip_equal(pred_frame, target_frame, "joints"):
            if len(pred_joint) != 3 or len(target_joint) != 3:
                raise ValueError("mpjpe expects 3D joint vectors")
            total += math.dist(pred_joint, target_joint)
            count += 1
    return _safe_mean(total, count)


def weighted_mpjpe(predicted: Motion3D, target: Motion3D, weights: Sequence[float]) -> float:
    """Weighted mean per-joint position error for motions shaped T x J x 3."""

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


def compute_metric_bundle(
    fields: MetricFields,
    metric_names: Sequence[str] | None = None,
) -> dict[str, MetricValue]:
    """Compute requested shared metrics with explicit availability status.

    This is intentionally small for the A0/LR-254 training lane. It uses the
    same MPJPE/W-MPJPE contract introduced by the LR-239 metric work: body
    position errors are valid only when paired G1 body/link positions and a
    pinned FK/link/root-alignment contract are present.
    """

    names = tuple(metric_names or DEFAULT_ONLINE_METRIC_NAMES)
    results: dict[str, MetricValue] = {}
    for name in names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"unknown metric: {name}")
        results[name] = METRIC_REGISTRY[name].compute(fields)
    return results


def flatten_metric_bundle(bundle: Mapping[str, MetricValue]) -> dict[str, object]:
    """Flatten metric results into scalar/status/reason columns."""

    row: dict[str, object] = {}
    for name, result in bundle.items():
        if result.available:
            row[name] = float(result.value)
        row[f"{name}_status"] = result.status
        if result.reason:
            row[f"{name}_reason"] = result.reason
    return row


def metric_metadata(metric_names: Sequence[str] | None = None) -> dict[str, dict[str, object]]:
    """Return registry metadata for reports and validation gates."""

    names = tuple(metric_names or DEFAULT_ONLINE_METRIC_NAMES)
    return {name: METRIC_REGISTRY[name].metadata.to_dict() for name in names}


def joint_mae(predicted: Motion1D, target: Motion1D) -> float:
    """Mean absolute error for joint vectors shaped T x D."""

    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_value, target_value in _zip_equal(pred_frame, target_frame, "joint dimensions"):
            total += abs(pred_value - target_value)
            count += 1
    return _safe_mean(total, count)


def joint_mse(predicted: Motion1D, target: Motion1D) -> float:
    """Mean squared error for joint vectors shaped T x D."""

    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_value, target_value in _zip_equal(pred_frame, target_frame, "joint dimensions"):
            total += (pred_value - target_value) ** 2
            count += 1
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
    """RMSE between predicted and target joint velocities."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(predicted) != len(target):
        raise ValueError(f"mismatched frames: {len(predicted)} != {len(target)}")
    if len(predicted) < 2:
        return 0.0
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
    for frame_index, (prev_frame, cur_frame, prev_ref_frame, cur_ref_frame) in enumerate(zip(
        body_positions,
        body_positions[1:],
        reference,
        reference[1:],
    )):
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
        raise ValueError("up_axis must be x, y, z, 0, 1, or 2")
    if axis in (0, 1, 2):
        return axis
    raise ValueError("up_axis must be x, y, z, 0, 1, or 2")


def _point_clearance(point: Vector, axis: int, ground_height: float) -> float:
    return point[axis] - ground_height


def _horizontal_speed(left: Vector, right: Vector, horizontal_axes: Sequence[int], fps: float) -> float:
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


def _first_present(fields: MetricFields, aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in fields and fields[alias] is not None:
            return fields[alias]
    return None


def _as_sequence(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _as_float_vector(value: Any) -> list[float]:
    value = _as_sequence(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("weights must be a numeric sequence")
    return [float(item) for item in value]


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


def _optional_mask(fields: MetricFields, key: str) -> list[bool] | None:
    value = fields.get(key)
    if value is None:
        return None
    value = _as_sequence(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")
    return [bool(item) for item in value]


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


def _ok(metadata: MetricMetadata, value: float) -> MetricValue:
    if not math.isfinite(float(value)):
        return _unavailable(metadata, "metric value is not finite")
    return MetricValue(metadata=metadata, status=METRIC_AVAILABLE, value=float(value))


def _unavailable(metadata: MetricMetadata, reason: str) -> MetricValue:
    return MetricValue(metadata=metadata, status=METRIC_UNAVAILABLE, reason=reason)


def _blocked(metadata: MetricMetadata, reason: str) -> MetricValue:
    return MetricValue(metadata=metadata, status=METRIC_BLOCKED, reason=reason)


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


def _contract_is_pinned(contract: Any) -> bool:
    if isinstance(contract, Mapping):
        return bool(contract.get("pinned") or contract.get("validated") or contract.get("accepted"))
    return bool(contract)


def _body_metric(metadata: MetricMetadata, fields: MetricFields, *, weighted: bool) -> MetricValue:
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
            "requires pinned FK/link/root-alignment contract before body-position metrics are valid",
        )
    try:
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
        if weighted:
            weights = _first_present(
                fields,
                ("body_position_weights", "mpjpe_weights", "joint_weights", "body_weights"),
            )
            if weights is None:
                return _unavailable(metadata, "requires body_position_weights or mpjpe_weights")
            return _ok(metadata, weighted_mpjpe(pred_motion, target_motion, _as_float_vector(weights)))
        return _ok(metadata, mpjpe(pred_motion, target_motion))
    except ValueError as exc:
        return _unavailable(metadata, str(exc))


_PREDICTED_BODY_ALIASES = (
    "predicted_g1_body_pos",
    "predicted_body_pos",
    "predicted_body_positions",
    "pred_body_pos",
)
_TARGET_BODY_ALIASES = (
    "target_g1_body_pos",
    "target_body_pos",
    "target_body_positions",
    "target_body_pos_w",
)
_MPJPE_METADATA = MetricMetadata(
    name="mpjpe",
    unit="m",
    direction=LOWER_IS_BETTER,
    required_fields=(
        "predicted_g1_body_pos",
        "target_g1_body_pos",
        "pinned FK/link/root-alignment contract",
    ),
    mask_semantics=BODY_MASK,
    reducer="mean Euclidean body/link error over retained frames and body/link positions",
    description="True G1 body/link MPJPE gated by a pinned FK/link/root-alignment contract.",
    source_ref="LR-239 shared online/offline metric registry",
)
_W_MPJPE_METADATA = MetricMetadata(
    name="w_mpjpe",
    unit="m",
    direction=LOWER_IS_BETTER,
    required_fields=(
        "predicted_g1_body_pos",
        "target_g1_body_pos",
        "body_position_weights",
        "pinned FK/link/root-alignment contract",
    ),
    mask_semantics=BODY_MASK,
    reducer="weighted mean Euclidean body/link error over retained frames and body/link positions",
    description="Weighted G1 body/link MPJPE gated by a pinned FK/link/root-alignment contract.",
    source_ref="LR-239 shared online/offline metric registry",
)

METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "mpjpe": MetricDefinition(
        _MPJPE_METADATA,
        lambda fields: _body_metric(_MPJPE_METADATA, fields, weighted=False),
    ),
    "body_position_mpjpe": MetricDefinition(
        _MPJPE_METADATA,
        lambda fields: _body_metric(_MPJPE_METADATA, fields, weighted=False),
    ),
    "w_mpjpe": MetricDefinition(
        _W_MPJPE_METADATA,
        lambda fields: _body_metric(_W_MPJPE_METADATA, fields, weighted=True),
    ),
    "weighted_mpjpe": MetricDefinition(
        _W_MPJPE_METADATA,
        lambda fields: _body_metric(_W_MPJPE_METADATA, fields, weighted=True),
    ),
}
