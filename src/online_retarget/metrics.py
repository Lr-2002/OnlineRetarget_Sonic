"""Evaluation metrics for offline retargeting checks.

The functions accept Python sequences so they can be tested before installing
NumPy/PyTorch. Training code can wrap them or replace the internals with tensor
implementations later while keeping the same metric definitions.
"""

from __future__ import annotations

import math
from typing import Sequence


Vector = Sequence[float]
JointFrame = Sequence[Vector]
Motion3D = Sequence[JointFrame]
Motion1D = Sequence[Sequence[float]]


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


def joint_rmse(predicted: Motion1D, target: Motion1D) -> float:
    """Root mean squared error for joint vectors shaped T x D."""

    total = 0.0
    count = 0
    for pred_frame, target_frame in _zip_equal(predicted, target, "frames"):
        for pred_value, target_value in _zip_equal(pred_frame, target_frame, "joint dimensions"):
            total += (pred_value - target_value) ** 2
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
    for prev_frame, cur_frame, prev_ref_frame, cur_ref_frame in zip(
        body_positions,
        body_positions[1:],
        reference,
        reference[1:],
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
                continue
            speed = _horizontal_speed(
                prev_frame[foot_index],
                cur_frame[foot_index],
                horizontal_axes,
                fps,
            )
            max_slide_speed = max(max_slide_speed, speed)
            slide_violations += speed > max_contact_slide_speed
            slide_samples += 1

    return {
        "contact_frame_ratio": _zero_mean(float(contact_samples), total_foot_samples),
        "foot_float_rate": _zero_mean(float(foot_float_samples), contact_samples),
        "contact_slide_rate": _zero_mean(float(slide_violations), slide_samples),
        "max_contact_slide_speed": max_slide_speed,
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
    squared = sum((right[index] - left[index]) ** 2 for index in horizontal_axes)
    return math.sqrt(squared) * fps


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
