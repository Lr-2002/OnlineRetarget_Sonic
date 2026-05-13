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


def _safe_mean(total: float, count: int) -> float:
    if count == 0:
        raise ValueError("metric received no samples")
    return total / count


def _zip_equal(left: Sequence, right: Sequence, label: str):
    if len(left) != len(right):
        raise ValueError(f"mismatched {label}: {len(left)} != {len(right)}")
    return zip(left, right)
