"""A0 visual-validation data boundary.

The trainer owns model inference and sample selection. This module owns the
visualization-facing coordinate and backend contract for A0 frozen-Skeleton-AE
validation clips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_G1_USD = Path("/home/user/project/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd")
PRIMARY_VISUAL_BACKEND = "somamesh_global_soma_plus_isaaclab_g1_kinematic_playback"
DEBUG_CAPSULE_BACKEND = "software_capsule_debug_fallback"
SOMA_DISPLAY_TRANSFORM = "(x,y,z)_display=(x,-z,y)_soma"


class A0VisualValidationRenderer:
    """Stable A0 visual-validation boundary used by the trainer.

    The primary acceptance backend is SomaMesh/global-SOMA source playback plus
    IsaacLab G1 kinematic playback. The legacy in-process capsule renderer is
    allowed only as debug fallback and is recorded as such in clip metadata.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        g1_usd_path: Path | str | None = None,
    ) -> None:
        self._config = config
        visual_cfg = config.get("visual_validation", {})
        if not isinstance(visual_cfg, Mapping):
            visual_cfg = {}
        configured_usd = visual_cfg.get("g1_robot_usd") or visual_cfg.get("g1_usd") or g1_usd_path
        self.g1_usd_path = Path(str(configured_usd)) if configured_usd else DEFAULT_G1_USD

    def compose_prediction_root(
        self,
        predicted_root_pos: np.ndarray,
        fallback_root_pos: np.ndarray,
    ) -> np.ndarray:
        """Return visualization-world root positions for model predictions."""

        root = np.asarray(predicted_root_pos, dtype=np.float32).copy()
        if self.should_compose_soma_root_xy:
            root[:, :2] += np.asarray(fallback_root_pos, dtype=np.float32)[:, :2]
        return root

    def compose_prediction_state(
        self,
        prediction: Mapping[str, np.ndarray],
        *,
        fallback_root_pos: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Copy prediction state and compose root XY at the visual boundary."""

        updated = {key: np.asarray(value, dtype=np.float32) for key, value in prediction.items()}
        if "root_pos" in updated:
            updated["root_pos"] = self.compose_prediction_root(updated["root_pos"], fallback_root_pos)
        return updated

    @property
    def should_compose_soma_root_xy(self) -> bool:
        input_data = self._config.get("input_data", {})
        return (
            isinstance(input_data, Mapping)
            and input_data.get("format") == "soma_motionlib"
            and _include_root_pos_target(self._config)
        )

    def root_composition_metadata(self) -> dict[str, Any]:
        return {
            "compose_soma_root_local_xy_to_world": bool(self.should_compose_soma_root_xy),
            "condition": 'input_data.format == "soma_motionlib" and include_root_pos_target == true',
            "xy_operation": "pred_root[:, :2] += fallback_root_pos[:, :2]",
            "z_semantics": "predicted_absolute_z",
            "root_rot6d_semantics": "predicted_absolute_rot6d",
        }

    def backend_manifest(self, *, active_backend: str) -> dict[str, Any]:
        return {
            "primary_backend": PRIMARY_VISUAL_BACKEND,
            "active_backend": active_backend,
            "debug_fallback_backend": DEBUG_CAPSULE_BACKEND,
            "source_human_backend": "SomaMesh/global-SOMA display",
            "source_display_transform": SOMA_DISPLAY_TRANSFORM,
            "g1_backend": "IsaacLab G1 kinematic playback",
            "g1_asset_usd": str(self.g1_usd_path),
            "required_overlays": ["world_axes", "root_axes", "semantic_left_right"],
            "active_backend_is_acceptance_backend": active_backend != DEBUG_CAPSULE_BACKEND,
            "debug_fallback_is_acceptance_backend": False,
        }

    def isaaclab_g1_render_command(
        self,
        *,
        python_bin: Path | str,
        script_path: Path | str,
        motion_path: Path | str,
        output_path: Path | str,
        duration_sec: float,
        width: int,
        height: int,
    ) -> list[str]:
        """Build the stable G1 IsaacLab playback command for rerender handoff."""

        return [
            str(python_bin),
            str(script_path),
            "--g1-motion",
            str(motion_path),
            "--format",
            "npz",
            "--output",
            str(output_path),
            "--duration-sec",
            f"{float(duration_sec):g}",
            "--robot-usd",
            str(self.g1_usd_path),
            "--preserve-world-root",
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
        ]

    @staticmethod
    def soma_point_to_display(point: Sequence[float]) -> tuple[float, float, float]:
        x, y, z = (float(point[index]) if index < len(point) else 0.0 for index in range(3))
        return (x, -z, y)

    @classmethod
    def soma_frame_maps_to_display(
        cls,
        frames: Sequence[Mapping[str, Sequence[float]]],
    ) -> list[dict[str, tuple[float, float, float]]]:
        return [
            {name: cls.soma_point_to_display(point) for name, point in frame.items()}
            for frame in frames
        ]

    @classmethod
    def soma_motionlib_source_frames(
        cls,
        soma_joints: np.ndarray,
        joint_names: Sequence[str],
    ) -> list[dict[str, tuple[float, float, float]]]:
        joints = np.asarray(soma_joints, dtype=np.float32)
        usable = min(len(joint_names), joints.shape[1])
        return [
            {
                joint_names[index]: cls.soma_point_to_display(frame[index])
                for index in range(usable)
            }
            for frame in joints
        ]


def _include_root_pos_target(config: Mapping[str, Any]) -> bool:
    features = config.get("features", {})
    if not isinstance(features, Mapping):
        return False
    explicit = features.get("include_root_pos_target")
    if explicit is not None:
        return bool(explicit)
    target_text = " ".join(
        str(features.get(key, ""))
        for key in ("target_feature", "target_features", "target_pose_feature")
    )
    return "root_pos" in target_text
