"""Web-facing motion retarget and MuJoCo preview pipeline.

The web app intentionally exposes a conservative pipeline surface: BVH loading
and G1 kinematic preview are implemented locally, while learned retargeting and
MuJoCo physics report explicit availability/status instead of pretending a
trained controller is present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import uuid
from typing import Iterable, Mapping, Sequence
import warnings
import zipfile

from .data.bones_seed import G1_CSV_COLUMNS, G1_JOINT_COLUMNS
from .data.g1_quality import (
    G1KinematicModel,
    g1_fk_body_positions,
    load_g1_kinematic_model,
)
from .data.windowed_builder import (
    DEFAULT_SOURCE_BODY_NAMES,
    global_body_position_maps_from_bvh,
    parse_bvh_motion,
)


os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_G1_MJCF = Path("/home/user/repos/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
DEFAULT_GMR_ROOT = Path(os.environ.get("ONLINE_RETARGET_GMR_ROOT", "/home/user/repos/GMR"))
DEFAULT_GMR_ROBOT = "unitree_g1"
DEFAULT_MAX_FRAMES = 0
DEFAULT_ROOT_HEIGHT = 0.793
WEB_RUN_ROOT = Path("outputs/web_runs")
MUJOCO_VIDEO_ARTIFACT = "mujoco_g1_render_mp4"
MUJOCO_VIDEO_FILENAME = "mujoco_g1_render.mp4"
ROOT_XY_LOCKED_FOR_VIS = True
GROUND_ALIGNMENT_FOOT_ROOT_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
SOMA_GMR_BVH_JOINT_ALIASES = {
    "LeftLeg": "LeftUpLeg",
    "LeftShin": "LeftLeg",
    "RightLeg": "RightUpLeg",
    "RightShin": "RightLeg",
}
SOMA_GMR_BVH_REQUIRED_JOINTS = frozenset(
    (*SOMA_GMR_BVH_JOINT_ALIASES.keys(), "LeftToeBase", "RightToeBase")
)

STAGE_ORDER = ("load", "retarget", "kinematic_sim", "physics_sim")
KINEMATIC_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "head_link",
    "left_ankle_roll_link",
    "left_toe_link",
    "right_ankle_roll_link",
    "right_toe_link",
    "left_rubber_hand",
    "right_rubber_hand",
)

SMPL_PREVIEW_BODY_NAMES = (
    "Hips",
    "Spine1",
    "Chest",
    "Head",
    "LeftHand",
    "RightHand",
    "LeftFoot",
    "RightFoot",
)

SOURCE_CAPSULE_EDGES = (
    ("Hips", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine1", "Chest"),
    ("Spine2", "Chest"),
    ("Chest", "Neck1"),
    ("Neck1", "Head"),
    ("Chest", "Head"),
    ("Chest", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Chest", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftLeg"),
    ("Hips", "LeftFoot"),
    ("LeftLeg", "LeftShin"),
    ("LeftShin", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightLeg"),
    ("Hips", "RightFoot"),
    ("RightLeg", "RightShin"),
    ("RightShin", "RightFoot"),
    ("RightFoot", "RightToeBase"),
    ("Chest", "LeftHand"),
    ("Chest", "RightHand"),
)


def _frame_limit(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _limited_sequence(items: Sequence[object], max_frames: int | None) -> Sequence[object]:
    if max_frames is None:
        return items
    return items[:max_frames]


@dataclass(frozen=True)
class StageResult:
    status: str
    message: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WebPipelineResult:
    run_id: str
    output_dir: Path
    input_format: str
    stages: dict[str, StageResult]
    artifacts: dict[str, str]
    preview: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "input_format": self.input_format,
            "stages": {key: value.to_dict() for key, value in self.stages.items()},
            "artifacts": dict(self.artifacts),
            "preview": self.preview,
        }


def run_web_pipeline(
    source_bytes: bytes,
    filename: str,
    *,
    output_root: Path = WEB_RUN_ROOT,
    model_xml: Path = DEFAULT_G1_MJCF,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    render_frames: bool = False,
    compare_retargeters: bool = False,
) -> WebPipelineResult:
    """Run the upload -> retarget -> kinematic preview -> physics status pipeline."""

    run_id = _run_id(filename)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    stages: dict[str, StageResult] = {}
    artifacts: dict[str, str] = {}
    preview: dict[str, object] = {"source": {}, "robot": {}, "timeline": []}

    source_path = output_dir / _safe_filename(filename or "motion.bvh")
    source_path.write_bytes(source_bytes)
    artifacts["source"] = str(source_path)

    frame_limit = _frame_limit(max_frames)

    input_format = _detect_format(filename, source_bytes)
    loaded = _load_preview_source(source_bytes, input_format, max_frames=frame_limit)
    if loaded["status"] != "ok":
        stages["load"] = StageResult(
            status=str(loaded["status"]),
            message=str(loaded["message"]),
            details={
                "input_format": input_format,
                "bytes": len(source_bytes),
                **_dict_value(loaded.get("details")),
            },
        )
        for stage in STAGE_ORDER[1:]:
            stages[stage] = StageResult(status="blocked", message="Blocked by load stage.", details={})
        result = WebPipelineResult(
            run_id=run_id,
            output_dir=output_dir,
            input_format=input_format,
            stages=stages,
            artifacts=artifacts,
            preview=preview,
        )
        _write_result(output_dir, result)
        return result

    frame_time = float(loaded["frame_time"])
    source_frames = _frame_list(loaded["source_frames"])
    preview["source"] = {
        "frames": source_frames,
        "body_names": list(loaded["body_names"]),
        "frame_time": frame_time,
        "render_style": "capsule",
        "capsule_edges": _source_capsule_edges(tuple(str(item) for item in loaded["body_names"])),
    }
    stages["load"] = StageResult(
        status="ok",
        message=str(loaded["message"]),
        details=_dict_value(loaded.get("details")),
    )

    primary_retarget = _retarget_source_to_g1(
        source_path,
        input_format=input_format,
        source_frames=source_frames,
        body_names=tuple(str(item) for item in loaded["body_names"]),
        max_frames=frame_limit,
        mode=str(loaded["retarget_source"]),
    )
    retargets = [primary_retarget]
    if compare_retargeters and _retarget_id(primary_retarget) != "rule_based_preview":
        comparison = _retarget_preview_frames_to_g1(
            source_frames,
            body_names=tuple(str(item) for item in loaded["body_names"]),
            max_frames=frame_limit,
            mode=str(loaded["retarget_source"]),
        )
        comparison["message"] = "Generated rule-based preview retarget for comparison."
        retargets.append(comparison)

    retarget = retargets[0]
    for index, item in enumerate(retargets):
        method_id = _retarget_id(item)
        g1_csv = output_dir / ("retargeted_g1_preview.csv" if index == 0 else f"retargeted_g1_{method_id}.csv")
        report_json = output_dir / ("retarget_report.json" if index == 0 else f"retarget_report_{method_id}.json")
        _write_g1_csv(g1_csv, item["trajectory"])
        if index == 0:
            artifacts["retargeted_g1_csv"] = str(g1_csv)
            artifacts["retarget_report"] = str(report_json)
        else:
            artifacts[f"retargeted_g1_{method_id}_csv"] = str(g1_csv)
            artifacts[f"retarget_{method_id}_report"] = str(report_json)
        report_json.write_text(
            json.dumps(item["report"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    retarget_details = dict(retarget["report"])
    retarget_details["comparison_enabled"] = compare_retargeters
    retarget_details["retargeters"] = [
        {
            "id": _retarget_id(item),
            "title": _retarget_title(item),
            "selected_retargeter": _dict_value(item.get("report")).get("selected_retargeter", ""),
        }
        for item in retargets
    ]
    stages["retarget"] = StageResult(
        status="ok",
        message=str(retarget["message"]),
        details=retarget_details,
    )

    if model_xml.exists():
        try:
            model = load_g1_kinematic_model(model_xml)
            robot_frames = _robot_preview_frames(model, retarget["trajectory"])
            preview["robot"] = {
                "frames": robot_frames,
                "body_names": list(KINEMATIC_BODY_NAMES),
                "model_xml": str(model_xml),
            }
            preview["timeline"] = _timeline(retarget["trajectory"], frame_time)
            kinematic_report = {
                "model_xml": str(model_xml),
                "frames": len(robot_frames),
                "bodies": list(KINEMATIC_BODY_NAMES),
            }
            stages["kinematic_sim"] = StageResult(
                status="ok",
                message="Generated MuJoCo-model kinematic FK preview without physics stepping.",
                details=kinematic_report,
            )
        except Exception as exc:  # pragma: no cover - defensive status surfacing.
            stages["kinematic_sim"] = StageResult(
                status="failed",
                message=f"G1 MJCF kinematic preview failed: {exc}",
                details={"model_xml": str(model_xml)},
            )
    else:
        stages["kinematic_sim"] = StageResult(
            status="blocked",
            message="G1 MJCF model file was not found.",
            details={"model_xml": str(model_xml)},
        )

    physics = _run_mujoco_physics_preview(
        model_xml=model_xml,
        trajectory=retarget["trajectory"],
        frame_time=frame_time,
        output_dir=output_dir,
        render_frames=render_frames,
        video_filename=MUJOCO_VIDEO_FILENAME,
    )
    physics_panels = [
        _panel_from_physics(
            run_id=run_id,
            retarget=retarget,
            physics=physics,
            artifact_name=MUJOCO_VIDEO_ARTIFACT,
            model_xml=model_xml,
            render_frames=render_frames,
        )
    ]
    if isinstance(physics.get("video_path"), str) and physics["video_path"]:
        artifacts[MUJOCO_VIDEO_ARTIFACT] = str(physics["video_path"])

    for item in retargets[1:]:
        method_id = _retarget_id(item)
        artifact_name = f"mujoco_{method_id}_render_mp4"
        comparison_physics = _run_mujoco_physics_preview(
            model_xml=model_xml,
            trajectory=item["trajectory"],
            frame_time=frame_time,
            output_dir=output_dir,
            render_frames=render_frames,
            video_filename=f"mujoco_g1_render_{method_id}.mp4",
        )
        if isinstance(comparison_physics.get("video_path"), str) and comparison_physics["video_path"]:
            artifacts[artifact_name] = str(comparison_physics["video_path"])
        physics_panels.append(
            _panel_from_physics(
                run_id=run_id,
                retarget=item,
                physics=comparison_physics,
                artifact_name=artifact_name,
                model_xml=model_xml,
                render_frames=render_frames,
            )
        )

    preview["panels"] = physics_panels
    if physics_panels:
        preview["mujoco"] = physics_panels[0]
    physics_details = {key: value for key, value in physics.items() if key not in {"status", "message"}}
    physics_details["panels"] = physics_panels
    physics_details["root_xy_locked"] = ROOT_XY_LOCKED_FOR_VIS
    physics_status = _combined_panel_status(physics_panels)
    stages["physics_sim"] = StageResult(
        status=physics_status,
        message=(
            f"Generated {len(physics_panels)} MuJoCo G1 render panel(s)."
            if physics_status == "ok"
            else str(physics["message"])
        ),
        details=physics_details,
    )

    result = WebPipelineResult(
        run_id=run_id,
        output_dir=output_dir,
        input_format=input_format,
        stages=stages,
        artifacts=artifacts,
        preview=preview,
    )
    _write_result(output_dir, result)
    return result


def _load_preview_source(source_bytes: bytes, input_format: str, max_frames: int | None) -> dict[str, object]:
    if input_format == "bvh":
        try:
            text = source_bytes.decode("utf-8")
            motion = parse_bvh_motion(text)
        except (UnicodeDecodeError, ValueError) as exc:
            return {
                "status": "failed",
                "message": f"BVH load failed: {exc}",
                "details": {},
            }
        source_frames = _source_preview_frames(motion, max_frames=max_frames)
        return {
            "status": "ok",
            "message": "BVH parsed and source FK preview frames were generated.",
            "source_frames": source_frames,
            "body_names": list(DEFAULT_SOURCE_BODY_NAMES),
            "frame_time": motion.frame_time,
            "retarget_source": "bvh_fk",
            "details": {
                "frames": len(motion.frames),
                "used_frames": len(source_frames),
                "frame_time": motion.frame_time,
                "joints": len(motion.joints),
                "channels": motion.channel_count,
            },
        }
    if input_format == "smpl":
        return _load_smpl_preview_source(source_bytes, max_frames=max_frames)
    return {
        "status": "blocked",
        "message": f"Unsupported upload format for preview: {input_format}",
        "details": {"input_format": input_format},
    }


def _load_smpl_preview_source(source_bytes: bytes, max_frames: int | None) -> dict[str, object]:
    if importlib.util.find_spec("numpy") is None:
        return {
            "status": "blocked",
            "message": "SMPL-like .npz preview requires numpy in the active Python environment.",
            "details": {"input_format": "smpl"},
        }
    try:
        import numpy as np  # type: ignore[import-not-found]

        with np.load(io.BytesIO(source_bytes), allow_pickle=False) as data:
            keys = set(data.files)
            pose_key = _first_present(keys, ("poses", "pose_body", "body_pose", "fullpose"))
            trans_key = _first_present(keys, ("trans", "transl", "translation", "root_trans"))
            if pose_key is None:
                return {
                    "status": "failed",
                    "message": "SMPL-like .npz missing poses/pose_body/body_pose/fullpose array.",
                    "details": {"keys": sorted(keys)},
                }
            poses = np.asarray(data[pose_key], dtype=float)
            if poses.ndim == 1:
                poses = poses.reshape(1, -1)
            trans = (
                np.asarray(data[trans_key], dtype=float)
                if trans_key is not None
                else np.zeros((poses.shape[0], 3), dtype=float)
            )
            if trans.ndim == 1:
                trans = trans.reshape(1, -1)
            fps = float(np.asarray(data["mocap_framerate"]).reshape(-1)[0]) if "mocap_framerate" in keys else 30.0
            source_frames = _smpl_preview_frames_from_arrays(poses, trans, max_frames=max_frames)
            return {
                "status": "ok",
                "message": "SMPL-like NPZ arrays parsed into an approximate preview skeleton.",
                "source_frames": source_frames,
                "body_names": list(SMPL_PREVIEW_BODY_NAMES),
                "frame_time": 1.0 / fps if fps > 0 else 1.0 / 30.0,
                "retarget_source": "smpl_npz_preview",
                "details": {
                    "frames": int(poses.shape[0]),
                    "used_frames": len(source_frames),
                    "pose_key": pose_key,
                    "trans_key": trans_key or "",
                    "pose_dim": int(poses.shape[1]) if poses.ndim > 1 else 0,
                    "fps": fps,
                    "approximation": "SMPL body-model mesh/joint decoding is not implemented; this uses root translation plus low-dimensional pose cues.",
                },
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return {
            "status": "failed",
            "message": f"SMPL-like NPZ load failed: {exc}",
            "details": {},
        }


def _smpl_preview_frames_from_arrays(
    poses: object,
    trans: object,
    max_frames: int | None,
) -> list[dict[str, list[float]]]:
    frame_count = len(poses) if max_frames is None else min(len(poses), max_frames)  # type: ignore[arg-type]
    frames: list[dict[str, list[float]]] = []
    for index in range(frame_count):
        pose = [float(value) for value in poses[index].reshape(-1)]  # type: ignore[index,union-attr]
        root = [float(value) for value in trans[min(index, len(trans) - 1)].reshape(-1)[:3]]  # type: ignore[arg-type,index,union-attr]
        while len(root) < 3:
            root.append(0.0)
        sway = _pose_value(pose, 2) * 0.08
        arm = _pose_value(pose, 8) * 0.12
        step = _pose_value(pose, 15) * 0.10
        crouch = abs(_pose_value(pose, 5)) * 0.04
        hips = [root[0], root[1], root[2] + 0.9 - crouch]
        frames.append(
            {
                "Hips": _round_point(hips),
                "Spine1": _round_point((hips[0], hips[1], hips[2] + 0.22)),
                "Chest": _round_point((hips[0], hips[1], hips[2] + 0.46)),
                "Head": _round_point((hips[0], hips[1], hips[2] + 0.72)),
                "LeftHand": _round_point((hips[0] - 0.34, hips[1] + 0.12 + arm, hips[2] + 0.36)),
                "RightHand": _round_point((hips[0] + 0.34, hips[1] - 0.12 - arm, hips[2] + 0.36)),
                "LeftFoot": _round_point((hips[0] - 0.09 + step, hips[1] + 0.08 + sway, hips[2] - 0.9)),
                "RightFoot": _round_point((hips[0] + 0.09 - step, hips[1] - 0.08 + sway, hips[2] - 0.9)),
            }
        )
    return frames


def _retarget_source_to_g1(
    source_path: Path,
    *,
    input_format: str,
    source_frames: Sequence[Mapping[str, Sequence[float]]],
    body_names: Sequence[str],
    max_frames: int | None,
    mode: str,
) -> dict[str, object]:
    gmr = _retarget_with_gmr(source_path, input_format=input_format, max_frames=max_frames)
    if gmr["status"] == "ok":
        return {
            "trajectory": gmr["trajectory"],
            "report": gmr["report"],
            "message": "Retargeted to Unitree G1 with GMR.",
        }

    fallback = _retarget_preview_frames_to_g1(
        source_frames,
        body_names=body_names,
        max_frames=max_frames,
        mode=mode,
    )
    report = dict(fallback["report"])
    report["requested_retargeter"] = "gmr"
    report["selected_retargeter"] = "rule_based_preview"
    report["gmr_status"] = gmr["status"]
    report["gmr_message"] = gmr["message"]
    report["gmr_details"] = _dict_value(gmr.get("details"))
    return {
        "trajectory": fallback["trajectory"],
        "report": report,
        "message": (
            "GMR was not available for this upload, so a deterministic G1 preview "
            "fallback was generated. See retarget details for the GMR blocker."
        ),
    }


def _retarget_id(retarget: Mapping[str, object]) -> str:
    report = _dict_value(retarget.get("report"))
    raw = str(report.get("selected_retargeter") or report.get("mode") or "retargeter")
    return _safe_artifact_stem(raw)


def _retarget_title(retarget: Mapping[str, object]) -> str:
    report = _dict_value(retarget.get("report"))
    method = str(report.get("selected_retargeter") or report.get("mode") or "retargeter")
    robot = str(report.get("target_robot") or DEFAULT_GMR_ROBOT)
    return f"{robot} / {method}"


def _safe_artifact_stem(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.lower())
    return safe.strip("_") or "retargeter"


def _panel_from_physics(
    *,
    run_id: str,
    retarget: Mapping[str, object],
    physics: Mapping[str, object],
    artifact_name: str,
    model_xml: Path,
    render_frames: bool,
) -> dict[str, object]:
    return {
        "title": _retarget_title(retarget),
        "method": _retarget_id(retarget),
        "status": str(physics.get("status", "")),
        "message": str(physics.get("message", "")),
        "video_artifact": artifact_name,
        "video_url": f"/api/artifact?run_id={run_id}&name={artifact_name}",
        "robot": DEFAULT_GMR_ROBOT,
        "model_xml": str(model_xml),
        "renderer": "mujoco.Renderer",
        "render_frames": render_frames,
        "root_xy_locked": ROOT_XY_LOCKED_FOR_VIS,
    }


def _combined_panel_status(panels: Sequence[Mapping[str, object]]) -> str:
    statuses = [str(panel.get("status", "")) for panel in panels]
    if statuses and all(status == "ok" for status in statuses):
        return "ok"
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "blocked" for status in statuses):
        return "blocked"
    return statuses[0] if statuses else "blocked"


def _retarget_with_gmr(source_path: Path, *, input_format: str, max_frames: int | None) -> dict[str, object]:
    if not DEFAULT_GMR_ROOT.exists():
        return {
            "status": "blocked",
            "message": "GMR repository was not found.",
            "details": {"gmr_root": str(DEFAULT_GMR_ROOT)},
        }

    source_kind = _gmr_source_kind(source_path, input_format)
    if source_kind is None:
        return {
            "status": "blocked",
            "message": "GMR supports this web path only for BVH or full SMPL-X NPZ inputs.",
            "details": {"input_format": input_format},
        }

    if str(DEFAULT_GMR_ROOT) not in sys.path:
        sys.path.insert(0, str(DEFAULT_GMR_ROOT))

    try:
        import numpy as np  # type: ignore[import-not-found]

        GeneralMotionRetargeting = _load_gmr_retargeter_class()
    except Exception as exc:
        return {
            "status": "blocked",
            "message": f"GMR Python dependencies are unavailable: {exc}",
            "details": {
                "gmr_root": str(DEFAULT_GMR_ROOT),
                "required": "general_motion_retargeting, mink, scipy, qpsolvers, numpy",
            },
        }

    try:
        if source_kind.startswith("bvh_"):
            frames, human_height, source_fps, bvh_adapter = _load_gmr_bvh_frames(source_path, source_kind)
            gmr_config_source = source_kind
        elif source_kind == "smplx_preview":
            frames, human_height, source_fps = _load_gmr_smpl_preview_frames(source_path, max_frames=max_frames)
            gmr_config_source = "smplx"
            bvh_adapter = {}
        else:
            frames, human_height, source_fps = _load_gmr_smplx_frames(source_path)
            gmr_config_source = source_kind
            bvh_adapter = {}
        retargeter = GeneralMotionRetargeting(
            src_human=gmr_config_source,
            tgt_robot=DEFAULT_GMR_ROBOT,
            actual_human_height=human_height,
            verbose=False,
        )
        trajectory: list[dict[str, object]] = []
        for frame_index, human_frame in enumerate(_limited_sequence(frames, max_frames)):
            qpos = np.asarray(retargeter.retarget(human_frame), dtype=float).reshape(-1)
            trajectory.append(_trajectory_item_from_gmr_qpos(qpos, frame_index))

        limitations = [
            "This is kinematic GMR IK output; controller-grade physical tracking is still separate.",
        ]
        if source_kind == "smplx_preview":
            limitations.append("SMPL preview inputs use approximate joint targets, not full SMPL-X body-model decoding.")
        return {
            "status": "ok",
            "message": "GMR retarget completed.",
            "trajectory": trajectory,
            "report": {
                "mode": "gmr",
                "selected_retargeter": "gmr",
                "src_human": source_kind,
                "gmr_config_source": gmr_config_source,
                "target_robot": DEFAULT_GMR_ROBOT,
                "gmr_root": str(DEFAULT_GMR_ROOT),
                "frames": len(trajectory),
                "source_fps": source_fps,
                "output_joint_count": len(G1_JOINT_COLUMNS),
                "model_xml": str(DEFAULT_G1_MJCF),
                "limitations": limitations,
                "gmr_bvh_adapter": bvh_adapter,
            },
        }
    except Exception as exc:
        return {
            "status": "failed",
            "message": f"GMR retarget failed: {exc}",
            "details": {"src_human": source_kind, "robot": DEFAULT_GMR_ROBOT},
        }


def _gmr_source_kind(source_path: Path, input_format: str) -> str | None:
    if input_format == "bvh":
        bvh_joints = _bvh_joint_names(source_path)
        if "nokov" in source_path.name.lower() or {"LeftToeBase", "RightToeBase"}.issubset(bvh_joints):
            return "bvh_nokov"
        return "bvh_lafan1"
    if input_format == "smpl":
        if _looks_like_full_smplx_npz(source_path):
            return "smplx"
        if _looks_like_smpl_preview_npz(source_path):
            return "smplx_preview"
    return None


def _bvh_joint_names(source_path: Path) -> set[str]:
    names: set[str] = set()
    try:
        with source_path.open(encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line == "MOTION":
                    break
                parts = line.split()
                if len(parts) >= 2 and parts[0] in {"ROOT", "JOINT"}:
                    names.add(parts[1])
    except OSError:
        return set()
    return names


def _looks_like_full_smplx_npz(source_path: Path) -> bool:
    if importlib.util.find_spec("numpy") is None:
        return False
    try:
        import numpy as np  # type: ignore[import-not-found]

        with np.load(source_path, allow_pickle=True) as data:
            return {"pose_body", "root_orient", "trans", "betas", "gender"}.issubset(set(data.files))
    except Exception:
        return False


def _looks_like_smpl_preview_npz(source_path: Path) -> bool:
    if importlib.util.find_spec("numpy") is None:
        return False
    try:
        import numpy as np  # type: ignore[import-not-found]

        with np.load(source_path, allow_pickle=False) as data:
            keys = set(data.files)
            return _first_present(keys, ("poses", "pose_body", "body_pose", "fullpose")) is not None
    except Exception:
        return False


def _load_gmr_bvh_frames(source_path: Path, source_kind: str) -> tuple[Sequence[object], float, float, dict[str, object]]:
    _ensure_gmr_package_stub()
    from general_motion_retargeting.utils.lafan1 import load_bvh_file  # type: ignore[import-not-found]

    bvh_format = source_kind.removeprefix("bvh_")
    gmr_source_path, adapter_report = _prepare_gmr_bvh_source(source_path, source_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        try:
            frames, human_height = load_bvh_file(str(gmr_source_path), format=bvh_format)
        finally:
            if gmr_source_path != source_path:
                try:
                    gmr_source_path.unlink()
                except OSError:
                    pass
    frame_time = _bvh_frame_time(source_path)
    fps = 1.0 / frame_time if frame_time > 0 else 30.0
    return frames, float(human_height), fps, adapter_report


def _prepare_gmr_bvh_source(source_path: Path, source_kind: str) -> tuple[Path, dict[str, object]]:
    if source_kind != "bvh_nokov":
        return source_path, {"applied": False, "reason": f"source_kind_{source_kind}_preserved"}

    try:
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
        adapted_text, report = _adapt_soma_bvh_for_gmr(source_text)
    except ValueError as exc:
        return source_path, {"applied": False, "reason": str(exc)}
    if not bool(report.get("applied")):
        return source_path, report

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".bvh",
        prefix=f"{_safe_artifact_stem(source_path.stem)}-gmr-soma-",
        dir=source_path.parent,
        delete=False,
    ) as handle:
        handle.write(adapted_text)
        adapted_path = Path(handle.name)
    return adapted_path, report


def _adapt_soma_bvh_for_gmr(text: str) -> tuple[str, dict[str, object]]:
    lines = text.splitlines(keepends=True)
    motion_index = _bvh_motion_line_index(lines)
    if motion_index is None:
        return text, {"applied": False, "reason": "missing_motion_section"}

    header_lines = lines[:motion_index]
    motion_lines = lines[motion_index:]
    original_channels = _bvh_header_channel_counts(header_lines)
    if len(original_channels) < 2:
        return text, {"applied": False, "reason": "not_soma_root_hips_bvh"}
    if original_channels[0] != ("Root", 6) or original_channels[1] != ("Hips", 6):
        return text, {"applied": False, "reason": "not_soma_root_hips_bvh"}

    span = _dummy_root_hips_span(header_lines)
    if span is None:
        return text, {"applied": False, "reason": "not_soma_root_hips_bvh"}
    root_index, hips_index, root_close_index = span

    input_width = sum(count for _, count in original_channels)
    adapted_motion, frame_count, nonzero_dummy = _drop_dummy_root_motion_channels(
        motion_lines,
        input_width=input_width,
    )
    if nonzero_dummy > 0:
        return text, {
            "applied": False,
            "reason": "dummy_root_channels_nonzero",
            "input_channels": input_width,
            "frames": frame_count,
            "dummy_root_nonzero_frames": nonzero_dummy,
        }

    joint_names = {name for name, _ in original_channels}
    missing_soma_joints = sorted(SOMA_GMR_BVH_REQUIRED_JOINTS.difference(joint_names))
    if missing_soma_joints:
        return text, {
            "applied": False,
            "reason": "missing_soma_gmr_joints",
            "missing_joints": missing_soma_joints,
        }

    adapted_header = _adapt_soma_bvh_header(header_lines, root_index, hips_index, root_close_index)
    adapted_channels = _bvh_header_channel_counts(adapted_header)
    output_width = sum(count for _, count in adapted_channels)
    if output_width != input_width - 6:
        raise ValueError("SOMA GMR adapter could not account for the dummy Root channels")

    alias_map = {
        source: target
        for source, target in SOMA_GMR_BVH_JOINT_ALIASES.items()
        if any(name == target for name, _ in adapted_channels)
    }
    return "".join(adapted_header + adapted_motion), {
        "applied": True,
        "mode": "soma_root_hips_unwrap",
        "dropped_dummy_root_channels": 6,
        "input_channels": input_width,
        "output_channels": output_width,
        "frames": frame_count,
        "dummy_root_nonzero_frames": nonzero_dummy,
        "aliases": alias_map,
        "foot_alias_source": "GMR nokov loader creates LeftFootMod/RightFootMod from LeftFoot and ToeBase joints",
    }


def _bvh_motion_line_index(lines: Sequence[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip() == "MOTION":
            return index
    return None


def _bvh_header_channel_counts(header_lines: Sequence[str]) -> list[tuple[str, int]]:
    channels: list[tuple[str, int]] = []
    current_joint = ""
    for raw_line in header_lines:
        parts = raw_line.strip().split()
        if len(parts) >= 2 and parts[0] in {"ROOT", "JOINT"}:
            current_joint = parts[1]
        elif len(parts) >= 2 and parts[0] == "CHANNELS" and current_joint:
            try:
                channels.append((current_joint, int(parts[1])))
            except ValueError:
                continue
    return channels


def _dummy_root_hips_span(header_lines: Sequence[str]) -> tuple[int, int, int] | None:
    root_index: int | None = None
    hips_index: int | None = None
    root_close_index: int | None = None
    depth = 0
    for index, raw_line in enumerate(header_lines):
        line = raw_line.strip()
        parts = line.split()
        if root_index is None and depth == 0 and len(parts) >= 2 and parts[:2] == ["ROOT", "Root"]:
            root_index = index
        elif (
            root_index is not None
            and hips_index is None
            and depth == 1
            and len(parts) >= 2
            and parts[:2] == ["JOINT", "Hips"]
        ):
            hips_index = index

        depth += raw_line.count("{")
        if root_index is not None:
            depth -= raw_line.count("}")
            if index > root_index and root_close_index is None and depth == 0 and "}" in raw_line:
                root_close_index = index
                break
        else:
            depth -= raw_line.count("}")

    if root_index is None or hips_index is None or root_close_index is None:
        return None
    return root_index, hips_index, root_close_index


def _adapt_soma_bvh_header(
    header_lines: Sequence[str],
    root_index: int,
    hips_index: int,
    root_close_index: int,
) -> list[str]:
    adapted: list[str] = []
    for index, raw_line in enumerate(header_lines):
        if root_index <= index < hips_index:
            continue
        if index == hips_index:
            adapted.append(_replace_bvh_joint_decl(raw_line, "ROOT", "Hips"))
            continue
        if index == root_close_index:
            continue
        adapted.append(_rename_bvh_joint_decl(raw_line, SOMA_GMR_BVH_JOINT_ALIASES))
    return adapted


def _replace_bvh_joint_decl(raw_line: str, kind: str, name: str) -> str:
    newline = "\n" if raw_line.endswith("\n") else ""
    return f"{kind} {name}{newline}"


def _rename_bvh_joint_decl(raw_line: str, aliases: Mapping[str, str]) -> str:
    parts = raw_line.strip().split()
    if len(parts) >= 2 and parts[0] in {"ROOT", "JOINT"} and parts[1] in aliases:
        indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        newline = "\n" if raw_line.endswith("\n") else ""
        return f"{indent}{parts[0]} {aliases[parts[1]]}{newline}"
    return raw_line


def _drop_dummy_root_motion_channels(
    motion_lines: Sequence[str],
    *,
    input_width: int,
) -> tuple[list[str], int, int]:
    adapted: list[str] = []
    frame_rows = 0
    nonzero_dummy = 0
    seen_frame_time = False
    for raw_line in motion_lines:
        stripped = raw_line.strip()
        if not stripped:
            adapted.append(raw_line)
            continue
        if stripped.startswith("Frame Time:"):
            seen_frame_time = True
            adapted.append(raw_line)
            continue
        if not seen_frame_time or stripped == "MOTION" or stripped.startswith("Frames:"):
            adapted.append(raw_line)
            continue

        values = stripped.split()
        if len(values) != input_width:
            raise ValueError(
                f"SOMA GMR adapter expected {input_width} motion channels, got {len(values)}"
            )
        dummy_values = [float(value) for value in values[:6]]
        if any(abs(value) > 1e-8 for value in dummy_values):
            nonzero_dummy += 1
        adapted.append(" ".join(values[6:]) + ("\n" if raw_line.endswith("\n") else ""))
        frame_rows += 1
    return adapted, frame_rows, nonzero_dummy


def _load_gmr_smplx_frames(source_path: Path) -> tuple[Sequence[object], float, float]:
    _ensure_gmr_package_stub()
    from general_motion_retargeting.utils.smpl import (  # type: ignore[import-not-found]
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    body_model_root = DEFAULT_GMR_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(str(source_path), body_model_root)
    frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    return frames, float(human_height), float(aligned_fps)


def _load_gmr_smpl_preview_frames(source_path: Path, max_frames: int | None) -> tuple[Sequence[object], float, float]:
    import numpy as np  # type: ignore[import-not-found]

    with np.load(source_path, allow_pickle=False) as data:
        keys = set(data.files)
        pose_key = _first_present(keys, ("poses", "pose_body", "body_pose", "fullpose"))
        if pose_key is None:
            raise ValueError("SMPL preview input is missing poses/pose_body/body_pose/fullpose.")
        trans_key = _first_present(keys, ("trans", "transl", "translation", "root_trans"))
        poses = np.asarray(data[pose_key], dtype=float)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        trans = (
            np.asarray(data[trans_key], dtype=float)
            if trans_key is not None
            else np.zeros((poses.shape[0], 3), dtype=float)
        )
        if trans.ndim == 1:
            trans = trans.reshape(1, -1)
        fps = float(np.asarray(data["mocap_framerate"]).reshape(-1)[0]) if "mocap_framerate" in keys else 30.0

    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    frames: list[dict[str, list[object]]] = []
    frame_count = len(poses) if max_frames is None else min(len(poses), max_frames)
    for index in range(frame_count):
        pose = [float(value) for value in poses[index].reshape(-1)]
        root = [float(value) for value in trans[min(index, len(trans) - 1)].reshape(-1)[:3]]
        while len(root) < 3:
            root.append(0.0)
        sway = _pose_value(pose, 2) * 0.08
        arm = _pose_value(pose, 8) * 0.12
        step = _pose_value(pose, 15) * 0.10
        crouch = abs(_pose_value(pose, 5)) * 0.04
        pelvis = np.array([root[0], root[1], root[2] + 0.92 - crouch], dtype=float)
        frames.append(
            {
                "pelvis": [pelvis, identity],
                "spine3": [pelvis + np.array([0.0, 0.0, 0.46]), identity],
                "left_hip": [pelvis + np.array([0.0, 0.10, -0.08]), identity],
                "right_hip": [pelvis + np.array([0.0, -0.10, -0.08]), identity],
                "left_knee": [pelvis + np.array([0.03 + step, 0.11 + sway, -0.48]), identity],
                "right_knee": [pelvis + np.array([-0.03 - step, -0.11 + sway, -0.48]), identity],
                "left_foot": [pelvis + np.array([0.08 + step, 0.12 + sway, -0.92]), identity],
                "right_foot": [pelvis + np.array([-0.08 - step, -0.12 + sway, -0.92]), identity],
                "left_shoulder": [pelvis + np.array([0.0, 0.22, 0.42]), identity],
                "right_shoulder": [pelvis + np.array([0.0, -0.22, 0.42]), identity],
                "left_elbow": [pelvis + np.array([0.02, 0.48 + arm, 0.25]), identity],
                "right_elbow": [pelvis + np.array([0.02, -0.48 - arm, 0.25]), identity],
                "left_wrist": [pelvis + np.array([0.04, 0.70 + arm, 0.12]), identity],
                "right_wrist": [pelvis + np.array([0.04, -0.70 - arm, 0.12]), identity],
            }
        )
    return frames, 1.70, fps if fps > 0 else 30.0


def _load_gmr_retargeter_class() -> object:
    _ensure_gmr_package_stub()
    from general_motion_retargeting.motion_retarget import (  # type: ignore[import-not-found]
        GeneralMotionRetargeting,
    )

    return GeneralMotionRetargeting


def _ensure_gmr_package_stub() -> None:
    """Expose GMR modules without importing its viewer-heavy package __init__."""

    package = sys.modules.get("general_motion_retargeting")
    if package is not None and hasattr(package, "__path__"):
        return
    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = [str(DEFAULT_GMR_ROOT / "general_motion_retargeting")]  # type: ignore[attr-defined]
    sys.modules["general_motion_retargeting"] = package


def _trajectory_item_from_gmr_qpos(qpos: object, frame_index: int) -> dict[str, object]:
    values = [float(value) for value in qpos]  # type: ignore[union-attr]
    root = values[:3] if len(values) >= 3 else [0.0, 0.0, DEFAULT_ROOT_HEIGHT]
    root_quat = values[3:7] if len(values) >= 7 else [1.0, 0.0, 0.0, 0.0]
    joints = {
        column: values[index + 7] if index + 7 < len(values) else 0.0
        for index, column in enumerate(G1_JOINT_COLUMNS)
    }
    return {
        "frame": frame_index,
        "root": _float_sequence(root, 3),
        "root_quat": _normalize_quat_wxyz(root_quat),
        "root_euler": [0.0, 0.0, 0.0],
        "joints": joints,
    }


def _retarget_preview_frames_to_g1(
    source_frames: Sequence[Mapping[str, Sequence[float]]],
    *,
    body_names: Sequence[str],
    max_frames: int | None,
    mode: str,
) -> dict[str, object]:
    frame_count = len(source_frames) if max_frames is None else min(len(source_frames), max_frames)
    trajectory: list[dict[str, object]] = []
    previous_root_x = 0.0
    for frame_index, frame in enumerate(source_frames[:frame_count]):
        root = _frame_point(frame, "Hips")
        root_x = root[0]
        root_z = root[2]
        left_foot_y = _frame_point(frame, "LeftFoot")[1]
        right_foot_y = _frame_point(frame, "RightFoot")[1]
        left_hand_y = _frame_point(frame, "LeftHand")[1]
        right_hand_y = _frame_point(frame, "RightHand")[1]
        forward_delta = root_x - previous_root_x if frame_index else 0.0
        previous_root_x = root_x

        joints = {name: 0.0 for name in G1_JOINT_COLUMNS}
        stride = _clamp(forward_delta * 8.0, -0.35, 0.35)
        lift_delta = _clamp((left_foot_y - right_foot_y) * 0.8, -0.25, 0.25)
        arm_swing = _clamp((left_hand_y - right_hand_y) * 0.35, -0.35, 0.35)

        joints["left_hip_pitch_joint_dof"] = stride + lift_delta
        joints["right_hip_pitch_joint_dof"] = -stride - lift_delta
        joints["left_knee_joint_dof"] = max(0.0, -lift_delta) + abs(stride) * 0.4
        joints["right_knee_joint_dof"] = max(0.0, lift_delta) + abs(stride) * 0.4
        joints["left_ankle_pitch_joint_dof"] = -joints["left_knee_joint_dof"] * 0.35
        joints["right_ankle_pitch_joint_dof"] = -joints["right_knee_joint_dof"] * 0.35
        joints["left_shoulder_pitch_joint_dof"] = -arm_swing
        joints["right_shoulder_pitch_joint_dof"] = arm_swing
        joints["left_elbow_joint_dof"] = 0.25 + max(0.0, arm_swing)
        joints["right_elbow_joint_dof"] = 0.25 + max(0.0, -arm_swing)
        joints["waist_yaw_joint_dof"] = _clamp(root_z * 0.15, -0.2, 0.2)

        trajectory.append(
            {
                "frame": frame_index,
                "root": [root_x, 0.0, DEFAULT_ROOT_HEIGHT],
                "root_euler": [0.0, 0.0, 0.0],
                "joints": joints,
            }
        )

    return {
        "trajectory": trajectory,
        "report": {
            "mode": "rule_based_preview",
            "selected_retargeter": "rule_based_preview",
            "input_mode": mode,
            "target_robot": DEFAULT_GMR_ROBOT,
            "frames": len(trajectory),
            "output_joint_count": len(G1_JOINT_COLUMNS),
            "source_body_count": len(body_names),
            "limitations": [
                "Not a learned model prediction.",
                "SMPL NPZ support is approximate and does not decode a body model mesh.",
                "Physical tracking control is not implemented in this placeholder retargeter.",
            ],
        },
    }


def _robot_preview_frames(
    model: G1KinematicModel,
    trajectory: Sequence[Mapping[str, object]],
) -> list[dict[str, list[float]]]:
    frames: list[dict[str, list[float]]] = []
    for item in trajectory:
        joints = item.get("joints", {})
        if not isinstance(joints, Mapping):
            joints = {}
        joint_values = [float(joints.get(column, 0.0)) for column in G1_JOINT_COLUMNS]
        root = _visual_root(item.get("root", (0.0, 0.0, DEFAULT_ROOT_HEIGHT)))
        root_euler = item.get("root_euler", (0.0, 0.0, 0.0))
        body_points = g1_fk_body_positions(
            model,
            joint_values,
            root_position=_float_sequence(root, 3),
            root_euler=_float_sequence(root_euler, 3),
        )
        frame: dict[str, list[float]] = {}
        for body_name in KINEMATIC_BODY_NAMES:
            points = body_points.get(body_name)
            if points:
                frame[body_name] = [round(float(value), 5) for value in points[0]]
        frames.append(frame)
    return frames


def _run_mujoco_physics_preview(
    *,
    model_xml: Path,
    trajectory: Sequence[Mapping[str, object]],
    frame_time: float,
    output_dir: Path,
    render_frames: bool,
    video_filename: str,
) -> dict[str, object]:
    if not model_xml.exists():
        return {
            "status": "blocked",
            "message": "G1 MJCF model file was not found.",
            "model_xml": str(model_xml),
        }
    if importlib.util.find_spec("mujoco") is None:
        return {
            "status": "blocked",
            "message": "Python package 'mujoco' is not installed in this environment.",
            "model_xml": str(model_xml),
        }

    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco  # type: ignore[import-not-found]

        model = mujoco.MjModel.from_xml_path(str(model_xml))
        data = mujoco.MjData(model)
        steps = len(trajectory)
        ground_alignment = _compute_mujoco_ground_alignment(mujoco, model, data, trajectory)
        root_z_offsets = _root_z_offsets_from_ground_alignment(ground_alignment, steps)
        video_path = output_dir / video_filename
        render = _render_mujoco_g1_video(
            mujoco,
            model,
            data,
            trajectory,
            video_path=video_path,
            frame_time=frame_time,
            render_frames=render_frames,
            root_z_offsets=root_z_offsets,
        )
        finite = True
        for index, item in enumerate(trajectory):
            _set_mujoco_state(
                model,
                data,
                item,
                lock_root_xy=ROOT_XY_LOCKED_FOR_VIS,
                root_z_offset=root_z_offsets[index] if index < len(root_z_offsets) else 0.0,
            )
            mujoco.mj_forward(model, data)
            finite = finite and all(math.isfinite(float(value)) for value in data.qpos)
        if render["status"] != "ok":
            return {
                "status": "failed",
                "message": str(render["message"]),
                "model_xml": str(model_xml),
                "robot": DEFAULT_GMR_ROBOT,
                "steps": steps,
                "qpos_dim": int(model.nq),
                "ctrl_dim": int(model.nu),
                "ground_alignment": _public_ground_alignment(ground_alignment),
                "render": render,
            }
        return {
            "status": "ok" if finite else "failed",
            "message": (
                "MuJoCo G1 forward playback and offscreen render completed."
                if finite
                else "MuJoCo produced non-finite qpos."
            ),
            "model_xml": str(model_xml),
            "robot": DEFAULT_GMR_ROBOT,
            "steps": steps,
            "frame_time": frame_time,
            "qpos_dim": int(model.nq),
            "ctrl_dim": int(model.nu),
            "rendered_by_mujoco": True,
            "render_backend": "mujoco.Renderer",
            "validation_mode": "mj_forward",
            "render_frames": render_frames,
            "root_xy_locked": ROOT_XY_LOCKED_FOR_VIS,
            "ground_alignment": _public_ground_alignment(ground_alignment),
            "video_path": str(video_path),
            "video_bytes": video_path.stat().st_size if video_path.exists() else 0,
            "render": render,
        }
    except Exception as exc:  # pragma: no cover - depends on optional mujoco runtime.
        return {
            "status": "failed",
            "message": f"MuJoCo physics rollout failed: {exc}",
            "model_xml": str(model_xml),
        }


def _render_mujoco_g1_video(
    mujoco: object,
    model: object,
    data: object,
    trajectory: Sequence[Mapping[str, object]],
    *,
    video_path: Path,
    frame_time: float,
    render_frames: bool,
    root_z_offsets: Sequence[float] = (),
    width: int = 960,
    height: int = 540,
) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"status": "blocked", "message": "ffmpeg is required to encode the MuJoCo render video."}
    if not trajectory:
        return {"status": "blocked", "message": "No trajectory frames were available for MuJoCo rendering."}

    fps = int(_clamp(round(1.0 / frame_time) if frame_time > 0 else 30, 1, 240))
    video_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = None
    process = None
    frame_count = 0
    frame_sums: list[int] = []
    frame_stds: list[float] = []
    previous_frame: bytes | None = None
    changed_frames = 0
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)  # type: ignore[attr-defined]
        camera = mujoco.MjvCamera()  # type: ignore[attr-defined]
        mujoco.mjv_defaultCamera(camera)  # type: ignore[attr-defined]
        camera.lookat = [0.0, 0.0, 0.8]
        camera.distance = 3.0
        camera.elevation = -12
        camera.azimuth = 135
        scene_option = mujoco.MjvOption()  # type: ignore[attr-defined]
        mujoco.mjv_defaultOption(scene_option)  # type: ignore[attr-defined]
        scene_option.frame = (
            mujoco.mjtFrame.mjFRAME_BODY  # type: ignore[attr-defined]
            if render_frames
            else mujoco.mjtFrame.mjFRAME_NONE  # type: ignore[attr-defined]
        )
        command = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdin is None:
            return {"status": "failed", "message": "ffmpeg stdin was not available."}

        for index, item in enumerate(trajectory):
            _set_mujoco_state(
                model,
                data,
                item,
                lock_root_xy=ROOT_XY_LOCKED_FOR_VIS,
                root_z_offset=root_z_offsets[index] if index < len(root_z_offsets) else 0.0,
            )
            mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
            _center_camera_on_robot(model, data, camera)
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            image = renderer.render()
            frame_bytes = image.tobytes()
            process.stdin.write(frame_bytes)
            frame_count += 1
            frame_sums.append(int(image.sum()))
            frame_stds.append(float(image.std()))
            if previous_frame is not None and frame_bytes != previous_frame:
                changed_frames += 1
            previous_frame = frame_bytes

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            return {
                "status": "failed",
                "message": "ffmpeg failed while encoding MuJoCo render video.",
                "ffmpeg_tail": stderr[-800:],
            }
        if not video_path.exists() or video_path.stat().st_size == 0:
            return {"status": "failed", "message": "MuJoCo render video was not written."}
        if not frame_sums or max(frame_stds) <= 0.0:
            return {"status": "failed", "message": "MuJoCo renderer produced blank frames."}
        return {
            "status": "ok",
            "message": "Encoded MuJoCo offscreen render video.",
            "video_path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
            "frame_sum_min": min(frame_sums),
            "frame_sum_max": max(frame_sums),
            "frame_std_max": round(max(frame_stds), 4),
            "changed_frames": changed_frames,
            "render_frames": render_frames,
            "root_xy_locked": ROOT_XY_LOCKED_FOR_VIS,
            "root_z_aligned_to_ground": bool(root_z_offsets),
        }
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.kill()
        return {"status": "failed", "message": f"MuJoCo video render failed: {exc}"}
    finally:
        if renderer is not None:
            renderer.close()


def _set_mujoco_state(
    model: object,
    data: object,
    item: Mapping[str, object],
    *,
    lock_root_xy: bool,
    root_z_offset: float = 0.0,
) -> None:
    root = _float_sequence(item.get("root", (0.0, 0.0, DEFAULT_ROOT_HEIGHT)), 3)
    if lock_root_xy:
        root[0] = 0.0
        root[1] = 0.0
    root[2] += root_z_offset
    root_quat = _normalize_quat_wxyz(item.get("root_quat", (1.0, 0.0, 0.0, 0.0)))
    joints = item.get("joints", {})
    if not isinstance(joints, Mapping):
        joints = {}
    qpos = data.qpos  # type: ignore[attr-defined]
    if len(qpos) >= 7:
        qpos[0:3] = root
        qpos[3:7] = root_quat
    for column in G1_JOINT_COLUMNS:
        joint_name = column[:-4] if column.endswith("_dof") else column
        try:
            joint_id = model.joint(joint_name).id  # type: ignore[attr-defined]
            qpos_address = model.jnt_qposadr[joint_id]  # type: ignore[attr-defined]
        except Exception:
            continue
        if 0 <= qpos_address < len(qpos):
            qpos[qpos_address] = float(joints.get(column, 0.0))


def _compute_mujoco_ground_alignment(
    mujoco: object,
    model: object,
    data: object,
    trajectory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    foot_geom_ids = _mujoco_foot_geom_ids(mujoco, model)
    if not trajectory:
        return {"applied": False, "mode": "sequence_foot_geom_min_z", "reason": "empty_trajectory"}
    if not foot_geom_ids:
        return {"applied": False, "mode": "sequence_foot_geom_min_z", "reason": "no_foot_geoms"}

    pre_heights: list[float] = []
    for item in trajectory:
        _set_mujoco_state(model, data, item, lock_root_xy=ROOT_XY_LOCKED_FOR_VIS)
        mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
        pre_heights.append(_lowest_mujoco_geom_z(mujoco, model, data, foot_geom_ids))

    finite_pre_heights = [value for value in pre_heights if math.isfinite(value)]
    if not finite_pre_heights:
        return {
            "applied": False,
            "mode": "sequence_foot_geom_min_z",
            "reason": "non_finite_foot_heights",
            "foot_geom_count": len(foot_geom_ids),
        }
    sequence_offset = -min(finite_pre_heights)
    post_heights = [value + sequence_offset for value in finite_pre_heights]
    root_z_offsets = [sequence_offset if math.isfinite(value) else 0.0 for value in pre_heights]
    return {
        "applied": True,
        "mode": "sequence_foot_geom_min_z",
        "ground_height": 0.0,
        "frames": len(root_z_offsets),
        "foot_geom_count": len(foot_geom_ids),
        "foot_root_bodies": list(GROUND_ALIGNMENT_FOOT_ROOT_BODIES),
        "pre_min_foot_z": round(min(finite_pre_heights), 6),
        "pre_max_foot_z": round(max(finite_pre_heights), 6),
        "pre_mean_foot_z": round(sum(finite_pre_heights) / len(finite_pre_heights), 6),
        "post_min_foot_z": round(min(post_heights), 6),
        "post_max_foot_z": round(max(post_heights), 6),
        "post_mean_foot_z": round(sum(post_heights) / len(post_heights), 6),
        "root_z_offset_min": round(sequence_offset, 6),
        "root_z_offset_max": round(sequence_offset, 6),
        "root_z_offset_mean": round(sequence_offset, 6),
        "root_z_offset_delta_abs_max": 0.0,
        "_root_z_offsets": [round(value, 8) for value in root_z_offsets],
    }


def _root_z_offsets_from_ground_alignment(alignment: Mapping[str, object], frame_count: int) -> list[float]:
    raw_offsets = alignment.get("_root_z_offsets")
    if not isinstance(raw_offsets, Sequence) or isinstance(raw_offsets, (str, bytes)):
        return [0.0] * frame_count
    offsets = [float(value) if isinstance(value, (int, float)) else 0.0 for value in raw_offsets]
    if len(offsets) < frame_count:
        offsets.extend([0.0] * (frame_count - len(offsets)))
    return offsets[:frame_count]


def _public_ground_alignment(alignment: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in alignment.items() if not str(key).startswith("_")}


def _mujoco_foot_geom_ids(mujoco: object, model: object) -> tuple[int, ...]:
    body_ids: set[int] = set()
    for body_name in GROUND_ALIGNMENT_FOOT_ROOT_BODIES:
        try:
            body_ids.update(_mujoco_body_descendants(model, int(model.body(body_name).id)))  # type: ignore[attr-defined]
        except Exception:
            continue
    if not body_ids:
        for body_id in range(int(model.nbody)):  # type: ignore[attr-defined]
            body_name = _mujoco_object_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore[attr-defined]
            if any(token in body_name.lower() for token in ("foot", "toe", "ankle")):
                body_ids.add(body_id)
    geom_ids = [
        geom_id
        for geom_id in range(int(model.ngeom))  # type: ignore[attr-defined]
        if int(model.geom_bodyid[geom_id]) in body_ids  # type: ignore[attr-defined]
    ]
    return tuple(geom_ids)


def _mujoco_body_descendants(model: object, root_body_id: int) -> set[int]:
    descendants: set[int] = set()
    stack = [root_body_id]
    while stack:
        body_id = stack.pop()
        descendants.add(body_id)
        for child_id in range(int(model.nbody)):  # type: ignore[attr-defined]
            if int(model.body_parentid[child_id]) == body_id:  # type: ignore[attr-defined]
                stack.append(child_id)
    return descendants


def _lowest_mujoco_geom_z(
    mujoco: object,
    model: object,
    data: object,
    geom_ids: Sequence[int],
) -> float:
    if not geom_ids:
        return math.inf
    return min(_mujoco_geom_min_z(mujoco, model, data, geom_id) for geom_id in geom_ids)


def _mujoco_geom_min_z(mujoco: object, model: object, data: object, geom_id: int) -> float:
    import numpy as np  # type: ignore[import-not-found]

    geom_type = int(model.geom_type[geom_id])  # type: ignore[attr-defined]
    position = np.asarray(data.geom_xpos[geom_id], dtype=float)  # type: ignore[attr-defined]
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)  # type: ignore[attr-defined]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):  # type: ignore[attr-defined]
        return float(position[2] - float(model.geom_size[geom_id][0]))  # type: ignore[attr-defined]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):  # type: ignore[attr-defined]
        radius = float(model.geom_size[geom_id][0])  # type: ignore[attr-defined]
        half_length = float(model.geom_size[geom_id][1])  # type: ignore[attr-defined]
        endpoints = (
            position + rotation @ np.array([0.0, 0.0, half_length]),
            position + rotation @ np.array([0.0, 0.0, -half_length]),
        )
        return float(min(point[2] for point in endpoints) - radius)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):  # type: ignore[attr-defined]
        radius = float(model.geom_size[geom_id][0])  # type: ignore[attr-defined]
        half_length = float(model.geom_size[geom_id][1])  # type: ignore[attr-defined]
        local_points = [
            np.array([x, y, z])
            for z in (-half_length, half_length)
            for x, y in ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius))
        ]
        return float(min((position + rotation @ point)[2] for point in local_points))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):  # type: ignore[attr-defined]
        size_x, size_y, size_z = (float(value) for value in model.geom_size[geom_id])  # type: ignore[attr-defined]
        local_points = [
            np.array([x, y, z])
            for x in (-size_x, size_x)
            for y in (-size_y, size_y)
            for z in (-size_z, size_z)
        ]
        return float(min((position + rotation @ point)[2] for point in local_points))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):  # type: ignore[attr-defined]
        mesh_id = int(model.geom_dataid[geom_id])  # type: ignore[attr-defined]
        vertex_start = int(model.mesh_vertadr[mesh_id])  # type: ignore[attr-defined]
        vertex_count = int(model.mesh_vertnum[mesh_id])  # type: ignore[attr-defined]
        vertices = np.asarray(model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=float)  # type: ignore[attr-defined]
        if len(vertices) > 0:
            return float(np.min((position + vertices @ rotation.T)[:, 2]))
    return float(position[2] - float(model.geom_rbound[geom_id]))  # type: ignore[attr-defined]


def _mujoco_object_name(mujoco: object, model: object, obj_type: object, obj_id: int) -> str:
    try:
        return str(mujoco.mj_id2name(model, obj_type, obj_id) or "")  # type: ignore[attr-defined]
    except Exception:
        return ""


def _visual_root(value: object) -> list[float]:
    root = _float_sequence(value, 3)
    if ROOT_XY_LOCKED_FOR_VIS:
        root[0] = 0.0
        root[1] = 0.0
    return root


def _center_camera_on_robot(model: object, data: object, camera: object) -> None:
    try:
        body_id = model.body("pelvis").id  # type: ignore[attr-defined]
        pos = data.xpos[body_id]  # type: ignore[attr-defined]
        camera.lookat = [0.0, 0.0, float(pos[2])]  # type: ignore[attr-defined]
    except Exception:
        camera.lookat = [0.0, 0.0, DEFAULT_ROOT_HEIGHT]  # type: ignore[attr-defined]


def _source_preview_frames(motion: object, max_frames: int | None) -> list[dict[str, list[float]]]:
    maps = global_body_position_maps_from_bvh(
        motion,  # type: ignore[arg-type]
        body_names=DEFAULT_SOURCE_BODY_NAMES,
        position_scale=0.01,
    )
    frames: list[dict[str, list[float]]] = []
    for frame in _limited_sequence(maps, max_frames):
        frames.append(
            {
                body: [round(float(value), 5) for value in coords]
                for body, coords in frame.items()
            }
        )
    return frames


def _source_capsule_edges(body_names: Sequence[str]) -> list[list[str]]:
    available = set(body_names)
    return [[start, end] for start, end in SOURCE_CAPSULE_EDGES if start in available and end in available]


def _write_g1_csv(path: Path, trajectory: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=G1_CSV_COLUMNS)
        writer.writeheader()
        for item in trajectory:
            joints = item.get("joints", {})
            root = _float_sequence(item.get("root", (0.0, 0.0, DEFAULT_ROOT_HEIGHT)), 3)
            root_euler = _float_sequence(item.get("root_euler", (0.0, 0.0, 0.0)), 3)
            row = {
                "Frame": str(item.get("frame", 0)),
                "root_translateX": f"{root[0]:.8f}",
                "root_translateY": f"{root[1]:.8f}",
                "root_translateZ": f"{root[2]:.8f}",
                "root_rotateX": f"{root_euler[0]:.8f}",
                "root_rotateY": f"{root_euler[1]:.8f}",
                "root_rotateZ": f"{root_euler[2]:.8f}",
            }
            if isinstance(joints, Mapping):
                for column in G1_JOINT_COLUMNS:
                    row[column] = f"{float(joints.get(column, 0.0)):.8f}"
            writer.writerow(row)


def _write_result(output_dir: Path, result: WebPipelineResult) -> None:
    (output_dir / "pipeline_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _timeline(trajectory: Sequence[Mapping[str, object]], frame_time: float) -> list[dict[str, float]]:
    return [
        {"frame": float(item.get("frame", index)), "time": round(index * frame_time, 5)}
        for index, item in enumerate(trajectory)
    ]


def _dict_value(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _frame_list(value: object) -> list[dict[str, list[float]]]:
    if not isinstance(value, Sequence):
        return []
    frames: list[dict[str, list[float]]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        frame: dict[str, list[float]] = {}
        for name, point in item.items():
            frame[str(name)] = _float_sequence(point, 3)
        frames.append(frame)
    return frames


def _frame_point(frame: Mapping[str, Sequence[float]], body_name: str) -> list[float]:
    value = frame.get(body_name)
    return _float_sequence(value, 3)


def _first_present(keys: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def _pose_value(values: Sequence[float], index: int) -> float:
    return values[index] if index < len(values) and math.isfinite(values[index]) else 0.0


def _round_point(values: Sequence[float]) -> list[float]:
    return [round(float(value), 5) for value in values[:3]]


def _normalize_quat_wxyz(value: object) -> list[float]:
    quat = _float_sequence(value, 4)
    norm = math.sqrt(sum(item * item for item in quat))
    if norm <= 0.0 or not math.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [item / norm for item in quat]


def _bvh_frame_time(path: Path) -> float:
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip().lower().startswith("frame time:"):
                    return float(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return 1.0 / 30.0
    return 1.0 / 30.0


def _detect_format(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".bvh" or content.lstrip().startswith(b"HIERARCHY"):
        return "bvh"
    if suffix in {".npz", ".pkl", ".smpl", ".smplx"}:
        return "smpl"
    return suffix.lstrip(".") or "unknown"


def _safe_filename(filename: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in filename)
    return safe or "motion.bvh"


def _run_id(filename: str) -> str:
    stem = Path(_safe_filename(filename)).stem[:32] or "motion"
    return f"{int(time.time())}-{stem}-{uuid.uuid4().hex[:8]}"


def _float_sequence(value: object, length: int) -> list[float]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        parsed = [float(item) for item in value]
    else:
        parsed = []
    while len(parsed) < length:
        parsed.append(0.0)
    return parsed[:length]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
