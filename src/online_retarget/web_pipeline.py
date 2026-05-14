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
from pathlib import Path
import time
import uuid
from typing import Iterable, Mapping, Sequence
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


DEFAULT_G1_MJCF = Path("/home/user/repos/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
DEFAULT_MAX_FRAMES = 240
DEFAULT_ROOT_HEIGHT = 0.793
WEB_RUN_ROOT = Path("outputs/web_runs")

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
    max_frames: int = DEFAULT_MAX_FRAMES,
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

    input_format = _detect_format(filename, source_bytes)
    loaded = _load_preview_source(source_bytes, input_format, max_frames=max_frames)
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
    }
    stages["load"] = StageResult(
        status="ok",
        message=str(loaded["message"]),
        details=_dict_value(loaded.get("details")),
    )

    retarget = _retarget_preview_frames_to_g1(
        source_frames,
        body_names=tuple(str(item) for item in loaded["body_names"]),
        max_frames=max_frames,
        mode=str(loaded["retarget_source"]),
    )
    g1_csv = output_dir / "retargeted_g1_preview.csv"
    _write_g1_csv(g1_csv, retarget["trajectory"])
    artifacts["retargeted_g1_csv"] = str(g1_csv)
    artifacts["retarget_report"] = str(output_dir / "retarget_report.json")
    (output_dir / "retarget_report.json").write_text(
        json.dumps(retarget["report"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stages["retarget"] = StageResult(
        status="ok",
        message=(
            "Generated a deterministic G1 joint reference preview. "
            "This is a rule-based placeholder until a trained retarget model is available."
        ),
        details=retarget["report"],
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
    )
    stages["physics_sim"] = StageResult(
        status=str(physics["status"]),
        message=str(physics["message"]),
        details={key: value for key, value in physics.items() if key not in {"status", "message"}},
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


def _load_preview_source(source_bytes: bytes, input_format: str, max_frames: int) -> dict[str, object]:
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


def _load_smpl_preview_source(source_bytes: bytes, max_frames: int) -> dict[str, object]:
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
    max_frames: int,
) -> list[dict[str, list[float]]]:
    frame_count = min(len(poses), max_frames)  # type: ignore[arg-type]
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


def _retarget_preview_frames_to_g1(
    source_frames: Sequence[Mapping[str, Sequence[float]]],
    *,
    body_names: Sequence[str],
    max_frames: int,
    mode: str,
) -> dict[str, object]:
    frame_count = min(len(source_frames), max_frames)
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
            "input_mode": mode,
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
        root = item.get("root", (0.0, 0.0, DEFAULT_ROOT_HEIGHT))
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
        import mujoco  # type: ignore[import-not-found]

        model = mujoco.MjModel.from_xml_path(str(model_xml))
        data = mujoco.MjData(model)
        steps = min(len(trajectory), 120)
        finite = True
        for item in trajectory[:steps]:
            _set_mujoco_state(model, data, item)
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            finite = finite and all(math.isfinite(float(value)) for value in data.qpos)
        return {
            "status": "ok" if finite else "failed",
            "message": "MuJoCo physics rollout completed." if finite else "MuJoCo produced non-finite qpos.",
            "model_xml": str(model_xml),
            "steps": steps,
            "frame_time": frame_time,
            "qpos_dim": int(model.nq),
            "ctrl_dim": int(model.nu),
        }
    except Exception as exc:  # pragma: no cover - depends on optional mujoco runtime.
        return {
            "status": "failed",
            "message": f"MuJoCo physics rollout failed: {exc}",
            "model_xml": str(model_xml),
        }


def _set_mujoco_state(model: object, data: object, item: Mapping[str, object]) -> None:
    root = _float_sequence(item.get("root", (0.0, 0.0, DEFAULT_ROOT_HEIGHT)), 3)
    joints = item.get("joints", {})
    if not isinstance(joints, Mapping):
        joints = {}
    qpos = data.qpos  # type: ignore[attr-defined]
    if len(qpos) >= 7:
        qpos[0:3] = root
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for column in G1_JOINT_COLUMNS:
        joint_name = column[:-4] if column.endswith("_dof") else column
        try:
            joint_id = model.joint(joint_name).id  # type: ignore[attr-defined]
            qpos_address = model.jnt_qposadr[joint_id]  # type: ignore[attr-defined]
        except Exception:
            continue
        if 0 <= qpos_address < len(qpos):
            qpos[qpos_address] = float(joints.get(column, 0.0))


def _source_preview_frames(motion: object, max_frames: int) -> list[dict[str, list[float]]]:
    maps = global_body_position_maps_from_bvh(
        motion,  # type: ignore[arg-type]
        body_names=DEFAULT_SOURCE_BODY_NAMES,
        position_scale=0.01,
    )
    frames: list[dict[str, list[float]]] = []
    for frame in maps[:max_frames]:
        frames.append(
            {
                body: [round(float(value), 5) for value in coords]
                for body, coords in frame.items()
            }
        )
    return frames


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
