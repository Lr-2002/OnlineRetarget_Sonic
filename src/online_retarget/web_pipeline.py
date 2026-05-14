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

from .data.bones_seed import G1_CSV_COLUMNS, G1_JOINT_COLUMNS
from .data.g1_quality import (
    G1KinematicModel,
    g1_fk_body_positions,
    load_g1_kinematic_model,
)
from .data.windowed_builder import (
    DEFAULT_SOURCE_BODY_NAMES,
    body_positions_from_bvh,
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
    if input_format != "bvh":
        message = (
            "Only BVH upload is implemented in this local preview. "
            "SMPL/SMPL-X input needs a body-model decoder before retargeting."
        )
        stages["load"] = StageResult(
            status="blocked",
            message=message,
            details={"input_format": input_format, "bytes": len(source_bytes)},
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

    try:
        text = source_bytes.decode("utf-8")
        motion = parse_bvh_motion(text)
    except (UnicodeDecodeError, ValueError) as exc:
        stages["load"] = StageResult(
            status="failed",
            message=f"BVH load failed: {exc}",
            details={"input_format": input_format, "bytes": len(source_bytes)},
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

    source_frames = _source_preview_frames(motion, max_frames=max_frames)
    preview["source"] = {
        "frames": source_frames,
        "body_names": list(DEFAULT_SOURCE_BODY_NAMES),
        "frame_time": motion.frame_time,
    }
    stages["load"] = StageResult(
        status="ok",
        message="BVH parsed and source FK preview frames were generated.",
        details={
            "frames": len(motion.frames),
            "used_frames": len(source_frames),
            "frame_time": motion.frame_time,
            "joints": len(motion.joints),
            "channels": motion.channel_count,
        },
    )

    retarget = _retarget_bvh_to_g1(motion, max_frames=max_frames)
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
            preview["timeline"] = _timeline(retarget["trajectory"], motion.frame_time)
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
        frame_time=motion.frame_time,
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


def _retarget_bvh_to_g1(motion: object, max_frames: int) -> dict[str, object]:
    source_positions = body_positions_from_bvh(
        motion,  # type: ignore[arg-type]
        body_names=DEFAULT_SOURCE_BODY_NAMES,
        root_body="Hips",
        position_scale=0.01,
    )
    frame_count = min(len(source_positions), max_frames)
    trajectory: list[dict[str, object]] = []
    previous_root_x = 0.0
    for frame_index, positions in enumerate(source_positions[:frame_count]):
        root_x = _body_axis(positions, 0, 0)
        root_z = _body_axis(positions, 0, 2)
        left_foot_y = _named_axis(positions, "LeftFoot", 1)
        right_foot_y = _named_axis(positions, "RightFoot", 1)
        left_hand_y = _named_axis(positions, "LeftHand", 1)
        right_hand_y = _named_axis(positions, "RightHand", 1)
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
            "frames": len(trajectory),
            "output_joint_count": len(G1_JOINT_COLUMNS),
            "limitations": [
                "Not a learned model prediction.",
                "SMPL inputs require a decoder before this stage.",
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


def _body_axis(flat_positions: Sequence[float], body_index: int, axis: int) -> float:
    index = body_index * 3 + axis
    return float(flat_positions[index]) if 0 <= index < len(flat_positions) else 0.0


def _named_axis(flat_positions: Sequence[float], body_name: str, axis: int) -> float:
    try:
        body_index = DEFAULT_SOURCE_BODY_NAMES.index(body_name)
    except ValueError:
        return 0.0
    return _body_axis(flat_positions, body_index, axis)


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
