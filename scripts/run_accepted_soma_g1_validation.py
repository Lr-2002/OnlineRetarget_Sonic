#!/usr/bin/env python3
"""Run accepted SOMA source + IsaacLab G1 visual validation on delta.

The output contract matches the LR-106/LR-117 accepted visualization: left
panel is SomaMesh/global-SOMA source, right panel is IsaacLab G1 kinematic
playback with root-zeroed target XY, a follow camera, large ground plane,
world/root axes, and semantic left/right markers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_PYTHON = Path("/home/user/venvs/isaaclab-210/bin/python")
DEFAULT_SOMA_PYTHON = Path("/home/user/project/ContextRetarget/third_party/soma-retargeter/.venv/bin/python")
DEFAULT_SOMA_RETARGETER = Path("/home/user/project/ContextRetarget/third_party/soma-retargeter")
DEFAULT_BVH_ROOT = Path("/home/user/data/motion_data/clean_data/soma_proportional/bvh")
DEFAULT_STAGE_ROOT = ROOT / "outputs/lr106_stage"
DEFAULT_ROBOT_MOTION_SUBDIR = "robot_soma_paired_v1"
DEFAULT_SOMA_MOTION_SUBDIR = "soma_filtered_v1"
DEFAULT_ROBOT_USD = ROOT / "runs/isaaclab_urdf_cache/g1_main/main.usd"
DEFAULT_G1_MJCF = Path(
    "/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/data/assets/"
    "robot_description/mjcf/g1_29dof_rev_1_0.xml"
)
DEFAULT_AGENTHUB_URL = "http://10.1.11.30:5175"
DEFAULT_UPLOAD_SCRIPT = Path("/home/user/agent-hub/scripts/remote-upload.sh")
DEFAULT_GROUND_COLOR = (0.08, 0.20, 0.72)
DEFAULT_GROUND_SIZE = 80.0


@dataclass(frozen=True)
class ValidationSample:
    date: str
    stem: str
    frames: int

    @property
    def key(self) -> str:
        return f"{self.date}__{self.stem}"

    @property
    def safe_stem(self) -> str:
        return safe_name(self.stem)


DEFAULT_LR106_SAMPLES: tuple[ValidationSample, ...] = (
    ValidationSample("220720", "itching_neck_003__A032_M", 200),
    ValidationSample("220720", "step_rotate_idle_045_001__A030", 200),
    ValidationSample("221115", "walk_arc_cw_stop_006__A060_M", 200),
    ValidationSample("221115", "idle_hands_on_back_start_001__A058_M", 200),
    ValidationSample("220720", "body_stretch_2_002__A029", 200),
    ValidationSample("220713", "turn_jump_0000_002__A023_M", 107),
    ValidationSample("220713", "jog_forward_loop_003__A021_M", 200),
    ValidationSample("221115", "walk_ff_loop_180_R_001__A060_M", 151),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="JSON/CSV sample manifest.")
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Sample as DATE:STEM:FRAMES. May be repeated. Overrides default LR-106 samples.",
    )
    parser.add_argument("--sample-limit", type=int, default=0, help="Limit selected samples after manifest/default expansion.")
    parser.add_argument("--run-name", default="", help="Output run name under --output-root.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--bvh-root", type=Path, default=DEFAULT_BVH_ROOT)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--robot-motion-subdir", default=DEFAULT_ROBOT_MOTION_SUBDIR)
    parser.add_argument("--soma-motion-subdir", default=DEFAULT_SOMA_MOTION_SUBDIR)
    parser.add_argument("--robot-usd", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument("--g1-mjcf", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--soma-python", type=Path, default=DEFAULT_SOMA_PYTHON)
    parser.add_argument("--soma-retargeter-root", type=Path, default=DEFAULT_SOMA_RETARGETER)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--stride-triangles", type=int, default=3)
    parser.add_argument("--ground-size", type=float, default=DEFAULT_GROUND_SIZE)
    parser.add_argument("--ground-color", type=float, nargs=3, default=DEFAULT_GROUND_COLOR)
    parser.add_argument("--camera-mode", choices=("follow", "trajectory", "fixed"), default="follow")
    parser.add_argument("--camera-offset", type=float, nargs=3, default=(3.4, -4.4, 2.2))
    parser.add_argument("--camera-follow-smoothing", type=int, default=4)
    parser.add_argument("--camera-framing-margin", type=float, default=1.35)
    parser.add_argument("--no-agenthub", action="store_true", help="Skip Agent Hub upload.")
    parser.add_argument("--agenthub-url", default=os.environ.get("AGENT_HUB_URL", DEFAULT_AGENTHUB_URL))
    parser.add_argument("--agenthub-project", default="online-retarget")
    parser.add_argument("--agenthub-title", default="Accepted SOMA/G1 validation smoke")
    parser.add_argument(
        "--agenthub-summary",
        default=(
            "SomaMesh/global-SOMA source plus IsaacLab G1 kinematic playback with world/root axes "
            "and semantic L/R overlays."
        ),
    )
    parser.add_argument("--upload-script", type=Path, default=DEFAULT_UPLOAD_SCRIPT)
    return parser.parse_args()


def parse_sample(value: str) -> ValidationSample:
    sep = ":" if ":" in value else ","
    parts = [part.strip() for part in value.split(sep)]
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("sample must be DATE:STEM:FRAMES")
    try:
        frames = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample FRAMES must be an integer") from exc
    if frames <= 0:
        raise argparse.ArgumentTypeError("sample FRAMES must be positive")
    return ValidationSample(parts[0], parts[1], frames)


def load_samples_from_manifest(path: Path) -> list[ValidationSample]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("samples", raw) if isinstance(raw, dict) else raw
        return [
            ValidationSample(str(row["date"]), str(row["stem"]), int(row["frames"]))
            for row in rows
        ]

    samples: list[ValidationSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(ValidationSample(str(row["date"]), str(row["stem"]), int(row["frames"])))
    return samples


def select_samples(args: argparse.Namespace) -> list[ValidationSample]:
    if args.sample:
        samples = [parse_sample(value) for value in args.sample]
    elif args.manifest is not None:
        samples = load_samples_from_manifest(args.manifest)
    else:
        samples = list(DEFAULT_LR106_SAMPLES)

    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    if not samples:
        raise SystemExit("no validation samples selected")
    return samples


def main() -> int:
    args = parse_args()
    samples = select_samples(args)
    run_name = args.run_name or time.strftime("accepted_soma_g1_validation_%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "batch_render.log"
    rows: list[dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"run_name={run_name}\noutput_dir={output_dir}\n")
        log.write("contract=accepted_somamesh_source_isaaclab_g1_overlay\n")
        for index, sample in enumerate(samples):
            report = render_one(index, sample, args, output_dir, log)
            rows.append(report)
            write_json(output_dir / "summary.json", summary(output_dir, samples, rows, None))
            print(json.dumps({"event": "clip_done", **report}, sort_keys=True), flush=True)

    report_md = output_dir / "accepted_soma_g1_validation_report.md"
    write_json(output_dir / "summary.json", summary(output_dir, samples, rows, None))
    report_md.write_text(markdown_report(output_dir, samples, rows, None), encoding="utf-8")

    upload_report = None
    if not args.no_agenthub:
        files = collect_upload_files(output_dir, rows, report_md, log_path)
        upload_report = upload_agenthub(args, files)
        write_json(output_dir / "agenthub_upload.json", upload_report)
        write_json(output_dir / "summary.json", summary(output_dir, samples, rows, upload_report))
        report_md.write_text(markdown_report(output_dir, samples, rows, upload_report), encoding="utf-8")

    final_summary = summary(output_dir, samples, rows, upload_report)
    print(json.dumps({"event": "done", **final_summary}, indent=2, sort_keys=True))
    return 0 if final_summary["status"] == "ok" else 2


def render_one(
    index: int,
    sample: ValidationSample,
    args: argparse.Namespace,
    output_dir: Path,
    log: Any,
) -> dict[str, Any]:
    clip_dir = output_dir / f"{index:02d}_{sample.safe_stem}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    paths = sample_paths(sample, args)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {
            "index": index,
            "sample_name": sample.key,
            "status": "blocked",
            "failure_category": "data_or_asset_missing",
            "missing": missing,
        }

    source_video = clip_dir / "source_somabvh_somamesh.mp4"
    source_report_path = clip_dir / "source_somabvh_somamesh.json"
    source_cmd = [
        str(args.soma_python),
        str(ROOT / "scripts/render_somamesh_source.py"),
        "--bvh",
        str(paths["source_bvh"]),
        "--output",
        str(source_video),
        "--report",
        str(source_report_path),
        "--retargeter-root",
        str(args.soma_retargeter_root),
        "--fps",
        str(args.fps),
        "--frame-count",
        str(sample.frames),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--stride-triangles",
        str(args.stride_triangles),
        "--title",
        sample.stem,
    ]
    source_env = os.environ.copy()
    source_env["PYTHONPATH"] = f"{args.soma_retargeter_root}:{ROOT}:{ROOT / 'src'}"
    if run_logged(source_cmd, log, cwd=ROOT, env=source_env) != 0:
        return failed(index, sample, "source_somamesh")

    target_video = clip_dir / "target_g1_isaaclab.mp4"
    target_cmd = [
        str(args.isaac_python),
        str(ROOT / "scripts/render_g1_isaac_pair.py"),
        "--g1-motion",
        str(paths["target_motion"]),
        "--format",
        "motionlib",
        "--output",
        str(target_video),
        "--robot-usd",
        str(args.robot_usd),
        "--duration-sec",
        str(args.duration_sec),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--camera-mode",
        str(args.camera_mode),
        "--camera-offset",
        *[str(value) for value in args.camera_offset],
        "--ground-size",
        str(args.ground_size),
        "--ground-color",
        *[str(value) for value in args.ground_color],
        "--camera-follow-smoothing",
        str(args.camera_follow_smoothing),
        "--camera-framing-margin",
        str(args.camera_framing_margin),
        "--draw-orientation-labels",
        "--root-rot-format",
        "xyzw",
        "--fast-exit-after-report",
    ]
    if run_logged(target_cmd, log, cwd=ROOT) != 0:
        return failed(index, sample, "target_isaaclab")

    overlay_npz = clip_dir / "target_g1_overlay_body_pos_w.npz"
    target_json = clip_dir / "target_g1_isaaclab.json"
    if run_overlay_fk(args, paths["target_motion"], overlay_npz, log) != 0:
        return failed(index, sample, "overlay_fk")
    target_data = json.loads(target_json.read_text(encoding="utf-8"))
    target_data["npz"] = str(overlay_npz)
    target_data["overlay_npz_contract"] = (
        "G1 FK body_pos_w derived from the same motionlib root/joint tensors used for IsaacLab playback"
    )
    write_json(target_json, target_data)

    overlay_root = clip_dir / "overlay"
    overlay_cmd = [
        str(args.isaac_python),
        str(ROOT / "scripts/overlay_isaac_orientation_debug.py"),
        "--input-root",
        str(clip_dir.parent),
        "--output-root",
        str(overlay_root),
        "--run-name",
        "orientation",
        "--clip",
        clip_dir.name,
    ]
    if run_logged(overlay_cmd, log, cwd=ROOT) != 0:
        return failed(index, sample, "orientation_overlay")

    overlay_video = overlay_root / "orientation" / clip_dir.name / "target_g1_isaaclab_orientation_debug.mp4"
    final_video = clip_dir / f"{sample.safe_stem}_somamesh_source_g1_isaac_with_axes.mp4"
    combine_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-i",
        str(overlay_video),
        "-filter_complex",
        f"[0:v][1:v]hstack=inputs=2,fps={int(round(args.fps))}[v]",
        "-map",
        "[v]",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_video),
    ]
    if run_logged(combine_cmd, log, cwd=ROOT) != 0:
        return failed(index, sample, "combine")

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    target_report = json.loads(target_json.read_text(encoding="utf-8"))
    overlay_report_path = overlay_root / "orientation" / clip_dir.name / "target_g1_isaaclab_orientation_debug.json"
    overlay_report = json.loads(overlay_report_path.read_text(encoding="utf-8"))
    final_report = final_clip_report(
        index=index,
        sample=sample,
        paths=paths,
        clip_dir=clip_dir,
        source_video=source_video,
        target_video=target_video,
        overlay_video=overlay_video,
        final_video=final_video,
        source_report=source_report,
        target_report=target_report,
        overlay_report=overlay_report,
    )
    write_json(clip_dir / "final_report.json", final_report)
    return final_report


def sample_paths(sample: ValidationSample, args: argparse.Namespace) -> dict[str, Path]:
    return {
        "source_bvh": args.bvh_root / sample.date / f"{sample.stem}.bvh",
        "source_motion": args.stage_root / args.soma_motion_subdir / f"{sample.key}.pkl",
        "target_motion": args.stage_root / args.robot_motion_subdir / f"{sample.key}.pkl",
        "robot_usd": args.robot_usd,
        "g1_mjcf": args.g1_mjcf,
    }


def run_overlay_fk(args: argparse.Namespace, target_motion: Path, overlay_npz: Path, log: Any) -> int:
    code = "\n".join(
        [
            "from pathlib import Path",
            "import numpy as np",
            "from render_bvh_g1_mujoco_pair import load_g1_motion, zero_initial_root_xy",
            "from train_sonic_kin_skeleton_ae import _quat_wxyz_to_euler_xyz",
            "from online_retarget.data.bones_sonic import SONIC_BODY_NAMES, sonic_joint_values_to_g1_columns",
            "from online_retarget.data.g1_quality import g1_fk_body_positions, load_g1_kinematic_model",
            (
                f"motion = load_g1_motion(Path({str(target_motion)!r}), fmt='motionlib', "
                f"max_frames=0, duration_sec={float(args.duration_sec)!r}, root_position_scale=0.01, "
                "angle_scale=np.pi / 180.0, root_rot_format='xyzw')"
            ),
            "motion = zero_initial_root_xy(motion)",
            f"model = load_g1_kinematic_model(Path({str(args.g1_mjcf)!r}))",
            "frames = []",
            "for joints, root, quat in zip(motion['joint_pos'], motion['root_pos'], motion['root_quat']):",
            "    points = g1_fk_body_positions(",
            "        model,",
            "        sonic_joint_values_to_g1_columns(joints),",
            "        root_position=root,",
            "        root_euler=_quat_wxyz_to_euler_xyz(quat),",
            "        include_empty_body_origin=True,",
            "    )",
            "    frames.append([np.asarray(points[name], dtype=np.float32).mean(axis=0) for name in SONIC_BODY_NAMES])",
            (
                f"np.savez(Path({str(overlay_npz)!r}), fps=np.asarray(motion['fps'], dtype=np.float32), "
                "body_pos_w=np.asarray(frames, dtype=np.float32))"
            ),
        ]
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts'}"
    return run_logged([str(args.isaac_python), "-c", code], log, cwd=ROOT, env=env)


def final_clip_report(
    *,
    index: int,
    sample: ValidationSample,
    paths: dict[str, Path],
    clip_dir: Path,
    source_video: Path,
    target_video: Path,
    overlay_video: Path,
    final_video: Path,
    source_report: dict[str, Any],
    target_report: dict[str, Any],
    overlay_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "index": index,
        "sample_name": sample.key,
        "status": "ok",
        "fps": target_report.get("fps"),
        "frames": target_report.get("frames"),
        "expected_frames": sample.frames,
        "source_bvh": str(paths["source_bvh"]),
        "source_motionlib": str(paths["source_motion"]),
        "target_motionlib": str(paths["target_motion"]),
        "final_video": str(final_video),
        "final_report": str(clip_dir / "final_report.json"),
        "source_video": str(source_video),
        "target_video": str(target_video),
        "overlay_video": str(overlay_video),
        "source_display_conversion": "(x, y, z)_display = (x, -z, y)_soma",
        "source_renderer": "SomaMesh LBS",
        "source_root_camera_reference": "Hips/pelvis; horizontal follow, fixed look-at height",
        "target_backend": target_report.get("backend"),
        "g1_asset_path": target_report.get("robot_asset"),
        "target_camera_mode": target_report.get("camera_mode"),
        "target_camera_policy": target_report.get("camera_policy"),
        "target_ground_size": target_report.get("ground_size"),
        "target_ground_color": target_report.get("ground_color"),
        "target_initial_root_xy_zeroed": target_report.get("initial_root_xy_zeroed"),
        "root_quaternion_convention": (
            "motionlib root_rot xyzw, converted to IsaacLab wxyz before write_root_state_to_sim"
        ),
        "joint_order": target_report.get("motion_joint_names"),
        "semantic_lr_marker": (
            "target overlay uses semantic body names for L/R; source panel derives labels from BVH/SOMA semantics"
        ),
        "world_root_axes": "world axes and root local axes visible in target overlay; source panel draws world/root axes",
        "changed_frames_source": source_report.get("changed_frames"),
        "changed_frames_target": target_report.get("changed_frames"),
        "changed_frames_overlay": (overlay_report.get("overlay") or {}).get("changed_frames"),
        "ffprobe": ffprobe(final_video),
    }


def failed(index: int, sample: ValidationSample, stage: str) -> dict[str, Any]:
    return {
        "index": index,
        "sample_name": sample.key,
        "status": "failed",
        "stage": stage,
        "failure_category": stage,
    }


def run_logged(
    command: Sequence[str],
    log: Any,
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> int:
    log.write("\n$ " + " ".join(command) + "\n")
    log.flush()
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return int(result.returncode)


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "r_frame_rate": stream.get("r_frame_rate", ""),
        "duration": stream.get("duration", ""),
        "size_bytes": path.stat().st_size,
    }


def collect_upload_files(output_dir: Path, rows: Iterable[dict[str, Any]], report_md: Path, log_path: Path) -> list[Path]:
    files = [report_md, output_dir / "summary.json", log_path]
    for row in rows:
        if row.get("status") == "ok":
            files.append(Path(str(row["final_video"])))
            files.append(Path(str(row["final_report"])))
    return [path for path in files if path.exists()]


def upload_agenthub(args: argparse.Namespace, files: list[Path]) -> dict[str, Any]:
    if not args.upload_script.exists():
        return {"status": "blocked", "message": f"missing upload script: {args.upload_script}"}
    command = [
        str(args.upload_script),
        "-p",
        args.agenthub_project,
        "-t",
        args.agenthub_title,
        "-s",
        args.agenthub_summary,
        *[str(path) for path in files],
    ]
    env = os.environ.copy()
    env.setdefault("AGENT_HUB_URL", args.agenthub_url)
    env.setdefault("AGENT_HUB_DEVICE_TYPE", "server")
    env.setdefault("AGENT_HUB_DEVICE_NAME", os.uname().nodename)
    env.setdefault("AGENT_HUB_AGENT_IDENTITY", f"{os.environ.get('USER', 'codex')}@{os.uname().nodename}")
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    parsed = parse_agenthub_upload_output(result.stdout)
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "dashboard_url": parsed.get("dashboard_url", ""),
        "run_id": parsed.get("run_id", ""),
        "output": result.stdout[-6000:],
    }


def parse_agenthub_upload_output(output: str) -> dict[str, str]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return {}
    data = payload.get("data") or {}
    return {
        "dashboard_url": str(data.get("dashboard_url", "")),
        "run_id": str(data.get("run_id", "")),
    }


def summary(
    output_dir: Path,
    samples: Sequence[ValidationSample],
    rows: Sequence[dict[str, Any]],
    upload: dict[str, Any] | None,
) -> dict[str, Any]:
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    upload_ok = upload is None or upload.get("status") == "ok"
    return {
        "status": "ok" if len(rows) == len(samples) and ok_count == len(samples) and upload_ok else "partial",
        "output_dir": str(output_dir),
        "requested_count": len(samples),
        "rendered_count": len(rows),
        "ok_count": ok_count,
        "execution_host": os.uname().nodename,
        "execution_device": "delta / IsaacLab venv / soma-retargeter",
        "source_contract": (
            "SomaBVH/SomaMesh LBS, SOMA Y-up to display Z-up, "
            "Hips/pelvis camera reference, fixed look-at height"
        ),
        "target_contract": "IsaacLab G1 kinematic playback from motionlib PKL",
        "standard_output_contract": {
            "source": "SomaMesh/global-SOMA source panel",
            "target": "IsaacLab G1 kinematic playback panel",
            "axes": "world axes and root local axes visible",
            "semantic_lr": "L/R labels derive from semantic body names",
            "source_display_conversion": "(x, y, z)_display = (x, -z, y)_soma",
            "target_camera": "follow mode with smoothed root XY and stable look-at height",
            "target_root_xy_policy": "root-zeroed relative XY by renderer default",
            "target_ground": f"{DEFAULT_GROUND_SIZE:.1f}m ground plane default",
            "root_quaternion": "motionlib root_rot xyzw -> IsaacLab wxyz",
            "g1_asset": str(DEFAULT_ROBOT_USD),
        },
        "retarget_mapping_changed": False,
        "agenthub_upload": upload,
        "rows": list(rows),
    }


def markdown_report(
    output_dir: Path,
    samples: Sequence[ValidationSample],
    rows: Sequence[dict[str, Any]],
    upload: dict[str, Any] | None,
) -> str:
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    lines = [
        "# Accepted SOMA/G1 Validation",
        "",
        f"- Status: {ok_count}/{len(samples)} rendered OK",
        "- Source: SomaBVH/SomaMesh LBS; SOMA native Y-up to display Z-up as `(x, y, z)_display = (x, -z, y)_soma`.",
        "- Camera: Hips/pelvis root reference; horizontal follow; fixed look-at height.",
        "- Target: IsaacLab G1 kinematic playback; root XY is zeroed by renderer default; "
        f"camera follows smoothed root XY; ground plane defaults to {DEFAULT_GROUND_SIZE:.1f}m; "
        "motionlib `root_rot` is `xyzw`, written to IsaacLab as `wxyz`.",
        "- Overlays: world axes, root axes, front/L/R markers; L/R labels come from semantic body names.",
        "- Scope: kinematic visualization validation only; not policy tracking, dynamics, balance, or sim2real proof.",
        f"- Output dir: `{output_dir}`",
    ]
    if upload:
        lines.append(f"- Agent Hub upload status: {upload.get('status')}")
        if upload.get("dashboard_url"):
            lines.append(f"- Agent Hub URL: {upload.get('dashboard_url')}")
    lines.extend(
        [
            "",
            "| # | sample | status | frames | fps | changed source/target/overlay | video |",
            "| ---: | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('index', '')} | `{row.get('sample_name', '')}` | {row.get('status', '')} | "
            f"{row.get('frames', '')} | {row.get('fps', '')} | "
            f"{row.get('changed_frames_source', '')}/{row.get('changed_frames_target', '')}/"
            f"{row.get('changed_frames_overlay', '')} | `{row.get('final_video', '')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
