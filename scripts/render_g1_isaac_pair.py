#!/usr/bin/env python3
"""Render BONES-SONIC G1 NPZ motion in Isaac Lab as kinematic playback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=None, help="Legacy BONES-SONIC NPZ input.")
    parser.add_argument("--g1-motion", type=Path, default=None, help="G1 motion: motionlib .pkl, .npz, or .csv.")
    parser.add_argument("--bvh", type=Path, default=None, help="Optional source BVH to render as the left panel.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config for source render defaults.")
    parser.add_argument("--format", choices=("auto", "motionlib", "npz", "csv"), default="auto")
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/data/assets"),
    )
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=None,
    )
    parser.add_argument("--robot-usd", type=Path, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--source-position-scale", type=float, default=None)
    parser.add_argument("--root-position-scale", type=float, default=0.01, help="CSV root position scale only.")
    parser.add_argument("--angle-scale", type=float, default=float(3.141592653589793 / 180.0), help="CSV angle scale only.")
    parser.add_argument("--preserve-world-root", action="store_true")
    parser.add_argument("--camera-mode", choices=("trajectory", "follow", "fixed"), default="trajectory")
    parser.add_argument("--camera-offset", type=float, nargs=3, default=(2.5, -3.0, 1.6))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.g1_motion is None and args.npz is None:
        parser.error("one of --g1-motion or --npz is required")
    args.headless = True
    args.enable_cameras = True
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402

from render_bvh_g1_mujoco_pair import (  # noqa: E402
    combine_two_videos,
    load_g1_motion,
    render_source_bvh_panel,
    root_xy_span,
    zero_initial_root_xy,
)


SONIC_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


def main() -> None:
    os.environ.setdefault("ISAACLAB_USE_CACHED_CUDA_KERNELS", "1")
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)

    motion_path = args_cli.g1_motion or args_cli.npz
    if motion_path is None:
        raise SystemExit("one of --g1-motion or --npz is required")
    config = _read_json(args_cli.config) if args_cli.config is not None else {}
    visual_cfg = config.get("visual_validation", {}) if isinstance(config.get("visual_validation", {}), dict) else {}
    source_scale = (
        float(args_cli.source_position_scale)
        if args_cli.source_position_scale is not None
        else float(visual_cfg.get("source_position_scale", 0.01))
    )
    motion = load_g1_motion(
        motion_path,
        fmt=args_cli.format,
        max_frames=args_cli.max_frames,
        duration_sec=args_cli.duration_sec,
        root_position_scale=args_cli.root_position_scale,
        angle_scale=args_cli.angle_scale,
    )
    if not args_cli.preserve_world_root:
        motion = zero_initial_root_xy(motion)
    target_video = args_cli.output
    source_report = None
    combine_report = None
    if args_cli.bvh is not None:
        panel_dir = args_cli.output.with_suffix("")
        panel_dir.mkdir(parents=True, exist_ok=True)
        source_video = panel_dir / "source_bvh_capsules.mp4"
        target_video = panel_dir / "target_g1_isaac.mp4"
        source_report = render_source_bvh_panel(
            args_cli.bvh,
            source_video,
            fps=float(motion["fps"]),
            frame_count=int(motion["frame_count"]),
            width=args_cli.width,
            height=args_cli.height,
            source_position_scale=source_scale,
        )
    stage_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / motion["fps"], render_interval=1, device="cuda:0")
    )
    _spawn_scene(args_cli.asset_root)
    robot = _spawn_robot(args_cli.asset_root, args_cli.robot_urdf, args_cli.robot_usd)
    camera = _spawn_camera(args_cli.width, args_cli.height)
    stage_utils.update_stage()
    sim.reset()
    robot.update(sim.cfg.dt)

    joint_order = _joint_order(robot.joint_names, motion["joint_names"])
    camera_plan = _camera_plan(motion, args_cli.camera_mode, np.asarray(args_cli.camera_offset, dtype=np.float32))
    frames_written = 0
    changed_frames = 0
    previous_frame = None
    frame_sums: list[int] = []

    with imageio.get_writer(target_video, fps=motion["fps"], codec="libx264", quality=8) as writer:
        for frame_index in range(motion["frame_count"]):
            root_pos = motion["root_pos"][frame_index].copy()
            root_quat = motion["root_quat"][frame_index]
            joint_pos = motion["joint_pos"][frame_index, joint_order]
            joint_vel = motion["joint_vel"][frame_index, joint_order]

            _write_robot_state(robot, root_pos, root_quat, joint_pos, joint_vel, sim.device)
            robot.set_joint_position_target(
                torch.as_tensor(joint_pos[None, :], dtype=torch.float32, device=sim.device)
            )
            robot.write_data_to_sim()
            target_np = root_pos if args_cli.camera_mode == "follow" else camera_plan["target"]
            eye_np = target_np + camera_plan["offset"]
            target = torch.as_tensor(target_np[None, :], dtype=torch.float32, device=sim.device)
            eye = torch.as_tensor(eye_np[None, :], dtype=torch.float32, device=sim.device)
            camera.set_world_poses_from_view(eye, target + torch.tensor([[0.0, 0.0, 0.35]], device=sim.device))
            sim.step(render=True)
            robot.update(sim.cfg.dt)
            camera.update(dt=0.0, force_recompute=True)
            frame = camera.data.output["rgb"][0].detach().cpu().numpy()
            frame = np.asarray(frame[..., :3], dtype=np.uint8)
            frame = cv2.putText(
                frame.copy(),
                f"IsaacLab G1 kinematic playback frame {frame_index:04d}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 42, 48),
                2,
                cv2.LINE_AA,
            )
            writer.append_data(frame)
            frame_bytes = frame.tobytes()
            frame_sums.append(int(frame.sum()))
            if previous_frame is not None and frame_bytes != previous_frame:
                changed_frames += 1
            previous_frame = frame_bytes
            frames_written += 1

    if args_cli.bvh is not None:
        combine_report = combine_two_videos((source_video, target_video), args_cli.output, fps=int(round(float(motion["fps"]))))

    status = "ok" if combine_report is None or combine_report.get("status") == "ok" else "failed"
    report = {
        "status": status,
        "backend": "isaaclab_kinematic_playback",
        "g1_motion": str(motion_path),
        "bvh": str(args_cli.bvh) if args_cli.bvh is not None else "",
        "output": str(args_cli.output),
        "target_video": str(target_video),
        "fps": motion["fps"],
        "frames": frames_written,
        "changed_frames": changed_frames,
        "width": args_cli.width,
        "height": args_cli.height,
        "camera_mode": args_cli.camera_mode,
        "root_xy_preserved": True,
        "initial_root_xy_zeroed": not args_cli.preserve_world_root,
        "target_root_xy_span_m": root_xy_span(motion["root_pos"]),
        "source_position_scale": source_scale,
        "source_render": source_report,
        "combine": combine_report,
        "robot_asset": str(_robot_asset(args_cli.asset_root, args_cli.robot_urdf, args_cli.robot_usd)),
        "robot_joint_names": robot.joint_names,
        "motion_joint_names": list(motion["joint_names"]),
        "frame_sum_min": min(frame_sums) if frame_sums else None,
        "frame_sum_max": max(frame_sums) if frame_sums else None,
    }
    args_cli.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    rep.vp_manager.destroy_hydra_textures("Replicator")
    sim.stop()
    sim.clear_all_callbacks()
    sim.clear_instance()
    simulation_app.close()


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _camera_plan(motion: dict[str, object], mode: str, camera_offset: np.ndarray) -> dict[str, np.ndarray]:
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    if mode == "fixed":
        target = np.asarray([0.0, 0.0, 0.8], dtype=np.float32)
        offset = camera_offset
    elif mode == "trajectory":
        min_xy = np.min(root_pos[:, :2], axis=0)
        max_xy = np.max(root_pos[:, :2], axis=0)
        center_xy = (min_xy + max_xy) * 0.5
        span = max(float(np.linalg.norm(max_xy - min_xy)), 1.0)
        target = np.asarray(
            [float(center_xy[0]), float(center_xy[1]), max(0.8, float(np.mean(root_pos[:, 2])))],
            dtype=np.float32,
        )
        offset = camera_offset * max(1.0, span / 2.0)
    else:
        target = root_pos[0].copy()
        offset = camera_offset
    return {"target": target, "offset": offset.astype(np.float32, copy=False)}


def _spawn_scene(asset_root: Path) -> None:
    sim_utils.GroundPlaneCfg(size=(12.0, 12.0), color=(0.62, 0.62, 0.60)).func(
        "/World/Ground",
        sim_utils.GroundPlaneCfg(size=(12.0, 12.0), color=(0.62, 0.62, 0.60)),
    )
    sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)).func(
        "/World/DomeLight",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
    )
    os.environ.setdefault("ROS_PACKAGE_PATH", str(asset_root))


def _spawn_robot(asset_root: Path, robot_urdf: Path | None, robot_usd: Path | None) -> Articulation:
    spawn_cfg = _robot_spawn_cfg(asset_root, robot_urdf, robot_usd)
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    return Articulation(cfg)


def _robot_spawn_cfg(asset_root: Path, robot_urdf: Path | None, robot_usd: Path | None):
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    if robot_usd is not None:
        return sim_utils.UsdFileCfg(
            usd_path=str(robot_usd),
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    return sim_utils.UrdfFileCfg(
        asset_path=str(_robot_urdf(asset_root, robot_urdf)),
        fix_base=False,
        replace_cylinders_with_capsules=True,
        merge_fixed_joints=True,
        force_usd_conversion=False,
        usd_dir=str(Path("runs/isaaclab_urdf_cache/g1_main").resolve()),
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    )


def _robot_urdf(asset_root: Path, robot_urdf: Path | None) -> Path:
    if robot_urdf is not None:
        return robot_urdf
    return asset_root / "robot_description/urdf/g1/main.urdf"


def _robot_asset(asset_root: Path, robot_urdf: Path | None, robot_usd: Path | None) -> Path:
    if robot_usd is not None:
        return robot_usd
    return _robot_urdf(asset_root, robot_urdf)


def _spawn_camera(width: int, height: int) -> Camera:
    cfg = CameraCfg(
        height=height,
        width=width,
        prim_path="/World/Camera",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=26.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 100.0),
        ),
    )
    return Camera(cfg)


def _joint_order(robot_joint_names: list[str], motion_joint_names_raw: object) -> np.ndarray:
    motion_joint_names = tuple(str(name) for name in motion_joint_names_raw)
    missing = [name for name in robot_joint_names if name not in motion_joint_names]
    if missing:
        raise RuntimeError(f"Robot has joints not present in motion order: {missing}")
    return np.asarray([motion_joint_names.index(name) for name in robot_joint_names], dtype=np.int64)


def _write_robot_state(
    robot: Articulation,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    device: str,
) -> None:
    root_state = torch.zeros((1, 13), dtype=torch.float32, device=device)
    root_state[0, :3] = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    root_state[0, 3:7] = torch.as_tensor(root_quat_wxyz, dtype=torch.float32, device=device)
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(
        torch.as_tensor(joint_pos[None, :], dtype=torch.float32, device=device),
        torch.as_tensor(joint_vel[None, :], dtype=torch.float32, device=device),
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app.is_running():
            simulation_app.close()
