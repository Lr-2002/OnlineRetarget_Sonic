"""Sonic callback for integrated OnlineRetarget visual validation.

The callback is instantiated by Sonic's Hydra ``callbacks`` config and runs
inside the training loop.  It intentionally renders from tensors collected in
the Sonic environment rather than from copied files, so source SOMA, dataset G1
target, and inferred G1 motion share the same 50 Hz physical timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence


try:  # Keep this module importable in the lightweight local test env.
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - exercised when transformers is absent.
    class TrainerCallback:  # type: ignore[no-redef]
        pass


DEFAULT_TARGET_FPS = 50.0
DEFAULT_EVERY_STEPS = 20_000
DEFAULT_NUM_VIDEOS = 8
DEFAULT_DURATION_SEC = 4.0
DEFAULT_LOG_PREFIX = "online_retarget_visual_validation"


@dataclass(frozen=True)
class ValidationClipSpec:
    """One global validation clip assigned to one local rank/env slot."""

    clip_index: int
    local_env_index: int


def should_run_visual_validation(
    global_step: int,
    every_steps: int = DEFAULT_EVERY_STEPS,
    *,
    last_step: int | None = None,
    now: float | None = None,
    every_seconds: float | None = None,
    last_time: float | None = None,
) -> bool:
    """Return true on step or optional wall-clock validation cadence."""

    if global_step <= 0:
        return False
    if last_step == global_step:
        return False
    if every_steps > 0 and global_step % every_steps == 0:
        return True
    if every_seconds is None or every_seconds <= 0 or now is None or last_time is None:
        return False
    return now - last_time >= every_seconds


def rank_video_indices(num_videos: int, rank: int, world_size: int) -> tuple[int, ...]:
    """Split global validation clip indices deterministically across ranks."""

    if num_videos <= 0:
        return ()
    world_size = max(1, int(world_size))
    rank = max(0, int(rank))
    return tuple(index for index in range(int(num_videos)) if index % world_size == rank)


def validation_frame_count(duration_sec: float, target_fps: float) -> int:
    """Number of 50 Hz frames to render for a validation clip."""

    return max(1, int(round(float(duration_sec) * float(target_fps))))


class SonicVisualValidationCallback(TrainerCallback):
    """Render and upload time-aligned source/target/inference validation clips."""

    def __init__(
        self,
        every_steps: int = DEFAULT_EVERY_STEPS,
        num_videos: int = DEFAULT_NUM_VIDEOS,
        duration_sec: float = DEFAULT_DURATION_SEC,
        target_fps: float = DEFAULT_TARGET_FPS,
        output_dir: str | None = None,
        wandb_upload: bool = True,
        log_prefix: str = DEFAULT_LOG_PREFIX,
        fail_on_render_error: bool = False,
        every_minutes: float | None = None,
        every_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self.every_steps = int(every_steps)
        self.num_videos = int(num_videos)
        self.duration_sec = float(duration_sec)
        self.target_fps = float(target_fps)
        self.output_dir = output_dir
        self.wandb_upload = bool(wandb_upload)
        self.log_prefix = log_prefix
        self.fail_on_render_error = bool(fail_on_render_error)
        if every_seconds is not None:
            self.every_seconds = float(every_seconds)
        elif every_minutes is not None:
            self.every_seconds = float(every_minutes) * 60.0
        else:
            self.every_seconds = None
        self._last_step: int | None = None
        self._last_validation_time: float = time.time()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(getattr(state, "global_step", 0))
        now = time.time()
        if not should_run_visual_validation(
            step,
            self.every_steps,
            last_step=self._last_step,
            now=now,
            every_seconds=self.every_seconds,
            last_time=self._last_validation_time,
        ):
            return control
        self._last_step = step
        self._last_validation_time = now

        env = kwargs.get("env")
        model = kwargs.get("model")
        accelerator = kwargs.get("accelerator")
        rank = _rank(accelerator, args)
        world_size = _world_size(accelerator, args)
        step_dir = self._step_output_dir(args, step)
        rank_dir = step_dir / f"rank_{rank:03d}"
        rank_dir.mkdir(parents=True, exist_ok=True)

        started = time.time()
        report: dict[str, Any]
        try:
            report = self._run_rank_validation(
                env=env,
                model=model,
                args=args,
                step=step,
                rank=rank,
                world_size=world_size,
                rank_dir=rank_dir,
            )
        except Exception as exc:  # noqa: BLE001
            if self.fail_on_render_error:
                raise
            report = {
                "status": "failed",
                "rank": rank,
                "world_size": world_size,
                "step": step,
                "message": str(exc),
                "videos_ok": 0,
                "videos_failed": len(rank_video_indices(self.num_videos, rank, world_size)),
            }
            _write_json(rank_dir / "rank_report.json", report)

        report["elapsed_sec"] = time.time() - started
        _write_json(rank_dir / "rank_report.json", report)

        _barrier(accelerator)
        if _is_main_process(accelerator, state) and self.wandb_upload:
            upload_metrics = self._upload_wandb(step_dir=step_dir, step=step)
            report.update(upload_metrics)
            _write_json(step_dir / "main_upload_report.json", upload_metrics)
        _barrier(accelerator)

        return control

    def _run_rank_validation(
        self,
        *,
        env: Any,
        model: Any,
        args: Any,
        step: int,
        rank: int,
        world_size: int,
        rank_dir: Path,
    ) -> dict[str, Any]:
        if env is None:
            raise RuntimeError("SonicVisualValidationCallback requires env in callback kwargs")
        if model is None:
            raise RuntimeError("SonicVisualValidationCallback requires model in callback kwargs")

        clip_indices = rank_video_indices(self.num_videos, rank, world_size)
        specs = [
            ValidationClipSpec(clip_index=index, local_env_index=slot)
            for slot, index in enumerate(clip_indices)
        ]
        if not specs:
            return {
                "status": "ok",
                "rank": rank,
                "world_size": world_size,
                "step": step,
                "videos_ok": 0,
                "videos_failed": 0,
                "clips": [],
            }

        frame_count = validation_frame_count(self.duration_sec, self.target_fps)
        trajectories = _collect_rollout(
            env=env,
            model=model,
            args=args,
            specs=specs,
            frame_count=frame_count,
            target_fps=self.target_fps,
        )

        clip_reports = []
        ok_count = 0
        failed_count = 0
        for spec in specs:
            trajectory = trajectories.get(spec.clip_index)
            if trajectory is None:
                failed_count += 1
                clip_reports.append(
                    {
                        "clip_index": spec.clip_index,
                        "status": "failed",
                        "message": "no trajectory collected",
                    }
                )
                continue

            safe_key = _safe_name(str(trajectory.get("motion_key", "unknown")))
            video_path = rank_dir / f"clip_{spec.clip_index:02d}_{safe_key}.mp4"
            try:
                _render_triplet_video(
                    trajectory=trajectory,
                    video_path=video_path,
                    target_fps=self.target_fps,
                    duration_sec=self.duration_sec,
                )
                clip_report = _clip_report(
                    trajectory=trajectory,
                    video_path=video_path,
                    step=step,
                    rank=rank,
                    world_size=world_size,
                    target_fps=self.target_fps,
                    duration_sec=self.duration_sec,
                )
                ok_count += 1
            except Exception as exc:  # noqa: BLE001
                if self.fail_on_render_error:
                    raise
                clip_report = {
                    "clip_index": spec.clip_index,
                    "status": "failed",
                    "message": str(exc),
                    "video_path": str(video_path),
                }
                failed_count += 1
            clip_reports.append(clip_report)
            _write_json(rank_dir / f"clip_{spec.clip_index:02d}_report.json", clip_report)

        return {
            "status": "ok" if failed_count == 0 else "partial",
            "rank": rank,
            "world_size": world_size,
            "step": step,
            "target_fps": self.target_fps,
            "duration_sec": self.duration_sec,
            "target_frame_count": frame_count,
            "videos_ok": ok_count,
            "videos_failed": failed_count,
            "clips": clip_reports,
        }

    def _step_output_dir(self, args: Any, step: int) -> Path:
        if self.output_dir:
            root = Path(os.path.expandvars(str(self.output_dir)))
        else:
            output_dir = getattr(args, "output_dir", None) or os.environ.get(
                "ONLINE_RETARGET_OUTPUT_DIR",
                "outputs",
            )
            root = Path(str(output_dir)) / DEFAULT_LOG_PREFIX
        return root / f"step_{step:08d}"

    def _upload_wandb(self, *, step_dir: Path, step: int) -> dict[str, Any]:
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001
            return {
                f"{self.log_prefix}/wandb_upload_status": "failed",
                f"{self.log_prefix}/wandb_upload_message": f"wandb import failed: {exc}",
            }

        if getattr(wandb, "run", None) is None:
            return {
                f"{self.log_prefix}/wandb_upload_status": "skipped",
                f"{self.log_prefix}/wandb_upload_message": "no active wandb run",
            }

        payload: dict[str, Any] = {}
        videos = sorted(step_dir.glob("rank_*/clip_*.mp4"))[: self.num_videos]
        for video_path in videos:
            key = f"{self.log_prefix}/{video_path.parent.name}_{video_path.stem}"
            payload[key] = wandb.Video(
                str(video_path),
                fps=int(round(self.target_fps)),
                format="mp4",
            )

        if not payload:
            return {
                f"{self.log_prefix}/wandb_upload_status": "skipped",
                f"{self.log_prefix}/wandb_upload_message": "no videos found",
            }

        payload[f"{self.log_prefix}/videos_uploaded"] = len(videos)
        payload[f"{self.log_prefix}/git_sha"] = _git_sha()
        wandb.log(payload, step=step)
        return {
            f"{self.log_prefix}/wandb_upload_status": "ok",
            f"{self.log_prefix}/videos_uploaded": len(videos),
        }


def _collect_rollout(
    *,
    env: Any,
    model: Any,
    args: Any,
    specs: Sequence[ValidationClipSpec],
    frame_count: int,
    target_fps: float,
) -> dict[int, dict[str, Any]]:
    import torch

    policy = getattr(model, "policy", model)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(policy, "eval"):
        policy.eval()
    if hasattr(policy, "eval_mode"):
        policy.eval_mode()
    if hasattr(env, "set_is_evaluating"):
        env.set_is_evaluating(True, global_rank=getattr(args, "global_rank", 0))

    obs = _reset_env(env, args)
    if hasattr(policy, "init_rollout"):
        policy.init_rollout()

    num_envs = int(getattr(env, "num_envs", len(specs)))
    device = getattr(env, "device", "cpu")
    dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
    trajectories = {
        spec.clip_index: {
            "clip_index": spec.clip_index,
            "local_env_index": spec.local_env_index,
            "source_soma": [],
            "target_g1": [],
            "inferred_g1": [],
            "source_frame_indices": [],
            "encoder_routes": [],
            "motion_id": None,
            "motion_key": None,
            "source_fps": target_fps,
            "target_fps": target_fps,
            "physical_time_aligned": True,
        }
        for spec in specs
    }

    try:
        with torch.no_grad():
            for _ in range(frame_count):
                _append_current_frame(env, specs, trajectories)
                actions = _act(policy, obs, dones)
                _append_encoder_routes(policy, specs, trajectories)
                actor_state = {"obs": obs, "obs_dict": obs, "actions": actions}
                obs, _rewards, dones, _extras = env.step(actor_state)
    finally:
        if hasattr(env, "set_is_evaluating"):
            env.set_is_evaluating(False)
        if hasattr(env, "set_is_training"):
            env.set_is_training()
        if hasattr(policy, "train_mode"):
            policy.train_mode()
        _reset_policy_rollout_buffer(policy)
        if hasattr(model, "train"):
            model.train()

    return trajectories


def _reset_policy_rollout_buffer(policy: Any) -> None:
    """Reset inference history without deleting Sonic PPO aux-loss state."""

    if hasattr(policy, "init_rollout"):
        policy.init_rollout()
    elif hasattr(policy, "clear_rollout"):
        policy.clear_rollout()


def _reset_env(env: Any, args: Any) -> Any:
    if hasattr(env, "reset_all"):
        return env.reset_all(global_rank=getattr(args, "global_rank", 0))
    if hasattr(env, "reset"):
        return env.reset()
    raise RuntimeError("env has neither reset_all nor reset")


def _act(policy: Any, obs: Any, dones: Any) -> Any:
    if hasattr(policy, "act_inference"):
        return policy.act_inference(
            obs_dict=obs,
            cur_dones=dones,
            skip_episode_attnmask=True,
        )
    if callable(policy):
        return policy(obs)
    raise RuntimeError("policy has neither act_inference nor callable interface")


def _append_current_frame(
    env: Any,
    specs: Sequence[ValidationClipSpec],
    trajectories: dict[int, dict[str, Any]],
) -> None:
    command = _motion_command(env)
    source_soma = _maybe_tensor(_get_attr(command, "soma_joints"))
    target_g1 = _maybe_tensor(_get_attr(command, "body_pos_w"))
    inferred_g1 = _maybe_tensor(_get_attr(command, "robot_body_pos_w"))
    motion_ids = _maybe_tensor(_get_attr(command, "motion_ids"))
    time_steps = _motion_time_steps(command)
    motion_keys = _motion_keys(command, motion_ids)

    for spec in specs:
        env_idx = spec.local_env_index
        trajectory = trajectories[spec.clip_index]
        if source_soma is not None and env_idx < len(source_soma):
            trajectory["source_soma"].append(source_soma[env_idx])
        if target_g1 is not None and env_idx < len(target_g1):
            trajectory["target_g1"].append(target_g1[env_idx])
        if inferred_g1 is not None and env_idx < len(inferred_g1):
            trajectory["inferred_g1"].append(inferred_g1[env_idx])
        if time_steps is not None and env_idx < len(time_steps):
            trajectory["source_frame_indices"].append(int(time_steps[env_idx]))
        if motion_ids is not None and env_idx < len(motion_ids):
            trajectory["motion_id"] = int(motion_ids[env_idx])
        if motion_keys and trajectory.get("motion_id") is not None:
            motion_id = int(trajectory["motion_id"])
            if motion_id < len(motion_keys):
                trajectory["motion_key"] = str(motion_keys[motion_id])


def _append_encoder_routes(
    policy: Any,
    specs: Sequence[ValidationClipSpec],
    trajectories: dict[int, dict[str, Any]],
) -> None:
    routes = _current_soma_routes(policy)
    if routes is None:
        return
    for spec in specs:
        env_idx = spec.local_env_index
        if env_idx < len(routes):
            trajectories[spec.clip_index]["encoder_routes"].append(int(routes[env_idx]))


def _current_soma_routes(policy: Any) -> Any:
    actor_module = _get_attr(policy, "actor_module")
    encoders = _get_attr(actor_module, "encoders")
    soma_encoder = None
    if encoders is not None:
        try:
            soma_encoder = encoders["soma"]
        except Exception:  # noqa: BLE001
            soma_encoder = _get_attr(encoders, "soma")
    routes = _maybe_tensor(_get_attr(soma_encoder, "last_routes")) if soma_encoder is not None else None
    if routes is None:
        return None
    if isinstance(routes, Sequence) and routes and isinstance(routes[0], Sequence):
        return [row[-1] for row in routes if row]
    while getattr(routes, "ndim", 0) > 1:
        routes = routes[..., -1]
    return routes


def _motion_command(env: Any) -> Any:
    command = getattr(env, "motion_command", None)
    if command is not None:
        return command
    inner = getattr(env, "env", None)
    manager = getattr(inner, "command_manager", None)
    if manager is not None:
        return manager.get_term("motion")
    raise RuntimeError("no Sonic motion command found on env")


def _motion_time_steps(command: Any) -> Any:
    start = _maybe_tensor(_get_attr(command, "motion_start_time_steps"))
    steps = _maybe_tensor(_get_attr(command, "time_steps"))
    if start is None or steps is None:
        return None
    return start + steps


def _motion_keys(command: Any, motion_ids: Any) -> Sequence[str]:
    motion_lib = _get_attr(command, "motion_lib")
    if motion_lib is None or motion_ids is None:
        return ()
    keys = getattr(motion_lib, "curr_motion_keys", None)
    if keys is None:
        keys = getattr(motion_lib, "_motion_data_keys", None)
    return keys if keys is not None else ()


def _render_triplet_video(
    *,
    trajectory: Mapping[str, Any],
    video_path: Path,
    target_fps: float,
    duration_sec: float,
) -> None:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    source = _stack_or_none(trajectory.get("source_soma"))
    target = _stack_or_none(trajectory.get("target_g1"))
    inferred = _stack_or_none(trajectory.get("inferred_g1"))
    if source is None or target is None or inferred is None:
        missing = [
            name
            for name, value in (
                ("source_soma", source),
                ("target_g1", target),
                ("inferred_g1", inferred),
            )
            if value is None
        ]
        raise RuntimeError(f"missing validation panel data: {', '.join(missing)}")

    frame_count = min(len(source), len(target), len(inferred))
    expected = validation_frame_count(duration_sec, target_fps)
    frame_count = min(frame_count, expected)
    bounds = _axis_bounds([source[:frame_count], target[:frame_count], inferred[:frame_count]])

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        video_path,
        fps=int(round(target_fps)),
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
    ) as writer:
        for frame_idx in range(frame_count):
            fig = plt.figure(figsize=(12, 4), dpi=100)
            titles = (
                "Source SOMA/BVH",
                "Dataset G1 Target",
                "Inferred G1",
            )
            panels = (source[frame_idx], target[frame_idx], inferred[frame_idx])
            colors = ("#2f6f9f", "#222222", "#b23a48")
            for panel_idx, (title, points, color) in enumerate(
                zip(titles, panels, colors, strict=False),
                start=1,
            ):
                ax = fig.add_subplot(1, 3, panel_idx, projection="3d")
                _plot_points(ax, points, title, color, bounds)
            fig.tight_layout(pad=0.4)
            writer.append_data(_figure_canvas_rgb(fig))
            plt.close(fig)


def _figure_canvas_rgb(fig: Any) -> Any:
    import numpy as np

    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[..., :3].copy()
    if hasattr(fig.canvas, "tostring_rgb"):
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return image.reshape(height, width, 3)
    raise RuntimeError("Matplotlib canvas cannot export RGB image data")


def _plot_points(ax: Any, points: Any, title: str, color: str, bounds: tuple[Any, Any, Any]) -> None:
    import numpy as np

    arr = np.asarray(points, dtype=float).reshape(-1, 3)
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=8, c=color, depthshade=False)
    for start, end in _chain_edges(len(arr)):
        ax.plot(
            arr[[start, end], 0],
            arr[[start, end], 1],
            arr[[start, end], 2],
            color=color,
            linewidth=1.0,
            alpha=0.75,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off()


def _chain_edges(num_points: int) -> tuple[tuple[int, int], ...]:
    if num_points <= 1:
        return ()
    if num_points >= 14:
        g1_like = (
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (0, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (7, 11),
            (11, 12),
            (12, 13),
        )
        return tuple(edge for edge in g1_like if edge[1] < num_points)
    return tuple((idx, idx + 1) for idx in range(num_points - 1))


def _axis_bounds(arrays: Sequence[Any]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    import numpy as np

    valid = [np.asarray(array, dtype=float).reshape(-1, 3) for array in arrays if array is not None]
    points = np.concatenate(valid, axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2.0
    span = max(float((hi - lo).max()), 1.0)
    half = span * 0.55
    return tuple((float(c - half), float(c + half)) for c in center)  # type: ignore[return-value]


def _clip_report(
    *,
    trajectory: Mapping[str, Any],
    video_path: Path,
    step: int,
    rank: int,
    world_size: int,
    target_fps: float,
    duration_sec: float,
) -> dict[str, Any]:
    frame_indices = list(trajectory.get("source_frame_indices") or [])
    encoder_routes = [int(route) for route in (trajectory.get("encoder_routes") or [])]
    return {
        "status": "ok",
        "step": step,
        "rank": rank,
        "world_size": world_size,
        "clip_index": trajectory.get("clip_index"),
        "local_env_index": trajectory.get("local_env_index"),
        "motion_id": trajectory.get("motion_id"),
        "motion_key": trajectory.get("motion_key"),
        "source_fps": trajectory.get("source_fps", target_fps),
        "target_fps": target_fps,
        "source_frame_indices": frame_indices,
        "encoder_routes": encoder_routes,
        "encoder_route_first": encoder_routes[0] if encoder_routes else None,
        "encoder_route_last": encoder_routes[-1] if encoder_routes else None,
        "encoder_route_counts": _route_counts(encoder_routes),
        "target_frame_count": len(trajectory.get("target_g1") or []),
        "duration_sec": duration_sec,
        "physical_time_aligned": bool(trajectory.get("physical_time_aligned", False)),
        "git_sha": _git_sha(),
        "online_retarget_git_sha": os.environ.get("ONLINE_RETARGET_GIT_SHA"),
        "sonic_git_sha": os.environ.get("SONIC_GIT_SHA"),
        "video_path": str(video_path),
    }


def _route_counts(routes: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        key = str(int(route))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _stack_or_none(values: Any) -> Any:
    if not values:
        return None
    import numpy as np

    return np.asarray(values, dtype=float)


def _maybe_tensor(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _get_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:  # noqa: BLE001
        return None


def _rank(accelerator: Any, args: Any) -> int:
    if accelerator is not None and hasattr(accelerator, "process_index"):
        return int(accelerator.process_index)
    if hasattr(args, "global_rank"):
        return int(args.global_rank)
    return int(os.environ.get("RANK", "0"))


def _world_size(accelerator: Any, args: Any) -> int:
    if accelerator is not None and hasattr(accelerator, "num_processes"):
        return int(accelerator.num_processes)
    if hasattr(args, "world_size"):
        return int(args.world_size)
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process(accelerator: Any, state: Any) -> bool:
    if accelerator is not None and hasattr(accelerator, "is_main_process"):
        return bool(accelerator.is_main_process)
    return bool(getattr(state, "is_world_process_zero", True))


def _barrier(accelerator: Any) -> None:
    if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
        accelerator.wait_for_everyone()


def _git_sha() -> str | None:
    for key in ("ONLINE_RETARGET_GIT_SHA", "GIT_COMMIT", "WANDB_GIT_COMMIT"):
        value = os.environ.get(key)
        if value:
            return value
    root = os.environ.get("ONLINE_RETARGET_ROOT")
    if not root:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _safe_name(value: str) -> str:
    value = Path(value).stem or "unknown"
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:80] or "unknown"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
