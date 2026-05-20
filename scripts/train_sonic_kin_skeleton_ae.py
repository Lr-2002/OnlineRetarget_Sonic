#!/usr/bin/env python3
"""Train skeleton-conditioned kin-only retargeting MLPs.

This is a supervised, simulator-free training lane.  It predicts the same
kinematic target fields used by SONIC's ``g1_kin`` decoder:

* ``command_multi_future_nonflat``: 29 joint positions + 29 joint velocities
* ``motion_anchor_ori_b_mf_nonflat``: 6D root orientation per future frame
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REQUIRED_NPZ_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "owner",
    "purpose",
    "source_repo",
    "source_rev",
    "input_data",
    "output_dir",
    "validation_command",
    "expected_artifacts",
    "variant",
    "features",
    "model",
    "training",
    "runtime",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp_compact() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stats(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def read_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS - set(data))
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")
    return data


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_status_short(root: Path) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ["not-a-git-worktree"]
    return out.splitlines()[:200]


def git_has_tracked_changes(root: Path) -> bool:
    checks = (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    )
    for cmd in checks:
        result = subprocess.run(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return True
    return False


def require_latest_git(root: Path, label: str) -> None:
    try:
        upstream = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        raise RuntimeError(f"{label} has no upstream tracking branch: {root}") from exc
    if "/" not in upstream:
        raise RuntimeError(f"{label} has unsupported upstream {upstream!r}: {root}")
    remote, branch = upstream.split("/", 1)
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", remote, branch],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{label} could not fetch {upstream}; refusing to train without a latest-code check"
        ) from exc

    head = git_revision(root)
    upstream_head = subprocess.check_output(
        ["git", "rev-parse", "FETCH_HEAD"],
        cwd=root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if head != upstream_head:
        raise RuntimeError(f"{label} is not latest: HEAD={head}, {upstream}={upstream_head}")


def resolve_text(value: str, config: dict[str, Any], run_group: str) -> str:
    return (
        value.replace("{run_group}", run_group)
        .replace("{variant}", str(config["variant"]["name"]))
        .replace("{timestamp}", timestamp_compact())
    )


def resolve_output_dir(config: dict[str, Any], run_group: str) -> Path:
    return Path(resolve_text(str(config["output_dir"]), config, run_group))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(norm < 1e-8, 1.0, norm)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )
    return quat_normalize(out)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                axis=-1,
            ),
            np.stack(
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                axis=-1,
            ),
            np.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def future_indices(frame_indices: np.ndarray, total_frames: int, window: int, step: int) -> np.ndarray:
    offsets = np.arange(window, dtype=np.int64) * int(step)
    return np.minimum(frame_indices[:, None] + offsets[None, :], total_frames - 1)


def relative_root_rot6d(root_quat: np.ndarray, frame_indices: np.ndarray, idx: np.ndarray) -> np.ndarray:
    base = quat_normalize(root_quat[frame_indices])
    refs = quat_normalize(root_quat[idx])
    rel = quat_mul(quat_conjugate(base[:, None, :]), refs)
    mat = quat_to_matrix(rel)
    return mat[..., :, :2].reshape(frame_indices.shape[0], idx.shape[1], 6)


def build_features(
    arrays: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    window: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_pos = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    body_pos = arrays["body_pos_w"]
    body_quat = arrays["body_quat_w"]
    total_frames = min(joint_pos.shape[0], joint_vel.shape[0], body_pos.shape[0], body_quat.shape[0])
    idx = future_indices(frame_indices, total_frames, window, step)

    root_pos0 = body_pos[frame_indices, 0]
    root_quat0 = quat_normalize(body_quat[frame_indices, 0])
    root_rot0_t = np.swapaxes(quat_to_matrix(root_quat0), -1, -2)
    rel_body_pos = body_pos[idx] - root_pos0[:, None, None, :]
    local_body_pos = np.einsum("nij,nwbj->nwbi", root_rot0_t, rel_body_pos)
    root_ori6d = relative_root_rot6d(body_quat[:, 0], frame_indices, idx)
    root_z_rel = (body_pos[idx, 0, 2] - root_pos0[:, None, 2])[..., None]

    motion = np.concatenate(
        [
            local_body_pos.reshape(frame_indices.shape[0], -1),
            root_ori6d.reshape(frame_indices.shape[0], -1),
            root_z_rel.reshape(frame_indices.shape[0], -1),
        ],
        axis=-1,
    )

    skeleton_anchor = local_body_pos[:, 0]
    skeleton_lengths = np.linalg.norm(skeleton_anchor, axis=-1)
    skeleton = np.concatenate(
        [skeleton_anchor.reshape(frame_indices.shape[0], -1), skeleton_lengths],
        axis=-1,
    )

    command = np.concatenate([joint_pos[idx], joint_vel[idx]], axis=-1)
    target = np.concatenate(
        [command.reshape(frame_indices.shape[0], -1), root_ori6d.reshape(frame_indices.shape[0], -1)],
        axis=-1,
    )
    return (
        motion.astype(np.float32, copy=False),
        skeleton.astype(np.float32, copy=False),
        target.astype(np.float32, copy=False),
    )


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key].astype(np.float32, copy=False) for key in REQUIRED_NPZ_KEYS}


def index_path_from_config(config: dict[str, Any]) -> Path:
    indexing = config["input_data"].get("indexing", {})
    raw_path = indexing.get("index_csv") or indexing.get("prebuilt_index_jsonl")
    if not raw_path:
        raise ValueError("input_data.indexing must define index_csv or prebuilt_index_jsonl")
    return Path(raw_path)


def remap_source_path(source_path: str, indexing: dict[str, Any]) -> Path:
    source_prefix = indexing.get("source_path_prefix", "")
    target_prefix = indexing.get("target_path_prefix", "")
    if source_prefix and target_prefix and source_path.startswith(source_prefix):
        return Path(target_prefix + source_path[len(source_prefix) :])
    return Path(source_path)


def rows_from_jsonl_index(config: dict[str, Any], data_root: Path) -> tuple[list[dict[str, Any]], int]:
    indexing = config["input_data"].get("indexing", {})
    index_path = Path(indexing["prebuilt_index_jsonl"])
    max_clips = int(indexing.get("max_clips", 0))
    rows = []
    skipped = 0
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            source = json.loads(line)
            if not source.get("source_npz_exists", False):
                skipped += 1
                continue
            source_path = str(source["source_npz_path"])
            path = remap_source_path(source_path, indexing)
            try:
                relative_path = str(path.relative_to(data_root))
            except ValueError:
                skipped += 1
                continue
            frame_count = int(float(source.get("move_duration_frames") or -1))
            if frame_count <= 1:
                skipped += 1
                continue
            rows.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "frame_count": frame_count,
                    "source_move_name": source.get("move_name"),
                    "source_labels": source.get("labels", {}),
                }
            )
            if max_clips > 0 and len(rows) >= max_clips:
                break
    return rows, skipped


def rows_from_csv_index(config: dict[str, Any], data_root: Path) -> tuple[list[dict[str, Any]], int]:
    indexing = config["input_data"].get("indexing", {})
    index_path = Path(indexing["index_csv"])
    max_clips = int(indexing.get("max_clips", 0))
    rows = []
    skipped = 0
    with index_path.open("r", encoding="utf-8", newline="") as f:
        for source in csv.DictReader(f):
            if source.get("schema_status") not in {"", "ok", None}:
                skipped += 1
                continue
            source_path = str(source.get("sonic_path") or source.get("source_npz_path") or "")
            if not source_path:
                skipped += 1
                continue
            path = remap_source_path(source_path, indexing)
            try:
                relative_path = str(path.relative_to(data_root))
            except ValueError:
                skipped += 1
                continue
            frame_count_text = source.get("frame_count") or source.get("move_duration_frames") or ""
            try:
                frame_count = int(float(frame_count_text))
            except ValueError:
                skipped += 1
                continue
            if frame_count <= 1:
                skipped += 1
                continue
            rows.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "frame_count": frame_count,
                    "source_move_name": source.get("filename") or source.get("move_name"),
                    "source_labels": {
                        "package": source.get("package", ""),
                        "category": source.get("category", ""),
                        "actor_uid": source.get("actor_uid", ""),
                    },
                }
            )
            if max_clips > 0 and len(rows) >= max_clips:
                break
    return rows, skipped


def rows_from_index(config: dict[str, Any], data_root: Path) -> tuple[list[dict[str, Any]], int]:
    indexing = config["input_data"].get("indexing", {})
    if indexing.get("index_csv"):
        return rows_from_csv_index(config, data_root)
    return rows_from_jsonl_index(config, data_root)


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def split_rows(rows: list[dict[str, Any]], validation_ratio: float, hash_salt: str) -> None:
    threshold = int(validation_ratio * 1_000_000)
    for row in rows:
        value = stable_hash_int(row["relative_path"] + hash_salt) % 1_000_000
        row["split"] = "validation" if value < threshold else "train"
    if not any(row["split"] == "validation" for row in rows):
        rows[-1]["split"] = "validation"
    if not any(row["split"] == "train" for row in rows):
        rows[0]["split"] = "train"


class KinWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], split: str, config: dict[str, Any]) -> None:
        self.rows = [row for row in rows if row["split"] == split]
        self.data_root = Path(config["input_data"]["data_root"])
        self.window = int(config["features"]["future_window_frames"])
        self.step = int(config["features"]["future_step"])
        self.frame_stride = int(config["training"]["frame_stride"])
        self.chunk_frames = int(config["training"]["loader_chunk_frames"])
        self.samples: list[tuple[int, int, int]] = []
        for row_idx, row in enumerate(self.rows):
            frame_count = int(row["frame_count"])
            max_start = max(0, frame_count - 1)
            start = 0
            while start <= max_start:
                stop = min(frame_count, start + self.chunk_frames * self.frame_stride)
                if stop > start:
                    self.samples.append((row_idx, start, stop))
                start = stop
        if not self.samples:
            raise ValueError(f"no samples for split {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_idx, start, stop = self.samples[index]
        row = self.rows[row_idx]
        arrays = load_arrays(self.data_root / row["relative_path"])
        total_frames = min(
            arrays["joint_pos"].shape[0],
            arrays["joint_vel"].shape[0],
            arrays["body_pos_w"].shape[0],
            arrays["body_quat_w"].shape[0],
        )
        stop = min(stop, total_frames)
        if start >= stop:
            start = 0
            stop = total_frames
        frame_indices = np.arange(start, stop, self.frame_stride, dtype=np.int64)
        motion, skeleton, target = build_features(arrays, frame_indices, self.window, self.step)
        return torch.from_numpy(motion), torch.from_numpy(skeleton), torch.from_numpy(target)


def collate_chunks(
    chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    motion = torch.cat([item[0] for item in chunks], dim=0)
    skeleton = torch.cat([item[1] for item in chunks], dim=0)
    target = torch.cat([item[2] for item in chunks], dim=0)
    return motion, skeleton, target


class RunningStats:
    def __init__(self, dim: int, device: torch.device) -> None:
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.sum = torch.zeros(dim, dtype=torch.float64, device=device)
        self.sumsq = torch.zeros(dim, dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor) -> None:
        data = x.to(torch.float64)
        self.count += data.shape[0]
        self.sum += data.sum(dim=0)
        self.sumsq += (data * data).sum(dim=0)

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.sum / self.count.clamp_min(1.0)
        var = self.sumsq / self.count.clamp_min(1.0) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return mean.to(torch.float32), std.to(torch.float32)


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(prev, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ConcatRetargeter(nn.Module):
    def __init__(self, motion_dim: int, skeleton_dim: int, output_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.net = build_mlp(
            motion_dim + skeleton_dim,
            int(cfg["hidden_dim"]),
            output_dim,
            int(cfg["num_layers"]),
            float(cfg.get("dropout", 0.0)),
        )

    def forward(self, motion: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([motion, skeleton], dim=-1))


class FilmRetargeter(nn.Module):
    def __init__(self, motion_dim: int, skeleton_dim: int, output_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        num_layers = int(cfg["num_layers"])
        self.input = nn.Linear(motion_dim, hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, num_layers - 1)))
        self.film = nn.Linear(skeleton_dim, 2 * hidden_dim * max(1, num_layers))
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.SiLU()
        self.num_layers = max(1, num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, motion: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        film = self.film(skeleton).view(skeleton.shape[0], self.num_layers, 2, self.hidden_dim)
        h = self.activation(self.input(motion))
        gamma, beta = film[:, 0, 0], film[:, 0, 1]
        h = h * (1.0 + gamma) + beta
        for idx, layer in enumerate(self.layers, start=1):
            h = self.activation(layer(h))
            gamma, beta = film[:, idx, 0], film[:, idx, 1]
            h = h * (1.0 + gamma) + beta
        return self.output(h)


class AdapterRetargeter(nn.Module):
    def __init__(self, motion_dim: int, skeleton_dim: int, output_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        adapter_dim = int(cfg.get("adapter_dim", max(32, hidden_dim // 4)))
        self.motion = build_mlp(motion_dim, hidden_dim, hidden_dim, int(cfg["num_layers"]), float(cfg.get("dropout", 0.0)))
        self.adapter = nn.Sequential(
            nn.Linear(skeleton_dim, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, motion: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        return self.output(self.motion(motion) + self.adapter(skeleton))


class ExpertRetargeter(nn.Module):
    def __init__(self, motion_dim: int, skeleton_dim: int, output_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        num_experts = int(cfg.get("num_experts", 4))
        self.experts = nn.ModuleList(
            build_mlp(motion_dim, hidden_dim, hidden_dim, int(cfg["num_layers"]), float(cfg.get("dropout", 0.0)))
            for _ in range(num_experts)
        )
        self.gate = nn.Linear(skeleton_dim, num_experts)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, motion: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(skeleton), dim=-1)
        expert_h = torch.stack([expert(motion) for expert in self.experts], dim=1)
        h = torch.sum(expert_h * weights.unsqueeze(-1), dim=1)
        return self.output(h)


def make_model(motion_dim: int, skeleton_dim: int, output_dim: int, config: dict[str, Any]) -> nn.Module:
    variant_type = config["variant"]["type"]
    cfg = config["model"]
    if variant_type == "concat":
        return ConcatRetargeter(motion_dim, skeleton_dim, output_dim, cfg)
    if variant_type == "film":
        return FilmRetargeter(motion_dim, skeleton_dim, output_dim, cfg)
    if variant_type == "adapter":
        return AdapterRetargeter(motion_dim, skeleton_dim, output_dim, cfg)
    if variant_type == "expert":
        return ExpertRetargeter(motion_dim, skeleton_dim, output_dim, cfg)
    raise ValueError(f"unknown variant type: {variant_type}")


def compute_or_load_stats(
    output_dir: Path,
    train_dataset: KinWindowDataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    stats_path = output_dir / "stats" / "normalization.pt"
    if stats_path.exists():
        payload = torch.load(stats_path, map_location=device, weights_only=False)
        return {key: value.to(device) for key, value in payload.items() if torch.is_tensor(value)}

    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_chunks,
    )
    motion_stats = None
    skeleton_stats = None
    target_stats = None
    seen = 0
    max_frames = int(config["normalization"]["max_frames"])
    frames_per_chunk = int(config["normalization"].get("frames_per_chunk", 1024))
    for motion, skeleton, target in loader:
        if motion.shape[0] > frames_per_chunk:
            select = torch.randperm(motion.shape[0])[:frames_per_chunk]
            motion, skeleton, target = motion[select], skeleton[select], target[select]
        motion = motion.to(device)
        skeleton = skeleton.to(device)
        target = target.to(device)
        if motion_stats is None:
            motion_stats = RunningStats(motion.shape[-1], device)
            skeleton_stats = RunningStats(skeleton.shape[-1], device)
            target_stats = RunningStats(target.shape[-1], device)
        motion_stats.update(motion)
        skeleton_stats.update(skeleton)
        target_stats.update(target)
        seen += int(motion.shape[0])
        if seen >= max_frames:
            break

    if motion_stats is None or skeleton_stats is None or target_stats is None:
        raise RuntimeError("could not compute normalization stats")
    motion_mean, motion_std = motion_stats.finalize()
    skeleton_mean, skeleton_std = skeleton_stats.finalize()
    target_mean, target_std = target_stats.finalize()
    payload = {
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "skeleton_mean": skeleton_mean,
        "skeleton_std": skeleton_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in payload.items()}, stats_path)
    return payload


def normalize_batch(
    motion: torch.Tensor,
    skeleton: torch.Tensor,
    target: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        (motion - stats["motion_mean"]) / stats["motion_std"],
        (skeleton - stats["skeleton_mean"]) / stats["skeleton_std"],
        (target - stats["target_mean"]) / stats["target_std"],
    )


def loss_and_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    target_raw: torch.Tensor,
    stats: dict[str, torch.Tensor],
    command_dim: int,
    joint_dim: int,
    command_weight: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_command = pred_norm[:, :command_dim]
    target_command = target_norm[:, :command_dim]
    pred_anchor = pred_norm[:, command_dim:]
    target_anchor = target_norm[:, command_dim:]
    command_loss = torch.mean((pred_command - target_command) ** 2)
    anchor_loss = torch.mean((pred_anchor - target_anchor) ** 2)
    loss = command_weight * command_loss + anchor_weight * anchor_loss

    with torch.no_grad():
        pred_raw = pred_norm * stats["target_std"] + stats["target_mean"]
        raw_error = pred_raw - target_raw
        command_raw = raw_error[:, :command_dim]
        anchor_raw = raw_error[:, command_dim:]
        command_frame_dim = joint_dim * 2
        joint_error = command_raw.reshape(command_raw.shape[0], -1, command_frame_dim)
        joint_pos_error = joint_error[..., :joint_dim]
        joint_vel_error = joint_error[..., joint_dim:]
        metrics = {
            "loss": float(loss.detach().item()),
            "command_mse_norm": float(command_loss.detach().item()),
            "anchor_mse_norm": float(anchor_loss.detach().item()),
            "joint_pos_rmse_raw": float(torch.sqrt(torch.mean(joint_pos_error**2)).item()),
            "joint_vel_rmse_raw": float(torch.sqrt(torch.mean(joint_vel_error**2)).item()),
            "anchor_rmse_raw": float(torch.sqrt(torch.mean(anchor_raw**2)).item()),
        }
    return loss, metrics


def validate(
    model: nn.Module,
    loader: DataLoader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    command_dim: int,
    joint_dim: int,
) -> dict[str, float]:
    model.eval()
    rows = []
    max_batches = int(config["training"]["validation_batches"])
    command_weight = float(config["training"].get("command_loss_weight", 1.0))
    anchor_weight = float(config["training"].get("anchor_loss_weight", 1.0))
    with torch.no_grad():
        for batch_idx, (motion, skeleton, target) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            motion = motion.to(device, non_blocking=True)
            skeleton = skeleton.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            motion_n, skeleton_n, target_n = normalize_batch(motion, skeleton, target, stats)
            pred_n = model(motion_n, skeleton_n)
            _, metrics = loss_and_metrics(
                pred_n,
                target_n,
                target,
                stats,
                command_dim,
                joint_dim,
                command_weight,
                anchor_weight,
            )
            rows.append(metrics)
    model.train()
    if not rows:
        return {}
    return {f"validation/{key}": float(np.mean([row[key] for row in rows])) for key in rows[0]}


def write_loss_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "elapsed_sec",
                "train_loss",
                "train_command_mse_norm",
                "train_anchor_mse_norm",
                "train_joint_pos_rmse_raw",
                "train_joint_vel_rmse_raw",
                "train_anchor_rmse_raw",
            ]
        )


def append_loss_row(path: Path, step: int, elapsed: float, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                step,
                f"{elapsed:.4f}",
                f"{metrics['loss']:.10f}",
                f"{metrics['command_mse_norm']:.10f}",
                f"{metrics['anchor_mse_norm']:.10f}",
                f"{metrics['joint_pos_rmse_raw']:.10f}",
                f"{metrics['joint_vel_rmse_raw']:.10f}",
                f"{metrics['anchor_rmse_raw']:.10f}",
            ]
        )


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, Any],
    keep_last: int,
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "saved_at": utc_now(),
    }
    step_path = ckpt_dir / f"step_{step:08d}.pt"
    torch.save(payload, step_path)
    tmp_latest = ckpt_dir / "latest.tmp.pt"
    torch.save(payload, tmp_latest)
    tmp_latest.replace(ckpt_dir / "latest.pt")
    for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep_last]:
        old.unlink(missing_ok=True)


def write_manifest(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    run_group: str,
    rows: list[dict[str, Any]],
    train_dataset: KinWindowDataset,
    validation_dataset: KinWindowDataset,
    motion_dim: int,
    skeleton_dim: int,
    target_dim: int,
    skipped_index_rows: int,
) -> dict[str, Any]:
    source_root = Path(config["source_repo"])
    control_root = Path.cwd()
    manifest = {
        "run_id": f"sonic-kin-skeleton-{config['variant']['name']}-{timestamp_compact()}",
        "run_group": run_group,
        "host": socket.gethostname(),
        "timestamp": utc_now(),
        "environment_prefix": sys.prefix,
        "command_line": " ".join(sys.argv),
        "config_path": str(config_path),
        "source_repo": str(source_root),
        "source_revision_declared": config["source_rev"],
        "source_revision_actual": git_revision(source_root),
        "source_status_short": git_status_short(source_root),
        "control_repo": str(control_root),
        "control_revision_actual": git_revision(control_root),
        "control_status_short": git_status_short(control_root),
        "variant": config["variant"],
        "feature_dims": {
            "motion": motion_dim,
            "skeleton": skeleton_dim,
            "target": target_dim,
        },
        "data_snapshot": {
            "data_root": config["input_data"]["data_root"],
            "index": file_stats(index_path_from_config(config)),
            "row_count": len(rows),
            "skipped_index_rows": skipped_index_rows,
            "train_chunks": len(train_dataset),
            "validation_chunks": len(validation_dataset),
        },
        "metrics_path": str(output_dir / "loss_curve.csv"),
        "notes": config["purpose"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def init_wandb(config: dict[str, Any], manifest: dict[str, Any], output_dir: Path, run_group: str):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    import wandb

    wandb_dir = Path(resolve_text(wandb_cfg.get("dir", str(output_dir / "wandb")), config, run_group))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    name = resolve_text(wandb_cfg.get("name", config["variant"]["name"]), config, run_group)
    group = resolve_text(wandb_cfg.get("group", run_group), config, run_group)
    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity"),
        name=name,
        group=group,
        dir=wandb_dir,
        config={**config, "manifest": manifest},
        tags=wandb_cfg.get("tags", []),
        resume="allow",
    )
    if run is not None:
        run.summary["git_commit"] = manifest["control_revision_actual"]
        run.summary["sonic_git_commit"] = manifest["source_revision_actual"]
    return run


def validate_runtime(config: dict[str, Any], output_dir: Path) -> None:
    write_root = Path(config["runtime"]["write_root"])
    if output_dir != write_root and write_root not in output_dir.parents:
        raise ValueError(f"output_dir must be under write_root {write_root}: {output_dir}")
    for forbidden in config["runtime"].get("forbid_write_roots", []):
        forbidden_path = Path(forbidden)
        if output_dir == forbidden_path or forbidden_path in output_dir.parents:
            raise ValueError(f"output_dir must not be under {forbidden_path}: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    required_gpu_count = int(config["runtime"].get("required_gpu_count", 1))
    if torch.cuda.device_count() < required_gpu_count:
        raise RuntimeError(f"expected at least {required_gpu_count} visible GPU(s), found {torch.cuda.device_count()}")
    data_root = Path(config["input_data"]["data_root"])
    if not data_root.exists():
        raise FileNotFoundError(f"data_root is missing: {data_root}")
    if not index_path_from_config(config).exists():
        raise FileNotFoundError(f"index is missing: {index_path_from_config(config)}")
    source_root = Path(config["source_repo"])
    if config["runtime"].get("require_committed_code", True):
        control_root = Path.cwd()
        if git_revision(control_root) is None:
            raise RuntimeError(f"control repo is not a git worktree: {control_root}")
        if git_has_tracked_changes(control_root):
            raise RuntimeError(f"control repo has uncommitted tracked changes: {control_root}")
        if git_revision(source_root) is None:
            raise RuntimeError(f"source repo is not a git worktree: {source_root}")
        if git_has_tracked_changes(source_root):
            raise RuntimeError(f"source repo has uncommitted tracked changes: {source_root}")
    if config["runtime"].get("require_latest_code", True):
        require_latest_git(Path.cwd(), "control repo")
        require_latest_git(source_root, "source repo")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = read_config(args.config)
    run_group = os.environ.get("KIN_RUN_GROUP", timestamp_compact())
    output_dir = resolve_output_dir(config, run_group)
    validate_runtime(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows, skipped = rows_from_index(config, Path(config["input_data"]["data_root"]))
    split_rows(rows, float(config["split"]["validation_ratio"]), str(config["split"]["hash_salt"]))
    train_dataset = KinWindowDataset(rows, "train", config)
    validation_dataset = KinWindowDataset(rows, "validation", config)
    stats = compute_or_load_stats(output_dir, train_dataset, config, device)

    feature_cfg = config["features"]
    window = int(feature_cfg["future_window_frames"])
    motion_dim = int(stats["motion_mean"].numel())
    skeleton_dim = int(stats["skeleton_mean"].numel())
    target_dim = int(stats["target_mean"].numel())
    anchor_dim = window * 6
    command_dim = target_dim - anchor_dim
    if command_dim <= 0 or command_dim % (window * 2) != 0:
        raise ValueError(
            f"target_dim={target_dim} is incompatible with window={window} and root 6D anchor target"
        )
    joint_dim = command_dim // (window * 2)
    model = make_model(motion_dim, skeleton_dim, target_dim, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_chunks,
        pin_memory=True,
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(config["training"]["num_workers"]) // 2),
        collate_fn=collate_chunks,
        pin_memory=True,
    )
    loss_curve = output_dir / "loss_curve.csv"
    write_loss_header(loss_curve)
    manifest = write_manifest(
        output_dir,
        args.config,
        config,
        run_group,
        rows,
        train_dataset,
        validation_dataset,
        motion_dim,
        skeleton_dim,
        target_dim,
        skipped,
    )
    wandb_run = init_wandb(config, manifest, output_dir, run_group)

    max_steps = int(config["training"]["max_steps"])
    per_batch_frames = int(config["training"]["batch_frames"])
    log_every = int(config["training"]["log_every"])
    validate_every = int(config["training"]["validate_every"])
    checkpoint_every = int(config["training"]["checkpoint_every"])
    keep_last = int(config["training"]["keep_last_checkpoints"])
    grad_clip = float(config["training"]["grad_clip_norm"])
    command_weight = float(config["training"].get("command_loss_weight", 1.0))
    anchor_weight = float(config["training"].get("anchor_loss_weight", 1.0))
    precision = config["training"].get("precision", "bf16")
    use_amp = precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    recent: deque[dict[str, float]] = deque(maxlen=log_every)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + 20260520)
    start = time.perf_counter()
    step = 0

    print(
        json.dumps(
            {
                "event": "start",
                "variant": config["variant"],
                "run_group": run_group,
                "output_dir": str(output_dir),
                "control_commit": manifest["control_revision_actual"],
                "sonic_commit": manifest["source_revision_actual"],
                "train_chunks": len(train_dataset),
                "validation_chunks": len(validation_dataset),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    while step < max_steps:
        for motion, skeleton, target in train_loader:
            if step >= max_steps:
                break
            motion = motion.to(device, non_blocking=True)
            skeleton = skeleton.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if motion.shape[0] > per_batch_frames:
                select = torch.randint(0, motion.shape[0], (per_batch_frames,), generator=rng, device=device)
                motion = motion[select]
                skeleton = skeleton[select]
                target = target[select]
            motion_n, skeleton_n, target_n = normalize_batch(motion, skeleton, target, stats)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred_n = model(motion_n, skeleton_n)
                loss, metrics = loss_and_metrics(
                    pred_n,
                    target_n,
                    target,
                    stats,
                    command_dim,
                    joint_dim,
                    command_weight,
                    anchor_weight,
                )
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            recent.append(metrics)

            if step % log_every == 0 or step == 1:
                elapsed = time.perf_counter() - start
                avg = {key: float(np.mean([row[key] for row in recent])) for key in metrics}
                append_loss_row(loss_curve, step, elapsed, avg)
                log_payload = {f"train/{key}": value for key, value in avg.items()}
                log_payload["elapsed_sec"] = elapsed
                log_payload["step"] = step
                print(json.dumps({"event": "train", **log_payload}, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(log_payload, step=step)

            if step % validate_every == 0 or step == 1:
                val_metrics = validate(model, val_loader, stats, device, config, command_dim, joint_dim)
                if val_metrics:
                    print(json.dumps({"event": "validation", "step": step, **val_metrics}, sort_keys=True), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(val_metrics, step=step)

            if step % checkpoint_every == 0 or step == 1:
                save_checkpoint(output_dir, model, optimizer, step, metrics, keep_last)

    final_metrics = {"step": step, "elapsed_sec": time.perf_counter() - start, "finished": True}
    save_checkpoint(output_dir, model, optimizer, step, final_metrics, keep_last)
    if wandb_run is not None:
        wandb_run.summary.update(final_metrics)
        wandb_run.finish()
    print(json.dumps({"event": "finished", **final_metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
