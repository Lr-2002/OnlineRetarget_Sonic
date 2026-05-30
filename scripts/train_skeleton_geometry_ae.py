#!/usr/bin/env python3
"""Train the Skeleton AE gate on continuous 104D SOMA geometry."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online_retarget.data.skeleton_ae_registry import SKELETON_GEOMETRY_DIM  # noqa: E402
from online_retarget.models.skeleton_geometry_ae import SkeletonGeometryAE  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp_compact() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "training_lane", "input_data", "output_dir", "model", "training", "runtime"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    if config["training_lane"] != "skeleton_geometry_ae_only":
        raise ValueError("training_lane must be skeleton_geometry_ae_only")
    model = config["model"]
    expected = {
        "input_dim": 104,
        "hidden_dims": [256, 128],
        "latent_dim": 64,
        "decoder_hidden_dims": [128, 256],
        "output_dim": 104,
        "dropout": 0.0,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"model.{key} must be {value!r}")
    return config


def resolve_output_dir(config: Mapping[str, Any], run_group: str) -> Path:
    text = str(config["output_dir"])
    return Path(text.replace("{run_group}", run_group).replace("{timestamp}", timestamp_compact()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(command, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return True
    return False


def load_registry_rows(path: Path, expected_dim: int = SKELETON_GEOMETRY_DIM) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            geometry = json.loads(row["geometry_json"])
            if len(geometry) != expected_dim:
                raise ValueError(f"{row.get('actor_uid', '')} has geometry dim {len(geometry)}")
            split = str(row.get("split", "")).strip()
            if split == "val":
                split = "validation"
            if split not in {"train", "validation"}:
                raise ValueError(f"unsupported split {split!r}")
            rows.append(
                {
                    "actor_uid": row.get("actor_uid") or row.get("encoder_id") or "",
                    "encoder_id": row.get("encoder_id") or row.get("actor_uid") or "",
                    "split": split,
                    "source_soma_proportional_path": row.get("source_soma_proportional_path", ""),
                    "geometry_shape": row.get("geometry_shape", ""),
                    "x": torch.tensor(geometry, dtype=torch.float32),
                }
            )
    if not rows:
        raise ValueError(f"registry has no geometry rows: {path}")
    return rows


class SkeletonGeometryDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]], split: str) -> None:
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows[index]
        return row["x"], str(row["encoder_id"])


def collate_skeletons(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
    return torch.stack([item[0] for item in batch], dim=0), [item[1] for item in batch]


def compute_or_load_normalization(
    output_dir: Path,
    train_dataset: SkeletonGeometryDataset,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    stats_path = output_dir / "stats" / "skeleton_geometry_normalization.pt"
    if stats_path.exists():
        payload = torch.load(stats_path, map_location=device, weights_only=False)
        return {key: value.to(device) for key, value in payload.items() if torch.is_tensor(value)}
    train_x = torch.stack([row["x"] for row in train_dataset.rows], dim=0).to(device)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    payload = {
        "skeleton_mean": mean,
        "skeleton_std": std,
        "fit_split": torch.tensor([1], dtype=torch.int64, device=device),
        "fit_count": torch.tensor([train_x.shape[0]], dtype=torch.int64, device=device),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in payload.items()}, stats_path)
    return payload


def normalize_skeleton(x: torch.Tensor, stats: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (x - stats["skeleton_mean"]) / stats["skeleton_std"]


def ae_metrics(
    reconstructed_norm: torch.Tensor,
    x_norm: torch.Tensor,
    x_raw: torch.Tensor,
    stats: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = torch.mean((reconstructed_norm - x_norm) ** 2)
    with torch.no_grad():
        reconstructed_raw = reconstructed_norm * stats["skeleton_std"] + stats["skeleton_mean"]
        raw_error = reconstructed_raw - x_raw
        offset_error = raw_error[:, :78].reshape(raw_error.shape[0], 26, 3)
        length_error = raw_error[:, 78:]
        z = {
            "loss": float(mse.detach().item()),
            "normalized_mse": float(mse.detach().item()),
            "offset_rmse_raw": float(torch.sqrt(torch.mean(offset_error**2)).item()),
            "length_rmse_raw": float(torch.sqrt(torch.mean(length_error**2)).item()),
        }
    return mse, z


def validate(
    model: SkeletonGeometryAE,
    loader: DataLoader,
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for x, _ids in loader:
            x = x.to(device)
            x_norm = normalize_skeleton(x, stats)
            reconstructed_norm, z = model(x_norm)
            _loss, metrics = ae_metrics(reconstructed_norm, x_norm, x, stats)
            metrics["latent_std_mean"] = float(torch.std(z, dim=0, unbiased=False).mean().item())
            rows.append(metrics)
    model.train()
    if not rows:
        return {}
    return {f"validation/{key}": float(np.mean([row[key] for row in rows])) for key in rows[0]}


def write_loss_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "elapsed_sec", "train_loss", "train_offset_rmse_raw", "train_length_rmse_raw"])


def append_loss_row(path: Path, step: int, elapsed_sec: float, metrics: Mapping[str, float]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                step,
                f"{elapsed_sec:.4f}",
                f"{metrics['loss']:.10f}",
                f"{metrics['offset_rmse_raw']:.10f}",
                f"{metrics['length_rmse_raw']:.10f}",
            ]
        )


def write_manifest(
    *,
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    registry_rows: Sequence[Mapping[str, Any]],
    train_dataset: SkeletonGeometryDataset,
    validation_dataset: SkeletonGeometryDataset,
    dry_run: bool,
) -> dict[str, Any]:
    manifest = {
        "run_id": f"skeleton-geometry-ae-{timestamp_compact()}",
        "host": socket.gethostname(),
        "timestamp": utc_now(),
        "command_line": " ".join(sys.argv),
        "config_path": str(config_path),
        "control_repo": str(ROOT),
        "control_revision_actual": git_revision(ROOT),
        "control_status_short": git_status_short(ROOT),
        "training_lane": config["training_lane"],
        "dry_run": dry_run,
        "registry_csv": str(config["input_data"]["registry_csv"]),
        "row_count": len(registry_rows),
        "train_skeleton_count": len(train_dataset),
        "validation_skeleton_count": len(validation_dataset),
        "normalization": {
            "stats_path": str(output_dir / "stats" / "skeleton_geometry_normalization.pt"),
            "fit_split": "train",
        },
        "geometry": {
            "input_shape": [SKELETON_GEOMETRY_DIM],
            "latent_shape": [64],
            "output_shape": [SKELETON_GEOMETRY_DIM],
            "source_field": "geometry_json",
        },
        "model": {
            "architecture": [104, 256, 128, 64, 128, 256, 104],
            "activation": "SiLU",
            "dropout": 0.0,
        },
        "metrics_path": str(output_dir / "loss_curve.csv"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def save_checkpoint(
    output_dir: Path,
    model: SkeletonGeometryAE,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Mapping[str, Any],
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": dict(metrics),
        "saved_at": utc_now(),
        "architecture": [104, 256, 128, 64, 128, 256, 104],
    }
    torch.save(payload, ckpt_dir / "latest.pt")


def runtime_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "auto")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_runtime(config: Mapping[str, Any], output_dir: Path) -> None:
    write_root = Path(str(config["runtime"]["write_root"]))
    if output_dir != write_root and write_root not in output_dir.parents:
        raise ValueError(f"output_dir must be under write_root {write_root}: {output_dir}")
    registry_csv = Path(str(config["input_data"]["registry_csv"]))
    if not registry_csv.exists():
        raise FileNotFoundError(f"registry_csv is missing: {registry_csv}")
    expected_shape = list(config["input_data"].get("expected_geometry_shape", [SKELETON_GEOMETRY_DIM]))
    if expected_shape != [SKELETON_GEOMETRY_DIM]:
        raise ValueError(f"expected_geometry_shape must be [{SKELETON_GEOMETRY_DIM}]")
    if config.get("runtime", {}).get("require_committed_code", False) and git_has_tracked_changes(ROOT):
        raise RuntimeError(f"control repo has uncommitted tracked changes: {ROOT}")


def run(config_path: Path, *, max_steps_override: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    config = read_config(config_path)
    run_group = os.environ.get("SKELETON_AE_RUN_GROUP", timestamp_compact())
    output_dir = resolve_output_dir(config, run_group)
    validate_runtime(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["training"]["seed"]))
    device = runtime_device(config)

    registry_rows = load_registry_rows(Path(str(config["input_data"]["registry_csv"])))
    train_dataset = SkeletonGeometryDataset(registry_rows, "train")
    validation_dataset = SkeletonGeometryDataset(registry_rows, "validation")
    stats = compute_or_load_normalization(output_dir, train_dataset, device)
    model = SkeletonGeometryAE(
        input_dim=int(config["model"]["input_dim"]),
        latent_dim=int(config["model"]["latent_dim"]),
    ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_skeletons,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"].get("validation_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        collate_fn=collate_skeletons,
    )
    manifest = write_manifest(
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        registry_rows=registry_rows,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        dry_run=dry_run,
    )
    if dry_run:
        validation_metrics = validate(model, validation_loader, stats, device)
        summary = {
            "event": "dry_run",
            "output_dir": str(output_dir),
            "manifest": str(output_dir / "manifest.json"),
            "normalization": str(output_dir / "stats" / "skeleton_geometry_normalization.pt"),
            "train_skeleton_count": len(train_dataset),
            "validation_skeleton_count": len(validation_dataset),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            **validation_metrics,
        }
        (output_dir / "dry_run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        return {**manifest, "dry_run_summary": summary}

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    max_steps = int(max_steps_override or config["training"]["max_steps"])
    validate_every = int(config["training"]["validate_every"])
    log_every = int(config["training"]["log_every"])
    loss_curve = output_dir / "loss_curve.csv"
    write_loss_header(loss_curve)
    start = time.perf_counter()
    step = 0
    last_metrics: dict[str, float] = {}
    while step < max_steps:
        for x, _ids in train_loader:
            if step >= max_steps:
                break
            x = x.to(device)
            x_norm = normalize_skeleton(x, stats)
            optimizer.zero_grad(set_to_none=True)
            reconstructed_norm, _z = model(x_norm)
            mse, metrics = ae_metrics(reconstructed_norm, x_norm, x, stats)
            mse.backward()
            grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            last_metrics = metrics
            if step % log_every == 0 or step == 1:
                append_loss_row(loss_curve, step, time.perf_counter() - start, metrics)
                print(json.dumps({"event": "train", "step": step, **metrics}, sort_keys=True), flush=True)
            if step % validate_every == 0 or step == 1:
                val_metrics = validate(model, validation_loader, stats, device)
                print(json.dumps({"event": "validation", "step": step, **val_metrics}, sort_keys=True), flush=True)
        if len(train_loader) == 0:
            break
    final_metrics = {"step": step, "elapsed_sec": time.perf_counter() - start, **last_metrics}
    save_checkpoint(output_dir, model, optimizer, step, final_metrics)
    print(json.dumps({"event": "finished", **final_metrics}, sort_keys=True), flush=True)
    return {**manifest, "final_metrics": final_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.config, max_steps_override=args.max_steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
