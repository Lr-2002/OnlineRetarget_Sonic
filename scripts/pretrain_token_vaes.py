#!/usr/bin/env python3
"""Pretrain independent continuous-token VAEs from supervised BONES JSONL samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - environment blocker path
    yaml = None

from online_retarget.data.schema import ObservationSpec
from online_retarget.models.token_vae import MLPTokenVAE, vae_loss
from scripts import train as train_entry


COMPONENTS = ("skeleton", "motion", "action")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--wandb-mode",
        help="Override tracking.wandb_mode, e.g. disabled, offline, or online.",
    )
    parser.add_argument("--allow-debug-data", action="store_true")
    args = parser.parse_args()

    config = _apply_wandb_mode_override(_load_config(args.config), args.wandb_mode)
    samples_jsonl = args.samples_jsonl or _path_from_config(config, ("data", "samples_jsonl"))
    if samples_jsonl is None:
        raise SystemExit("Set --samples-jsonl or data.samples_jsonl")
    output_dir = args.output_dir or (
        Path(str(_nested_get(config, ("experiment", "output_root"), "runs")))
        / "pretrain"
        / str(_nested_get(config, ("experiment", "name"), "token_vae_pretrain"))
    )
    components = tuple(args.components or _components_from_config(config))
    _validate_components(components)

    quality_gate = train_entry._quality_gate_context(
        config,
        index_csv=_path_from_config(config, ("data", "index_csv")),
        samples_jsonl=samples_jsonl,
        allow_debug_data=args.allow_debug_data,
    )
    train_entry._validate_quality_gate(quality_gate)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Token VAE pretraining requires torch.") from exc

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    runtime = train_entry._setup_torch_runtime(torch, rank=rank, world_size=world_size)
    try:
        _pretrain(
            torch=torch,
            config=config,
            samples_jsonl=samples_jsonl,
            output_dir=output_dir,
            components=components,
            max_steps=args.max_steps or int(_nested_get(config, ("train", "max_steps"), 1000)),
            batch_size=args.batch_size or int(_nested_get(config, ("train", "batch_size"), 256)),
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
    finally:
        train_entry._cleanup_torch_runtime(torch, runtime)


def _pretrain(
    *,
    torch,
    config: dict[str, Any],
    samples_jsonl: Path,
    output_dir: Path,
    components: tuple[str, ...],
    max_steps: int,
    batch_size: int,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    samples = train_entry._load_supervised_samples(samples_jsonl)
    if not samples:
        raise SystemExit(f"no supervised samples found in {samples_jsonl}")
    observation_spec = train_entry._observation_spec_from_config_and_manifest(
        config,
        train_entry._load_sample_manifest(samples_jsonl),
        input_dim=len(samples[0]["observation"]),
    )
    tensors = _component_tensors(torch, samples=samples, observation_spec=observation_spec)
    device = runtime["device"]
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = train_entry._init_wandb(
        config=config,
        quality_gate=quality_gate,
        output_dir=output_dir,
        enabled=rank == 0,
    )

    reports = {}
    for component in components:
        report = _train_component(
            torch=torch,
            config=config,
            component=component,
            data=tensors[component],
            output_dir=output_dir / component,
            max_steps=max_steps,
            batch_size=batch_size,
            device=device,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
            wandb_run=wandb_run,
        )
        reports[component] = report

    if rank == 0:
        summary = {
            "mode": "token_vae_pretrain",
            "samples_jsonl": str(samples_jsonl),
            "output_dir": str(output_dir),
            "components": list(components),
            "sample_count": len(samples),
            "latent_dim": int(_nested_get(config, ("model", "latent_dim"), 128)),
            "observation_spec": observation_spec.to_dict(),
            "quality_gate": quality_gate,
            "component_reports": reports,
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "world_size": world_size,
            "rank": rank,
            "distributed_runtime": train_entry._runtime_report(runtime),
            "wandb": train_entry._wandb_summary(wandb_run),
        }
        report_path = output_dir / "pretrain_report.json"
        report_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        train_entry._wandb_save(wandb_run, report_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
    train_entry._wandb_finish(wandb_run)


def _train_component(
    *,
    torch,
    config: dict[str, Any],
    component: str,
    data,
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    device,
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
    wandb_run,
) -> dict[str, Any]:
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    train_cfg = config.get("train", {}) if isinstance(config.get("train", {}), dict) else {}
    vae = MLPTokenVAE(
        input_dim=data.shape[1],
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        hidden_dims=tuple(int(value) for value in model_cfg.get("hidden_dims", [256, 256])),
        activation=str(model_cfg.get("activation", "gelu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    if runtime["distributed"]:
        vae = torch.nn.parallel.DistributedDataParallel(
            vae,
            device_ids=[runtime["local_rank"]] if runtime["device_type"] == "cuda" else None,
        )

    mean = data.mean(dim=0)
    std = data.std(dim=0, unbiased=False).clamp_min(float(train_cfg.get("std_epsilon", 1.0e-6)))
    normalized = (data - mean) / std
    dataset = torch.utils.data.TensorDataset(normalized)
    sampler = None
    seed = int(_nested_get(config, ("experiment", "seed"), 17))
    if runtime["distributed"]:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        sampler=sampler,
        shuffle=sampler is None,
        pin_memory=runtime["device_type"] == "cuda",
    )
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=float(train_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    beta = float(train_cfg.get("kl_beta", 1.0e-4))
    log_every = max(1, int(train_cfg.get("log_every", 100)))
    steps = max(1, max_steps)
    step = 0
    epoch = 0
    last_log = {}
    while step < steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for (batch,) in loader:
            step += 1
            batch = batch.to(device, non_blocking=True)
            reconstruction, mu, logvar, _ = vae(batch)
            loss, reconstruction_mse, kl = vae_loss(
                reconstruction,
                batch,
                mu,
                logvar,
                beta=beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if rank == 0 and (step == 1 or step == steps or step % log_every == 0):
                last_log = {
                    "component": component,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "reconstruction_mse": float(reconstruction_mse.detach().cpu()),
                    "kl": float(kl.detach().cpu()),
                }
                print(json.dumps(last_log, sort_keys=True))
                train_entry._wandb_log(
                    wandb_run,
                    {
                        f"{component}/loss": last_log["loss"],
                        f"{component}/reconstruction_mse": last_log["reconstruction_mse"],
                        f"{component}/kl": last_log["kl"],
                    },
                    step=step,
                )
            if step >= steps:
                break
        epoch += 1

    if runtime["distributed"]:
        torch.distributed.barrier()
    if rank != 0:
        return {}

    with torch.no_grad():
        full = normalized.to(device)
        reconstruction, mu, logvar, _ = vae(full, sample=False)
        final_loss, final_reconstruction_mse, final_kl = vae_loss(
            reconstruction,
            full,
            mu,
            logvar,
            beta=beta,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"
    report = {
        "component": component,
        "checkpoint": str(checkpoint),
        "report": str(report_path),
        "input_dim": int(data.shape[1]),
        "sample_count": int(data.shape[0]),
        "latent_dim": int(model_cfg.get("latent_dim", 128)),
        "hidden_dims": list(int(value) for value in model_cfg.get("hidden_dims", [256, 256])),
        "kl_beta": beta,
        "max_steps": steps,
        "batch_size": batch_size,
        "learning_rate": float(train_cfg.get("learning_rate", 3.0e-4)),
        "final_loss": float(final_loss.detach().cpu()),
        "final_reconstruction_mse": float(final_reconstruction_mse.detach().cpu()),
        "final_kl": float(final_kl.detach().cpu()),
        "last_log": last_log,
    }
    torch.save(
        {
            "component": component,
            "model_state_dict": train_entry._unwrap_model(vae).state_dict(),
            "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "report": report,
        },
        checkpoint,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_entry._wandb_save(wandb_run, checkpoint)
    train_entry._wandb_save(wandb_run, report_path)
    return report


def _component_tensors(
    torch,
    *,
    samples: list[dict[str, Any]],
    observation_spec: ObservationSpec,
) -> dict[str, Any]:
    observations = torch.tensor([sample["observation"] for sample in samples], dtype=torch.float32)
    targets = torch.tensor([sample["target_joints"] for sample in samples], dtype=torch.float32)
    source_dim = observation_spec.source_feature_dim()
    morphology_dim = observation_spec.morphology_dim()
    skeleton = observations[:, source_dim : source_dim + morphology_dim]
    if skeleton.shape[1] == 0:
        skeleton = observations.new_zeros((observations.shape[0], 1))
    return {
        "motion": observations[:, :source_dim],
        "skeleton": skeleton,
        "action": targets,
    }


def _components_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    payload = _nested_get(config, ("pretrain", "components"), list(COMPONENTS))
    if isinstance(payload, str):
        return (payload,)
    return tuple(str(item) for item in payload)


def _validate_components(components: tuple[str, ...]) -> None:
    unknown = [component for component in components if component not in COMPONENTS]
    if unknown:
        raise ValueError(f"unknown token VAE components: {unknown}")


def _load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _nested_get(mapping: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _apply_wandb_mode_override(config: dict[str, Any], wandb_mode: str | None) -> dict[str, Any]:
    if not wandb_mode:
        return config
    updated = dict(config)
    tracking = updated.get("tracking", {})
    tracking = dict(tracking) if isinstance(tracking, dict) else {}
    tracking["wandb_mode"] = wandb_mode
    updated["tracking"] = tracking
    return updated


def _path_from_config(config: dict[str, Any], path: tuple[str, ...]) -> Path | None:
    value = _nested_get(config, path, "")
    return Path(str(value)) if value else None


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(result.strip())
    except Exception:
        return False


if __name__ == "__main__":
    main()
