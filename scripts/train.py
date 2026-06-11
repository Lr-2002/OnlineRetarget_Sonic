#!/usr/bin/env python3
"""Training entry point for the direct-output baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - environment blocker path
    yaml = None

from online_retarget.data.schema import ObservationSpec, OutputSpec, iter_motion_pair_refs
from online_retarget.evaluation import EvaluationConfig, evaluate_jsonl


FORMAL_SAMPLE_BUILDER = "bvh_fk_30body_window"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--index-csv", type=Path)
    parser.add_argument("--samples-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quality-policy-id")
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--quality-policy-audit", type=Path)
    parser.add_argument("--action-column")
    parser.add_argument(
        "--allow-debug-data",
        action="store_true",
        help="Allow non-formal training on debug samples without the full M2Q quality gate.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--wandb-mode",
        help="Override tracking.wandb_mode, e.g. disabled, offline, or online.",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Load a checkpoint and write predictions/eval for --samples-jsonl without optimizing.",
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    git_sha = _git_sha()
    config = _apply_wandb_mode_override(_load_config(args.config), args.wandb_mode)
    index_csv = args.index_csv or _nested_get(config, ("data", "index_csv"), None)
    samples_jsonl = args.samples_jsonl or _nested_get(config, ("data", "samples_jsonl"), None)
    sample_manifest = _load_sample_manifest(Path(samples_jsonl) if samples_jsonl else None)
    observation = _observation_spec_from_config_and_manifest(config, sample_manifest)
    output = OutputSpec(target=str(_nested_get(config, ("model", "output"), "g1_joint_position")))
    reported_output_dim = _reported_output_dim(
        config,
        output_spec=output,
        sample_manifest=sample_manifest,
    )
    action_column = args.action_column or str(_nested_get(config, ("data", "action_column"), "curation_action"))
    quality_gate = _quality_gate_context(
        config,
        index_csv=Path(index_csv) if index_csv else None,
        samples_jsonl=Path(samples_jsonl) if samples_jsonl else None,
        quality_policy_id=args.quality_policy_id,
        quality_report=args.quality_report,
        quality_policy_audit=args.quality_policy_audit,
        action_column=action_column,
        allow_debug_data=args.allow_debug_data,
    )

    print(f"config={args.config}")
    print(f"rank={rank} world_size={world_size}")
    print(f"git_sha={git_sha}")
    print(f"git_dirty={_git_dirty()}")
    print(f"observation_dim={observation.flattened_dim()}")
    print(f"output_dim={reported_output_dim}")
    print(f"quality_gate={json.dumps(quality_gate, sort_keys=True)}")
    if index_csv:
        index_path = Path(index_csv)
        if not _index_supports_motion_pair_refs(index_path, action_column=action_column):
            print(
                "train_refs=not_applicable "
                f"index_csv={index_csv} reason=index lacks split/{action_column} motion-pair columns"
            )
        else:
            ref_count = 0
            ref_samples = []
            for ref in iter_motion_pair_refs(
                index_path,
                splits=("train",),
                actions=("keep", "downweight"),
                action_column=action_column,
            ):
                ref_count += 1
                if len(ref_samples) < args.limit:
                    ref_samples.append(ref)
            print(f"train_refs={ref_count} index_csv={index_csv}")
            for ref in ref_samples:
                print(json.dumps(ref.to_dict(), sort_keys=True))
    else:
        print("index_csv=unset")
    if samples_jsonl:
        print(f"samples_jsonl={samples_jsonl}")

    if args.dry_run:
        return

    _validate_quality_gate(quality_gate)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Training requires the conda environment from environment.yml with torch installed."
        ) from exc
    runtime = _setup_torch_runtime(torch, rank=rank, world_size=world_size)

    if not samples_jsonl:
        raise SystemExit(
            "Set --samples-jsonl or data.samples_jsonl to train. The current training loop "
            "consumes supervised JSONL artifacts produced by build-supervised-jsonl."
        )
    output_dir = args.output_dir or (
        Path(str(_nested_get(config, ("experiment", "output_root"), "runs")))
        / "train"
        / str(_nested_get(config, ("experiment", "name"), "baseline_mlp_direct_g1"))
    )
    if args.predict_only:
        if args.checkpoint is None:
            raise SystemExit("--predict-only requires --checkpoint")
        _predict_jsonl(
            torch=torch,
            config=config,
            samples_jsonl=Path(samples_jsonl),
            checkpoint=args.checkpoint,
            output_dir=output_dir,
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
        _cleanup_torch_runtime(torch, runtime)
        return

    try:
        _train_jsonl(
            torch=torch,
            config=config,
            samples_jsonl=Path(samples_jsonl),
            output_dir=output_dir,
            resume_checkpoint=args.checkpoint,
            max_steps=args.max_steps or int(_nested_get(config, ("train", "max_steps"), 1000)),
            batch_size=args.batch_size or int(_nested_get(config, ("train", "batch_size"), 64)),
            learning_rate=float(_nested_get(config, ("train", "learning_rate"), 3e-4)),
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
    finally:
        _cleanup_torch_runtime(torch, runtime)


def _index_supports_motion_pair_refs(index_csv: Path, *, action_column: str) -> bool:
    if not index_csv.exists():
        return False
    with index_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False
    columns = set(header)
    required = {
        "split",
        "actor_uid",
        "filename",
        "move_soma_proportional_path",
        "move_g1_path",
        action_column,
    }
    return required.issubset(columns)


def _load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read --config. Install the project environment "
            f"from environment.yml or pass a Python environment with pyyaml available: {path}"
        )
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


def _configured_model_family(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    key = str(model_cfg.get("family", "temporal_mlp")).lower().replace("-", "_")
    aliases = {
        "mlp": "temporal_mlp",
        "temporal_mlp": "temporal_mlp",
        "tf": "temporal_transformer",
        "transformer": "temporal_transformer",
        "temporal_transformer": "temporal_transformer",
        "token_tf": "token_transformer",
        "token_transformer": "token_transformer",
        "tokenized_transformer": "token_transformer",
        "cross_attention_transformer": "token_transformer",
        "fm": "flow_matching",
        "flow": "flow_matching",
        "flow_matching": "flow_matching",
        "dp": "diffusion_policy",
        "diffusion": "diffusion_policy",
        "diffusion_policy": "diffusion_policy",
        "dp_temporal": "temporal_diffusion_policy",
        "temporal_dp": "temporal_diffusion_policy",
        "temporal_diffusion": "temporal_diffusion_policy",
        "temporal_diffusion_policy": "temporal_diffusion_policy",
    }
    return aliases.get(key, key)


def _reported_output_dim(
    config: dict[str, Any],
    *,
    output_spec: OutputSpec,
    sample_manifest: dict[str, Any],
) -> int:
    manifest_output_dim = sample_manifest.get("output_dim")
    if isinstance(manifest_output_dim, int) and manifest_output_dim > 0:
        return manifest_output_dim
    try:
        horizon = int(_nested_get(config, ("data", "target_horizon_frames"), 1))
    except (TypeError, ValueError):
        horizon = 1
    return output_spec.output_dim() * max(1, horizon)


def _apply_wandb_mode_override(config: dict[str, Any], wandb_mode: str | None) -> dict[str, Any]:
    if not wandb_mode:
        return config
    updated = dict(config)
    tracking = updated.get("tracking", {})
    tracking = dict(tracking) if isinstance(tracking, dict) else {}
    tracking["wandb_mode"] = wandb_mode
    updated["tracking"] = tracking
    return updated


def _quality_gate_context(
    config: dict[str, Any],
    *,
    index_csv: Path | None,
    samples_jsonl: Path | None,
    quality_policy_id: str | None = None,
    quality_report: Path | None = None,
    quality_policy_audit: Path | None = None,
    action_column: str = "curation_action",
    allow_debug_data: bool = False,
) -> dict[str, Any]:
    """Collect the M2Q training gate inputs from config, CLI, and sample manifest."""

    manifest = _load_sample_manifest(samples_jsonl)
    manifest_config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    manifest_index = _path_or_none(str(manifest.get("index_csv", "")) if manifest else "")
    effective_index = index_csv or manifest_index
    effective_action_column = action_column or str(manifest_config.get("action_column", "curation_action"))
    if action_column == "curation_action" and manifest_config.get("action_column"):
        effective_action_column = str(manifest_config["action_column"])

    policy_id = quality_policy_id or _nested_get(config, ("data", "quality_policy_id"), "")
    report_path = quality_report or _path_or_none(str(_nested_get(config, ("data", "quality_report"), "")))
    audit_path = quality_policy_audit or _path_or_none(
        str(_nested_get(config, ("data", "quality_policy_audit"), ""))
    )
    audit_payload = _load_policy_audit(audit_path)
    allow_debug = allow_debug_data or bool(_nested_get(config, ("data", "allow_debug_data"), False))

    index_text = str(effective_index or "")
    manifest_path = samples_jsonl.parent / "manifest.json" if samples_jsonl else None
    uses_curated_index = (
        "curated_index.csv" in Path(index_text).name
        or "/curated/" in index_text.replace("\\", "/")
    )
    uses_merged_action = effective_action_column == "merged_quality_action"
    report_exists = bool(report_path and report_path.exists())
    audit_exists = bool(audit_path and audit_path.exists())
    audit_error = str(audit_payload.get("_error", "")) if audit_payload else ""
    audit_policy_id = str(audit_payload.get("policy_id", "")) if audit_payload else ""
    audit_promotable = bool(audit_payload.get("promotable", False)) if audit_payload else False
    audit_status = str(audit_payload.get("status", "")) if audit_payload else ""
    audit_blockers = audit_payload.get("blockers", []) if audit_payload else []
    if not isinstance(audit_blockers, list):
        audit_blockers = []
    return {
        "policy_id": str(policy_id),
        "quality_report": str(report_path or ""),
        "quality_report_exists": report_exists,
        "quality_policy_audit": str(audit_path or ""),
        "quality_policy_audit_exists": audit_exists,
        "quality_policy_audit_error": audit_error,
        "quality_policy_audit_policy_id": audit_policy_id,
        "quality_policy_audit_promotable": audit_promotable,
        "quality_policy_audit_status": audit_status,
        "quality_policy_audit_blockers": audit_blockers[:5],
        "index_csv": index_text,
        "samples_jsonl": str(samples_jsonl or ""),
        "samples_manifest": str(manifest_path or ""),
        "samples_manifest_exists": bool(manifest_path and manifest_path.exists()),
        "samples_builder": str(manifest.get("builder", "")) if manifest else "",
        "samples_builder_is_formal": manifest.get("builder") == FORMAL_SAMPLE_BUILDER if manifest else False,
        "formal_sample_builder": FORMAL_SAMPLE_BUILDER,
        "action_column": effective_action_column,
        "allow_debug_data": allow_debug,
        "uses_curated_index": uses_curated_index,
        "uses_merged_action": uses_merged_action,
        "status": "debug_override" if allow_debug else "formal_required",
    }


def _validate_quality_gate(context: dict[str, Any]) -> None:
    """Refuse formal optimization when M2Q provenance is missing."""

    if context.get("allow_debug_data"):
        print(
            "quality_gate_warning=debug data override enabled; "
            "this run is not a formal M2Q-gated training run."
        )
        return

    missing = []
    if not context.get("policy_id"):
        missing.append("data.quality_policy_id or --quality-policy-id")
    if not context.get("quality_report"):
        missing.append("data.quality_report or --quality-report")
    elif not context.get("quality_report_exists"):
        missing.append(f"existing quality report at {context.get('quality_report')}")
    if not context.get("quality_policy_audit"):
        missing.append("data.quality_policy_audit or --quality-policy-audit")
    elif not context.get("quality_policy_audit_exists"):
        missing.append(f"existing quality policy audit at {context.get('quality_policy_audit')}")
    elif context.get("quality_policy_audit_error"):
        missing.append(
            "readable quality policy audit "
            f"({context.get('quality_policy_audit_error')})"
        )
    elif not context.get("quality_policy_audit_promotable"):
        blockers = context.get("quality_policy_audit_blockers") or []
        blocker_text = f": {'; '.join(str(item) for item in blockers)}" if blockers else ""
        missing.append(f"promotable quality policy audit{blocker_text}")
    elif (
        context.get("policy_id")
        and context.get("quality_policy_audit_policy_id")
        and context.get("quality_policy_audit_policy_id") != context.get("policy_id")
    ):
        missing.append(
            "quality policy audit matching policy_id "
            f"{context.get('policy_id')} "
            f"(found {context.get('quality_policy_audit_policy_id')})"
        )
    if not context.get("uses_curated_index"):
        missing.append("curated index path generated by merge-quality")
    if not context.get("uses_merged_action"):
        missing.append("data.action_column=merged_quality_action or --action-column merged_quality_action")
    if context.get("samples_jsonl"):
        if not context.get("samples_manifest_exists"):
            missing.append("samples manifest next to supervised JSONL")
        elif not context.get("samples_builder_is_formal"):
            missing.append(
                "formal samples built by "
                f"{context.get('formal_sample_builder')} "
                f"(found {context.get('samples_builder') or 'unknown'})"
            )

    if missing:
        joined = "; ".join(missing)
        raise SystemExit(
            "Formal training quality gate failed. Missing: "
            f"{joined}. Use --allow-debug-data only for explicitly labeled debug runs."
        )


def _load_sample_manifest(samples_jsonl: Path | None) -> dict[str, Any]:
    if samples_jsonl is None:
        return {}
    manifest_path = samples_jsonl.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _load_policy_audit(audit_path: Path | None) -> dict[str, Any]:
    if audit_path is None or not audit_path.exists():
        return {}
    try:
        with audit_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_error": "audit JSON must be an object"}


def _path_or_none(value: str) -> Path | None:
    return Path(value) if value else None


def _observation_spec_from_config_and_manifest(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    input_dim: int | None = None,
) -> ObservationSpec:
    spec_payload = manifest.get("observation_spec", {})
    if not isinstance(spec_payload, dict):
        spec_payload = {}
    robot_payload = spec_payload.get("robot_state", {})
    robot_state = ObservationSpec().robot_state
    if isinstance(robot_payload, dict):
        robot_state = replace(
            robot_state,
            joint_dim=int(robot_payload.get("joint_dim", robot_state.joint_dim)),
            include_joint_position=bool(
                robot_payload.get("include_joint_position", robot_state.include_joint_position)
            ),
            include_joint_velocity=bool(
                robot_payload.get("include_joint_velocity", robot_state.include_joint_velocity)
            ),
            include_previous_action=bool(
                robot_payload.get("include_previous_action", robot_state.include_previous_action)
            ),
            include_imu_orientation=bool(
                robot_payload.get("include_imu_orientation", robot_state.include_imu_orientation)
            ),
            include_base_angular_velocity=bool(
                robot_payload.get(
                    "include_base_angular_velocity",
                    robot_state.include_base_angular_velocity,
                )
            ),
        )
    spec = ObservationSpec(
        history_frames=int(
            spec_payload.get(
                "history_frames",
                _nested_get(config, ("data", "history_frames"), 8),
            )
        ),
        source_body_count=int(
            spec_payload.get(
                "source_body_count",
                _nested_get(config, ("data", "source_body_count"), 30),
            )
        ),
        source_position_dim=int(spec_payload.get("source_position_dim", 3)),
        include_source_velocity=bool(spec_payload.get("include_source_velocity", True)),
        include_morphology=bool(spec_payload.get("include_morphology", True)),
        robot_state=robot_state,
    )
    if input_dim is None or spec.flattened_dim() == input_dim:
        return spec
    inferred = _infer_source_body_count(spec, input_dim)
    return replace(spec, source_body_count=inferred) if inferred else spec


def _infer_source_body_count(spec: ObservationSpec, input_dim: int) -> int | None:
    side_dim = spec.morphology_dim() + spec.robot_state_dim()
    source_dim = input_dim - side_dim
    per_body_per_frame = spec.source_position_dim * (2 if spec.include_source_velocity else 1)
    denom = spec.history_frames * per_body_per_frame
    if source_dim <= 0 or denom <= 0 or source_dim % denom != 0:
        return None
    return source_dim // denom


def _setup_torch_runtime(torch, *, rank: int, world_size: int) -> dict[str, Any]:
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    distributed = world_size > 1
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(local_rank if distributed else 0)
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")
    if distributed and not torch.distributed.is_initialized():
        backend = "nccl" if device_type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return {
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "device_type": device_type,
    }


def _cleanup_torch_runtime(torch, runtime: dict[str, Any]) -> None:
    if runtime.get("distributed") and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _runtime_report(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "distributed": bool(runtime.get("distributed")),
        "rank": int(runtime.get("rank", 0)),
        "world_size": int(runtime.get("world_size", 1)),
        "local_rank": int(runtime.get("local_rank", 0)),
        "device_type": str(runtime.get("device_type", "cpu")),
    }


def _train_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples_jsonl: Path,
    output_dir: Path,
    resume_checkpoint: Path | None,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    from online_retarget.models.registry import build_model

    samples = _load_supervised_samples(samples_jsonl)
    if not samples:
        raise SystemExit(f"no supervised samples found in {samples_jsonl}")
    if _configured_model_family(config) == "temporal_diffusion_policy":
        _train_temporal_diffusion_jsonl(
            torch=torch,
            config=config,
            samples=samples,
            samples_jsonl=samples_jsonl,
            output_dir=output_dir,
            resume_checkpoint=resume_checkpoint,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
        return
    input_dim = len(samples[0]["observation"])
    output_dim = len(_target_vector(samples[0]))
    observation_spec = _observation_spec_from_config_and_manifest(
        config,
        _load_sample_manifest(samples_jsonl),
        input_dim=input_dim,
    )
    device = runtime["device"]
    x = torch.tensor([sample["observation"] for sample in samples], dtype=torch.float32)
    y = torch.tensor([_target_vector(sample) for sample in samples], dtype=torch.float32)
    prev_y = torch.tensor(
        [_previous_target_vector(sample, output_dim) for sample in samples],
        dtype=torch.float32,
    )
    samples, x, y, prev_y, sample_filter = _filter_finite_supervised_tensors(
        torch,
        samples=samples,
        x=x,
        y=y,
        prev_y=prev_y,
    )
    if not samples:
        raise SystemExit(f"all supervised samples contain non-finite values: {samples_jsonl}")
    if rank == 0:
        print(f"sample_filter={json.dumps(sample_filter, sort_keys=True)}")

    seed = int(_nested_get(config, ("experiment", "seed"), 17))
    torch.manual_seed(seed)
    model_build = build_model(
        config,
        input_dim=input_dim,
        output_dim=output_dim,
        observation_spec=observation_spec,
    )
    model = model_build.model.to(device)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=device)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if state_dict is None:
            raise SystemExit(f"checkpoint lacks model_state_dict: {resume_checkpoint}")
        model.load_state_dict(state_dict)
    if runtime["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime["local_rank"]] if runtime["device_type"] == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    wandb_run = _init_wandb(
        config=config,
        quality_gate=quality_gate,
        output_dir=output_dir,
        enabled=rank == 0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_every = max(1, int(_nested_get(config, ("train", "log_every"), 100)))
    steps = min(max_steps, max_steps if max_steps > 0 else 1)
    dataset = torch.utils.data.TensorDataset(x, y, prev_y)
    sampler = None
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
    step = 0
    epoch = 0
    while step < steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_x, batch_y, batch_prev_y in loader:
            step += 1
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_prev_y = batch_prev_y.to(device, non_blocking=True)
            loss = _training_loss(
                torch,
                model,
                model_build.family,
                batch_x,
                batch_y,
                config,
                prev_target=batch_prev_y,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if rank == 0 and (step == 1 or step == steps or step % log_every == 0):
                step_log = {"step": step, "loss": float(loss.detach().cpu())}
                print(json.dumps(step_log, sort_keys=True))
                _wandb_log(wandb_run, step_log, step=step)
            if step >= steps:
                break
        epoch += 1

    if runtime["distributed"]:
        torch.distributed.barrier()
    if rank != 0:
        _wandb_finish(wandb_run)
        return

    with torch.no_grad():
        full_pred = _predict_tensor(
            torch,
            model,
            x,
            family=model_build.family,
            config=config,
            batch_size=batch_size,
            device=device,
            prev_y=prev_y,
        )
        final_loss = float(torch.nn.functional.mse_loss(full_pred, y.to(device)).detach().cpu())
    checkpoint = output_dir / "checkpoint.pt"
    predictions_jsonl = output_dir / "train_predictions.jsonl"
    _write_prediction_jsonl(
        predictions_jsonl,
        samples=samples,
        predictions=full_pred.detach().cpu().tolist(),
    )
    eval_result = None
    if bool(_nested_get(config, ("tracking", "auto_offline_eval"), True)):
        eval_result = evaluate_jsonl(
            input_jsonl=predictions_jsonl,
            output_root=output_dir,
            config=_evaluation_config(config, run_name="train_offline_eval"),
        )
    report = _build_train_report(
        samples_jsonl=samples_jsonl,
        output_dir=output_dir,
        checkpoint=checkpoint,
        predictions_jsonl=predictions_jsonl,
        offline_eval=eval_result.to_dict() if eval_result is not None else {},
        sample_count=len(samples),
        input_dim=input_dim,
        output_dim=output_dim,
        max_steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dims=tuple(int(value) for value in model_build.config.get("hidden_dims", [])),
        dropout=float(model_build.config.get("dropout", 0.0)),
        quality_gate=quality_gate,
        device=str(device),
        world_size=world_size,
        rank=rank,
        final_train_mse=final_loss,
        wandb_summary=_wandb_summary(wandb_run),
        model_family=model_build.family,
        model_config=model_build.config,
        loss_config=_loss_config(config),
        evaluation_config=_evaluation_config(config, run_name="train_offline_eval").to_dict(),
        distributed_runtime=_runtime_report(runtime),
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else "",
        sample_filter=sample_filter,
    )
    torch.save(
        {
            "model_state_dict": _unwrap_model(model).state_dict(),
            "report": report,
        },
        checkpoint,
    )
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _wandb_log(wandb_run, {"final_train_mse": final_loss})
    _wandb_save(wandb_run, checkpoint)
    _wandb_save(wandb_run, output_dir / "train_report.json")
    if eval_result is not None:
        _wandb_save(wandb_run, eval_result.summary_json)
    _wandb_finish(wandb_run)
    print(json.dumps(report, indent=2, sort_keys=True))


def _train_temporal_diffusion_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    samples_jsonl: Path,
    output_dir: Path,
    resume_checkpoint: Path | None,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    from online_retarget.models.registry import build_model

    input_dim = len(samples[0]["observation"])
    target_shape = _target_action_shape(samples[0])
    action_horizon, action_dim = target_shape
    observation_spec = _observation_spec_from_config_and_manifest(
        config,
        _load_sample_manifest(samples_jsonl),
        input_dim=input_dim,
    )
    tensors = _temporal_condition_tensors(torch, samples)
    samples, tensors, sample_filter = _filter_finite_temporal_tensors(
        torch,
        samples=samples,
        tensors=tensors,
    )
    if not samples:
        raise SystemExit(f"all temporal samples contain non-finite values: {samples_jsonl}")
    if rank == 0:
        print(f"sample_filter={json.dumps(sample_filter, sort_keys=True)}")

    seed = int(_nested_get(config, ("experiment", "seed"), 17))
    torch.manual_seed(seed)
    model_build = build_model(
        config,
        input_dim=input_dim,
        output_dim=action_dim,
        observation_spec=observation_spec,
    )
    model = model_build.model.to(runtime["device"])
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=runtime["device"])
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if state_dict is None:
            raise SystemExit(f"checkpoint lacks model_state_dict: {resume_checkpoint}")
        model.load_state_dict(state_dict)
    if runtime["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime["local_rank"]] if runtime["device_type"] == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    wandb_run = _init_wandb(
        config=config,
        quality_gate=quality_gate,
        output_dir=output_dir,
        enabled=rank == 0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_every = max(1, int(_nested_get(config, ("train", "log_every"), 100)))
    steps = min(max_steps, max_steps if max_steps > 0 else 1)
    dataset = torch.utils.data.TensorDataset(
        tensors["source_body_tokens"],
        tensors["source_skeleton"],
        tensors["morphology"],
        tensors["robot_state"],
        tensors["prev_action"],
        tensors["target_action"],
    )
    sampler = None
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
    step = 0
    epoch = 0
    while step < steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            step += 1
            condition = _temporal_batch_to_device(batch, runtime["device"])
            batch_y = condition.pop("target_action")
            loss = _training_loss(
                torch,
                model,
                model_build.family,
                condition,
                batch_y,
                config,
                prev_target=condition.get("prev_action"),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if rank == 0 and (step == 1 or step == steps or step % log_every == 0):
                step_log = {"step": step, "loss": float(loss.detach().cpu())}
                print(json.dumps(step_log, sort_keys=True))
                _wandb_log(wandb_run, step_log, step=step)
            if step >= steps:
                break
        epoch += 1

    if runtime["distributed"]:
        torch.distributed.barrier()
    if rank != 0:
        _wandb_finish(wandb_run)
        return

    with torch.no_grad():
        full_pred = _predict_tensor(
            torch,
            model,
            tensors,
            family=model_build.family,
            config=config,
            batch_size=batch_size,
            device=runtime["device"],
        )
        final_loss = float(
            torch.nn.functional.mse_loss(
                full_pred,
                tensors["target_action"].to(runtime["device"]),
            )
            .detach()
            .cpu()
        )
    checkpoint = output_dir / "checkpoint.pt"
    predictions_jsonl = output_dir / "train_predictions.jsonl"
    _write_prediction_jsonl(
        predictions_jsonl,
        samples=samples,
        predictions=full_pred.detach().cpu().tolist(),
    )
    eval_result = None
    if bool(_nested_get(config, ("tracking", "auto_offline_eval"), True)):
        eval_result = evaluate_jsonl(
            input_jsonl=predictions_jsonl,
            output_root=output_dir,
            config=_evaluation_config(config, run_name="train_offline_eval"),
        )
    report = _build_train_report(
        samples_jsonl=samples_jsonl,
        output_dir=output_dir,
        checkpoint=checkpoint,
        predictions_jsonl=predictions_jsonl,
        offline_eval=eval_result.to_dict() if eval_result is not None else {},
        sample_count=len(samples),
        input_dim=input_dim,
        output_dim=action_dim,
        max_steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dims=(),
        dropout=float(model_build.config.get("dropout", 0.0)),
        quality_gate=quality_gate,
        device=str(runtime["device"]),
        world_size=world_size,
        rank=rank,
        final_train_mse=final_loss,
        wandb_summary=_wandb_summary(wandb_run),
        model_family=model_build.family,
        model_config={
            **model_build.config,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
        },
        loss_config=_loss_config(config),
        evaluation_config=_evaluation_config(config, run_name="train_offline_eval").to_dict(),
        distributed_runtime=_runtime_report(runtime),
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else "",
        sample_filter=sample_filter,
    )
    torch.save(
        {
            "model_state_dict": _unwrap_model(model).state_dict(),
            "report": report,
        },
        checkpoint,
    )
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _wandb_log(wandb_run, {"final_train_mse": final_loss})
    _wandb_save(wandb_run, checkpoint)
    _wandb_save(wandb_run, output_dir / "train_report.json")
    if eval_result is not None:
        _wandb_save(wandb_run, eval_result.summary_json)
    _wandb_finish(wandb_run)
    print(json.dumps(report, indent=2, sort_keys=True))


def _training_loss(
    torch,
    model,
    family: str,
    observation,
    target,
    config: dict[str, Any],
    *,
    prev_target=None,
):
    if family == "flow_matching":
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], 1, device=target.device, dtype=target.dtype)
        state = (1.0 - time) * noise + time * target
        target_velocity = target - noise
        pred_velocity = model(observation, state, time)
        return torch.nn.functional.mse_loss(pred_velocity, target_velocity)
    if family == "diffusion_policy":
        weight = float(_loss_config(config).get("diffusion_policy", 1.0))
        return weight * _unwrap_model(model).diffusion_loss(observation, target)
    if family == "temporal_diffusion_policy":
        weight = float(_loss_config(config).get("temporal_diffusion_policy", 1.0))
        return weight * _unwrap_model(model).diffusion_loss(
            observation["source_body_tokens"],
            target,
            source_skeleton=observation.get("source_skeleton"),
            morphology=observation.get("morphology"),
            robot_state=observation.get("robot_state"),
            prev_action=observation.get("prev_action"),
        )
    if family == "token_transformer":
        unwrapped = _unwrap_model(model)
        if prev_target is None:
            prev_target = torch.zeros_like(target)
        prediction, aux = unwrapped.forward_with_aux(observation, prev_state=prev_target)
        return _supervised_loss(torch, prediction, target, config) + _auxiliary_token_loss(
            torch,
            aux,
            config,
        )
    return _supervised_loss(torch, model(observation), target, config)


def _supervised_loss(torch, prediction, target, config: dict[str, Any]):
    loss_cfg = _loss_config(config)
    total = prediction.new_tensor(0.0)
    total_weight = 0.0
    has_explicit_loss = any(key in loss_cfg for key in ("mse", "joint_position", "l1", "smooth_l1"))
    mse_weight = float(loss_cfg.get("mse", loss_cfg.get("joint_position", 0.0 if has_explicit_loss else 1.0)))
    if mse_weight:
        total = total + mse_weight * torch.nn.functional.mse_loss(prediction, target)
        total_weight += abs(mse_weight)
    l1_weight = float(loss_cfg.get("l1", 0.0))
    if l1_weight:
        total = total + l1_weight * torch.nn.functional.l1_loss(prediction, target)
        total_weight += abs(l1_weight)
    smooth_l1_weight = float(loss_cfg.get("smooth_l1", 0.0))
    if smooth_l1_weight:
        total = total + smooth_l1_weight * torch.nn.functional.smooth_l1_loss(prediction, target)
        total_weight += abs(smooth_l1_weight)
    if total_weight == 0.0:
        raise ValueError("loss config must enable at least one of mse, l1, smooth_l1")
    return total


def _auxiliary_token_loss(torch, aux: dict[str, Any], config: dict[str, Any]):
    loss_cfg = _loss_config(config)
    token_cfg = loss_cfg.get("token_autoencoder", {})
    if not isinstance(token_cfg, dict):
        token_cfg = {}
    total = aux["source"].new_tensor(0.0)
    weights = {
        "skeleton": float(token_cfg.get("skeleton", 0.0)),
        "motion": float(token_cfg.get("motion", 0.0)),
        "state": float(token_cfg.get("state", 0.0)),
        "latent_alignment": float(token_cfg.get("latent_alignment", 0.0)),
    }
    if weights["skeleton"] and aux["skeleton"].shape[-1] > 0:
        total = total + weights["skeleton"] * torch.nn.functional.mse_loss(
            aux["skeleton_reconstruction"],
            aux["skeleton"],
        )
    if weights["motion"]:
        total = total + weights["motion"] * torch.nn.functional.mse_loss(
            aux["motion_reconstruction"],
            aux["source"],
        )
    if weights["state"]:
        total = total + weights["state"] * torch.nn.functional.mse_loss(
            aux["state_reconstruction"],
            aux["prev_state"],
        )
    if weights["latent_alignment"]:
        total = total + weights["latent_alignment"] * torch.nn.functional.mse_loss(
            aux["z_motion"],
            aux["z_state"],
        )
    return total


def _predict_tensor(
    torch,
    model,
    x,
    *,
    family: str,
    config: dict[str, Any],
    batch_size: int,
    device,
    prev_y=None,
):
    model.eval()
    predictions = []
    if family == "temporal_diffusion_policy":
        count = x["source_body_tokens"].shape[0]
        diffusion_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
        for start in range(0, count, max(1, batch_size)):
            batch = {
                key: value[start : start + batch_size].to(device, non_blocking=True)
                for key, value in x.items()
                if key != "target_action"
            }
            pred = _unwrap_model(model).sample(
                batch["source_body_tokens"],
                source_skeleton=batch.get("source_skeleton"),
                morphology=batch.get("morphology"),
                robot_state=batch.get("robot_state"),
                prev_action=batch.get("prev_action"),
                steps=int(
                    diffusion_cfg.get(
                        "inference_steps",
                        diffusion_cfg.get("diffusion_steps", 32),
                    )
                ),
                start=str(diffusion_cfg.get("inference_start", "zeros")),
            )
            predictions.append(pred.detach())
        return torch.cat(predictions, dim=0)
    for start in range(0, x.shape[0], max(1, batch_size)):
        batch = x[start : start + batch_size].to(device, non_blocking=True)
        if family == "flow_matching":
            flow_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
            pred = _unwrap_model(model).sample(
                batch,
                steps=int(flow_cfg.get("inference_steps", 8)),
                start=str(flow_cfg.get("inference_start", "zeros")),
            )
        elif family == "diffusion_policy":
            diffusion_cfg = (
                config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
            )
            pred = _unwrap_model(model).sample(
                batch,
                steps=int(
                    diffusion_cfg.get(
                        "inference_steps",
                        diffusion_cfg.get("diffusion_steps", 32),
                    )
                ),
                start=str(diffusion_cfg.get("inference_start", "zeros")),
            )
        elif family == "token_transformer":
            if prev_y is None:
                prev_batch = torch.zeros(
                    batch.shape[0],
                    _unwrap_model(model).output_dim,
                    device=device,
                    dtype=batch.dtype,
                )
            else:
                prev_batch = prev_y[start : start + batch_size].to(device, non_blocking=True)
            pred = _unwrap_model(model)(batch, prev_state=prev_batch)
        else:
            pred = model(batch)
        predictions.append(pred.detach())
    return torch.cat(predictions, dim=0)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _target_vector(sample: dict[str, Any]) -> list[float]:
    future = sample.get("future_target_joints")
    if isinstance(future, list) and future and all(isinstance(row, list) for row in future):
        return [float(value) for row in future for value in row]
    target = sample.get("target_joints")
    if isinstance(target, list):
        return [float(value) for value in target]
    raise ValueError("sample lacks target_joints or future_target_joints")


def _previous_target_vector(sample: dict[str, Any], output_dim: int) -> list[float]:
    for key in ("prev_target_joints", "previous_target_joints", "prev_g1_joints"):
        value = sample.get(key)
        if isinstance(value, list) and len(value) == output_dim:
            return [float(item) for item in value]
        if isinstance(value, list) and value and output_dim % len(value) == 0:
            repeats = output_dim // len(value)
            return [float(item) for _ in range(repeats) for item in value]
    return [0.0] * output_dim


def _previous_target_joints(sample: dict[str, Any], output_dim: int) -> list[float]:
    return _previous_target_vector(sample, output_dim)


def _target_action_sequence(sample: dict[str, Any]) -> list[list[float]]:
    future = sample.get("future_target_joints")
    if isinstance(future, list) and future and all(isinstance(row, list) for row in future):
        return [[float(value) for value in row] for row in future]
    target = sample.get("target_joints")
    if isinstance(target, list):
        return [[float(value) for value in target]]
    raise ValueError("sample lacks target_joints or future_target_joints")


def _target_action_shape(sample: dict[str, Any]) -> tuple[int, int]:
    sequence = _target_action_sequence(sample)
    if not sequence or not sequence[0]:
        raise ValueError("target action sequence must be non-empty")
    return len(sequence), len(sequence[0])


def _temporal_condition_tensors(torch, samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must be non-empty")
    first_target_shape = _target_action_shape(samples[0])
    source_body_tokens = _source_body_token_sequence(samples[0])
    source_skeleton_dim = len(_source_skeleton_vector(samples[0], source_body_tokens))
    morphology_dim = len(_morphology_condition_vector(samples[0]))
    robot_state_dim = len(_robot_state_vector(samples[0]))
    action_dim = first_target_shape[1]
    rows = {
        "source_body_tokens": [],
        "source_skeleton": [],
        "morphology": [],
        "robot_state": [],
        "prev_action": [],
        "target_action": [],
    }
    for index, sample in enumerate(samples):
        target_action = _target_action_sequence(sample)
        shape = (len(target_action), len(target_action[0]) if target_action else 0)
        if shape != first_target_shape:
            raise ValueError(
                "temporal samples must share target shape "
                f"{first_target_shape}; sample {index} has {shape}"
            )
        source_tokens = _source_body_token_sequence(sample)
        if len(source_tokens) != first_target_shape[0]:
            raise ValueError(
                "source_body_tokens horizon must match future_target_joints; "
                f"sample {index} has {len(source_tokens)} vs {first_target_shape[0]}"
            )
        rows["source_body_tokens"].append(source_tokens)
        rows["source_skeleton"].append(
            _pad_or_trim(_source_skeleton_vector(sample, source_tokens), source_skeleton_dim)
        )
        rows["morphology"].append(_pad_or_trim(_morphology_condition_vector(sample), morphology_dim))
        rows["robot_state"].append(_pad_or_trim(_robot_state_vector(sample), robot_state_dim))
        rows["prev_action"].append(_pad_or_trim(_prev_action_vector(sample), action_dim))
        rows["target_action"].append(target_action)
    return {
        key: torch.tensor(value, dtype=torch.float32)
        for key, value in rows.items()
    }


def _source_body_token_sequence(sample: dict[str, Any]) -> list[list[list[float]]]:
    tokens = sample.get("source_body_tokens")
    if (
        isinstance(tokens, list)
        and tokens
        and all(isinstance(step, list) and step for step in tokens)
        and all(isinstance(body, list) for step in tokens for body in step)
    ):
        return [
            [[float(value) for value in body] for body in step]
            for step in tokens
        ]
    raise ValueError("temporal_diffusion_policy samples require source_body_tokens [T,N,D]")


def _source_skeleton_vector(
    sample: dict[str, Any],
    source_body_tokens: list[list[list[float]]] | None = None,
) -> list[float]:
    value = sample.get("source_skeleton")
    if isinstance(value, list):
        return [float(item) for item in value]
    body_count = len(source_body_tokens[0]) if source_body_tokens else 0
    return [0.0] * body_count * 4


def _morphology_condition_vector(sample: dict[str, Any]) -> list[float]:
    value = sample.get("morphology")
    if isinstance(value, list):
        return [float(item) for item in value]
    return [0.0] * 13


def _robot_state_vector(sample: dict[str, Any]) -> list[float]:
    value = sample.get("robot_state")
    if isinstance(value, list):
        return [float(item) for item in value]
    return [0.0] * 94


def _prev_action_vector(sample: dict[str, Any]) -> list[float]:
    for key in ("prev_target_joints", "previous_target_joints", "prev_g1_joints"):
        value = sample.get(key)
        if isinstance(value, list):
            return [float(item) for item in value]
    return []


def _pad_or_trim(values: list[float], width: int) -> list[float]:
    if len(values) >= width:
        return values[:width]
    return values + [0.0] * (width - len(values))


def _filter_finite_temporal_tensors(
    torch,
    *,
    samples: list[dict[str, Any]],
    tensors: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Drop JSONL rows with NaN/Inf values in any temporal tensor."""

    keep_mask = None
    finite_by_key = {}
    for key, tensor in tensors.items():
        finite = torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1)
        finite_by_key[key] = finite
        keep_mask = finite if keep_mask is None else keep_mask & finite
    if keep_mask is None:
        raise ValueError("no tensors to filter")
    dropped_indices = (~keep_mask).nonzero(as_tuple=False).flatten().tolist()
    report = {
        "input_count": len(samples),
        "filtered_count": int(keep_mask.sum().item()),
        "dropped_count": len(dropped_indices),
        "dropped_examples": [],
    }
    if not dropped_indices:
        return samples, tensors, report

    for index in dropped_indices[:20]:
        reasons = [
            f"{key}_nonfinite"
            for key, finite in finite_by_key.items()
            if not bool(finite[index])
        ]
        sample = samples[index]
        report["dropped_examples"].append(
            {
                "index": int(index),
                "sample_id": str(sample.get("sample_id", "")),
                "source_motion_path": str(sample.get("source_motion_path", "")),
                "target_g1_path": str(sample.get("target_g1_path", "")),
                "reasons": reasons,
            }
        )

    keep_indices = keep_mask.nonzero(as_tuple=False).flatten()
    filtered_samples = [samples[int(index)] for index in keep_indices.tolist()]
    filtered_tensors = {
        key: tensor.index_select(0, keep_indices)
        for key, tensor in tensors.items()
    }
    return filtered_samples, filtered_tensors, report


def _temporal_batch_to_device(batch, device) -> dict[str, Any]:
    keys = (
        "source_body_tokens",
        "source_skeleton",
        "morphology",
        "robot_state",
        "prev_action",
        "target_action",
    )
    return {
        key: value.to(device, non_blocking=True)
        for key, value in zip(keys, batch)
    }


def _filter_finite_supervised_tensors(
    torch,
    *,
    samples: list[dict[str, Any]],
    x,
    y,
    prev_y,
) -> tuple[list[dict[str, Any]], Any, Any, Any, dict[str, Any]]:
    """Drop JSONL rows with NaN/Inf tensors before they can poison optimization."""

    finite_x = torch.isfinite(x).all(dim=1)
    finite_y = torch.isfinite(y).all(dim=1)
    finite_prev_y = torch.isfinite(prev_y).all(dim=1)
    keep_mask = finite_x & finite_y & finite_prev_y
    dropped_indices = (~keep_mask).nonzero(as_tuple=False).flatten().tolist()
    report = {
        "input_count": len(samples),
        "filtered_count": int(keep_mask.sum().item()),
        "dropped_count": len(dropped_indices),
        "dropped_examples": [],
    }
    if not dropped_indices:
        return samples, x, y, prev_y, report

    for index in dropped_indices[:20]:
        reasons = []
        if not bool(finite_x[index]):
            reasons.append("observation_nonfinite")
        if not bool(finite_y[index]):
            reasons.append("target_joints_nonfinite")
        if not bool(finite_prev_y[index]):
            reasons.append("prev_target_joints_nonfinite")
        sample = samples[index]
        report["dropped_examples"].append(
            {
                "index": int(index),
                "sample_id": str(sample.get("sample_id", "")),
                "source_motion_path": str(sample.get("source_motion_path", "")),
                "target_g1_path": str(sample.get("target_g1_path", "")),
                "reasons": reasons,
            }
        )

    keep_indices = keep_mask.nonzero(as_tuple=False).flatten()
    filtered_samples = [samples[int(index)] for index in keep_indices.tolist()]
    return (
        filtered_samples,
        x.index_select(0, keep_indices),
        y.index_select(0, keep_indices),
        prev_y.index_select(0, keep_indices),
        report,
    )


def _loss_config(config: dict[str, Any]) -> dict[str, Any]:
    payload = config.get("loss", {})
    return dict(payload) if isinstance(payload, dict) else {}


def _evaluation_config(config: dict[str, Any], *, run_name: str) -> EvaluationConfig:
    payload = config.get("evaluation", {})
    if not isinstance(payload, dict):
        payload = {}
    metrics = payload.get("metrics", ())
    if isinstance(metrics, str):
        metric_tuple = (metrics,)
    else:
        metric_tuple = tuple(str(metric) for metric in metrics)
    return EvaluationConfig(
        metrics=metric_tuple,
        fps=float(payload.get("fps", 30.0)),
        joint_jump_velocity=float(payload.get("joint_jump_velocity", 20.0)),
        ground_height=float(payload.get("ground_height", 0.0)),
        up_axis=payload.get("up_axis", 2),
        contact_height_threshold=float(payload.get("contact_height_threshold", 0.04)),
        max_contact_slide_speed=float(payload.get("max_contact_slide_speed", 0.25)),
        max_contact_skate_distance=float(payload.get("max_contact_skate_distance", 0.02)),
        failure_metric=str(payload.get("failure_metric", "joint_rmse")),
        max_failures=int(payload.get("max_failures", 50)),
        run_name=run_name,
    )


def _load_supervised_samples(samples_jsonl: Path) -> list[dict[str, Any]]:
    samples = []
    with samples_jsonl.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            sample = json.loads(stripped)
            if "observation" not in sample:
                raise ValueError(f"sample on line {line_number} lacks observation")
            try:
                _target_vector(sample)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sample on line {line_number} lacks target_joints/future_target_joints"
                ) from exc
            samples.append(sample)
    return samples


def _predict_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples_jsonl: Path,
    checkpoint: Path,
    output_dir: Path,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    from online_retarget.models.registry import build_model

    if rank != 0:
        return
    samples = _load_supervised_samples(samples_jsonl)
    if not samples:
        raise SystemExit(f"no supervised samples found in {samples_jsonl}")
    if _configured_model_family(config) == "temporal_diffusion_policy":
        _predict_temporal_diffusion_jsonl(
            torch=torch,
            config=config,
            samples=samples,
            samples_jsonl=samples_jsonl,
            checkpoint=checkpoint,
            output_dir=output_dir,
            quality_gate=quality_gate,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
        return
    input_dim = len(samples[0]["observation"])
    output_dim = len(_target_vector(samples[0]))
    observation_spec = _observation_spec_from_config_and_manifest(
        config,
        _load_sample_manifest(samples_jsonl),
        input_dim=input_dim,
    )
    device = runtime["device"]
    model_build = build_model(
        config,
        input_dim=input_dim,
        output_dim=output_dim,
        observation_spec=observation_spec,
    )
    model = model_build.model.to(device)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if state_dict is None:
        raise SystemExit(f"checkpoint lacks model_state_dict: {checkpoint}")
    model.load_state_dict(state_dict)
    model.eval()
    x = torch.tensor([sample["observation"] for sample in samples], dtype=torch.float32)
    y = torch.tensor([_target_vector(sample) for sample in samples], dtype=torch.float32)
    prev_y = torch.tensor(
        [_previous_target_vector(sample, output_dim) for sample in samples],
        dtype=torch.float32,
    )
    samples, x, y, prev_y, sample_filter = _filter_finite_supervised_tensors(
        torch,
        samples=samples,
        x=x,
        y=y,
        prev_y=prev_y,
    )
    if not samples:
        raise SystemExit(f"all supervised samples contain non-finite values: {samples_jsonl}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        pred = _predict_tensor(
            torch,
            model,
            x,
            family=model_build.family,
            config=config,
            batch_size=int(_nested_get(config, ("train", "batch_size"), 64)),
            device=device,
            prev_y=prev_y,
        )
        predictions = pred.detach().cpu().tolist()
        final_mse = float(torch.nn.functional.mse_loss(pred, y.to(device)).detach().cpu())
    predictions_jsonl = output_dir / "predictions.jsonl"
    _write_prediction_jsonl(predictions_jsonl, samples=samples, predictions=predictions)
    eval_result = None
    if bool(_nested_get(config, ("tracking", "auto_offline_eval"), True)):
        eval_result = evaluate_jsonl(
            input_jsonl=predictions_jsonl,
            output_root=output_dir,
            config=_evaluation_config(config, run_name="offline_eval"),
        )
    report = {
        "mode": "predict_only",
        "samples_jsonl": str(samples_jsonl),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_jsonl),
        "offline_eval": eval_result.to_dict() if eval_result is not None else {},
        "sample_count": len(samples),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "model_family": model_build.family,
        "model_config": model_build.config,
        "loss_config": _loss_config(config),
        "evaluation_config": _evaluation_config(config, run_name="offline_eval").to_dict(),
        "sample_filter": sample_filter,
        "quality_gate": quality_gate,
        "device": str(device),
        "world_size": world_size,
        "rank": rank,
        "mse": final_mse,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    (output_dir / "predict_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _predict_temporal_diffusion_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    samples_jsonl: Path,
    checkpoint: Path,
    output_dir: Path,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
    runtime: dict[str, Any],
) -> None:
    from online_retarget.models.registry import build_model

    if rank != 0:
        return
    input_dim = len(samples[0]["observation"])
    action_horizon, action_dim = _target_action_shape(samples[0])
    observation_spec = _observation_spec_from_config_and_manifest(
        config,
        _load_sample_manifest(samples_jsonl),
        input_dim=input_dim,
    )
    device = runtime["device"]
    model_build = build_model(
        config,
        input_dim=input_dim,
        output_dim=action_dim,
        observation_spec=observation_spec,
    )
    model = model_build.model.to(device)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if state_dict is None:
        raise SystemExit(f"checkpoint lacks model_state_dict: {checkpoint}")
    model.load_state_dict(state_dict)
    model.eval()
    tensors = _temporal_condition_tensors(torch, samples)
    samples, tensors, sample_filter = _filter_finite_temporal_tensors(
        torch,
        samples=samples,
        tensors=tensors,
    )
    if not samples:
        raise SystemExit(f"all temporal samples contain non-finite values: {samples_jsonl}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        pred = _predict_tensor(
            torch,
            model,
            tensors,
            family=model_build.family,
            config=config,
            batch_size=int(_nested_get(config, ("train", "batch_size"), 64)),
            device=device,
        )
        predictions = pred.detach().cpu().tolist()
        final_mse = float(
            torch.nn.functional.mse_loss(
                pred,
                tensors["target_action"].to(device),
            )
            .detach()
            .cpu()
        )
    predictions_jsonl = output_dir / "predictions.jsonl"
    _write_prediction_jsonl(predictions_jsonl, samples=samples, predictions=predictions)
    eval_result = None
    if bool(_nested_get(config, ("tracking", "auto_offline_eval"), True)):
        eval_result = evaluate_jsonl(
            input_jsonl=predictions_jsonl,
            output_root=output_dir,
            config=_evaluation_config(config, run_name="offline_eval"),
        )
    report = {
        "mode": "predict_only",
        "samples_jsonl": str(samples_jsonl),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_jsonl),
        "offline_eval": eval_result.to_dict() if eval_result is not None else {},
        "sample_count": len(samples),
        "input_dim": input_dim,
        "output_dim": action_dim,
        "model_family": model_build.family,
        "model_config": {
            **model_build.config,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
        },
        "loss_config": _loss_config(config),
        "evaluation_config": _evaluation_config(config, run_name="offline_eval").to_dict(),
        "sample_filter": sample_filter,
        "quality_gate": quality_gate,
        "device": str(device),
        "world_size": world_size,
        "rank": rank,
        "mse": final_mse,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    (output_dir / "predict_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _write_prediction_jsonl(
    path: Path,
    *,
    samples: list[dict[str, Any]],
    predictions: list[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample, prediction in zip(samples, predictions):
            payload = {
                "sample_id": sample.get("sample_id", ""),
                "actor_uid": sample.get("actor_uid", ""),
                "category": sample.get("category", ""),
                "package": sample.get("package", ""),
                "quality_flags": sample.get("quality_flags", []),
                "predicted_joints": _prediction_sequence(sample, prediction),
                "target_joints": _target_sequence(sample),
            }
            for key in (
                "fps",
                "target_frame",
                "target_frame_indices",
                "target_horizon_frames",
                "source_motion_path",
                "target_g1_path",
                "sonic_relative_path",
                "target_joint_names",
            ):
                if key in sample:
                    payload[key] = sample[key]
            payload["sequence_id"] = str(
                sample.get(
                    "target_g1_path",
                    sample.get("sonic_relative_path", sample.get("source_motion_path", "")),
                )
            )
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")


def _target_sequence(sample: dict[str, Any]) -> list[list[float]]:
    future = sample.get("future_target_joints")
    if isinstance(future, list) and future and all(isinstance(row, list) for row in future):
        return [[float(value) for value in row] for row in future]
    return [[float(value) for value in sample["target_joints"]]]


def _prediction_sequence(sample: dict[str, Any], prediction: Any) -> list[list[float]]:
    if (
        isinstance(prediction, list)
        and prediction
        and all(isinstance(row, list) for row in prediction)
    ):
        return [[float(value) for value in row] for row in prediction]
    target = _target_sequence(sample)
    horizon = len(target)
    joint_dim = len(target[0]) if target else len(prediction)
    if horizon > 0 and joint_dim > 0 and len(prediction) == horizon * joint_dim:
        return [
            [float(value) for value in prediction[index * joint_dim : (index + 1) * joint_dim]]
            for index in range(horizon)
        ]
    return [[float(value) for value in prediction]]


def _build_train_report(
    *,
    samples_jsonl: Path,
    output_dir: Path,
    checkpoint: Path,
    predictions_jsonl: Path,
    offline_eval: dict[str, Any],
    sample_count: int,
    input_dim: int,
    output_dim: int,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: tuple[int, ...],
    dropout: float,
    quality_gate: dict[str, Any],
    device: str,
    world_size: int,
    rank: int,
    final_train_mse: float,
    wandb_summary: dict[str, Any],
    model_family: str = "temporal_mlp",
    model_config: dict[str, Any] | None = None,
    loss_config: dict[str, Any] | None = None,
    evaluation_config: dict[str, Any] | None = None,
    distributed_runtime: dict[str, Any] | None = None,
    resume_checkpoint: str = "",
    sample_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "samples_jsonl": str(samples_jsonl),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "predictions_jsonl": str(predictions_jsonl),
        "offline_eval": offline_eval,
        "sample_count": sample_count,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "model_family": model_family,
        "model_config": model_config or {},
        "loss_config": loss_config or {},
        "evaluation_config": evaluation_config or {},
        "quality_gate": quality_gate,
        "device": device,
        "world_size": world_size,
        "rank": rank,
        "distributed_runtime": distributed_runtime or {},
        "resume_checkpoint": resume_checkpoint,
        "sample_filter": sample_filter
        or {
            "input_count": sample_count,
            "filtered_count": sample_count,
            "dropped_count": 0,
            "dropped_examples": [],
        },
        "final_train_mse": final_train_mse,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "wandb": wandb_summary,
    }


def _init_wandb(
    *,
    config: dict[str, Any],
    quality_gate: dict[str, Any],
    output_dir: Path,
    enabled: bool,
):
    mode = str(_nested_get(config, ("tracking", "wandb_mode"), os.environ.get("WANDB_MODE", "disabled")))
    if not enabled or mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    project = str(_nested_get(config, ("experiment", "project"), "OnlineRetarget"))
    name = str(_nested_get(config, ("experiment", "name"), "baseline_mlp_direct_g1"))
    return wandb.init(
        project=project,
        name=name,
        mode=mode,
        dir=str(output_dir),
        config={
            "config": config,
            "quality_gate": quality_gate,
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
        },
    )


def _wandb_log(run, payload: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(payload, step=step)


def _wandb_save(run, path: Path) -> None:
    if run is not None and path.exists():
        run.save(str(path))


def _wandb_finish(run) -> None:
    if run is not None:
        run.finish()


def _wandb_summary(run) -> dict[str, Any]:
    if run is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "project": getattr(run, "project", ""),
        "name": getattr(run, "name", ""),
        "id": getattr(run, "id", ""),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(result.strip())
    except Exception:
        return False


if __name__ == "__main__":
    main()
