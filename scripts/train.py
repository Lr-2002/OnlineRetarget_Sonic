#!/usr/bin/env python3
"""Training entry point for the direct-output baseline."""

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
    parser.add_argument("--action-column")
    parser.add_argument(
        "--allow-debug-data",
        action="store_true",
        help="Allow non-formal training on debug samples without the full M2Q quality gate.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    git_sha = _git_sha()
    config = _load_config(args.config)
    observation = ObservationSpec(
        history_frames=int(_nested_get(config, ("data", "history_frames"), 8))
    )
    output = OutputSpec(target=str(_nested_get(config, ("model", "output"), "g1_joint_position_delta")))
    index_csv = args.index_csv or _nested_get(config, ("data", "index_csv"), None)
    samples_jsonl = args.samples_jsonl or _nested_get(config, ("data", "samples_jsonl"), None)
    action_column = args.action_column or str(_nested_get(config, ("data", "action_column"), "curation_action"))
    quality_gate = _quality_gate_context(
        config,
        index_csv=Path(index_csv) if index_csv else None,
        samples_jsonl=Path(samples_jsonl) if samples_jsonl else None,
        quality_policy_id=args.quality_policy_id,
        quality_report=args.quality_report,
        action_column=action_column,
        allow_debug_data=args.allow_debug_data,
    )

    print(f"config={args.config}")
    print(f"rank={rank} world_size={world_size}")
    print(f"git_sha={git_sha}")
    print(f"git_dirty={_git_dirty()}")
    print(f"observation_dim={observation.flattened_dim()}")
    print(f"output_dim={output.output_dim()}")
    print(f"quality_gate={json.dumps(quality_gate, sort_keys=True)}")
    if index_csv:
        ref_count = 0
        ref_samples = []
        for ref in iter_motion_pair_refs(
            Path(index_csv),
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

    if not samples_jsonl:
        raise SystemExit(
            "Set --samples-jsonl or data.samples_jsonl to train. The current training loop "
            "consumes supervised JSONL artifacts produced by build-supervised-jsonl."
        )

    _train_jsonl(
        torch=torch,
        config=config,
        samples_jsonl=Path(samples_jsonl),
        output_dir=args.output_dir
        or Path(str(_nested_get(config, ("experiment", "output_root"), "runs")))
        / "train"
        / str(_nested_get(config, ("experiment", "name"), "baseline_mlp_direct_g1")),
        max_steps=args.max_steps or int(_nested_get(config, ("train", "max_steps"), 1000)),
        batch_size=args.batch_size or int(_nested_get(config, ("train", "batch_size"), 64)),
        learning_rate=float(_nested_get(config, ("train", "learning_rate"), 3e-4)),
        hidden_dims=tuple(int(value) for value in _nested_get(config, ("model", "hidden_dims"), [512, 512, 256])),
        dropout=float(_nested_get(config, ("model", "dropout"), 0.0)),
        quality_gate=quality_gate,
        rank=rank,
        world_size=world_size,
    )


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


def _quality_gate_context(
    config: dict[str, Any],
    *,
    index_csv: Path | None,
    samples_jsonl: Path | None,
    quality_policy_id: str | None = None,
    quality_report: Path | None = None,
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
    allow_debug = allow_debug_data or bool(_nested_get(config, ("data", "allow_debug_data"), False))

    index_text = str(effective_index or "")
    manifest_path = samples_jsonl.parent / "manifest.json" if samples_jsonl else None
    uses_curated_index = (
        "curated_index.csv" in Path(index_text).name
        or "/curated/" in index_text.replace("\\", "/")
    )
    uses_merged_action = effective_action_column == "merged_quality_action"
    report_exists = bool(report_path and report_path.exists())
    return {
        "policy_id": str(policy_id),
        "quality_report": str(report_path or ""),
        "quality_report_exists": report_exists,
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


def _path_or_none(value: str) -> Path | None:
    return Path(value) if value else None


def _train_jsonl(
    *,
    torch,
    config: dict[str, Any],
    samples_jsonl: Path,
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: tuple[int, ...],
    dropout: float,
    quality_gate: dict[str, Any],
    rank: int,
    world_size: int,
) -> None:
    from online_retarget.models.mlp import OnlineRetargetMLP

    samples = _load_supervised_samples(samples_jsonl)
    if not samples:
        raise SystemExit(f"no supervised samples found in {samples_jsonl}")
    input_dim = len(samples[0]["observation"])
    output_dim = len(samples[0]["target_joints"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([sample["observation"] for sample in samples], dtype=torch.float32, device=device)
    y = torch.tensor([sample["target_joints"] for sample in samples], dtype=torch.float32, device=device)

    seed = int(_nested_get(config, ("experiment", "seed"), 17))
    torch.manual_seed(seed)
    model = OnlineRetargetMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    wandb_run = _init_wandb(
        config=config,
        quality_gate=quality_gate,
        output_dir=output_dir,
        enabled=rank == 0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_every = max(1, int(_nested_get(config, ("train", "log_every"), 100)))
    steps = min(max_steps, max_steps if max_steps > 0 else 1)
    for step in range(1, steps + 1):
        indices = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),), device=device)
        pred = model(x[indices])
        loss = loss_fn(pred, y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if rank == 0 and (step == 1 or step == steps or step % log_every == 0):
            step_log = {"step": step, "loss": float(loss.detach().cpu())}
            print(json.dumps(step_log, sort_keys=True))
            _wandb_log(wandb_run, step_log, step=step)

    with torch.no_grad():
        full_pred = model(x)
        final_loss = float(loss_fn(full_pred, y).detach().cpu())
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
            config=EvaluationConfig(run_name="train_offline_eval"),
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
        hidden_dims=hidden_dims,
        dropout=dropout,
        quality_gate=quality_gate,
        device=str(device),
        world_size=world_size,
        rank=rank,
        final_train_mse=final_loss,
        wandb_summary=_wandb_summary(wandb_run),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
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


def _load_supervised_samples(samples_jsonl: Path) -> list[dict[str, Any]]:
    samples = []
    with samples_jsonl.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            sample = json.loads(stripped)
            if "observation" not in sample or "target_joints" not in sample:
                raise ValueError(f"sample on line {line_number} lacks observation/target_joints")
            samples.append(sample)
    return samples


def _write_prediction_jsonl(
    path: Path,
    *,
    samples: list[dict[str, Any]],
    predictions: list[list[float]],
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
                "predicted_joints": [prediction],
                "target_joints": [sample["target_joints"]],
            }
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")


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
        "quality_gate": quality_gate,
        "device": device,
        "world_size": world_size,
        "rank": rank,
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
    project = str(_nested_get(config, ("experiment", "project"), "online-retarget"))
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
