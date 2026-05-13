#!/usr/bin/env python3
"""Benchmark direct retargeter forward latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_retarget.data.schema import ObservationSpec, OutputSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--hidden-dims", default="512,512,256")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    observation = ObservationSpec()
    output = OutputSpec()
    payload = {
        "observation_dim": observation.flattened_dim(),
        "output_dim": output.output_dim(),
        "device": args.device,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "hidden_dims": _hidden_dims(args.hidden_dims),
    }

    if args.dry_run:
        _emit(payload, args.output_json)
        return

    try:
        import torch
        from online_retarget.models.mlp import OnlineRetargetMLP
    except ImportError as exc:
        payload["blocked"] = "torch is required for latency benchmarking"
        _emit(payload, args.output_json)
        raise SystemExit(payload["blocked"]) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        payload["blocked"] = "CUDA device is not available"
        _emit(payload, args.output_json)
        raise SystemExit(payload["blocked"])

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model = OnlineRetargetMLP(
        input_dim=observation.flattened_dim(),
        output_dim=output.output_dim(),
        hidden_dims=tuple(payload["hidden_dims"]),
    ).to(device=device, dtype=dtype)
    model.eval()
    x = torch.zeros((1, observation.flattened_dim()), device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies = []
        for _ in range(args.iters):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

    payload.update(
        {
            "param_count": sum(param.numel() for param in model.parameters()),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "p99_ms": _percentile(latencies, 0.99),
            "mean_ms": statistics.mean(latencies),
            "max_ms": max(latencies),
        }
    )
    _emit(payload, args.output_json)


def _hidden_dims(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _emit(payload: dict[str, object], output_json: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
