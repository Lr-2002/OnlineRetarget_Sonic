from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .phase3 import BenchmarkConfig, run_phase3_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        adapter=args.adapter,
        packets=args.packets,
        workers=args.workers,
        gpu_devices=_parse_gpu_devices(args.gpu_devices),
        dry_run=args.dry_run,
        synthetic_smoke=args.synthetic_smoke,
        timeout_sec=args.timeout_sec,
        sample_interval_sec=args.sample_interval_sec,
    )
    report = run_phase3_benchmark(config)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    output_json = args.output_json or args.output_dir / "phase3_benchmark_report.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3 benchmark harness for VisPacket static/render adapters."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vis_phase3_benchmark"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--adapter",
        choices=("none", "isaac_render", "somamesh_source_render"),
        default="none",
    )
    parser.add_argument("--packets", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--gpu-devices",
        default="",
        help="Comma-separated CUDA_VISIBLE_DEVICES assignments, e.g. 0,1.",
    )
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--sample-interval-sec", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser


def _parse_gpu_devices(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
