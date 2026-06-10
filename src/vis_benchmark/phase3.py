from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import threading
from time import perf_counter, sleep
from typing import Any, Mapping

from vis_core import RenderRequest, run_static_packet


BENCHMARK_SCHEMA_VERSION = "phase3_vis_benchmark/v0.1"
ADAPTER_CHOICES = ("none", "isaac_render", "somamesh_source_render")


@dataclass(frozen=True)
class BenchmarkConfig:
    manifest_path: Path | None
    output_dir: Path
    adapter: str = "none"
    packets: int = 1
    workers: int = 1
    gpu_devices: tuple[str, ...] = ()
    dry_run: bool = False
    synthetic_smoke: bool = False
    timeout_sec: float | None = None
    sample_interval_sec: float = 0.25


def run_phase3_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    _validate_config(config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _resolve_manifest_path(config, output_dir=output_dir)

    if config.dry_run:
        return _run_dry_plan(config, manifest_path)

    monitor = _ResourceMonitor(sample_interval_sec=config.sample_interval_sec)
    batch_start = perf_counter()
    monitor.start()
    try:
        runs = _run_packet_batch(config, manifest_path)
    finally:
        resource_metrics = monitor.stop()
    batch_wall_sec = perf_counter() - batch_start

    completed = [run for run in runs if run["status"] == "ok"]
    failed = [run for run in runs if run["status"] != "ok"]
    report = _base_report(config, manifest_path)
    report.update(
        {
            "status": "failed" if failed else "ok",
            "runs": runs,
            "stage_times_sec": _stage_times(completed, dry_run=False, adapter=config.adapter),
            "resource_metrics": resource_metrics,
            "throughput": _throughput(completed, batch_wall_sec),
            "concurrency": _concurrency(config, runs, batch_wall_sec),
            "failures": failed,
        }
    )
    return report


def write_synthetic_packet(root: str | Path) -> Path:
    packet_root = Path(root)
    packet_root.mkdir(parents=True, exist_ok=True)
    (packet_root / "source.bvh").write_text("HIERARCHY\n", encoding="utf-8")
    (packet_root / "human.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"root_pos": [0, 0, 0], "root_rot": [0, 0, 0, 1], "pelvis": [0, 0, 0]},
                    {"root_pos": [0, 0, 1], "root_rot": [0, 0, 0, 1], "pelvis": [0, 0, 1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_csv(packet_root / "target.csv")
    _write_csv(packet_root / "play.csv")
    manifest_path = packet_root / "packet.json"
    manifest_path.write_text(json.dumps(_synthetic_manifest_payload()), encoding="utf-8")
    return manifest_path


def _run_dry_plan(config: BenchmarkConfig, manifest_path: Path) -> dict[str, Any]:
    static_result = run_static_packet(manifest_path, renderer=None)
    adapter_plan = _adapter_plan(config, static_result.loaded_packet)
    report = _base_report(config, manifest_path)
    report.update(
        {
            "status": "dry_run",
            "runs": [
                {
                    "packet_index": 0,
                    "status": "dry_run",
                    "diagnostics": static_result.diagnostics.as_dict(),
                }
            ],
            "adapter_plan": adapter_plan,
            "stage_times_sec": _stage_times(
                [{"diagnostics": static_result.diagnostics.as_dict()}],
                dry_run=True,
                adapter=config.adapter,
            ),
            "resource_metrics": {
                "observed": {},
                "unavailable": {
                    "gpu_utilization_percent": "dry_run_does_not_execute_renderer",
                    "vram_peak_mb": "dry_run_does_not_execute_renderer",
                    "cpu_ram_io": "dry_run_does_not_execute_packet_batch",
                    "worker_idle_sec": "dry_run_does_not_execute_packet_batch",
                },
            },
            "throughput": {
                "observed": {},
                "unavailable": {
                    "effective_fps": "dry_run_does_not_execute_renderer",
                    "packets_per_hour": "dry_run_does_not_execute_packet_batch",
                },
            },
            "concurrency": {
                "observed": {
                    "requested_packets": config.packets,
                    "workers": config.workers,
                    "gpu_devices_requested": list(config.gpu_devices),
                    "mode": "dry_run",
                },
                "unavailable": {
                    "packet_concurrency": "dry_run_does_not_execute_packet_batch",
                    "multi_gpu_assignment": "dry_run_does_not_execute_packet_batch",
                },
            },
        }
    )
    return report


def _run_packet_batch(config: BenchmarkConfig, manifest_path: Path) -> list[dict[str, Any]]:
    if config.workers <= 1:
        return [
            _run_one_packet_worker(
                str(manifest_path),
                config.adapter,
                str(config.output_dir / f"packet_{idx:04d}"),
                idx,
                config.timeout_sec,
                _gpu_device_for_index(config, idx),
            )
            for idx in range(config.packets)
        ]

    runs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                _run_one_packet_worker,
                str(manifest_path),
                config.adapter,
                str(config.output_dir / f"packet_{idx:04d}"),
                idx,
                config.timeout_sec,
                _gpu_device_for_index(config, idx),
            ): idx
            for idx in range(config.packets)
        }
        for future in as_completed(futures):
            try:
                runs.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive process-pool boundary
                runs.append(
                    {
                        "packet_index": futures[future],
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    return sorted(runs, key=lambda run: int(run.get("packet_index", -1)))


def _run_one_packet_worker(
    manifest_path: str,
    adapter_name: str,
    output_dir: str,
    packet_index: int,
    timeout_sec: float | None,
    gpu_device: str | None,
) -> dict[str, Any]:
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
    try:
        packet_output_dir = Path(output_dir)
        packet_output_dir.mkdir(parents=True, exist_ok=True)
        renderer = _renderer(adapter_name, timeout_sec=timeout_sec)
        result = run_static_packet(manifest_path, renderer=renderer, output_dir=packet_output_dir)
        adapter_report = _load_adapter_report(result.diagnostics.adapter_diagnostics)
        return {
            "packet_index": packet_index,
            "status": "ok",
            "gpu_device": gpu_device,
            "output_dir": str(packet_output_dir),
            "diagnostics": result.diagnostics.as_dict(),
            "adapter_report": adapter_report,
        }
    except Exception as exc:
        diagnostics = getattr(exc, "diagnostics", None)
        return {
            "packet_index": packet_index,
            "status": "failed",
            "gpu_device": gpu_device,
            "output_dir": output_dir,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "diagnostics": diagnostics if isinstance(diagnostics, Mapping) else {},
        }
    finally:
        if previous_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible_devices


def _renderer(adapter_name: str, *, timeout_sec: float | None) -> Any:
    if adapter_name == "none":
        return None
    if adapter_name == "isaac_render":
        from vis_adapters import IsaacRenderAdapter

        return IsaacRenderAdapter(timeout_sec=timeout_sec)
    if adapter_name == "somamesh_source_render":
        from vis_adapters import SomaMeshSourceRenderAdapter

        return SomaMeshSourceRenderAdapter(timeout_sec=timeout_sec)
    raise ValueError(f"unsupported adapter: {adapter_name}")


def _adapter_plan(config: BenchmarkConfig, loaded_packet: Any) -> dict[str, Any]:
    if config.adapter == "none":
        return {
            "adapter": "none",
            "preflight": None,
            "command": None,
            "note": "static runner load/validate only; no render adapter selected",
        }
    request = RenderRequest(loaded_packet=loaded_packet, output_dir=config.output_dir / "dry_run")
    renderer = _renderer(config.adapter, timeout_sec=config.timeout_sec)
    preflight = renderer.preflight(request)
    command = None
    if preflight.ok:
        command = renderer.build_command(request).as_dict()
    return {
        "adapter": config.adapter,
        "preflight": preflight.as_dict(),
        "command": command,
    }


def _base_report(config: BenchmarkConfig, manifest_path: Path) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_scope": "synthetic_smoke" if config.synthetic_smoke else "baseline_packet",
        "performance_conclusion": None,
        "performance_conclusion_status": "not_reported_by_harness",
        "inputs": {
            "manifest_path": str(manifest_path),
            "output_dir": str(config.output_dir),
            "adapter": config.adapter,
            "packets": config.packets,
            "workers": config.workers,
            "gpu_devices": list(config.gpu_devices),
            "dry_run": config.dry_run,
            "synthetic_smoke": config.synthetic_smoke,
        },
        "requires_real_measurement": [
            (
                "GPU utilization and VRAM peak require a real 4090/5090 host with "
                "nvidia-smi visible during non-dry-run adapter execution."
            ),
            (
                "Credible effective FPS and packets/hour require a real baseline "
                "VisPacket and referenced BVH/CSV/JSON assets."
            ),
            (
                "Dedicated timeline-align and encode wall times require the selected "
                "renderer script to emit those stage timers; this harness does not infer them."
            ),
            (
                "Multi-GPU throughput requires real GPU devices plus renderer/runtime "
                "support for the assigned CUDA_VISIBLE_DEVICES values."
            ),
        ],
    }


def _stage_times(
    runs: list[dict[str, Any]],
    *,
    dry_run: bool,
    adapter: str,
) -> dict[str, Any]:
    diagnostics = [run.get("diagnostics", {}) for run in runs]
    observed = {
        "packet_load_wall_sec": _sum_diagnostic(diagnostics, "load_sec"),
        "packet_validate_wall_sec": _sum_diagnostic(diagnostics, "validate_sec"),
        "static_runner_wall_sec": _sum_diagnostic(diagnostics, "wall_sec"),
    }
    if not dry_run and adapter != "none":
        observed["adapter_render_wall_sec"] = _sum_diagnostic(diagnostics, "render_sec")

    unavailable = {
        "timeline_align_wall_sec": (
            "no dedicated timeline-align stage timer is emitted by the current static "
            "runner or render adapters"
        ),
        "encode_wall_sec": (
            "no dedicated encode stage timer is emitted by the current render scripts; "
            "adapter_render_wall_sec includes render subprocess wall time when an adapter runs"
        ),
    }
    if dry_run:
        unavailable["adapter_render_wall_sec"] = "dry_run_does_not_execute_renderer"
    elif adapter == "none":
        unavailable["adapter_render_wall_sec"] = "no_render_adapter_selected"
    return {"observed": observed, "unavailable": unavailable}


def _throughput(runs: list[dict[str, Any]], batch_wall_sec: float) -> dict[str, Any]:
    frame_count = sum(int(run["diagnostics"].get("frame_count", 0)) for run in runs)
    observed: dict[str, Any] = {
        "completed_packets": len(runs),
        "total_frames": frame_count,
        "batch_wall_sec": batch_wall_sec,
    }
    if batch_wall_sec > 0.0:
        observed["effective_fps"] = frame_count / batch_wall_sec
        observed["packets_per_hour"] = len(runs) * 3600.0 / batch_wall_sec
    return {
        "observed": observed,
        "unavailable": {},
        "interpretation": "measurement_of_this_run_only",
    }


def _concurrency(
    config: BenchmarkConfig,
    runs: list[dict[str, Any]],
    batch_wall_sec: float,
) -> dict[str, Any]:
    completed = [run for run in runs if run["status"] == "ok"]
    failed = [run for run in runs if run["status"] != "ok"]
    packet_wall_total = sum(
        float(run.get("diagnostics", {}).get("wall_sec", 0.0)) for run in completed
    )
    worker_slot_wall_sec = max(config.workers, 1) * batch_wall_sec
    observed = {
        "requested_packets": config.packets,
        "completed_packets": len(completed),
        "failed_packets": len(failed),
        "workers": config.workers,
        "mode": "single_process" if config.workers <= 1 else "multi_process",
        "gpu_devices_requested": list(config.gpu_devices),
        "gpu_device_assignments": [
            {"packet_index": run.get("packet_index"), "gpu_device": run.get("gpu_device")}
            for run in runs
        ],
        "worker_slot_wall_sec": worker_slot_wall_sec,
        "worker_busy_wall_sec": packet_wall_total,
        "worker_idle_sec": max(0.0, worker_slot_wall_sec - packet_wall_total),
    }
    if worker_slot_wall_sec > 0.0:
        observed["worker_utilization_ratio"] = min(packet_wall_total / worker_slot_wall_sec, 1.0)
    unavailable = {}
    if not config.gpu_devices:
        unavailable["multi_gpu_packet_concurrency"] = "no_gpu_devices_requested"
    return {"observed": observed, "unavailable": unavailable}


class _ResourceMonitor:
    def __init__(self, *, sample_interval_sec: float) -> None:
        self._sample_interval_sec = sample_interval_sec
        self._before_self: resource.struct_rusage | None = None
        self._before_children: resource.struct_rusage | None = None
        self._gpu_sampler = _NvidiaSmiSampler(sample_interval_sec=sample_interval_sec)

    def start(self) -> None:
        self._before_self = resource.getrusage(resource.RUSAGE_SELF)
        self._before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._gpu_sampler.start()

    def stop(self) -> dict[str, Any]:
        after_self = resource.getrusage(resource.RUSAGE_SELF)
        after_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        gpu_metrics = self._gpu_sampler.stop()
        before_self = self._before_self or after_self
        before_children = self._before_children or after_children
        observed = {
            "cpu_user_sec": _rusage_delta(after_self, before_self, "ru_utime")
            + _rusage_delta(after_children, before_children, "ru_utime"),
            "cpu_system_sec": _rusage_delta(after_self, before_self, "ru_stime")
            + _rusage_delta(after_children, before_children, "ru_stime"),
            "process_max_rss_mb": max(
                _rss_mb(after_self.ru_maxrss),
                _rss_mb(after_children.ru_maxrss),
            ),
            "io_input_blocks": int(
                _rusage_delta(after_self, before_self, "ru_inblock")
                + _rusage_delta(after_children, before_children, "ru_inblock")
            ),
            "io_output_blocks": int(
                _rusage_delta(after_self, before_self, "ru_oublock")
                + _rusage_delta(after_children, before_children, "ru_oublock")
            ),
        }
        observed.update(gpu_metrics["observed"])
        return {
            "observed": observed,
            "unavailable": gpu_metrics["unavailable"],
        }


class _NvidiaSmiSampler:
    def __init__(self, *, sample_interval_sec: float) -> None:
        self._sample_interval_sec = sample_interval_sec
        self._samples: list[dict[str, int]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvidia_smi = shutil.which("nvidia-smi")

    def start(self) -> None:
        if self._nvidia_smi is None:
            return
        self._sample_once()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._nvidia_smi is None:
            return {
                "observed": {},
                "unavailable": {
                    "gpu_utilization_percent": "nvidia-smi_not_found",
                    "vram_peak_mb": "nvidia-smi_not_found",
                },
            }
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._sample_interval_sec * 4.0))
        if not self._samples:
            return {
                "observed": {},
                "unavailable": {
                    "gpu_utilization_percent": "nvidia-smi_returned_no_samples",
                    "vram_peak_mb": "nvidia-smi_returned_no_samples",
                },
            }
        util_values = [sample["gpu_util_percent"] for sample in self._samples]
        vram_values = [sample["memory_used_mb"] for sample in self._samples]
        return {
            "observed": {
                "gpu_utilization_peak_percent": max(util_values),
                "gpu_utilization_avg_percent": statistics.fmean(util_values),
                "vram_peak_mb": max(vram_values),
                "vram_avg_mb": statistics.fmean(vram_values),
                "gpu_sample_count": len(self._samples),
                "gpu_devices_sampled": sorted({sample["gpu_index"] for sample in self._samples}),
            },
            "unavailable": {},
        }

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            sleep(self._sample_interval_sec)

    def _sample_once(self) -> None:
        assert self._nvidia_smi is not None
        result = subprocess.run(
            [
                self._nvidia_smi,
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode != 0:
            return
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                self._samples.append(
                    {
                        "gpu_index": int(parts[0]),
                        "gpu_util_percent": int(parts[1]),
                        "memory_used_mb": int(parts[2]),
                    }
                )
            except ValueError:
                continue


def _resolve_manifest_path(config: BenchmarkConfig, *, output_dir: Path) -> Path:
    if config.synthetic_smoke:
        return write_synthetic_packet(output_dir / "synthetic_packet")
    if config.manifest_path is None:
        raise ValueError("--manifest is required unless --synthetic-smoke is set")
    return config.manifest_path


def _validate_config(config: BenchmarkConfig) -> None:
    if config.adapter not in ADAPTER_CHOICES:
        raise ValueError(f"adapter must be one of: {', '.join(ADAPTER_CHOICES)}")
    if config.packets < 1:
        raise ValueError("packets must be >= 1")
    if config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.sample_interval_sec <= 0.0:
        raise ValueError("sample_interval_sec must be > 0")


def _gpu_device_for_index(config: BenchmarkConfig, index: int) -> str | None:
    if not config.gpu_devices:
        return None
    return config.gpu_devices[index % len(config.gpu_devices)]


def _sum_diagnostic(diagnostics: list[Mapping[str, Any]], key: str) -> float:
    return sum(float(item.get(key, 0.0)) for item in diagnostics)


def _load_adapter_report(adapter_diagnostics: Mapping[str, Any]) -> dict[str, Any] | None:
    command = adapter_diagnostics.get("command")
    if not isinstance(command, Mapping):
        return None
    raw_report_path = command.get("report_path")
    if not isinstance(raw_report_path, str) or not raw_report_path:
        return None
    report_path = Path(raw_report_path)
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _rusage_delta(
    after: resource.struct_rusage,
    before: resource.struct_rusage,
    field_name: str,
) -> float:
    return float(getattr(after, field_name) - getattr(before, field_name))


def _rss_mb(raw_rss: int) -> float:
    if sys.platform == "darwin":
        return raw_rss / (1024.0 * 1024.0)
    return raw_rss / 1024.0


def _write_csv(path: Path) -> None:
    rows = [
        {"root_pos": "0 0 0", "root_rot": "0 0 0 1", "pelvis": "0 0 0"},
        {"root_pos": "0 0 1", "root_rot": "0 0 0 1", "pelvis": "0 0 1"},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "VisPacket v0.1",
        "coordinate_standard": "isaac",
        "timeline": {"fps": 50, "dt": 0.02, "frame_count": 2, "sim_dt": 0.005, "dte": 4},
        "tracks": {
            "human": {
                "uri": "human.json",
                "format": "json",
                "coordinate": {
                    "standard": "soma_bvh",
                    "up_axis": "Y",
                    "forward_axis": "Z",
                    "handedness": "right",
                    "unit_length": "meter",
                    "unit_angle": "degree",
                    "root_rotation": "euler_xyz",
                },
                "joint_names": ["pelvis"],
            },
            "target_g1": {
                "uri": "target.csv",
                "format": "csv",
                "coordinate": "isaac",
                "joint_names": ["pelvis"],
            },
            "play_g1": {
                "uri": "play.csv",
                "format": "csv",
                "coordinate": "isaac",
                "joint_names": ["pelvis"],
            },
        },
        "render": {
            "enabled": True,
            "interface": "StaticRenderer",
            "config": {"source_bvh_uri": "source.bvh", "width": 320, "height": 240},
        },
        "physics": {"enabled": False},
        "diagnose": {"enabled": True},
    }
