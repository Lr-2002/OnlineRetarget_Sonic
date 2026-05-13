"""Build small supervised JSONL samples for baseline debugging.

This builder intentionally uses raw BVH channels for early data-path validation.
It is not the final 30-body observation contract from :mod:`schema`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
from pathlib import Path
import subprocess
import tarfile
from typing import Mapping, Sequence

from .bones_seed import G1_JOINT_COLUMNS
from .bvh_quality import parse_bvh_text
from .schema import MotionPairRef, iter_motion_pair_refs


@dataclass(frozen=True)
class SupervisedBuildConfig:
    split: str = "train"
    actions: tuple[str, ...] = ("keep", "downweight")
    action_column: str = "curation_action"
    limit: int = 16
    history_frames: int = 8
    target_frame_offset: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupervisedBuildResult:
    output_dir: Path
    samples_jsonl: Path
    manifest_json: Path
    sample_count: int
    skipped_count: int
    input_dim: int
    output_dim: int
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["samples_jsonl"] = str(self.samples_jsonl)
        payload["manifest_json"] = str(self.manifest_json)
        return payload


def build_supervised_jsonl(
    data_root: Path,
    index_csv: Path,
    output_root: Path,
    config: SupervisedBuildConfig | None = None,
) -> SupervisedBuildResult:
    """Build fixed-window source-channel to G1-joint JSONL samples."""

    config = config or SupervisedBuildConfig()
    refs = iter_motion_pair_refs(
        index_csv,
        splits=(config.split,),
        actions=config.actions,
        action_column=config.action_column,
    )
    output_dir = output_root.expanduser() / "supervised" / _run_name(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_jsonl = output_dir / "samples.jsonl"
    manifest_json = output_dir / "manifest.json"

    sample_count = 0
    skipped_count = 0
    input_dim = 0
    output_dim = len(G1_JOINT_COLUMNS)
    with tarfile.open(data_root.expanduser() / "soma_proportional.tar", "r:*") as source_tar:
        with tarfile.open(data_root.expanduser() / "g1.tar", "r:*") as target_tar:
            with samples_jsonl.open("w", encoding="utf-8") as f:
                for ref in refs:
                    if sample_count >= config.limit:
                        break
                    sample = _build_sample(source_tar, target_tar, ref, config)
                    if sample is None:
                        skipped_count += 1
                        continue
                    input_dim = len(sample["observation"])
                    f.write(json.dumps(sample, sort_keys=True))
                    f.write("\n")
                    sample_count += 1

    manifest = {
        "data_root": str(data_root),
        "index_csv": str(index_csv),
        "samples_jsonl": str(samples_jsonl),
        "builder": "raw_bvh_channel_debug",
        "contract_note": (
            "Debug samples flatten raw source BVH channels plus morphology. "
            "Final model input still needs the 30-body observation contract."
        ),
        "config": config.to_dict(),
        "sample_count": sample_count,
        "skipped_count": skipped_count,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SupervisedBuildResult(
        output_dir=output_dir,
        samples_jsonl=samples_jsonl,
        manifest_json=manifest_json,
        sample_count=sample_count,
        skipped_count=skipped_count,
        input_dim=input_dim,
        output_dim=output_dim,
        git_sha=manifest["git_sha"],
        git_dirty=manifest["git_dirty"],
    )


def _build_sample(
    source_tar: tarfile.TarFile,
    target_tar: tarfile.TarFile,
    ref: MotionPairRef,
    config: SupervisedBuildConfig,
) -> dict[str, object] | None:
    source_values = _read_bvh_values(source_tar, ref.source_motion_path)
    target_values = _read_g1_joints(target_tar, ref.target_g1_path)
    if source_values is None or target_values is None:
        return None
    if len(source_values) < config.history_frames:
        return None
    target_index = config.history_frames - 1 + config.target_frame_offset
    if target_index >= len(target_values):
        return None

    history = source_values[: config.history_frames]
    observation = _flatten(history) + _morphology_vector(ref.morphology)
    target = target_values[target_index]
    return {
        "sample_id": ref.sample_id,
        "actor_uid": ref.actor_uid,
        "category": ref.category,
        "package": ref.package,
        "quality_action": ref.quality_action,
        "quality_flags": list(ref.quality_flags),
        "source_motion_path": ref.source_motion_path,
        "target_g1_path": ref.target_g1_path,
        "history_frames": config.history_frames,
        "target_frame": target_index,
        "observation": observation,
        "target_joints": target,
    }


def _read_bvh_values(tar: tarfile.TarFile, member_path: str) -> list[list[float]] | None:
    try:
        extracted = tar.extractfile(member_path)
    except (KeyError, tarfile.TarError):
        return None
    if extracted is None:
        return None
    with extracted:
        try:
            text = io.TextIOWrapper(extracted, encoding="utf-8").read()
        except UnicodeDecodeError:
            return None
    try:
        parsed = parse_bvh_text(text)
    except ValueError:
        return None
    values = parsed.get("values")
    channel_names = parsed.get("channel_names")
    if not isinstance(values, list) or not isinstance(channel_names, list):
        return None
    width = len(channel_names)
    return [row for row in values if isinstance(row, list) and len(row) == width]


def _read_g1_joints(tar: tarfile.TarFile, member_path: str) -> list[list[float]] | None:
    try:
        extracted = tar.extractfile(member_path)
    except (KeyError, tarfile.TarError):
        return None
    if extracted is None:
        return None
    with extracted:
        try:
            rows = list(csv.DictReader(io.TextIOWrapper(extracted, encoding="utf-8", newline="")))
        except UnicodeDecodeError:
            return None
    values = []
    for row in rows:
        frame = [_maybe_float(row.get(column)) for column in G1_JOINT_COLUMNS]
        if any(value is None for value in frame):
            continue
        values.append([float(value) for value in frame if value is not None])
    return values


def _morphology_vector(morphology: Mapping[str, float | None]) -> list[float]:
    return [0.0 if value is None else float(value) for _, value in sorted(morphology.items())]


def _flatten(rows: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for row in rows for value in row]


def _maybe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _run_name(config: SupervisedBuildConfig) -> str:
    action_tag = config.action_column.replace("_", "-")
    return f"{config.split}_{action_tag}_h{config.history_frames}_limit{config.limit}"


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
