from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .errors import VisPacketLoadError
from .schema import TrackSpec, VisPacket, load_vis_packet_manifest
from .timeline import validate_track_lengths


@dataclass(frozen=True)
class TrackData:
    spec: TrackSpec
    source_path: Path
    frames: tuple[Any, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class LoadedVisPacket:
    packet: VisPacket
    tracks: Mapping[str, TrackData]

    def track_lengths(self) -> dict[str, int]:
        return {role: data.frame_count for role, data in self.tracks.items()}


def load_vis_packet(path: str | Path, *, validate: bool = True) -> LoadedVisPacket:
    packet = load_vis_packet_manifest(path)
    tracks = {
        role: load_track_data(track, base_dir=packet.base_dir)
        for role, track in packet.tracks.items()
    }
    loaded = LoadedVisPacket(packet=packet, tracks=tracks)
    if validate:
        validate_loaded_packet(loaded)
    return loaded


def validate_loaded_packet(loaded: LoadedVisPacket) -> None:
    validate_track_lengths(loaded.track_lengths(), loaded.packet.timeline)
    for track in loaded.tracks.values():
        validate_track_payload(track)


def load_track_data(track: TrackSpec, *, base_dir: Path | None = None) -> TrackData:
    source_path = _resolve_track_path(track.uri, base_dir=base_dir)
    if track.format == "json":
        frames = _load_json_frames(source_path)
    elif track.format == "csv":
        frames = _load_csv_frames(source_path)
    else:
        raise VisPacketLoadError(f"unsupported track format '{track.format}' for {track.role}")
    return TrackData(spec=track, source_path=source_path, frames=frames)


def validate_track_payload(track: TrackData) -> None:
    declared_fields = tuple(track.spec.root_fields) + tuple(track.spec.joint_names)
    if not declared_fields:
        return
    for frame_idx, frame in enumerate(track.frames):
        for field_name in declared_fields:
            if not _frame_has_payload(frame, field_name):
                raise VisPacketLoadError(
                    f"track '{track.spec.role}' frame {frame_idx} missing declared payload "
                    f"'{field_name}'"
                )


def _resolve_track_path(uri: str, *, base_dir: Path | None) -> Path:
    path = Path(uri)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _load_json_frames(path: Path) -> tuple[Any, ...]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except OSError as exc:
        raise VisPacketLoadError(f"failed to read JSON track: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VisPacketLoadError(f"invalid JSON track: {path}") from exc

    if isinstance(payload, Mapping):
        frames = payload.get("frames")
    else:
        frames = payload
    if not isinstance(frames, list):
        raise VisPacketLoadError(f"JSON track must be a list or object with frames list: {path}")
    return tuple(frames)


def _load_csv_frames(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise VisPacketLoadError(f"CSV track must have a header: {path}")
            return tuple(dict(row) for row in reader)
    except OSError as exc:
        raise VisPacketLoadError(f"failed to read CSV track: {path}") from exc


def _frame_has_payload(frame: Any, field_name: str) -> bool:
    if not isinstance(frame, Mapping) or field_name not in frame:
        return False
    value = frame[field_name]
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return bool(value)
    return True
