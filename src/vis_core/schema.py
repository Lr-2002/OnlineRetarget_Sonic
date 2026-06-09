from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from .coordinates import (
    CoordinateConvention,
    ISAAC_COORDINATE_STANDARD,
    coordinate_from_mapping,
    require_isaac_standard,
)
from .errors import VisPacketLoadError, VisPacketSchemaError
from .timeline import Timeline, timeline_from_mapping


SCHEMA_VERSION = "VisPacket v0.1"
SCHEMA_VERSION_ALIASES = frozenset({SCHEMA_VERSION, "vis_packet/v0.1"})
TRACK_ROLES = ("human", "target_g1", "play_g1")
SUPPORTED_TRACK_FORMATS = frozenset({"json", "csv"})
OPTIONAL_STAGE_NAMES = frozenset({"render", "physics", "diagnose"})
RESERVED_CORE_KEYS = frozenset({"metrics"})


@dataclass(frozen=True)
class OptionalStage:
    enabled: bool = False
    interface: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interface": self.interface,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class TrackSpec:
    role: str
    uri: str
    format: str
    coordinate: CoordinateConvention
    joint_names: tuple[str, ...] = ()
    root_fields: tuple[str, ...] = ("root_pos", "root_rot")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "uri": self.uri,
            "format": self.format,
            "coordinate": self.coordinate.as_dict(),
            "joint_names": list(self.joint_names),
            "root_fields": list(self.root_fields),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VisPacket:
    schema_version: str
    timeline: Timeline
    coordinate_standard: CoordinateConvention
    tracks: Mapping[str, TrackSpec]
    render: OptionalStage = field(default_factory=OptionalStage)
    physics: OptionalStage = field(default_factory=OptionalStage)
    diagnose: OptionalStage = field(default_factory=OptionalStage)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    base_dir: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timeline": self.timeline.as_dict(),
            "coordinate_standard": self.coordinate_standard.as_dict(),
            "tracks": {role: track.as_dict() for role, track in self.tracks.items()},
            "render": self.render.as_dict(),
            "physics": self.physics.as_dict(),
            "diagnose": self.diagnose.as_dict(),
            "metadata": dict(self.metadata),
        }


def load_vis_packet_manifest(path: str | Path) -> VisPacket:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except OSError as exc:
        raise VisPacketLoadError(f"failed to read VisPacket manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise VisPacketLoadError(f"invalid JSON VisPacket manifest: {manifest_path}") from exc

    return parse_vis_packet_manifest(payload, base_dir=manifest_path.parent)


def parse_vis_packet_manifest(
    payload: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> VisPacket:
    if not isinstance(payload, Mapping):
        raise VisPacketSchemaError("VisPacket manifest must be a mapping")
    _reject_reserved_keys(payload, "VisPacket")

    schema_version = payload.get("schema_version")
    if schema_version not in SCHEMA_VERSION_ALIASES:
        raise VisPacketSchemaError(f"schema_version must be '{SCHEMA_VERSION}'")

    timeline = timeline_from_mapping(_required_mapping(payload, "timeline", "VisPacket"))
    coordinate_standard = require_isaac_standard(
        payload.get("coordinate_standard", ISAAC_COORDINATE_STANDARD.as_dict())
    )
    tracks = _parse_tracks(
        _required_mapping(payload, "tracks", "VisPacket"),
        default_coordinate=coordinate_standard,
    )
    render = optional_stage_from_mapping(payload.get("render"), "render")
    physics = optional_stage_from_mapping(payload.get("physics"), "physics")
    diagnose = optional_stage_from_mapping(payload.get("diagnose"), "diagnose")

    return VisPacket(
        schema_version=SCHEMA_VERSION,
        timeline=timeline,
        coordinate_standard=coordinate_standard,
        tracks=tracks,
        render=render,
        physics=physics,
        diagnose=diagnose,
        metadata=_optional_mapping(payload.get("metadata"), "metadata"),
        base_dir=Path(base_dir) if base_dir is not None else None,
    )


def optional_stage_from_mapping(value: Any, name: str) -> OptionalStage:
    if name not in OPTIONAL_STAGE_NAMES:
        raise VisPacketSchemaError(f"unknown optional stage '{name}'")
    if value is None:
        return OptionalStage()
    if isinstance(value, bool):
        return OptionalStage(enabled=value)
    if not isinstance(value, Mapping):
        raise VisPacketSchemaError(f"{name} must be a mapping or bool")
    _reject_reserved_keys(value, name)

    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise VisPacketSchemaError(f"{name}.enabled must be a boolean")

    interface = value.get("interface")
    if interface is not None and (not isinstance(interface, str) or not interface.strip()):
        raise VisPacketSchemaError(f"{name}.interface must be a non-empty string")

    config = value.get("config", {})
    if not isinstance(config, Mapping):
        raise VisPacketSchemaError(f"{name}.config must be a mapping")
    _reject_reserved_keys(config, f"{name}.config")
    return OptionalStage(
        enabled=enabled,
        interface=interface.strip() if isinstance(interface, str) else None,
        config=dict(config),
    )


def _parse_tracks(
    tracks_payload: Mapping[str, Any],
    *,
    default_coordinate: CoordinateConvention,
) -> dict[str, TrackSpec]:
    roles = set(tracks_payload)
    missing = sorted(set(TRACK_ROLES) - roles)
    extra = sorted(roles - set(TRACK_ROLES))
    if missing:
        raise VisPacketSchemaError(f"tracks missing required roles: {missing}")
    if extra:
        raise VisPacketSchemaError(f"tracks contains unsupported roles for v0.1: {extra}")

    tracks: dict[str, TrackSpec] = {}
    for role in TRACK_ROLES:
        raw_track = _required_mapping(tracks_payload, role, "tracks")
        _reject_reserved_keys(raw_track, f"tracks.{role}")
        track_role = raw_track.get("role", role)
        if track_role != role:
            raise VisPacketSchemaError(f"tracks.{role}.role must equal '{role}'")
        uri = _required_str(raw_track, "uri", f"tracks.{role}")
        fmt = _track_format(raw_track.get("format"), uri, role)
        coordinate = coordinate_from_mapping(
            raw_track.get("coordinate"),
            field_name=f"tracks.{role}.coordinate",
            default=default_coordinate,
        )
        tracks[role] = TrackSpec(
            role=role,
            uri=uri,
            format=fmt,
            coordinate=coordinate,
            joint_names=_optional_string_tuple(raw_track.get("joint_names"), f"tracks.{role}"),
            root_fields=_optional_string_tuple(
                raw_track.get("root_fields", ("root_pos", "root_rot")),
                f"tracks.{role}",
            ),
            metadata=_optional_mapping(raw_track.get("metadata"), f"tracks.{role}.metadata"),
        )
    return tracks


def _reject_reserved_keys(value: Mapping[str, Any], field_name: str) -> None:
    reserved = sorted(RESERVED_CORE_KEYS.intersection(value))
    if reserved:
        raise VisPacketSchemaError(
            f"{field_name} contains reserved keys {reserved}; metrics stay outside vis_core"
        )


def _required_mapping(value: Mapping[str, Any], key: str, field_name: str) -> Mapping[str, Any]:
    raw = value.get(key)
    if not isinstance(raw, Mapping):
        raise VisPacketSchemaError(f"{field_name}.{key} must be a mapping")
    return raw


def _optional_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise VisPacketSchemaError(f"{field_name} must be a mapping")
    _reject_reserved_keys(value, field_name)
    return dict(value)


def _required_str(value: Mapping[str, Any], key: str, field_name: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise VisPacketSchemaError(f"{field_name}.{key} is required")
    return raw.strip()


def _optional_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise VisPacketSchemaError(f"{field_name} value must be a list of strings")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise VisPacketSchemaError(f"{field_name} value must be a list of strings") from exc
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise VisPacketSchemaError(f"{field_name} value must contain only non-empty strings")
    return tuple(item.strip() for item in items)


def _track_format(raw_format: Any, uri: str, role: str) -> str:
    if raw_format is None:
        suffix = Path(uri).suffix.lower().lstrip(".")
        fmt = suffix
    elif isinstance(raw_format, str):
        fmt = raw_format.strip().lower()
    else:
        raise VisPacketSchemaError(f"tracks.{role}.format must be a string")

    if fmt not in SUPPORTED_TRACK_FORMATS:
        raise VisPacketSchemaError(
            f"tracks.{role}.format must be one of {sorted(SUPPORTED_TRACK_FORMATS)}"
        )
    return fmt
