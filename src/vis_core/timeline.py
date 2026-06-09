from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping

from .errors import TimelineValidationError


TIMELINE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Timeline:
    frame_count: int
    fps: float
    dt: float
    start_time: float = 0.0
    sim_dt: float | None = None
    dte: int | None = None

    @property
    def duration_sec(self) -> float:
        return self.frame_count * self.dt

    def frame_times(self) -> tuple[float, ...]:
        return tuple(self.start_time + idx * self.dt for idx in range(self.frame_count))

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "frame_count": self.frame_count,
            "fps": self.fps,
            "dt": self.dt,
            "start_time": self.start_time,
            "sim_dt": self.sim_dt,
            "dte": self.dte,
            "duration_sec": self.duration_sec,
        }


def timeline_from_mapping(value: Mapping[str, Any]) -> Timeline:
    if not isinstance(value, Mapping):
        raise TimelineValidationError("timeline must be a mapping")

    frame_count = _positive_int(value.get("frame_count"), "timeline.frame_count")
    fps = _optional_positive_float(value.get("fps"), "timeline.fps")
    dt = _optional_positive_float(value.get("dt"), "timeline.dt")
    if fps is None and dt is None:
        raise TimelineValidationError("timeline requires fps or dt")
    if fps is None:
        assert dt is not None
        fps = 1.0 / dt
    if dt is None:
        dt = 1.0 / fps
    if not isclose(dt, 1.0 / fps, rel_tol=TIMELINE_TOLERANCE, abs_tol=TIMELINE_TOLERANCE):
        raise TimelineValidationError("timeline.fps and timeline.dt are inconsistent")

    sim_dt = _optional_positive_float(value.get("sim_dt"), "timeline.sim_dt")
    dte = _optional_positive_int(value.get("dte"), "timeline.dte")
    if sim_dt is not None and dte is not None:
        expected_dt = sim_dt * dte
        if not isclose(dt, expected_dt, rel_tol=TIMELINE_TOLERANCE, abs_tol=TIMELINE_TOLERANCE):
            raise TimelineValidationError("timeline.dt must equal timeline.sim_dt * timeline.dte")
    elif sim_dt is not None:
        ratio = dt / sim_dt
        inferred_dte = round(ratio)
        if inferred_dte < 1 or not isclose(
            ratio,
            inferred_dte,
            rel_tol=TIMELINE_TOLERANCE,
            abs_tol=TIMELINE_TOLERANCE,
        ):
            raise TimelineValidationError("timeline.dt must be an integer multiple of sim_dt")
        dte = inferred_dte

    start_time = _optional_float(value.get("start_time", 0.0), "timeline.start_time")
    return Timeline(
        frame_count=frame_count,
        fps=fps,
        dt=dt,
        start_time=start_time,
        sim_dt=sim_dt,
        dte=dte,
    )


def validate_track_lengths(track_lengths: Mapping[str, int], timeline: Timeline) -> None:
    for role, length in track_lengths.items():
        if length != timeline.frame_count:
            raise TimelineValidationError(
                f"track '{role}' has {length} frames but timeline.frame_count is "
                f"{timeline.frame_count}"
            )


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TimelineValidationError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0.0:
        raise TimelineValidationError(f"{field_name} must be a positive number")
    return float(value)


def _optional_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimelineValidationError(f"{field_name} must be a number")
    return float(value)
