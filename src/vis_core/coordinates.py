from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import CoordinateValidationError


ISAAC_STANDARD = "isaac"
ALLOWED_UP_AXES = frozenset({"X", "Y", "Z"})
ALLOWED_HANDEDNESS = frozenset({"right", "left"})
ALLOWED_LENGTH_UNITS = frozenset({"meter", "centimeter", "millimeter"})
ALLOWED_ANGLE_UNITS = frozenset({"radian", "degree"})
ALLOWED_ROTATION_FORMATS = frozenset(
    {
        "xyzw",
        "wxyz",
        "axis_angle",
        "euler_xyz",
        "euler_xzy",
        "euler_yxz",
        "euler_yzx",
        "euler_zxy",
        "euler_zyx",
    }
)


@dataclass(frozen=True)
class CoordinateConvention:
    """Declared coordinate frame for one VisPacket track.

    Isaac canonical coordinates are Z-up, X-forward, right-handed, meters,
    radians, with quaternions stored as xyzw.
    """

    standard: str
    up_axis: str
    forward_axis: str
    handedness: str
    unit_length: str
    unit_angle: str
    root_rotation: str

    @property
    def is_isaac(self) -> bool:
        return self == ISAAC_COORDINATE_STANDARD

    def as_dict(self) -> dict[str, str]:
        return {
            "standard": self.standard,
            "up_axis": self.up_axis,
            "forward_axis": self.forward_axis,
            "handedness": self.handedness,
            "unit_length": self.unit_length,
            "unit_angle": self.unit_angle,
            "root_rotation": self.root_rotation,
        }


ISAAC_COORDINATE_STANDARD = CoordinateConvention(
    standard=ISAAC_STANDARD,
    up_axis="Z",
    forward_axis="X",
    handedness="right",
    unit_length="meter",
    unit_angle="radian",
    root_rotation="xyzw",
)


def coordinate_from_mapping(
    value: Mapping[str, Any] | str | None,
    *,
    field_name: str = "coordinate",
    default: CoordinateConvention = ISAAC_COORDINATE_STANDARD,
) -> CoordinateConvention:
    if value is None:
        return default
    if isinstance(value, str):
        if value.lower() != ISAAC_STANDARD:
            raise CoordinateValidationError(
                f"{field_name}='{value}' must be expanded into a coordinate mapping"
            )
        return default
    if not isinstance(value, Mapping):
        raise CoordinateValidationError(f"{field_name} must be a mapping or 'isaac'")

    standard = _required_str(value, "standard", field_name).lower()
    if standard == ISAAC_STANDARD:
        convention = CoordinateConvention(
            standard=ISAAC_STANDARD,
            up_axis=_optional_axis(value, "up_axis", default.up_axis, field_name),
            forward_axis=_optional_axis(
                value,
                "forward_axis",
                default.forward_axis,
                field_name,
                signed=True,
            ),
            handedness=_optional_choice(
                value,
                "handedness",
                default.handedness,
                ALLOWED_HANDEDNESS,
                field_name,
            ),
            unit_length=_optional_choice(
                value,
                "unit_length",
                default.unit_length,
                ALLOWED_LENGTH_UNITS,
                field_name,
            ),
            unit_angle=_optional_choice(
                value,
                "unit_angle",
                default.unit_angle,
                ALLOWED_ANGLE_UNITS,
                field_name,
            ),
            root_rotation=_optional_choice(
                value,
                "root_rotation",
                default.root_rotation,
                ALLOWED_ROTATION_FORMATS,
                field_name,
            ),
        )
        if convention != ISAAC_COORDINATE_STANDARD:
            raise CoordinateValidationError(
                f"{field_name} declares standard='isaac' but does not match the "
                "Isaac coordinate standard"
            )
        return convention

    convention = CoordinateConvention(
        standard=standard,
        up_axis=_required_axis(value, "up_axis", field_name),
        forward_axis=_required_axis(value, "forward_axis", field_name, signed=True),
        handedness=_required_choice(value, "handedness", ALLOWED_HANDEDNESS, field_name),
        unit_length=_required_choice(value, "unit_length", ALLOWED_LENGTH_UNITS, field_name),
        unit_angle=_required_choice(value, "unit_angle", ALLOWED_ANGLE_UNITS, field_name),
        root_rotation=_required_choice(
            value,
            "root_rotation",
            ALLOWED_ROTATION_FORMATS,
            field_name,
        ),
    )
    _validate_axis_pair(convention.up_axis, convention.forward_axis, field_name)
    return convention


def require_isaac_standard(
    value: Mapping[str, Any] | str | None,
    *,
    field_name: str = "coordinate_standard",
) -> CoordinateConvention:
    convention = coordinate_from_mapping(value, field_name=field_name)
    if not convention.is_isaac:
        raise CoordinateValidationError(f"{field_name} must be the Isaac coordinate standard")
    return convention


def _required_str(value: Mapping[str, Any], key: str, field_name: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise CoordinateValidationError(f"{field_name}.{key} is required")
    return raw.strip()


def _required_axis(
    value: Mapping[str, Any],
    key: str,
    field_name: str,
    *,
    signed: bool = False,
) -> str:
    return _normalize_axis(_required_str(value, key, field_name), field_name, key, signed=signed)


def _optional_axis(
    value: Mapping[str, Any],
    key: str,
    default: str,
    field_name: str,
    *,
    signed: bool = False,
) -> str:
    raw = value.get(key, default)
    if not isinstance(raw, str):
        raise CoordinateValidationError(f"{field_name}.{key} must be a string")
    return _normalize_axis(raw, field_name, key, signed=signed)


def _required_choice(
    value: Mapping[str, Any],
    key: str,
    allowed: frozenset[str],
    field_name: str,
) -> str:
    raw = _required_str(value, key, field_name).lower()
    if raw not in allowed:
        raise CoordinateValidationError(f"{field_name}.{key} must be one of {sorted(allowed)}")
    return raw


def _optional_choice(
    value: Mapping[str, Any],
    key: str,
    default: str,
    allowed: frozenset[str],
    field_name: str,
) -> str:
    raw = value.get(key, default)
    if not isinstance(raw, str):
        raise CoordinateValidationError(f"{field_name}.{key} must be a string")
    raw = raw.strip().lower()
    if raw not in allowed:
        raise CoordinateValidationError(f"{field_name}.{key} must be one of {sorted(allowed)}")
    return raw


def _normalize_axis(raw: str, field_name: str, key: str, *, signed: bool) -> str:
    axis = raw.strip().upper()
    if signed and axis.startswith(("+", "-")):
        sign = "-" if axis.startswith("-") else ""
        axis = sign + axis[1:]
    elif axis.startswith(("+", "-")):
        raise CoordinateValidationError(f"{field_name}.{key} must be an unsigned axis")

    base_axis = axis[1:] if axis.startswith("-") else axis
    if base_axis not in ALLOWED_UP_AXES:
        raise CoordinateValidationError(f"{field_name}.{key} must be X, Y, or Z")
    return axis


def _validate_axis_pair(up_axis: str, forward_axis: str, field_name: str) -> None:
    forward_base = forward_axis[1:] if forward_axis.startswith("-") else forward_axis
    if up_axis == forward_base:
        raise CoordinateValidationError(
            f"{field_name}.up_axis and {field_name}.forward_axis must differ"
        )
