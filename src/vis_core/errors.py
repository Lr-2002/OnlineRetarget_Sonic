from __future__ import annotations


class VisCoreError(ValueError):
    """Base class for VisPacket validation and loading failures."""


class VisPacketSchemaError(VisCoreError):
    """Raised when a VisPacket manifest violates the core schema."""


class CoordinateValidationError(VisPacketSchemaError):
    """Raised when a coordinate convention is incomplete or invalid."""


class TimelineValidationError(VisPacketSchemaError):
    """Raised when timeline fps/dt/DTE constraints are inconsistent."""


class VisPacketLoadError(VisCoreError):
    """Raised when a manifest or track payload cannot be loaded."""
