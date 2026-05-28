"""Shared feature packing for SONIC kin-only SOMA encoder training and inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .sonic_native_contract import (
    FORMAL_TRAINING_LANE,
    FORBIDDEN_SOURCE_FEATURES,
    TARGET_FPS,
    load_config,
    validate_config,
)


class FeatureContractError(ValueError):
    """Raised when source/target features violate the deployable contract."""


@dataclass(frozen=True)
class SonicNativeFeatureContract:
    """Feature keys shared by formal training, validation, and inference."""

    source_keys: tuple[str, ...]
    target_label_keys: tuple[str, ...]
    target_fps: float = TARGET_FPS
    optional_source_keys: tuple[str, ...] = ()
    contract_name: str = FORMAL_TRAINING_LANE
    variant: str = "unknown"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SonicNativeFeatureContract":
        """Build and validate a feature contract from a formal config."""

        validate_config(config, require_formal=True)
        source_keys = _unique_strings(config.get("source_features"))
        source_encoder = config.get("source_encoder")
        if isinstance(source_encoder, Mapping):
            source_keys.extend(_unique_strings(source_encoder.get("inputs")))
        source_keys = _dedupe(source_keys)
        target_label_keys = _dedupe(_unique_strings(config.get("target_features")))
        optional_source_keys = tuple(key for key in source_keys if _is_optional_source_key(key))
        frequency = config.get("frequency") if isinstance(config.get("frequency"), Mapping) else {}
        variant = config.get("variant") if isinstance(config.get("variant"), Mapping) else {}
        return cls(
            source_keys=tuple(source_keys),
            target_label_keys=tuple(target_label_keys),
            target_fps=float(frequency.get("target_fps", TARGET_FPS)),
            optional_source_keys=optional_source_keys,
            variant=str(variant.get("name", "unknown")),
        )

    @classmethod
    def from_config_path(cls, path: str | Path) -> "SonicNativeFeatureContract":
        return cls.from_config(load_config(path))

    @property
    def required_source_keys(self) -> tuple[str, ...]:
        optional = set(self.optional_source_keys)
        return tuple(key for key in self.source_keys if key not in optional)

    @property
    def digest(self) -> str:
        payload = {
            "contract_name": self.contract_name,
            "source_keys": self.source_keys,
            "target_label_keys": self.target_label_keys,
            "target_fps": self.target_fps,
            "optional_source_keys": self.optional_source_keys,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def manifest(self) -> dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "variant": self.variant,
            "source_keys": list(self.source_keys),
            "required_source_keys": list(self.required_source_keys),
            "optional_source_keys": list(self.optional_source_keys),
            "target_label_keys": list(self.target_label_keys),
            "target_fps": self.target_fps,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class PackedFeatureBatch:
    """Packed features for one train/validation sample or inference request."""

    source: dict[str, Any]
    target_labels: dict[str, Any]
    contract_digest: str
    target_fps: float


def pack_inference_features(
    source: Mapping[str, Any],
    contract: SonicNativeFeatureContract,
) -> PackedFeatureBatch:
    """Pack only deployable source features for inference."""

    packed_source = _pack_source(source, contract)
    return PackedFeatureBatch(
        source=packed_source,
        target_labels={},
        contract_digest=contract.digest,
        target_fps=contract.target_fps,
    )


def pack_source_motion_with_morphology(
    source_motion: Mapping[str, Any],
    morphology_features: Mapping[str, Any],
    contract: SonicNativeFeatureContract,
) -> PackedFeatureBatch:
    """Merge source motion and actor morphology before inference packing."""

    source = dict(source_motion)
    source.update(morphology_features)
    return pack_inference_features(source, contract)


def pack_training_pair(
    source: Mapping[str, Any],
    target_labels: Mapping[str, Any],
    contract: SonicNativeFeatureContract,
) -> PackedFeatureBatch:
    """Pack source and target labels using the same deployable source contract."""

    packed_source = _pack_source(source, contract)
    packed_targets = {
        key: target_labels[key]
        for key in contract.target_label_keys
        if key in target_labels
    }
    missing_targets = [
        key
        for key in contract.target_label_keys
        if key not in target_labels and not _is_optional_target_key(key)
    ]
    if missing_targets:
        raise FeatureContractError(f"missing target labels: {', '.join(missing_targets)}")
    return PackedFeatureBatch(
        source=packed_source,
        target_labels=packed_targets,
        contract_digest=contract.digest,
        target_fps=contract.target_fps,
    )


def assert_matching_contracts(
    training_contract: SonicNativeFeatureContract,
    inference_contract: SonicNativeFeatureContract,
) -> None:
    """Ensure training and inference use exactly the same deployable source contract."""

    if training_contract.digest != inference_contract.digest:
        raise FeatureContractError(
            "training/inference feature contract mismatch: "
            f"{training_contract.digest} != {inference_contract.digest}"
        )


def _pack_source(
    source: Mapping[str, Any],
    contract: SonicNativeFeatureContract,
) -> dict[str, Any]:
    forbidden_present = [key for key in FORBIDDEN_SOURCE_FEATURES if key in source]
    if forbidden_present:
        raise FeatureContractError(
            "target-only fields are forbidden in deployable source features: "
            + ", ".join(forbidden_present)
        )
    missing = [key for key in contract.required_source_keys if key not in source]
    if missing:
        raise FeatureContractError(f"missing source features: {', '.join(missing)}")
    return {key: source[key] for key in contract.source_keys if key in source}


def _unique_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            item
            for child in value.values()
            for item in _unique_strings(child)
        ]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [item for child in value for item in _unique_strings(child)]
    return [str(value)]


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _is_optional_source_key(key: str) -> bool:
    return "optional" in key or key.endswith("_optional")


def _is_optional_target_key(key: str) -> bool:
    return key in {"body_pos_w", "body_quat_w"} or key.endswith("_optional")
