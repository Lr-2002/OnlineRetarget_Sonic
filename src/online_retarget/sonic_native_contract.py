"""Validation contract for SONIC-native retargeting configs.

The formal OnlineRetarget lane is human/SOMA/BVH source motion plus skeleton
conditioning into SONIC's existing G1 decoder path.  Target-only G1 state fields
are allowed for labels and visualization, but not as deployable encoder inputs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml


FORMAL_TRAINING_LANE = "sonic_native_retarget"
LEGACY_DIAGNOSTIC_LANE = "legacy_kin_diagnostic"
FORBIDDEN_SOURCE_FEATURES = ("body_pos_w", "body_quat_w")
FORBIDDEN_DEPLOYABLE_SONIC_SOURCE_FEATURES = ("joint_pos_multi_future_wrist_for_soma",)
TARGET_FPS = 50.0
VISUAL_VALIDATION_EVERY_STEPS = 20_000
VISUAL_VALIDATION_NUM_VIDEOS = 8
VISUAL_VALIDATION_DURATION_SEC = 4.0
FORMAL_MAX_STEPS = 1_000_000
MOTION_FILE_SENTINELS = ("dummy", "zeros")
ACTOR_UID_PATTERN = re.compile(r"A\d{3,}")


class ContractError(ValueError):
    """Raised when a config violates the SONIC-native retargeting contract."""


@dataclass(frozen=True)
class ConfigValidationResult:
    """Small summary returned by the config validator."""

    path: str | None
    training_lane: str
    variant: str
    formal: bool
    source_feature_count: int
    target_decoder: str | None
    warnings: tuple[str, ...] = ()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
        data = loaded if loaded is not None else {}
    else:
        raise ContractError(f"unsupported config extension: {config_path}")
    if not isinstance(data, dict):
        raise ContractError(f"config root must be a mapping: {config_path}")
    return data


def classify_training_lane(config: Mapping[str, Any]) -> str:
    """Return the explicit or inferred training lane for a config."""

    explicit = config.get("training_lane")
    if explicit:
        return str(explicit)

    purpose = str(config.get("purpose", "")).lower()
    if "kinematics only" in purpose or "g1_kin" in purpose:
        return LEGACY_DIAGNOSTIC_LANE
    return "unknown"


def validate_config(
    config: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    require_formal: bool = False,
    check_paths: bool = False,
) -> ConfigValidationResult:
    """Validate one config and return a compact summary.

    Non-formal configs are accepted by default so that legacy diagnostics can
    still be inspected.  Pass ``require_formal=True`` for launchers that must not
    accidentally run the legacy reconstruction lane.
    """

    lane = classify_training_lane(config)
    variant = _variant_name(config)
    warnings: list[str] = []
    errors: list[str] = []

    if lane != FORMAL_TRAINING_LANE:
        if require_formal:
            errors.append(
                f"training_lane must be {FORMAL_TRAINING_LANE!r}, got {lane!r}"
            )
        else:
            warnings.append(
                f"config is {lane!r}; strict SONIC-native retarget checks were not applied"
            )
        if errors:
            raise ContractError(_format_errors(path, errors))
        return ConfigValidationResult(
            path=str(path) if path is not None else None,
            training_lane=lane,
            variant=variant,
            formal=False,
            source_feature_count=len(_source_feature_strings(config)),
            target_decoder=_target_decoder_name(config),
            warnings=tuple(warnings),
        )

    _require(config.get("sonic_native") is True, errors, "sonic_native must be true")
    _require(
        str(config.get("owner", "")) == "OnlineRetarget",
        errors,
        "owner must be OnlineRetarget",
    )
    _require(config.get("sonic_config"), errors, "sonic_config is required")
    _require(config.get("base_actor_critic_config"), errors, "base_actor_critic_config is required")
    input_data = _mapping(config.get("input_data"))
    _require(input_data, errors, "input_data is required")
    _require(input_data.get("robot_motion_file"), errors, "input_data.robot_motion_file is required")
    _require(input_data.get("soma_motion_file"), errors, "input_data.soma_motion_file is required")
    _require(input_data.get("skeleton_registry"), errors, "input_data.skeleton_registry is required")

    source_features = _source_feature_strings(config)
    source_text = " ".join(source_features).lower()
    source_encoder = _mapping(config.get("source_encoder"))
    module_target = str(source_encoder.get("module_target", ""))
    _require(
        module_target.startswith("online_retarget.sonic_encoder_modules."),
        errors,
        "source_encoder.module_target must point to an OnlineRetarget SONIC encoder module",
    )
    _require(source_features, errors, "source_features/source_encoder.inputs are required")
    _require(
        _has_any(source_text, ("soma_joints", "bvh", "proportional", "local_nonflat")),
        errors,
        "source features must include SOMA/BVH/proportional motion features",
    )
    _require(
        _has_any(source_text, ("root_ori", "root_orientation", "ori_b")),
        errors,
        "source features must include root orientation",
    )
    _require(
        _has_any(
            source_text,
            ("skeleton", "morphology", "actor_uid", "bone_lengths", "proportions"),
        ),
        errors,
        "source features must include skeleton/morphology conditioning",
    )

    for feature in FORBIDDEN_SOURCE_FEATURES:
        if _contains_token(source_text, feature):
            errors.append(f"{feature} is forbidden in formal source encoder features")
    for feature in FORBIDDEN_DEPLOYABLE_SONIC_SOURCE_FEATURES:
        if _contains_token(source_text, feature):
            errors.append(
                f"{feature} is target-derived teacher forcing and is forbidden "
                "in deployable formal source encoder features"
            )

    for ref_path, value in _forbidden_source_references(config):
        errors.append(f"target-only source reference at {'.'.join(ref_path)}: {value!r}")

    target_decoder = _target_decoder_name(config)
    _require(target_decoder == "g1_dyn", errors, "target_decoder.primary must be g1_dyn")
    decoder_targets = set(_decoder_targets(config))
    _require("g1_dyn" in decoder_targets, errors, "decoder targets must include g1_dyn")

    target_fps = _float_from_paths(
        config,
        (
            ("frequency", "target_fps"),
            ("motion_lib", "target_fps"),
            ("manager_env", "commands", "motion", "motion_lib_cfg", "target_fps"),
        ),
    )
    _require(target_fps == TARGET_FPS, errors, "frequency.target_fps must be 50")

    training = _mapping(config.get("training"))
    max_steps = _optional_int(training.get("max_steps"))
    _require(
        max_steps is not None and max_steps >= FORMAL_MAX_STEPS,
        errors,
        "training.max_steps must be at least 1000000",
    )

    visual = _mapping(config.get("visual_validation"))
    _require(visual.get("enabled") is True, errors, "visual_validation.enabled must be true")
    _require(
        _optional_int(visual.get("every_steps")) == VISUAL_VALIDATION_EVERY_STEPS,
        errors,
        "visual_validation.every_steps must be 20000",
    )
    _require(
        _optional_int(visual.get("num_videos")) == VISUAL_VALIDATION_NUM_VIDEOS,
        errors,
        "visual_validation.num_videos must be 8",
    )
    _require(
        _optional_float(visual.get("duration_sec")) == VISUAL_VALIDATION_DURATION_SEC,
        errors,
        "visual_validation.duration_sec must be 4.0",
    )
    _require(
        visual.get("wandb_upload") is True,
        errors,
        "visual_validation.wandb_upload must be true",
    )

    wandb_cfg = _mapping(config.get("wandb"))
    _require(wandb_cfg.get("enabled") is True, errors, "wandb.enabled must be true")
    _require(
        wandb_cfg.get("log_git_sha") is True or "git_sha" in _all_string_values(wandb_cfg),
        errors,
        "wandb must log the git SHA",
    )

    runtime = _mapping(config.get("runtime"))
    _require(
        runtime.get("require_committed_code") is True,
        errors,
        "runtime.require_committed_code must be true",
    )
    _require(
        runtime.get("require_latest_code") is True,
        errors,
        "runtime.require_latest_code must be true",
    )

    if check_paths:
        _check_sonic_paths(config, errors)

    _validate_sonic_hydra_wiring(config, errors)
    _validate_data_hydra_consistency(config, errors)

    if errors:
        raise ContractError(_format_errors(path, errors))

    return ConfigValidationResult(
        path=str(path) if path is not None else None,
        training_lane=lane,
        variant=variant,
        formal=True,
        source_feature_count=len(source_features),
        target_decoder=target_decoder,
        warnings=tuple(warnings),
    )


def validate_file(
    path: str | Path,
    *,
    require_formal: bool = False,
    check_paths: bool = False,
) -> ConfigValidationResult:
    """Load and validate one config file."""

    config = load_config(path)
    return validate_config(
        config,
        path=path,
        require_formal=require_formal,
        check_paths=check_paths,
    )


def result_to_dict(result: ConfigValidationResult) -> dict[str, Any]:
    """Convert a validation result to JSON-serializable data."""

    return {
        "path": result.path,
        "training_lane": result.training_lane,
        "variant": result.variant,
        "formal": result.formal,
        "source_feature_count": result.source_feature_count,
        "target_decoder": result.target_decoder,
        "warnings": list(result.warnings),
    }


def _variant_name(config: Mapping[str, Any]) -> str:
    variant = config.get("variant")
    if isinstance(variant, Mapping):
        return str(variant.get("name") or variant.get("id") or "unknown")
    if variant:
        return str(variant)
    source_encoder = config.get("source_encoder")
    if isinstance(source_encoder, Mapping) and source_encoder.get("variant"):
        return str(source_encoder["variant"])
    return "unknown"


def _target_decoder_name(config: Mapping[str, Any]) -> str | None:
    target_decoder = config.get("target_decoder")
    if isinstance(target_decoder, Mapping):
        primary = target_decoder.get("primary")
        return str(primary) if primary is not None else None
    if isinstance(target_decoder, str):
        return target_decoder
    return None


def _decoder_targets(config: Mapping[str, Any]) -> tuple[str, ...]:
    targets: list[str] = []
    target_decoder = config.get("target_decoder")
    if isinstance(target_decoder, Mapping):
        for key in ("primary", "auxiliary"):
            targets.extend(_string_list(target_decoder.get(key)))
    targets.extend(_string_list(config.get("decoder_targets")))
    return tuple(dict.fromkeys(targets))


def _source_feature_strings(config: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("source_features", "deployable_source_features"):
        values.extend(_string_list(config.get(key)))
    source_encoder = config.get("source_encoder")
    if isinstance(source_encoder, Mapping):
        for key in ("inputs", "input_features", "conditioning_features"):
            values.extend(_string_list(source_encoder.get(key)))
    feature_contract = config.get("feature_contract")
    if isinstance(feature_contract, Mapping):
        for key in ("source", "deployable_source"):
            values.extend(_string_list(feature_contract.get(key)))
    return tuple(dict.fromkeys(values))


def _forbidden_source_references(
    config: Mapping[str, Any],
) -> Iterable[tuple[tuple[str, ...], str]]:
    for path, value in _iter_strings(config):
        if not _is_source_context(path):
            continue
        for feature in FORBIDDEN_SOURCE_FEATURES:
            if _contains_token(value, feature):
                yield path, value


def _is_source_context(path: Sequence[str]) -> bool:
    joined = ".".join(str(part).lower() for part in path)
    target_contexts = (
        "target",
        "label",
        "loss",
        "visual",
        "render",
        "diagnostic",
        "teacher",
        "wandb",
        "validation",
        "expected_artifacts",
    )
    if any(token in joined for token in target_contexts):
        return False
    source_contexts = (
        "source_features",
        "deployable_source_features",
        "feature_contract.source",
        "feature_contract.deployable_source",
        "source_encoder",
        "encoder.inputs",
        "encoder.input_features",
    )
    return any(token in joined for token in source_contexts)


def _iter_strings(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_strings(child, path + (str(key),))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for index, child in enumerate(value):
            yield from _iter_strings(child, path + (str(index),))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for _, text in _iter_strings(value)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        result: list[str] = []
        for item in value:
            result.extend(_string_list(item))
        return result
    return [str(value)]


def _all_string_values(value: Any) -> str:
    return " ".join(text for _, text in _iter_strings(value)).lower()


def _contains_token(text: str, token: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text) is not None


def _has_any(text: str, tokens: Sequence[str]) -> bool:
    return any(token in text for token in tokens)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_get(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _float_from_paths(config: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> float | None:
    for path in paths:
        value = _optional_float(_nested_get(config, path))
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require(condition: Any, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def _check_sonic_paths(config: Mapping[str, Any], errors: list[str]) -> None:
    source_repo = Path(str(config.get("source_repo", "")))
    if not source_repo.exists():
        errors.append(f"source_repo does not exist: {source_repo}")
        return
    for key in ("sonic_config", "base_actor_critic_config"):
        rel = config.get(key)
        if not rel:
            continue
        path = source_repo / str(rel)
        if not path.exists():
            errors.append(f"{key} does not exist under source_repo: {path}")

    input_data = _mapping(config.get("input_data"))
    robot_motion_file = str(input_data.get("robot_motion_file") or "")
    soma_motion_file = str(input_data.get("soma_motion_file") or "")
    skeleton_registry = str(input_data.get("skeleton_registry") or "")

    if robot_motion_file:
        _check_motion_path(
            "input_data.robot_motion_file",
            _resolve_runtime_path(robot_motion_file, source_repo=source_repo),
            errors,
            allow_sentinel=False,
        )
    if soma_motion_file:
        _check_motion_path(
            "input_data.soma_motion_file",
            _resolve_runtime_path(soma_motion_file, source_repo=source_repo),
            errors,
            allow_sentinel=False,
        )
    if robot_motion_file and soma_motion_file:
        _check_motion_key_alignment(
            _resolve_runtime_path(robot_motion_file, source_repo=source_repo),
            _resolve_runtime_path(soma_motion_file, source_repo=source_repo),
            errors,
            robot_filter_motion_keys=_string_list(input_data.get("robot_filter_motion_keys")),
            robot_remove_motion_keys=_string_list(input_data.get("robot_remove_motion_keys")),
        )
    if skeleton_registry:
        registry_path = _resolve_runtime_path(skeleton_registry, source_repo=Path.cwd())
        _check_file_path("input_data.skeleton_registry", registry_path, errors)
        if robot_motion_file:
            _check_registry_motion_coverage(
                _resolve_runtime_path(robot_motion_file, source_repo=source_repo),
                registry_path,
                errors,
                robot_filter_motion_keys=_string_list(
                    input_data.get("robot_filter_motion_keys")
                ),
                robot_remove_motion_keys=_string_list(
                    input_data.get("robot_remove_motion_keys")
                ),
            )


def _validate_sonic_hydra_wiring(config: Mapping[str, Any], errors: list[str]) -> None:
    sonic_hydra = _mapping(config.get("sonic_hydra"))
    _require(
        sonic_hydra.get("variant_wired") is True,
        errors,
        "sonic_hydra.variant_wired must be true for formal training configs",
    )
    hydra_text = " ".join(_string_list(sonic_hydra.get("args")))
    source_encoder = _mapping(config.get("source_encoder"))
    module_target = str(source_encoder.get("module_target", ""))
    if module_target:
        _require(
            module_target in hydra_text,
            errors,
            "sonic_hydra.args must wire the source_encoder.module_target",
        )
    _require(
        "soma_morphology" in hydra_text,
        errors,
        "sonic_hydra.args must inject soma_morphology into the SONIC tokenizer/encoder",
    )
    _require(
        "g1_target_action" in hydra_text,
        errors,
        "sonic_hydra.args must inject g1_target_action for dynamics supervision",
    )
    _require(
        "online_retarget.sonic_losses.G1DynamicsActionLoss" in hydra_text,
        errors,
        "sonic_hydra.args must wire the dynamics action auxiliary loss",
    )
    _require(
        "online_retarget.sonic_validation_callback.SonicVisualValidationCallback" in hydra_text,
        errors,
        "sonic_hydra.args must wire the integrated visual validation callback",
    )
    _require(
        f"algo.config.num_learning_iterations={FORMAL_MAX_STEPS}" in hydra_text,
        errors,
        "sonic_hydra.args must set Sonic algo.config.num_learning_iterations=1000000",
    )
    _require(
        f"every_steps={VISUAL_VALIDATION_EVERY_STEPS}" in hydra_text,
        errors,
        "visual validation callback must run every 20000 steps",
    )
    _require(
        f"num_videos={VISUAL_VALIDATION_NUM_VIDEOS}" in hydra_text,
        errors,
        "visual validation callback must render 8 videos",
    )
    _require(
        "duration_sec=4.0" in hydra_text,
        errors,
        "visual validation callback must render 4 second clips",
    )
    for feature in FORBIDDEN_DEPLOYABLE_SONIC_SOURCE_FEATURES:
        if _contains_token(hydra_text, feature):
            errors.append(
                f"sonic_hydra.args still references forbidden deployable source feature {feature}"
            )

    variant_type = str(_mapping(config.get("variant")).get("type", "")).lower()
    if variant_type in {"adapter", "expert"}:
        _require(
            "routing=deterministic_cluster" in hydra_text,
            errors,
            "adapter/expert variants must wire deterministic skeleton-cluster routing",
        )


def _validate_data_hydra_consistency(config: Mapping[str, Any], errors: list[str]) -> None:
    input_data = _mapping(config.get("input_data"))
    hydra = _hydra_overrides(config)
    expected = {
        "manager_env.commands.motion.motion_lib_cfg.motion_file": input_data.get(
            "robot_motion_file"
        ),
        "manager_env.commands.motion.motion_lib_cfg.soma_motion_file": input_data.get(
            "soma_motion_file"
        ),
        "manager_env.observations.tokenizer.soma_morphology.params.registry_csv": input_data.get(
            "skeleton_registry"
        ),
    }
    for hydra_key, input_value in expected.items():
        if input_value in {"", None}:
            continue
        hydra_value = hydra.get(hydra_key)
        if hydra_value in {"", None}:
            errors.append(f"sonic_hydra.args missing {hydra_key}")
            continue
        if str(hydra_value) != str(input_value):
            errors.append(
                f"sonic_hydra.args {hydra_key}={hydra_value!r} does not match "
                f"input_data value {input_value!r}"
            )
    robot_remove_keys = _string_list(input_data.get("robot_remove_motion_keys"))
    robot_filter_keys = _string_list(input_data.get("robot_filter_motion_keys"))
    if robot_filter_keys:
        hydra_key = "manager_env.commands.motion.motion_lib_cfg.filter_motion_keys"
        hydra_value = _strip_hydra_quotes(hydra.get(hydra_key))
        if hydra_value in {"", None}:
            errors.append(f"sonic_hydra.args missing {hydra_key}")
        elif str(hydra_value) != str(robot_filter_keys[0]):
            errors.append(
                f"sonic_hydra.args {hydra_key}={hydra_value!r} does not match "
                f"input_data value {robot_filter_keys[0]!r}"
            )
    if robot_remove_keys:
        hydra_key = "manager_env.commands.motion.motion_lib_cfg.remove_motion_keys"
        hydra_value = hydra.get(hydra_key)
        if hydra_value in {"", None}:
            errors.append(f"sonic_hydra.args missing {hydra_key}")
        else:
            missing = [key for key in robot_remove_keys if key not in str(hydra_value)]
            if missing:
                errors.append(
                    f"sonic_hydra.args {hydra_key} missing remove keys: {missing[:5]}"
                )


def _hydra_overrides(config: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    sonic_hydra = _mapping(config.get("sonic_hydra"))
    for raw_arg in _string_list(sonic_hydra.get("args")):
        key, sep, value = raw_arg.partition("=")
        if not sep:
            continue
        key = key.lstrip("+")
        result[key] = value
    return result


def _strip_hydra_quotes(value: Any) -> Any:
    if not isinstance(value, str) or len(value) < 2:
        return value
    if value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _resolve_runtime_path(value: str, *, source_repo: Path) -> Path:
    if value in MOTION_FILE_SENTINELS:
        return Path(value)
    path = Path(value)
    if path.is_absolute():
        return path
    return source_repo / path


def _check_motion_path(
    label: str,
    path: Path,
    errors: list[str],
    *,
    allow_sentinel: bool,
) -> None:
    if str(path) in MOTION_FILE_SENTINELS:
        if allow_sentinel:
            return
        errors.append(f"{label} must be a real motion library path, got {path}")
        return
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")
        return
    if path.is_dir():
        try:
            motion_keys = _directory_motion_keys(path)
        except OSError as exc:
            errors.append(f"{label} could not be scanned for PKL files: {path} ({exc})")
            return
        if not motion_keys:
            errors.append(f"{label} has no Sonic-loadable per-motion PKL files: {path}")
            return
        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            loaded = _load_joblib(metadata_path)
            if isinstance(loaded, Mapping):
                metadata_keys = set(str(key) for key in loaded)
                if metadata_keys != motion_keys:
                    metadata_only = sorted(metadata_keys - motion_keys)
                    file_only = sorted(motion_keys - metadata_keys)
                    details = []
                    if metadata_only:
                        details.append(
                            f"metadata_only={len(metadata_only)} examples={metadata_only[:5]}"
                        )
                    if file_only:
                        details.append(
                            f"file_only={len(file_only)} examples={file_only[:5]}"
                        )
                    errors.append(
                        f"{label} metadata keys must match Sonic-loadable PKL files: "
                        + "; ".join(details)
                    )
        return
    if path.is_file():
        if path.suffix != ".pkl":
            errors.append(f"{label} must be a .pkl file or directory: {path}")
        return
    errors.append(f"{label} is not a file or directory: {path}")


def _check_motion_key_alignment(
    robot_path: Path,
    soma_path: Path,
    errors: list[str],
    *,
    robot_filter_motion_keys: Sequence[str] = (),
    robot_remove_motion_keys: Sequence[str] = (),
) -> None:
    if not robot_path.exists() or not soma_path.exists():
        return
    raw_robot_keys = _motion_keys(robot_path)
    try:
        robot_keys = _apply_filter_motion_keys(raw_robot_keys, robot_filter_motion_keys)
    except ValueError as exc:
        errors.append(str(exc))
        return
    robot_keys = _apply_remove_motion_keys(robot_keys, robot_remove_motion_keys)
    soma_keys = _motion_keys(soma_path)
    if not robot_keys or not soma_keys:
        return
    missing_remove_keys = [
        key
        for key in robot_remove_motion_keys
        if not any(item.startswith(key) for item in raw_robot_keys)
    ]
    if missing_remove_keys:
        errors.append(
            "robot_remove_motion_keys must match at least one robot motion: "
            + f"examples={missing_remove_keys[:5]}"
        )
    robot_only = sorted(robot_keys - soma_keys)
    if not robot_only:
        return
    errors.append(
        "final robot motionlib keys must all exist in soma motionlib: "
        f"robot_only={len(robot_only)} examples={robot_only[:5]}"
    )


def _check_registry_motion_coverage(
    robot_path: Path,
    registry_path: Path,
    errors: list[str],
    *,
    robot_filter_motion_keys: Sequence[str] = (),
    robot_remove_motion_keys: Sequence[str] = (),
) -> None:
    if not robot_path.exists() or not registry_path.exists() or not registry_path.is_file():
        return
    try:
        motion_keys = _apply_filter_motion_keys(_motion_keys(robot_path), robot_filter_motion_keys)
    except ValueError as exc:
        errors.append(str(exc))
        return
    motion_keys = _apply_remove_motion_keys(motion_keys, robot_remove_motion_keys)
    if not motion_keys:
        return
    registry_actors = _registry_actor_uids(registry_path)
    missing = []
    for motion_key in sorted(motion_keys):
        actor_uid = _actor_uid_from_motion_key(motion_key)
        if actor_uid and actor_uid not in registry_actors:
            missing.append((motion_key, actor_uid))
    if missing:
        missing_actors = sorted({actor for _, actor in missing})
        errors.append(
            "skeleton_registry must cover all final robot motion actors: "
            f"missing_motion_count={len(missing)} "
            f"missing_actors={missing_actors[:20]} "
            f"examples={[key for key, _ in missing[:5]]}"
        )


def _apply_filter_motion_keys(keys: set[str], filter_motion_keys: Sequence[str]) -> set[str]:
    if not filter_motion_keys:
        return keys
    patterns = list(filter_motion_keys)
    if all(pattern in keys for pattern in patterns):
        return {pattern for pattern in patterns if pattern in keys}
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid filter_motion_keys regex: {pattern}") from exc
    return {key for key in keys if any(regex.fullmatch(key) for regex in compiled)}


def _apply_remove_motion_keys(keys: set[str], remove_motion_keys: Sequence[str]) -> set[str]:
    if not remove_motion_keys:
        return keys
    return {
        key
        for key in keys
        if not any(key.startswith(prefix) for prefix in remove_motion_keys)
    }


def _motion_keys(path: Path) -> set[str]:
    if path.is_dir():
        return _directory_motion_keys(path)
    if path.is_file() and path.suffix == ".pkl":
        loaded = _load_joblib(path)
        if isinstance(loaded, Mapping):
            return set(str(key) for key in loaded)
    return set()


def _directory_motion_keys(path: Path) -> set[str]:
    resolved = path.resolve() if path.exists() else path
    return set(_directory_motion_keys_cached(str(resolved)))


@lru_cache(maxsize=32)
def _directory_motion_keys_cached(path_text: str) -> frozenset[str]:
    path = Path(path_text)
    return frozenset(
        item.stem for item in path.rglob("*.pkl") if item.name != "metadata.pkl"
    )


def _load_joblib(path: Path) -> Any:
    import joblib

    return joblib.load(path)


@lru_cache(maxsize=16)
def _registry_actor_uids(path_text: str | Path) -> frozenset[str]:
    path = Path(path_text)
    with path.open(newline="", encoding="utf-8") as file:
        return frozenset(
            row.get("actor_uid", "") for row in csv.DictReader(file) if row.get("actor_uid")
        )


def _actor_uid_from_motion_key(motion_key: str) -> str | None:
    match = ACTOR_UID_PATTERN.search(Path(motion_key).stem)
    return match.group(0) if match else None


def _check_file_path(label: str, path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")
    elif not path.is_file():
        errors.append(f"{label} must be a file: {path}")


def _format_errors(path: str | Path | None, errors: Sequence[str]) -> str:
    prefix = f"{path}: " if path is not None else ""
    return prefix + "; ".join(errors)
