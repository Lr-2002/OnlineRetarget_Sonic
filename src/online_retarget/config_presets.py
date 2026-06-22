"""Shared policy preset resolution for training and sample builders."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BUILD_SYNC_KEYS = (
    "history_frames",
    "target_horizon_frames",
    "target_future_step",
    "include_target_root_pose",
    "source_rotation",
)


def apply_config_preset(config: dict[str, Any]) -> dict[str, Any]:
    preset_name = selected_config_preset(config)
    if not preset_name:
        return config
    presets = config.get("policy_presets", config.get("config_presets", {}))
    if not isinstance(presets, dict):
        raise ValueError("policy_presets/config_presets must be a mapping")
    preset = presets.get(preset_name)
    if not isinstance(preset, dict):
        available = ", ".join(sorted(str(key) for key in presets)) or "none"
        raise ValueError(f"unknown config preset {preset_name!r}; available: {available}")
    validate_config_preset_payload(preset_name, preset)
    resolved = deep_merge_dicts(config, preset)
    resolved["policy_preset"] = preset_name
    data = resolved.get("data", {})
    data = dict(data) if isinstance(data, dict) else {}
    data["policy_preset"] = preset_name
    sync_build_config_from_data(data)
    resolved["data"] = data
    validate_resolved_config_preset(preset_name, resolved)
    return resolved


def selected_config_preset(config: dict[str, Any]) -> str:
    data = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    names = [
        config.get("policy_preset"),
        config.get("config_preset"),
        data.get("policy_preset"),
        data.get("config_preset"),
    ]
    selected = [str(name) for name in names if name not in (None, "")]
    if not selected:
        return ""
    if len(set(selected)) > 1:
        raise ValueError(f"conflicting config preset names: {selected}")
    return selected[0]


def deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def sync_build_config_from_data(data: dict[str, Any]) -> None:
    build = data.get("build", {})
    build = dict(build) if isinstance(build, dict) else {}
    for key in BUILD_SYNC_KEYS:
        if key in data:
            build[key] = data[key]
    data["build"] = build


def validate_config_preset_payload(name: str, preset: dict[str, Any]) -> None:
    data = preset.get("data", {})
    model = preset.get("model", {})
    if not isinstance(data, dict) or not isinstance(model, dict):
        raise ValueError(f"config preset {name!r} must define data and model mappings")
    required_data = (
        "samples_jsonl",
        "target_format",
        "history_frames",
        "target_horizon_frames",
        "target_future_step",
        "source_body_count",
        "source_body_token_dim",
        "source_rotation",
        "action_dim",
    )
    required_model = ("family", "output")
    missing = [f"data.{key}" for key in required_data if key not in data]
    missing.extend(f"model.{key}" for key in required_model if key not in model)
    family = str(model.get("family", "")).lower().replace("-", "_")
    if family in {"temporal_diffusion_policy", "dp_temporal", "temporal_dp", "temporal_diffusion"}:
        route_b_model_required = (
            "action_dim",
            "source_body_token_dim",
            "source_skeleton_dim",
            "morphology_dim",
            "robot_state_dim",
            "source_body_count",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
        )
        missing.extend(f"model.{key}" for key in route_b_model_required if key not in model)
    else:
        if "hidden_dims" not in model and "d_model" not in model:
            missing.append("model.hidden_dims or model.d_model")
    if missing:
        raise ValueError(f"config preset {name!r} missing controlled keys: {', '.join(missing)}")


def validate_resolved_config_preset(name: str, config: dict[str, Any]) -> None:
    data = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    build = data.get("build", {}) if isinstance(data.get("build", {}), dict) else {}
    family = configured_model_family(config)
    errors = []
    if not data.get("samples_jsonl"):
        errors.append("data.samples_jsonl")
    if not data.get("target_format"):
        errors.append("data.target_format")
    if not model.get("family"):
        errors.append("model.family")
    if not model.get("output"):
        errors.append("model.output")
    for key in BUILD_SYNC_KEYS:
        if key in data and key in build and build[key] != data[key]:
            errors.append(f"data.build.{key} must match data.{key}")
    if family == "temporal_diffusion_policy":
        if data.get("target_format") != "bones_sonic_joint_root_pos_future_window":
            errors.append("data.target_format must be bones_sonic_joint_root_pos_future_window")
        if model.get("output") != "g1_joint_root_position_future_window":
            errors.append("model.output must be g1_joint_root_position_future_window")
        if not bool(data.get("include_target_root_pose", False)):
            errors.append("data.include_target_root_pose must be true")
        for key in (
            "history_frames",
            "target_horizon_frames",
            "target_future_step",
            "source_body_count",
            "source_body_token_dim",
            "source_rotation",
            "action_dim",
        ):
            if key not in data:
                errors.append(f"data.{key}")
        for key in (
            "action_dim",
            "source_body_token_dim",
            "source_skeleton_dim",
            "morphology_dim",
            "robot_state_dim",
            "source_body_count",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
        ):
            if key not in model:
                errors.append(f"model.{key}")
        if data.get("action_dim") is not None and model.get("action_dim") is not None:
            if int(data["action_dim"]) != int(model["action_dim"]):
                errors.append("data.action_dim must match model.action_dim")
    elif "route_b" in str(name).lower() or "temporal_diffusion" in str(name).lower():
        errors.append("route-b preset must resolve model.family=temporal_diffusion_policy")
    else:
        if family == "temporal_diffusion_policy":
            errors.append("flat preset must not resolve temporal_diffusion_policy")
    if errors:
        raise ValueError(f"config preset {name!r} is inconsistent: {', '.join(errors)}")


def configured_model_family(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    key = str(model_cfg.get("family", "temporal_mlp")).lower().replace("-", "_")
    aliases = {
        "mlp": "temporal_mlp",
        "temporal_mlp": "temporal_mlp",
        "tf": "temporal_transformer",
        "transformer": "temporal_transformer",
        "temporal_transformer": "temporal_transformer",
        "token_tf": "token_transformer",
        "token_transformer": "token_transformer",
        "tokenized_transformer": "token_transformer",
        "cross_attention_transformer": "token_transformer",
        "fm": "flow_matching",
        "flow": "flow_matching",
        "flow_matching": "flow_matching",
        "dp": "diffusion_policy",
        "diffusion": "diffusion_policy",
        "diffusion_policy": "diffusion_policy",
        "dp_temporal": "temporal_diffusion_policy",
        "temporal_dp": "temporal_diffusion_policy",
        "temporal_diffusion": "temporal_diffusion_policy",
        "temporal_diffusion_policy": "temporal_diffusion_policy",
    }
    return aliases.get(key, key)
