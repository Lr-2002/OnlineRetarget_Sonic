"""Config-driven model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from online_retarget.data.schema import ObservationSpec


@dataclass(frozen=True)
class ModelBuildResult:
    model: Any
    family: str
    config: dict[str, Any]


def build_model(
    config: Mapping[str, Any],
    *,
    input_dim: int,
    output_dim: int,
    observation_spec: ObservationSpec,
) -> ModelBuildResult:
    """Build a retargeter from config without importing torch at module import time."""

    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    family = _canonical_family(str(model_cfg.get("family", "temporal_mlp")))
    if family == "temporal_mlp":
        from .mlp import OnlineRetargetMLP

        hidden_dims = tuple(int(value) for value in model_cfg.get("hidden_dims", [512, 512, 256]))
        model = OnlineRetargetMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=str(model_cfg.get("activation", "silu")),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif family == "temporal_transformer":
        from .temporal import TemporalTransformerRetargeter

        model = TemporalTransformerRetargeter(
            input_dim=input_dim,
            output_dim=output_dim,
            history_frames=observation_spec.history_frames,
            source_feature_dim=observation_spec.source_feature_dim(),
            d_model=int(model_cfg.get("d_model", 256)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pooling=str(model_cfg.get("pooling", "last")),
        )
    elif family == "token_transformer":
        from .temporal import TokenizedTransformerRetargeter

        model = TokenizedTransformerRetargeter(
            input_dim=input_dim,
            output_dim=output_dim,
            history_frames=observation_spec.history_frames,
            source_feature_dim=observation_spec.source_feature_dim(),
            morphology_dim=observation_spec.morphology_dim(),
            robot_state_dim=observation_spec.robot_state_dim(),
            latent_dim=int(model_cfg.get("latent_dim", 128)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_encoder_layers=int(model_cfg.get("num_encoder_layers", 2)),
            num_decoder_layers=int(model_cfg.get("num_decoder_layers", 2)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            output_mode=str(model_cfg.get("output_mode", "position")),
            use_prev_state=bool(model_cfg.get("use_prev_state", True)),
            skeleton_encoder_mode=str(model_cfg.get("skeleton_encoder_mode", "shared_mlp")),
            num_actor_encoders=int(model_cfg.get("num_actor_encoders", 0)),
        )
    elif family == "flow_matching":
        from .temporal import FlowMatchingRetargeter

        hidden_dims = tuple(int(value) for value in model_cfg.get("hidden_dims", [512, 512, 256]))
        model = FlowMatchingRetargeter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=str(model_cfg.get("activation", "silu")),
            dropout=float(model_cfg.get("dropout", 0.0)),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 32)),
            inference_steps=int(model_cfg.get("inference_steps", 8)),
        )
    else:
        raise ValueError(f"unsupported model family: {model_cfg.get('family')}")
    return ModelBuildResult(model=model, family=family, config=dict(model_cfg))


def _canonical_family(family: str) -> str:
    key = family.lower().replace("-", "_")
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
    }
    return aliases.get(key, key)
