"""Auxiliary losses for SONIC-native OnlineRetarget training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class G1DynamicsActionLoss(nn.Module):
    """Supervise the SONIC ``g1_dyn`` decoder against G1 target actions."""

    def __init__(
        self,
        loss_type: str = "mse",
        target_key: str = "g1_target_action",
        decoder_name: str = "g1_dyn",
        pred_key: str = "action",
        **_: Any,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.target_key = target_key
        self.decoder_name = decoder_name
        self.pred_key = pred_key

    def forward(self, loss_inputs: dict[str, Any]) -> torch.Tensor:
        tokenizer_obs = loss_inputs["tokenizer_obs"]
        target = tokenizer_obs[self.target_key]
        pred = _decoder_prediction(loss_inputs, self.decoder_name, self.pred_key)
        pred, target = _align_pred_target(pred, target)
        return _loss(pred, target, self.loss_type)


class ActionSmoothnessLoss(nn.Module):
    """Penalize frame-to-frame jumps in the decoded dynamics action."""

    def __init__(
        self,
        decoder_name: str = "g1_dyn",
        pred_key: str = "action",
        loss_type: str = "mse",
        **_: Any,
    ) -> None:
        super().__init__()
        self.decoder_name = decoder_name
        self.pred_key = pred_key
        self.loss_type = loss_type

    def forward(self, loss_inputs: dict[str, Any]) -> torch.Tensor:
        pred = _decoder_prediction(loss_inputs, self.decoder_name, self.pred_key)
        if pred.ndim < 3 or pred.shape[-2] < 2:
            return pred.new_tensor(0.0)
        delta = pred[..., 1:, :] - pred[..., :-1, :]
        return _loss(delta, torch.zeros_like(delta), self.loss_type)


def _decoder_prediction(
    loss_inputs: dict[str, Any],
    decoder_name: str,
    pred_key: str,
) -> torch.Tensor:
    decoded_outputs = loss_inputs["decoded_outputs"]
    decoder_output = decoded_outputs.get(decoder_name, {})
    if pred_key in decoder_output:
        return decoder_output[pred_key]
    if "body_action" in decoder_output:
        return decoder_output["body_action"]
    if "meta_action" in decoder_output:
        return decoder_output["meta_action"]
    return loss_inputs["action_mean"]


def _align_pred_target(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.ndim == target.ndim + 1 and pred.shape[-2] == 1:
        pred = pred.squeeze(-2)
    if target.ndim == pred.ndim + 1 and target.shape[-2] == 1:
        target = target.squeeze(-2)
    if pred.ndim == target.ndim + 1:
        target = target.unsqueeze(-2).expand(*pred.shape[:-1], target.shape[-1])
    elif target.ndim == pred.ndim + 1:
        pred = pred.unsqueeze(-2).expand(*target.shape[:-1], pred.shape[-1])

    common_dim = min(pred.shape[-1], target.shape[-1])
    return pred[..., :common_dim], target[..., :common_dim]


def _loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_tensor(0.0)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "huber":
        return F.huber_loss(pred, target)
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    raise ValueError(f"unsupported loss_type: {loss_type}")
