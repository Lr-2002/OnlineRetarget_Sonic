"""Runtime SONIC modules for OnlineRetarget formal training."""

from __future__ import annotations

from typing import Any


try:
    from gear_sonic.trl.modules.universal_token_modules import (
        UniversalTokenModule as _SonicUniversalTokenModule,
    )
except ModuleNotFoundError:  # pragma: no cover - local tests may not install SONIC.
    _SonicUniversalTokenModule = object


class KinematicActionUniversalTokenModule(_SonicUniversalTokenModule):
    """Use the SONIC ``g1_kin`` decoder as the actor action source.

    SONIC's upstream ``UniversalTokenModule`` only populates ``action_mean``
    from the ``g1_dyn`` decoder.  Formal OnlineRetarget runs intentionally
    filter active decoders to ``g1_kin`` only, so this wrapper derives the PPO
    action mean from the decoded kinematic G1 joint-position trajectory instead
    of re-enabling the dynamics decoder.
    """

    def __init__(
        self,
        *args: Any,
        kinematic_action_decoder: str = "g1_kin",
        kinematic_action_output: str = "command_multi_future_nonflat",
        kinematic_action_frame_index: int = 0,
        kinematic_action_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kinematic_action_decoder = str(kinematic_action_decoder)
        self.kinematic_action_output = str(kinematic_action_output)
        self.kinematic_action_frame_index = int(kinematic_action_frame_index)
        self.kinematic_action_dim = (
            int(kinematic_action_dim) if kinematic_action_dim is not None else None
        )

    def forward(
        self,
        input_data: Any,
        compute_aux_loss: bool = False,
        return_dict: bool = False,
        latent_residual: Any = None,
        latent_residual_mode: str = "post_quantization",
        **kwargs: Any,
    ) -> Any:
        output = super().forward(
            input_data,
            compute_aux_loss=compute_aux_loss,
            return_dict=True,
            latent_residual=latent_residual,
            latent_residual_mode=latent_residual_mode,
            **kwargs,
        )
        if output.get("action_mean") is None:
            output["action_mean"] = self._action_mean_from_g1_kin(output["decoded_outputs"])
        if compute_aux_loss or return_dict:
            return output
        return output["action_mean"]

    def _action_mean_from_g1_kin(self, decoded_outputs: dict[str, Any]) -> Any:
        decoder_output = decoded_outputs.get(self.kinematic_action_decoder)
        if decoder_output is None:
            raise RuntimeError(
                f"kinematic action decoder {self.kinematic_action_decoder!r} "
                "was not decoded; keep active_decoders=[g1_kin] for kin-only runs"
            )
        command = decoder_output.get(self.kinematic_action_output)
        if command is None:
            raise RuntimeError(
                f"kinematic action output {self.kinematic_action_output!r} is missing "
                f"from decoder {self.kinematic_action_decoder!r}"
            )

        action_dim = self.kinematic_action_dim or int(getattr(self, "actions_dim"))
        if command.shape[-1] < action_dim:
            raise RuntimeError(
                f"{self.kinematic_action_output} has trailing dim {command.shape[-1]}, "
                f"smaller than action dim {action_dim}"
            )

        if len(command.shape) >= 4:
            frame_count = command.shape[-2]
            frame_index = self.kinematic_action_frame_index
            if frame_index < 0:
                frame_index += frame_count
            if frame_index < 0 or frame_index >= frame_count:
                raise RuntimeError(
                    f"kinematic_action_frame_index={self.kinematic_action_frame_index} "
                    f"is outside decoded frame count {frame_count}"
                )
            command = command[..., frame_index, :]
        return command[..., :action_dim].contiguous()
