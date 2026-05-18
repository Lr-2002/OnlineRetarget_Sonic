# Token Transformer Baseline Status

Date: 2026-05-18

## Scope

This note records the first runnable continuous-token Transformer baseline for
SOMA proportional BVH-window observations to Unitree G1 joint targets.

The design choices are recorded in:

- `docs/design/tokenized_transformer_baseline.md`

The implementation is deliberately still a scaffold. It proves that the local
data path, independent token VAEs, joint auxiliary token training, Transformer
prediction, checkpoint writing, and offline eval plumbing can run. It is not a
long-training result or a claim that the Transformer beats the MLP baseline.

## Token Components

All token latents default to `128`.

| Component | Raw tensor | Current encoder target | Current evidence |
| --- | --- | --- | --- |
| Skeleton | `ObservationSpec` morphology slice, currently 13D | MLP VAE reconstruction | `runs/pretrain/bones_bvh_token_vae_debug_smoke/skeleton/checkpoint.pt` |
| Motion | Source BVH FK history window, currently 1440D | MLP VAE reconstruction | `runs/pretrain/bones_bvh_token_vae_debug_smoke/motion/checkpoint.pt` |
| Action/state | G1 target joint vector, 29D | MLP VAE reconstruction | `runs/pretrain/bones_bvh_token_vae_debug_smoke/action/checkpoint.pt` |
| Previous state | `prev_target_joints` when emitted; zero fallback for older JSONL | Teacher-forced previous G1 state token in Transformer | `scripts/train.py`, `src/online_retarget/data/sonic_windowed_builder.py` |
| Query | Learned next-frame token | Cross-attention decoder query | `src/online_retarget/models/temporal.py` |

## Implemented Code

- `src/online_retarget/models/token_vae.py`
  - `MLPTokenVAE`
  - `vae_loss`
- `scripts/pretrain_token_vaes.py`
  - Reads supervised JSONL.
  - Splits `observation` into skeleton and motion slices via `ObservationSpec`.
  - Uses `target_joints` as the action token training signal.
  - Writes one checkpoint/report per component plus `pretrain_report.json`.
- `src/online_retarget/models/temporal.py`
  - `TokenizedTransformerRetargeter`
  - Encodes skeleton/motion/prev-state into continuous tokens.
  - Uses source skeleton+motion memory and a previous-state-conditioned query.
  - Adds auxiliary reconstruction heads for joint training.
- `scripts/train.py`
  - Loads `prev_target_joints` for teacher forcing.
  - Supports token Transformer loss with retargeting plus auxiliary token losses.
  - Fixes predict-only token Transformer eval by constructing `prev_y`.
- `configs/bones_bvh_token_vae_debug.yaml`
- `configs/bones_bvh_token_transformer_debug.yaml`

## Local Smoke Evidence

Independent VAE pretrain smoke:

```bash
PYTHONPATH=src:. /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/pretrain_token_vaes.py \
  --config configs/bones_bvh_token_vae_debug.yaml \
  --allow-debug-data \
  --output-dir runs/pretrain/bones_bvh_token_vae_debug_smoke \
  --max-steps 2 \
  --batch-size 128
```

Artifacts:

- `runs/pretrain/bones_bvh_token_vae_debug_smoke/pretrain_report.json`
- `runs/pretrain/bones_bvh_token_vae_debug_smoke/skeleton/checkpoint.pt`
- `runs/pretrain/bones_bvh_token_vae_debug_smoke/motion/checkpoint.pt`
- `runs/pretrain/bones_bvh_token_vae_debug_smoke/action/checkpoint.pt`

Token Transformer smoke:

```bash
PYTHONPATH=src:. /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/train.py \
  --config configs/bones_bvh_token_transformer_debug.yaml \
  --allow-debug-data \
  --output-dir runs/train/bones_bvh_token_transformer_debug_smoke \
  --max-steps 2 \
  --batch-size 64
```

Artifacts:

- `runs/train/bones_bvh_token_transformer_debug_smoke/checkpoint.pt`
- `runs/train/bones_bvh_token_transformer_debug_smoke/train_report.json`
- `runs/train/bones_bvh_token_transformer_debug_smoke/train_predictions.jsonl`
- `runs/train/bones_bvh_token_transformer_debug_smoke/eval/train_offline_eval/eval_summary.json`

Validation predict-only smoke:

```bash
PYTHONPATH=src:. /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/train.py \
  --config configs/bones_bvh_token_transformer_debug.yaml \
  --allow-debug-data \
  --predict-only \
  --checkpoint runs/train/bones_bvh_token_transformer_debug_smoke/checkpoint.pt \
  --samples-jsonl runs/supervised/somabvh_task_val_h8_stride10_limit1000/samples.jsonl \
  --output-dir runs/eval/bones_bvh_token_transformer_debug_smoke_val_predict
```

Artifact:

- `runs/eval/bones_bvh_token_transformer_debug_smoke_val_predict/predict_report.json`

This smoke reported 1,000 val samples, MSE `0.1469971388578415`, joint RMSE
`0.3496978777276041`, and action similarity `0.6054776032982514`. The values are
not a performance claim because the checkpoint was trained for only two steps.

## Current Limitations

- Current supervised JSONL is still a debug artifact:
  `runs/supervised/somabvh_task_train_h8_stride10_limit5000/samples.jsonl`.
- The local smoke uses `--allow-debug-data`, so it bypasses the formal M2Q gate.
- The independent VAEs are not yet loaded into the Transformer model. The
  Transformer currently learns its own projection encoders with auxiliary
  reconstruction losses.
- The action VAE currently uses `target_joints`, while the previous-state token
  uses `prev_target_joints` when available.
- Closed-loop autoregressive rollout, FK-MPJPE, batch-1 latency, DDP launch, and
  WandB online sync remain pending.

## Remote Handoff

Do not expose WandB tokens in logs. Login or environment setup should happen on
the remote host without printing the token.

SSH target supplied by the user:

```bash
ssh -v -i /home/user/.ssh/id_aliyun root@106.14.35.26 -p 1050
```

Suggested remote execution order after code sync:

```bash
PYTHONPATH=src:. python scripts/pretrain_token_vaes.py \
  --config configs/bones_bvh_token_vae_debug.yaml \
  --allow-debug-data \
  --wandb-mode online \
  --output-dir runs/pretrain/bones_bvh_token_vae_debug_remote \
  --max-steps 1000 \
  --batch-size 512

torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config configs/bones_bvh_token_transformer_debug.yaml \
  --allow-debug-data \
  --wandb-mode online \
  --output-dir runs/train/bones_bvh_token_transformer_debug_remote_8gpu \
  --max-steps 1000 \
  --batch-size 512
```

Before reporting remote results, verify:

- `train_report.json` records `world_size=8` and the expected git SHA.
- WandB run is online or offline-synced with config, git SHA, and output paths.
- Offline eval summary exists.
- A validation/predict-only run exists on heldout samples, not only train eval.
