# Milestones

## M0 - Repository Initialization

Gate:

- Git repository initialized.
- Read-only data rule documented.
- Initial docs cover literature, data, architecture, milestones, and experiment tracking.
- Pure-Python smoke tests pass.

Status: in progress.

## M1 - Data Inventory and Splits

Gate:

- Loader can read BONES-SEED metadata and build actor-level train/val/test splits.
- Source-target pairing is reproducible from metadata.
- Derived index is written outside `/home/user/data/motion_data`.
- Data report includes actor/skeleton distribution and missing-file checks.

## M2 - Direct Supervised Baseline

Gate:

- Minimal temporal MLP trains on a small subset.
- WandB logs config, git SHA, metrics, and artifact paths.
- DDP launch works on one node.
- Inference benchmark exists and reports p50/p95/p99 on 4090.

## M3 - Offline Evaluation Suite

Gate:

- Metrics run independent of training.
- Evaluation outputs per-category and per-actor breakdowns.
- Failure clips are exported to a reviewable output folder.
- MPJPE, G1 joint RMSE, action similarity, joint jump rate, and limit violation rate are reported.

## M4 - Isaac Lab Tracking Evaluation

Gate:

- G1 simulator asset and joint limits are verified.
- Predicted references can be replayed in Isaac Lab.
- Tracking success/fall rate and world-frame MPJPE are logged.
- Evaluation can run in tmux with a reproducible config.

## M5 - Physical Refinement Branch

Gate:

- Compare direct supervised model to a physics-refined target source.
- Implement one refinement route only: BeyondMimic-style tracking rollout, GMR/SOMA retargeter filtering, or NMR-like clustered experts.
- Show improvement on simulator metrics, not only offline loss.

## M6 - Online Retargeting Demo

Gate:

- Live or streamed skeleton input runs through the retargeter.
- End-to-end p95 inference and packing latency satisfy the 1 ms model budget or documented system budget.
- Output drives G1 simulator without manual per-clip tuning.
- Failure cases are logged with input, output, metrics, and config.
