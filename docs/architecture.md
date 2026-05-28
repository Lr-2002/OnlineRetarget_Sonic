# Architecture

Goal: a compact online retargeter that maps BONES-SEED SOMA proportional human motion to Unitree G1 motion references with sub-1 ms inference on an RTX 4090.

Active LR-185 problem definition: compare the two formal kin-only SONIC SOMA
encoder baselines, `SOMA uniform -> G1` and `SOMA proportional -> G1`, with
each launched as one 4-GPU run. The proportional source is actor-specific SOMA
BVH plus morphology/shape conditioning; the uniform source is the shared-SOMA
skeleton control. The target is direct Unitree G1 kinematic motion, initially
29D joint position/velocity plus anchor orientation from the BONES-SONIC lane.

Current implementation boundary: model and training changes now start from the
SONIC codebase, not from a separate OnlineRetarget retargeter. OnlineRetarget is
kept as the data registry, curation/audit, metric, documentation, and experiment
tracking layer for this lane. New features such as proportional-skeleton routing,
actor/shape conditioning, supervisor logging, visualization hooks, or latent
alignment should be added through SONIC's existing `UniversalTokenModule`,
`soma` encoder path, shared token space, and G1 decoder/controller surfaces unless
a later decision explicitly reopens a standalone trainer branch.

## Baseline Scope

First active baseline: SONIC-native G1 controller training on filtered BONES
data, followed by SONIC-native proportional-skeleton adaptation.

Do not add a new Transformer/flow/VAE model in OnlineRetarget for the active
lane. Those branches are archived research scaffolds unless a measured SONIC
failure mode justifies reopening them.

## Training Pipeline

```mermaid
flowchart LR
    A[Filtered BONES / registry audit in OnlineRetarget] --> B[SONIC motion library / index]
    C[SOMA proportional source + actor shape] --> D[SONIC tokenizer observations]
    B --> D
    D --> E[SONIC soma encoder or adapter]
    E --> F[Shared SONIC latent/token space]
    F --> G[SONIC g1_kin + g1_dyn decoders]
    G --> H[Isaac Lab / controller rollout]
    H --> I[WandB + metrics + videos]
```

## Inference Pipeline

```mermaid
flowchart LR
    A[live SOMA-proportional-like skeleton frames] --> B[SONIC-compatible observation packer]
    C[current G1 state / controller context] --> B
    B --> D[SONIC policy module]
    D --> E[G1 action / reference]
    E --> F[controller or simulator]
```

## Observation Design

Initial observation blocks:

- Source skeleton history: local joint/body positions, orientations if available, velocities, and contact proxies.
- Source morphology: actor height, foot length, shoulder/hip/knee/ankle measurements, SOMA shape parameters, and skeleton ID embedding only if needed.
- Robot state: current G1 joint position, joint velocity, previous action, base orientation/IMU roll-pitch, angular velocity.
- Optional future: short future source window if online latency permits buffering.

For the active lane, these fields must be exposed as SONIC tokenizer observations
or conditioning features rather than a separate OnlineRetarget observation
contract. The old fixed-width flattened-window contract remains useful for
offline probes, but it is not the implementation path for new model features.

Mid-term design choice: add a skeleton encoder between raw source windows and the retargeter. The encoder would convert SOMA skeleton structure, bone proportions, joint topology, local motion features, contacts, and optional shape parameters into learned skeleton/motion features, then feed those features to the direct G1 retargeter. This is not part of the first baseline, but it is a likely next branch if flattened FK windows plus morphology are not enough to capture cross-person motion semantics and skeleton-specific constraints.

Current schema implementation:

- `MotionPairRef` in `src/online_retarget/data/schema.py` consumes `split_index.csv` rows.
- `ObservationSpec(history_frames=8, source_body_count=30)` has flattened dim 1,547:
  - source history positions + velocities: 1,440
  - morphology: 13
  - robot state side channel: 94
- `OutputSpec` defaults to direct 29D G1 joint position delta.

## Output Design

Default output: 29-dimensional G1 joint target delta or next joint position.

Alternatives:

- Full generalized coordinate target: root plus 29 joints. Useful for offline reference generation, less direct for onboard control.
- Latent output: requires VAE encoder/decoder and a clear metric showing it improves generalization or smoothness.
- Short-horizon output: may improve temporal consistency, but increases output size and latency.

## Model Families

| Family | Use | Risk |
| --- | --- | --- |
| SONIC shared `soma` encoder | Active baseline extension point | May underfit actor-specific proportional skeletons |
| SONIC shared trunk + actor/shape adapter | First proportional-skeleton adaptation branch | Requires stable actor/shape routing and actor-heldout eval |
| One encoder per actor/skeleton group | Escalation only if adapters underfit | Hundreds of modules are hard to manage and easy to overfit |
| OnlineRetarget temporal MLP/Transformer/flow | Archived scaffold / diagnostic baseline only | Not the active code path |
| PDF-HR-style pose prior | Regularizer/scorer for G1 plausibility | Needs high-quality positive pose set |

## Losses

First supervised loss set:

- G1 joint position loss.
- G1 joint velocity loss.
- Body MPJPE/body position loss when `body_pos_w` or FK is available.
- Smoothness penalty on output deltas.
- Joint limit penalty after simulator joint limits are confirmed.
- Action similarity loss as cosine alignment over action/joint-delta vectors.

Later physics-aware losses:

- Foot sliding and ground penetration.
- Self-collision/self-intersection.
- Tracking policy success in Isaac Lab.
- PDF-HR-style pose prior distance.

## Metrics

Offline metrics live in `src/online_retarget/metrics.py` and must remain training-independent.

Initial metrics:

- `mpjpe`: body/joint position error.
- `joint_mae`: mean absolute G1 joint-space error; this is the eval-side counterpart of the NMR-style L1 supervised baseline.
- `joint_mse`: mean squared G1 joint-space error for comparison against older MSE smoke runs.
- `joint_rmse`: G1 joint-space RMSE.
- `max_joint_abs_error`: worst per-sample joint residual for catching localized mapping failures hidden by averages.
- `joint_velocity_rmse`: velocity-space residual when sequence predictions contain at least two frames.
- `action_similarity`: cosine similarity over predicted vs target action/delta vectors.
- `predicted_joint_jump_rate`, `target_joint_jump_rate`, and `predicted_minus_target_joint_jump_rate`: thresholded velocity discontinuity diagnostics for separating target-data artifacts from model-introduced artifacts.
- `joint_limit_violation_rate`: thresholded limit violation rate.
- `contact_artifact_metrics`: target-contact-aware foot float, contact slide, ground penetration, and clearance metrics for JSONL samples with body positions and foot body metadata.

Current eval implementation:

- `src/online_retarget/evaluation.py` evaluates JSONL prediction/target records.
- CLI: `PYTHONPATH=src python3 scripts/inspect_bones_seed.py offline-eval --input-jsonl <path> --output-root runs --run-name <name>`.
- Outputs: `eval_summary.json`, `per_sample_metrics.csv`, and `failure_manifest.csv`.
- Grouping: overall, per actor, per category, per package, and per quality flag.
- Optional contact artifact metrics are emitted when a JSONL row provides `predicted_body_pos`, `target_body_pos`, and either `foot_indices` or `body_names` plus `foot_body_names`/`foot_names`. Contact frames are inferred from the target body positions so the metric catches predicted floating or skating during target support.

Future simulator metrics:

- Tracking success rate.
- Episode length / fall rate.
- World-frame MPJPE.
- Contact/foot sliding.
- Sim-to-sim robustness under noise and latency.

## Latency Gate

The online model is not accepted until measured on target hardware. The benchmark must report:

- batch size 1 latency
- warmup count
- p50/p95/p99 latency
- device, dtype, and compile/export mode
- parameter count and activation footprint

The default acceptance target is p95 under 1 ms on RTX 4090.

Current benchmark scaffold:

- `scripts/benchmark_latency.py --dry-run` records observation/output dimensions and model hidden dims without importing torch.
- Non-dry-run requires torch and target hardware, then reports p50/p95/p99/mean/max latency plus parameter count.

## Simulator Gate

Current Isaac Lab scaffold:

- `scripts/eval_isaac.py --dry-run` writes an `isaac_eval_status.json` with expected simulator metrics.
- Non-dry-run explicitly blocks until Isaac Lab and a G1 replay/tracking task binding are available.
