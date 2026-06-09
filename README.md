# OnlineRetarget Sonic

Learning-based online retargeting from BONES-SEED SOMA human motion to Unitree G1 robot motion.

The current checked-in execution surface is a strict supervised SONIC/SOMA
motionlib lane. It keeps the target decoder to `g1_kin`, uses one 4-GPU DDP job
per config, and excludes PPO, Isaac rollout, reward, and `g1_dyn` surfaces from
the active supervised training path.

- LR-280 kin/walk data-package configs:
  `sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu.json` and
  `sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu.json`. These
  pin the accepted paired SOMA motionlib kin/walk package digest and share the
  LR-270 metric/eval cohort.
- LR-273/LR-284 proportional treatment:
  `sonic_kin_soma_motionlib_proportional_4gpu.json`, with temporal-consistency
  and A/B command-overlap auxiliary losses enabled.
- LR-274 matched proportional loss-off baseline:
  `sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json`.
- LR-177 A0 SOMA motionlib ablations remain supporting evidence for frozen
  Skeleton Geometry AE conditioning versus no skeleton encoder. See
  `docs/status/lr177_a0_usage.md`.

Historical SONIC-native shared-token configs named
`sonic_kin_only_soma_encoder_{uniform,proportional}.json` remain in the repo as
contract and migration artifacts. They are not the default current launcher
surface; use `docs/status/online_retarget_sonic_training_boundary_2026-05-20.md`
for the active remote launch matrix.

## Repository Status

- Data source: `/home/user/data/motion_data` is read-only.
- Active source lanes: BONES-SEED `soma_uniform.tar` and `soma_proportional.tar`, with proportional grouped by `actor_uid`.
- Primary target lane: `/home/user/data/motion_data/bones_sonic` BONES-SONIC NPZ files, joined to BONES-SEED metadata.
- Debug lanes: legacy `g1.tar` and AMASS/GMR retargeted NPZ files remain useful for parser regression, but they are not the active SONIC kin-only baseline source/target definition.
- Initial target: Unitree G1 29-DoF joint trajectories.
- Simulator target: Isaac Lab, introduced after offline metrics are stable.

## Setup

Conda plus direnv is the intended environment boundary:

```bash
conda env create -f environment.yml
direnv allow
python -m pip install -e ".[dev]"
```

The current machine does not need to have this environment active to inspect docs or run the pure-Python smoke tests.

## Repo Hygiene

- Treat `/home/user/data/motion_data` as read-only. Derived artifacts belong
  under `runs/`, `outputs/`, or another explicit output root.
- Keep formal training traceable: commit before a launch, use one config per
  4-GPU job, and let the launcher record git SHAs and manifests.
- Prefer explicit `CONFIG=...` values when launching. The compatibility wrapper
  `scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh` currently forwards
  into the strict supervised SOMA motionlib launcher.
- Keep historical A1/A2/B1/B2 or legacy SONIC-native evidence labeled as
  historical when updating status docs.

## Useful Commands

Run the focused local checks for the current strict supervised launch surface:

```bash
PYTHONPATH=src:. python3 -m unittest \
  tests.test_remote_launcher_guardrails \
  tests.test_data_package_indicator \
  -q
```

Compile Python sources without importing optional runtime dependencies:

```bash
python3 -m compileall -q src scripts
```

Inventory the local BONES-SEED metadata without modifying data:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py inventory --data-root /home/user/data/motion_data
```

Build the active BONES-SONIC NPZ index:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-sonic-index \
  --sonic-root /home/user/data/motion_data/bones_sonic \
  --metadata-csv /home/user/data/motion_data/metadata/seed_metadata_v003.csv \
  --output-root runs \
  --run-name bones_sonic_index_full_v0
```

Current full artifact: `runs/indices/bones_sonic_index_full_v0/sonic_index_report.json` reports 142,220 NPZ files, 522 actors, all 50 Hz, and schema `ok` for every file.

Run a bounded SONIC-native quality smoke from the NPZ tensors:

```bash
PYTHONPATH=src /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/inspect_bones_seed.py scan-sonic-quality \
  --index-csv runs/indices/bones_sonic_index_full_v0/sonic_index.csv \
  --output-root runs \
  --limit 512 \
  --sample-by category \
  --sample-by date \
  --model-xml /home/user/repos/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
  --frame-stride 2
```

The scanner records body-origin contact, XML joint-limit, and body-origin self-collision metrics as metric-only by default; use explicit `--enable-*` flags only after calibration.

Build an actor-heldout split index and metadata-level curation report under `runs/`:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py split-index \
  --data-root /home/user/data/motion_data \
  --output-root runs \
  --seed 17 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --policy-name metadata_balanced_v0 \
  --min-duration-frames 60
```

Scan G1 target CSV quality stats from a split index:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-g1-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --fps 120 \
  --max-joint-velocity 20 \
  --max-root-speed 8
```

BONES-SEED source BVH files observed so far declare `Frame Time: 0.008333`, so paired source/G1 scans should use 120 Hz unless a specific clip proves otherwise. The G1 scanner defaults to 120 Hz.

Scan source SOMA BVH quality stats from a split index:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-source-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --max-channel-velocity 3000 \
  --max-root-speed 500
```

Scan source SOMA BVH FK/contact quality stats for M2Q calibration:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-source-fk-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --ground-height 0.0 \
  --frame-stride 2 \
  --max-frames 256 \
  --contact-height-threshold 0.04 \
  --max-contact-slide-speed 0.25 \
  --max-mean-foot-clearance 0.10 \
  --max-penetration-depth 0.03 \
  --min-contact-frame-ratio 0.05
```

`scan-source-fk-quality` derives contact/slide FPS from each BVH `Frame Time` by default. Use `--fps` only for an explicit override.

Scan source/G1 pair consistency and target provenance:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-pair-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 560 \
  --sample-by category \
  --sample-by split \
  --expected-source-frame-time 0.008333333333333333 \
  --g1-fps 120 \
  --max-frame-count-delta 0 \
  --max-duration-delta-sec 0.001 \
  --target-provenance kinematic_g1_csv
```

Evaluate prediction/target JSONL outputs offline:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py offline-eval \
  --input-jsonl runs/predictions/example.jsonl \
  --output-root runs \
  --run-name baseline_eval
```

Build a tiny raw-BVH-channel supervised JSONL for data-path debugging:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-supervised-jsonl \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --split train \
  --limit 8 \
  --history-frames 8
```

Build a schema-compatible 30-body window JSONL from the curated index:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-windowed-jsonl \
  --data-root /home/user/data/motion_data \
  --index-csv runs/curated/smoke_source_g1_limit100/curated_index.csv \
  --output-root runs \
  --split train \
  --curation-action keep \
  --curation-action downweight \
  --action-column merged_quality_action \
  --limit 4 \
  --history-frames 8
```

Merge split index with source/G1 quality stats into a curated index:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py merge-quality \
  --split-index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --source-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit100/source_bvh_quality_stats.jsonl \
  --source-fk-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit100/source_fk_quality_stats.jsonl \
  --g1-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_quality_stats.jsonl \
  --pair-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_pair_limit100/pair_quality_stats.jsonl \
  --output-root runs \
  --run-name smoke_source_g1_limit100
```

The merge writes:

- `runs/curated/<run-name>/curated_index.csv`
- `runs/curated/<run-name>/curated_report.json`
- `runs/curated/<run-name>/worst_clips.csv`

Run the M2Q promotion preflight for a curated run:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py preflight-curation-policy \
  --curated-run-dir runs/curated/representative_source_g1_pair_limit560_by_category_split
```

The preflight writes `policy_preflight.json`, discovers matching threshold proposal artifacts
from the stats paths in `curated_report.json`, and reports the exact blockers before formal M5
training can use the policy audit.

After threshold proposals have been reviewed, write the accepted threshold-policy artifact from the
same curated run:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py accept-curation-threshold-policy \
  --curated-run-dir runs/curated/representative_source_g1_pair_limit560_by_category_split \
  --accepted-by <reviewer> \
  --rationale "<why these proposal files are accepted>" \
  --representative
```

This writes `threshold_policy.json` next to `curated_report.json` and refuses to overwrite an
existing policy unless `--overwrite` is supplied. It does not bypass the preflight audit: scan
coverage and manual-review decisions still have to pass before the policy is promotable.

Run smoke tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Launch a future training run in tmux:

```bash
scripts/train_tmux.sh configs/baseline_mlp.yaml
```

Debug the training contract without starting optimization:

```bash
PYTHONPATH=src python3 scripts/train.py --config configs/baseline_mlp.yaml --dry-run --limit 1
```

Run a formal supervised JSONL training loop in an environment with torch. The config must point to a curated index, `merged_quality_action`, a quality policy ID, and an existing quality report:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --config configs/baseline_mlp.yaml \
  --samples-jsonl runs/supervised/train_merged-quality-action_30b_h8_limit4/samples.jsonl \
  --max-steps 100 \
  --batch-size 8
```

Successful torch training writes `checkpoint.pt`, `train_report.json`, `train_predictions.jsonl`, and a train-set offline eval report under the run output directory. WandB is disabled by default with `tracking.wandb_mode: disabled`; set it to `offline` or `online` only in the intended conda environment.

For data-path debugging only, bypass the M2Q gate explicitly:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --config configs/baseline_mlp.yaml \
  --samples-jsonl runs/supervised/train_h8_limit8/samples.jsonl \
  --allow-debug-data \
  --max-steps 10 \
  --batch-size 8
```

### LR-177 A0 SOMA Motionlib Runs

Detailed usage, config matrix, dimensions, and metric limitations are documented in
`docs/status/lr177_a0_usage.md`.

Dry-run one A0 config with the same 4-rank shape as formal training:

```bash
export CONFIG=configs/sonic_kin_soma_motionlib_a0_frozen_ae_uniform_4gpu.json
export KIN_RUN_GROUP=lr177_a0_frozen_ae_uniform_dryrun_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src:. /workspace/isaaclab/_isaac_sim/python.sh -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  scripts/train_sonic_kin_skeleton_ae.py \
  --config "${CONFIG}" \
  --dry-run \
  --wandb-mode disabled
```

Launch a formal 4-GPU run through the guarded tmux launcher:

```bash
CONFIG=configs/sonic_kin_soma_motionlib_a0_frozen_ae_uniform_4gpu.json \
KIN_RUN_GROUP=lr177_a0_frozen_ae_uniform_$(date -u +%Y%m%dT%H%M%SZ) \
scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
```

Use the same command shape for the proportional frozen-AE config and both no-skeleton-encoder
configs by changing `CONFIG`. The launcher refuses uncommitted or stale control-repo code before
starting formal training; do not restart or modify already-running remote jobs that were launched
from an earlier committed SHA.

### Current Strict Supervised SOMA Motionlib Runs

The guarded 4-GPU launcher for the current strict supervised lane is:

```bash
CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu.json \
scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh

CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu.json \
scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
```

For the broader proportional treatment/baseline comparison:

```bash
CONFIG=configs/sonic_kin_soma_motionlib_proportional_4gpu.json \
scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh

CONFIG=configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json \
scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
```

All four configs use `training_lane=soma_motionlib_kin_only`, one 4-GPU DDP job,
`g1_kin` targets only, committed/latest repo checks, and output roots under
`outputs/`.

Dry-run the latency benchmark contract:

```bash
python3 scripts/benchmark_latency.py --dry-run --output-json runs/benchmarks/baseline_mlp_dry_run.json
```

Dry-run the Isaac Lab evaluation contract:

```bash
python3 scripts/eval_isaac.py \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --run-name dry_run \
  --dry-run
```

Launch the local web console for upload -> retarget -> preview runs:

```bash
PYTHONPATH=src python3 scripts/run_web.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`, upload a `.bvh` or SMPL-like `.npz`, and the app writes a per-run bundle under
`outputs/web_runs/` with the source file, `retargeted_g1_preview.csv`, `retarget_report.json`,
`pipeline_result.json`, and `mujoco_g1_render.mp4` when MuJoCo rendering is available. The Web path
loads BVH/SMPL-like inputs, retargets to Unitree G1 with GMR when its dependencies are installed,
falls back with an explicit blocker report otherwise, and renders the G1 MJCF through
`mujoco.Renderer` rather than front-end point drawing. The upload form includes a `Frame axes`
toggle that controls whether MuJoCo body frames are included in the rendered video. Full SMPL/SMPL-X
body-model decoding still requires the SMPL-X dependencies; generic `.npz` files with
`poses`/`pose_body` plus `trans` use approximate SMPL joint targets for GMR.

## Key Docs

- `docs/research/literature_review.md` - paper/source survey and design implications.
- `docs/research/data_inventory.md` - local data scan and skeleton/target assumptions.
- `docs/architecture.md` - training and inference pipeline.
- `docs/milestones.md` - checkable gates.
- `docs/experiment_tracking.md` - WandB, git, DDP, and tmux rules.
- `docs/status/online_retarget_sonic_training_boundary_2026-05-20.md` - active remote launch boundary.
- `docs/status/lr291_repo_cleanup_2026-06-09.md` - current repo/doc cleanup note.
- `docs/status/lr177_a0_usage.md` - LR-177 frozen-AE/no-skeleton-encoder config usage.
