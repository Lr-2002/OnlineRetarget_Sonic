# OnlineRetarget

Learning-based online retargeting from heterogeneous human skeleton motion to Unitree G1 robot motion.

The immediate goal is a compact, evaluation-first retargeter that can ingest human/SOMA skeleton data plus robot state and produce G1 motion references fast enough for online use. The first baseline is direct G1 joint output; latent, flow, and diffusion variants are tracked as later design branches after the baseline is measurable.

## Repository Status

- Data source: `/home/user/data/motion_data` is read-only.
- Primary dataset: BONES-SEED metadata plus SOMA and G1 motion archives.
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

## Useful Commands

Inventory the local BONES-SEED metadata without modifying data:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py inventory --data-root /home/user/data/motion_data
```

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
and `pipeline_result.json`. The current web path implements BVH loading, a deterministic
rule-based G1 preview target, G1 MJCF kinematic playback, MuJoCo stepping, and approximate
SMPL-like `.npz` preview from common `poses`/`trans` arrays. Full SMPL/SMPL-X body-model decoding
and learned retarget inference remain explicit future work. MuJoCo physics rollout runs only when
the Python `mujoco` package is installed; otherwise the physics stage is reported as blocked rather
than marked successful.

## Key Docs

- `docs/research/literature_review.md` - paper/source survey and design implications.
- `docs/research/data_inventory.md` - local data scan and skeleton/target assumptions.
- `docs/architecture.md` - training and inference pipeline.
- `docs/milestones.md` - checkable gates.
- `docs/experiment_tracking.md` - WandB, git, DDP, and tmux rules.
