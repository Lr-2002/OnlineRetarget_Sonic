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

Run smoke tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Launch a future training run in tmux:

```bash
scripts/train_tmux.sh configs/baseline_mlp.yaml
```

## Key Docs

- `docs/research/literature_review.md` - paper/source survey and design implications.
- `docs/research/data_inventory.md` - local data scan and skeleton/target assumptions.
- `docs/architecture.md` - training and inference pipeline.
- `docs/milestones.md` - checkable gates.
- `docs/experiment_tracking.md` - WandB, git, DDP, and tmux rules.
