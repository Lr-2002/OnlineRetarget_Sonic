# Experiment Tracking

## Data Policy

`/home/user/data/motion_data` is read-only. All derived indexes, normalized arrays, model checkpoints, logs, videos, and simulator outputs must be written to repo-local `runs/`, `outputs/`, or an explicitly configured scratch path.

## Git Policy

Before meaningful training:

1. Commit code and config.
2. Record `git rev-parse HEAD`.
3. Store any dirty diff as an artifact or do not start the run.

Commit messages should follow the Lore protocol from the workspace instructions when commits are requested.

## WandB Policy

Each training/eval run must log:

- git SHA and dirty/clean state
- config file
- dataset root and derived index path
- actor split file or split seed
- M2Q quality policy ID, quality report, and policy audit path
- metric definitions/version
- model parameter count
- latency benchmark when applicable
- checkpoint and exported model artifact

Default project: `OnlineRetarget`.

## Tmux Policy

Long training runs must start in tmux:

```bash
scripts/train_tmux.sh configs/baseline_mlp.yaml
```

Debug and short dry runs can run directly on the main line:

```bash
PYTHONPATH=src python3 scripts/train.py --config configs/baseline_mlp.yaml --dry-run
```

## DDP Policy

Training code must be rank-aware:

- Only rank 0 writes global summaries or initializes non-shared WandB artifacts.
- Each rank writes to a unique temporary directory when needed.
- Config, seed, and split are identical across ranks.
- Evaluation should run deterministically on rank 0 unless explicitly distributed.

## Environment Policy

Primary environment boundary:

- conda environment from `environment.yml`
- direnv loading `.envrc`
- Isaac Lab integration in a Python 3.10 compatible environment

If the local network fails while installing or fetching references, use the proxy endpoint provided by the user:

```bash
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export ALL_PROXY=socks5://127.0.0.1:7897
```
