#!/usr/bin/env python3
"""Training entry point scaffold.

The full dataset loader is intentionally not implemented in the repo
initialization pass. This entry point establishes the DDP/WandB/git-sha
contract that future training code must preserve.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    git_sha = _git_sha()

    print(f"config={args.config}")
    print(f"rank={rank} world_size={world_size}")
    print(f"git_sha={git_sha}")

    if args.dry_run:
        return

    try:
        import torch  # noqa: F401
        import wandb  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Training requires the conda environment from environment.yml "
            "with torch and wandb installed."
        ) from exc

    raise SystemExit("Dataset construction and optimization loop are milestone M2 work.")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
