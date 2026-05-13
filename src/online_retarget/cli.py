"""Command line helpers for repository initialization and inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_retarget.config import Paths
from online_retarget.data.bones_seed import actor_skeletons, summarize_metadata


def main() -> None:
    parser = argparse.ArgumentParser(prog="online-retarget")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Summarize BONES-SEED metadata")
    inventory.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    inventory.add_argument("--sample-actors", type=int, default=3)

    args = parser.parse_args()
    if args.command == "inventory":
        _inventory(args.data_root, args.sample_actors)


def _inventory(data_root: Path, sample_actors: int) -> None:
    summary = summarize_metadata(data_root)
    actors = actor_skeletons(data_root)
    payload = {
        "summary": summary.to_dict(),
        "sample_actors": [actor.to_dict() for actor in actors[:sample_actors]],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
