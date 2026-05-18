#!/usr/bin/env python3
"""Hydra wrapper for the plain supervised training entry point."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

try:
    import hydra
    from omegaconf import OmegaConf
except ImportError as exc:  # pragma: no cover - environment setup guard.
    raise SystemExit(
        "Hydra training requires hydra-core. Install the environment from environment.yml "
        "or run `pip install hydra-core` in the active training environment."
    ) from exc

from scripts import train as train_entry


@hydra.main(version_base=None, config_path="../configs", config_name="bones_bvh_mlp_5000")
def main(cfg) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        OmegaConf.save(config=cfg, f=handle.name)
        config_path = Path(handle.name)
    try:
        argv = ["train.py", "--config", str(config_path)]
        if bool(
            OmegaConf.select(cfg, "run.allow_debug_data", default=False)
            or OmegaConf.select(cfg, "data.allow_debug_data", default=False)
        ):
            argv.append("--allow-debug-data")
        if bool(OmegaConf.select(cfg, "run.dry_run", default=False)):
            argv.append("--dry-run")
        if bool(OmegaConf.select(cfg, "run.predict_only", default=False)):
            argv.append("--predict-only")
        output_dir = str(OmegaConf.select(cfg, "run.output_dir", default=""))
        if output_dir:
            argv.extend(["--output-dir", output_dir])
        checkpoint = str(OmegaConf.select(cfg, "run.checkpoint", default=""))
        if checkpoint:
            argv.extend(["--checkpoint", checkpoint])
        limit = OmegaConf.select(cfg, "run.limit", default=None)
        if limit is not None:
            argv.extend(["--limit", str(limit)])
        sys.argv = argv
        train_entry.main()
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
