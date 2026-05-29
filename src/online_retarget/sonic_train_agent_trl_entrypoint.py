"""OnlineRetarget wrapper for SONIC's ``train_agent_trl.py`` entrypoint."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

from online_retarget.sonic_tokenizer_compat import install_tokenizer_cfg_compat


def main() -> int | None:
    """Install local launch compatibility, then delegate to SONIC training."""

    install_tokenizer_cfg_compat()
    sonic_root = str(Path.cwd())
    if sonic_root not in sys.path:
        sys.path.insert(0, sonic_root)
    sonic_script = Path(sonic_root) / "gear_sonic" / "train_agent_trl.py"
    if not sonic_script.is_file():
        raise FileNotFoundError(f"missing SONIC training script: {sonic_script}")
    sys.argv[0] = str(sonic_script)
    runpy.run_path(str(sonic_script), run_name="__main__")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
