"""OnlineRetarget wrapper for SONIC's ``train_agent_trl.py`` entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

from online_retarget.sonic_tokenizer_compat import install_tokenizer_cfg_compat


def main() -> int | None:
    """Install local launch compatibility, then delegate to SONIC training."""

    install_tokenizer_cfg_compat()
    sonic_root = str(Path.cwd())
    if sonic_root not in sys.path:
        sys.path.insert(0, sonic_root)
    from gear_sonic.train_agent_trl import main as sonic_main

    return sonic_main()


if __name__ == "__main__":
    raise SystemExit(main())
