#!/usr/bin/env python3
"""Validate SONIC-native retargeting configs before training launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online_retarget.sonic_native_contract import (  # noqa: E402
    ContractError,
    result_to_dict,
    validate_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="JSON/YAML config files to validate")
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help="Reject legacy diagnostic configs.",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Also verify source_repo-relative SONIC config paths exist.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print validation results as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    try:
        for config_path in args.configs:
            result = validate_file(
                config_path,
                require_formal=args.require_formal,
                check_paths=args.check_paths,
            )
            results.append(result_to_dict(result))
    except ContractError as exc:
        print(f"contract validation failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            status = "formal" if result["formal"] else "legacy"
            print(
                f"ok {status} variant={result['variant']} "
                f"lane={result['training_lane']} config={result['path']}"
            )
            for warning in result["warnings"]:
                print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
