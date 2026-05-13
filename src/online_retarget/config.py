"""Project-level runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path(os.environ.get("ONLINE_RETARGET_DATA_ROOT", "/home/user/data/motion_data"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ONLINE_RETARGET_OUTPUT_ROOT", "runs"))


@dataclass(frozen=True)
class Paths:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT

    @classmethod
    def from_env(cls) -> "Paths":
        return cls(
            data_root=Path(os.environ.get("ONLINE_RETARGET_DATA_ROOT", str(DEFAULT_DATA_ROOT))),
            output_root=Path(os.environ.get("ONLINE_RETARGET_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))),
        )

    def validate_output_not_in_data_root(self) -> None:
        data_root = self.data_root.resolve()
        output_root = self.output_root.resolve()
        if output_root == data_root or data_root in output_root.parents:
            raise ValueError(
                f"Output root {output_root} is inside read-only data root {data_root}; "
                "choose a repo-local or scratch output path."
            )
