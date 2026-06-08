import csv
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "lr272_bones_soma_candidate_runner.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("lr272_bones_soma_candidate_runner", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Lr272BonesSomaCandidateRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_script_module()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is required for candidate runner smoke")
    def test_runner_all_writes_csv_metrics_and_visual(self):
        np = importlib.util.find_spec("numpy")
        assert np is not None
        import numpy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g1_tar = root / "g1.tar"
            member = "g1/csv/230101/walk.csv"
            _write_g1_tar(g1_tar, member, self.runner.G1_CSV_COLUMNS)
            qpos = numpy.zeros((4, 36), dtype=numpy.float64)
            qpos[:, 0] = [0.0, 0.01, 0.02, 0.03]
            qpos[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
            soma_npy = root / "soma.npy"
            numpy.save(soma_npy, qpos)
            stage_csv = root / "stage.csv"
            _write_stage_csv(stage_csv, g1_tar, member, soma_npy)
            config = root / "candidate.json"
            _write_config(config, g1_tar)
            output_dir = root / "out"

            rc = self.runner.main(
                [
                    "--config",
                    str(config),
                    "--stage-csv",
                    str(stage_csv),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "all",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue((output_dir / "retarget_csv" / "230101__walk__A001.csv").exists())
            self.assertTrue((output_dir / "metrics" / "candidate_metrics.csv").exists())
            self.assertTrue((output_dir / "metrics" / "candidate_metrics.json").exists())
            self.assertTrue((output_dir / "visuals" / "230101__walk__A001_root_xy.svg").exists())
            self.assertTrue((output_dir / "visuals" / "isaac_mesh_renderer_blocker.json").exists())
            metrics = json.loads((output_dir / "metrics" / "candidate_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["candidate_id"], "a_root_xy_scale_global_1p10")
            self.assertEqual(len(metrics["rows"]), 1)


def _write_g1_tar(path: Path, member: str, fieldnames: list[str]) -> None:
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in range(4):
            row = {name: 0.0 for name in fieldnames}
            row["Frame"] = frame
            row["root_translateX"] = float(frame)
            writer.writerow(row)
    with tarfile.open(path, "w") as tar:
        tar.add(csv_path, arcname=member)


def _write_stage_csv(path: Path, g1_tar: Path, member: str, soma_npy: Path) -> None:
    row = {
        "lr271_key": "230101__walk__A001",
        "source_bvh_fps": "120.0048",
        "soma_online_npy": str(soma_npy),
        "official_bones_g1_tar": str(g1_tar),
        "official_bones_g1_csv_member": member,
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_config(path: Path, g1_tar: Path) -> None:
    payload = {
        "candidate": {
            "candidate_id": "a_root_xy_scale_global_1p10",
            "route": "A_root_world_adapter",
            "root_world": {"xy_scale_mode": "global", "xy_scale": 1.10, "yaw_alignment": "none"},
            "summarizer": {},
            "dof_convention": {"sign_overrides": {}, "axis_swaps": {}},
        },
        "provenance": {"inputs": {"g1_tar": str(g1_tar)}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
