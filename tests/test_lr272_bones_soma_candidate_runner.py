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
            self.assertIn(metrics["rows"][0]["full_evaluator_status"], ("ok", "blocked"))
            self.assertIn("root_rot_geodesic_rmse_rad", metrics["rows"][0])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is required for candidate runner smoke")
    def test_train_split_calibration_excludes_eval_clip(self):
        import numpy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g1_tar = root / "g1.tar"
            train_member = "g1/csv/train.csv"
            val_member = "g1/csv/val.csv"
            _write_g1_tar_members(
                g1_tar,
                {
                    train_member: 2.0,
                    val_member: 20.0,
                },
                self.runner.G1_CSV_COLUMNS,
            )
            train_soma = root / "train.npy"
            val_soma = root / "val.npy"
            numpy.save(train_soma, _qpos_with_x_step(numpy, 0.01))
            numpy.save(val_soma, _qpos_with_x_step(numpy, 0.01))
            pairing_csv = root / "pairing.csv"
            _write_pairing_csv(pairing_csv, g1_tar, train_member, train_soma, val_member, val_soma)
            stage_csv = root / "stage.csv"
            _write_stage_csv(stage_csv, g1_tar, val_member, val_soma, split="val", key="val_clip")
            config = root / "candidate.json"
            _write_train_split_config(config, g1_tar, pairing_csv)
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
                    "retarget",
                ]
            )

            self.assertEqual(rc, 0)
            calibration = json.loads(
                (output_dir / "calibration" / "train_split_root_front_calibration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(calibration["status"], "ok")
            self.assertFalse(calibration["target_leakage_on_eval"])
            self.assertEqual(calibration["train_keys_used"], ["train_clip"])
            self.assertEqual(calibration["eval_keys_excluded"], ["val_clip"])
            csv_path = output_dir / "retarget_csv" / "val_clip.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[-1]["root_translateX"]), 6.0, places=5)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is required for candidate runner smoke")
    def test_fk_rootrel_identity_and_known_rigid_root_transform_invariants(self):
        import numpy

        local = numpy.asarray([0.25, -0.10, 0.80], dtype=numpy.float64)
        ref_root = numpy.asarray([[0.0, 0.0, 1.0], [0.35, 0.15, 1.0]], dtype=numpy.float64)
        ref_euler = numpy.zeros((2, 3), dtype=numpy.float64)
        ref_fk = [{"pelvis": (tuple(root + local),)} for root in ref_root]

        identity = self.runner.fk_mpjpe_metrics(ref_fk, ref_fk, ref_root, ref_root, ref_euler, ref_euler)
        self.assertAlmostEqual(identity["fk_world_max_m"], 0.0, places=12)
        self.assertAlmostEqual(identity["fk_rootrel_max_m"], 0.0, places=12)

        yaw = 0.7
        translation = numpy.asarray([1.2, -0.4, 0.05], dtype=numpy.float64)
        rotated_root = self.runner.rotate_xy_array(ref_root[:, :2] - ref_root[:1, :2], yaw) + ref_root[:1, :2]
        pred_root = ref_root.copy()
        pred_root[:, :2] = rotated_root
        pred_root += translation
        pred_euler = ref_euler.copy()
        pred_euler[:, 2] += yaw
        pred_fk = [
            {"pelvis": (tuple(root + self.runner.euler_xyz_to_matrix(euler) @ local),)}
            for root, euler in zip(pred_root, pred_euler)
        ]

        rigid = self.runner.fk_mpjpe_metrics(pred_fk, ref_fk, pred_root, ref_root, pred_euler, ref_euler)
        self.assertGreater(rigid["fk_world_mpjpe_m"], 0.1)
        self.assertAlmostEqual(rigid["fk_rootrel_max_m"], 0.0, places=12)


def _write_g1_tar(path: Path, member: str, fieldnames: list[str]) -> None:
    _write_g1_tar_members(path, {member: 1.0}, fieldnames)


def _write_g1_tar_members(path: Path, members: dict[str, float], fieldnames: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        with tarfile.open(path, "w") as tar:
            for index, (member, root_x_step_cm) in enumerate(members.items()):
                csv_path = tmp_root / f"{index}.csv"
                _write_g1_csv(csv_path, fieldnames, root_x_step_cm=root_x_step_cm)
                tar.add(csv_path, arcname=member)


def _write_g1_csv(path: Path, fieldnames: list[str], *, root_x_step_cm: float) -> None:
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in range(4):
            row = {name: 0.0 for name in fieldnames}
            row["Frame"] = frame
            row["root_translateX"] = float(frame) * root_x_step_cm
            writer.writerow(row)


def _qpos_with_x_step(numpy, step_m: float):
    qpos = numpy.zeros((4, 36), dtype=numpy.float64)
    qpos[:, 0] = [0.0, step_m, step_m * 2, step_m * 3]
    qpos[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    return qpos


def _write_stage_csv(
    path: Path,
    g1_tar: Path,
    member: str,
    soma_npy: Path,
    *,
    split: str = "",
    key: str = "230101__walk__A001",
) -> None:
    row = {
        "lr271_key": key,
        "split": split,
        "source_bvh_fps": "120.0048",
        "soma_online_npy": str(soma_npy),
        "official_bones_g1_tar": str(g1_tar),
        "official_bones_g1_csv_member": member,
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_pairing_csv(
    path: Path,
    g1_tar: Path,
    train_member: str,
    train_soma: Path,
    val_member: str,
    val_soma: Path,
) -> None:
    rows = [
        {
            "lr271_key": "train_clip",
            "split": "train",
            "source_bvh_fps": "120.0048",
            "soma_online_npy": str(train_soma),
            "official_bones_g1_tar": str(g1_tar),
            "official_bones_g1_csv_member": train_member,
        },
        {
            "lr271_key": "val_clip",
            "split": "val",
            "source_bvh_fps": "120.0048",
            "soma_online_npy": str(val_soma),
            "official_bones_g1_tar": str(g1_tar),
            "official_bones_g1_csv_member": val_member,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def _write_train_split_config(path: Path, g1_tar: Path, pairing_csv: Path) -> None:
    payload = {
        "candidate": {
            "candidate_id": "a_root_front_train_split_calibrated",
            "route": "A_root_world_adapter",
            "root_world": {
                "xy_scale_mode": "train_split_calibrated",
                "yaw_alignment": "train_split_calibrated",
                "calibration_max_rows": 4,
                "calibration_max_frames": 4,
            },
            "summarizer": {},
            "dof_convention": {"sign_overrides": {}, "axis_swaps": {}},
        },
        "provenance": {"inputs": {"g1_tar": str(g1_tar), "pairing_csv": str(pairing_csv)}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
