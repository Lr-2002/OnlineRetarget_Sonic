from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from online_retarget.data.skeleton_ae_registry import (
    SKELETON_GEOMETRY_DIM,
    SOMA_AE_JOINT_NAMES,
    build_all_skeleton_ae_registry,
    skeleton_geometry_from_bvh_text,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    import scripts.train_skeleton_geometry_ae as skeleton_ae
else:
    skeleton_ae = None


REPO_ROOT = Path(__file__).resolve().parents[1]


class SkeletonAERegistryTests(unittest.TestCase):
    def test_all_skeleton_registry_builds_geometry_and_actor_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_csv = root / "sonic_index.csv"
            rows = []
            for index in range(10):
                actor = f"A{index:03d}"
                bvh_path = root / "soma_proportional" / "bvh" / f"{actor}.bvh"
                bvh_path.parent.mkdir(parents=True, exist_ok=True)
                bvh_path.write_text(_bvh_text(offset_scale=float(index + 1)), encoding="utf-8")
                rows.append(
                    {
                        "actor_uid": actor,
                        "filename": f"move_{actor}",
                        "move_name": f"move_{actor}",
                        "package": "Locomotion",
                        "category": "Walk",
                        "source_soma_proportional_path": (
                            f"soma_proportional/bvh/{actor}.bvh"
                        ),
                        "source_soma_proportional_shape_path": f"shapes/{actor}.npz",
                    }
                )
            with index_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            result = build_all_skeleton_ae_registry(
                index_csv=index_csv,
                data_root=root,
                output_root=root / "outputs",
                validation_ratio=0.1,
                seed=2026053001,
            )

            self.assertEqual(result.skeleton_count, 10)
            self.assertEqual(result.train_skeleton_count, 9)
            self.assertEqual(result.validation_skeleton_count, 1)
            self.assertEqual(result.split_leakage_count, 0)
            with result.registry_csv.open(newline="", encoding="utf-8") as handle:
                registry_rows = list(csv.DictReader(handle))
            self.assertEqual(len(registry_rows), 10)
            self.assertEqual(registry_rows[0]["geometry_shape"], "[104]")
            geometry = json.loads(registry_rows[0]["geometry_json"])
            self.assertEqual(len(geometry), SKELETON_GEOMETRY_DIM)
            report = json.loads(result.report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["geometry"]["shape"], [104])
            self.assertEqual(report["split_leakage_count"], 0)

    def test_bvh_geometry_is_root_local_offsets_plus_lengths(self) -> None:
        geometry = skeleton_geometry_from_bvh_text(_bvh_text(offset_scale=2.0), position_scale=0.5)

        self.assertEqual(len(geometry), 104)
        hips_offset = geometry[:3]
        spine_offset = geometry[3:6]
        lengths = geometry[78:]
        self.assertEqual(hips_offset, [0.0, 0.0, 0.0])
        self.assertEqual(spine_offset, [1.0, 1.0, 0.1])
        self.assertEqual(lengths[0], 0.0)
        self.assertAlmostEqual(lengths[1], (1.0**2 + 1.0**2 + 0.1**2) ** 0.5)

    def test_ae_config_and_trainer_guardrail_tokens_are_absent(self) -> None:
        text = "\n".join(
            [
                (REPO_ROOT / "scripts" / "train_skeleton_geometry_ae.py").read_text(
                    encoding="utf-8"
                ),
                (REPO_ROOT / "configs" / "skeleton_geometry_ae_all_skeletons.json").read_text(
                    encoding="utf-8"
                ),
                (REPO_ROOT / "src" / "online_retarget" / "data" / "skeleton_ae_registry.py").read_text(
                    encoding="utf-8"
                ),
            ]
        )
        forbidden = [
            "NumClasses",
            "skeleton_cluster_id",
            "stable_skeleton_cluster",
            "one_hot",
            "classification_head",
            "reward",
            "PPO",
            "g1_dyn",
            "g1_target_action",
            "action_loss",
            "dynamics_loss",
            "dynamics_action_mse",
            "retargeter",
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, text)


@unittest.skipIf(torch is None, "torch is required for Skeleton AE trainer tests")
class SkeletonAETrainerTests(unittest.TestCase):
    def test_model_architecture_and_shapes_are_fixed(self) -> None:
        model = skeleton_ae.SkeletonGeometryAE(input_dim=104, latent_dim=64)
        x = torch.zeros((2, 104), dtype=torch.float32)

        reconstructed, z = model(x)

        self.assertEqual(tuple(z.shape), (2, 64))
        self.assertEqual(tuple(reconstructed.shape), (2, 104))
        linear_shapes = [
            (module.in_features, module.out_features)
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(
            linear_shapes,
            [(104, 256), (256, 128), (128, 64), (64, 128), (128, 256), (256, 104)],
        )

    def test_dry_run_fits_normalization_on_train_split_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_csv = root / "skeleton_ae_registry.csv"
            rows = []
            for index in range(10):
                split = "validation" if index == 9 else "train"
                value = 100.0 if split == "validation" else 2.0
                rows.append(
                    {
                        "actor_uid": f"A{index:03d}",
                        "encoder_id": f"A{index:03d}",
                        "split": split,
                        "source_soma_proportional_path": f"soma/{index}.bvh",
                        "geometry_shape": "[104]",
                        "geometry_json": json.dumps([value] * 104),
                    }
                )
            with registry_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "training_lane": "skeleton_geometry_ae_only",
                        "input_data": {
                            "registry_csv": str(registry_csv),
                            "expected_geometry_shape": [104],
                        },
                        "output_dir": str(root / "runs" / "{run_group}"),
                        "model": {
                            "input_dim": 104,
                            "hidden_dims": [256, 128],
                            "latent_dim": 64,
                            "decoder_hidden_dims": [128, 256],
                            "output_dim": 104,
                            "dropout": 0.0,
                        },
                        "training": {
                            "seed": 2026053001,
                            "batch_size": 4,
                            "max_steps": 1,
                            "learning_rate": 0.001,
                            "weight_decay": 0.0,
                            "log_every": 1,
                            "validate_every": 1,
                        },
                        "runtime": {
                            "write_root": str(root / "runs"),
                            "device": "cpu",
                            "require_committed_code": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            old_group = os.environ.get("SKELETON_AE_RUN_GROUP")
            os.environ["SKELETON_AE_RUN_GROUP"] = "dry_run_test"
            try:
                result = skeleton_ae.run(config_path, dry_run=True)
            finally:
                if old_group is None:
                    os.environ.pop("SKELETON_AE_RUN_GROUP", None)
                else:
                    os.environ["SKELETON_AE_RUN_GROUP"] = old_group

            stats_path = Path(result["normalization"]["stats_path"])
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            self.assertTrue(torch.allclose(stats["skeleton_mean"], torch.full((104,), 2.0)))
            self.assertEqual(int(stats["fit_count"].item()), 9)
            self.assertTrue((stats_path.parent.parent / "dry_run_summary.json").exists())

def _bvh_text(offset_scale: float = 1.0) -> str:
    lines = ["HIERARCHY", "ROOT Hips", "{", "  OFFSET 0.0 0.0 0.0"]
    lines.append("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    for index, name in enumerate(SOMA_AE_JOINT_NAMES[1:], start=1):
        x = float(index % 3) * offset_scale
        y = offset_scale
        z = float(index % 5) * 0.1 * offset_scale
        lines.extend(
            [
                f"  JOINT {name}",
                "  {",
                f"    OFFSET {x:.6f} {y:.6f} {z:.6f}",
                "    CHANNELS 3 Zrotation Xrotation Yrotation",
                "  }",
            ]
        )
    channel_count = 6 + (len(SOMA_AE_JOINT_NAMES) - 1) * 3
    lines.extend(["}", "MOTION", "Frames: 1", "Frame Time: 0.008333333333333333"])
    lines.append(" ".join("0.0" for _ in range(channel_count)))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
