from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
A0_CONFIGS = (
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_frozen_ae_uniform_4gpu.json",
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_frozen_ae_proportional_4gpu.json",
)

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    import scripts.train_sonic_kin_skeleton_ae as sonic_train
    from online_retarget.models.skeleton_geometry_ae import (
        SKELETON_GEOMETRY_AE_ARCHITECTURE,
        SkeletonGeometryAE,
    )
else:
    sonic_train = None
    SKELETON_GEOMETRY_AE_ARCHITECTURE = [104, 256, 128, 64, 128, 256, 104]
    SkeletonGeometryAE = None


class A0FrozenAEConfigTests(unittest.TestCase):
    def test_a0_configs_are_explicit_frozen_encoder_contracts(self) -> None:
        for path in A0_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                topology = config["input_data"]["soma_topology"]
                self.assertEqual(config["training_lane"], "soma_motionlib_kin_only")
                self.assertTrue(config["skeleton_ae"]["enabled"])
                self.assertTrue(config["skeleton_ae"]["freeze_encoder"])
                self.assertEqual(config["skeleton_ae"]["expected_architecture"], SKELETON_GEOMETRY_AE_ARCHITECTURE)
                self.assertEqual(config["skeleton_ae"]["x_skel_dim"], 104)
                self.assertEqual(config["skeleton_ae"]["z_skel_dim"], 64)
                self.assertEqual(config["skeleton_ae"]["cache_device"], "cpu")
                self.assertEqual(config["features"]["expected_dims"]["motion_token"], 840)
                self.assertEqual(config["features"]["expected_dims"]["model_input"], 904)
                self.assertEqual(config["features"]["expected_dims"]["target"], 670)
                self.assertFalse(config["ddp"]["init_sync"])
                self.assertIn(f"soma_{topology}_filtered_v1", config["input_data"]["soma_motion_dir"])
                self.assertEqual(config["variant"]["family"], "A0_frozen_skeleton_ae")
                text = path.read_text(encoding="utf-8")
                for token in (
                    "reward",
                    "PPO",
                    "g1_dyn",
                    "g1_target_action",
                    "action_loss",
                    "dynamics_loss",
                    "NumClasses",
                    "skeleton_cluster_id",
                    "classification_head",
                ):
                    self.assertNotIn(token, text)

    def test_a0_stage_trace_covers_requested_dry_run_boundaries(self) -> None:
        text = (REPO_ROOT / "scripts" / "train_sonic_kin_skeleton_ae.py").read_text(encoding="utf-8")
        for token in (
            "--stage-trace",
            "--index-only",
            "distributed_runtime_setup",
            "distributed_init_process_group",
            "cuda_set_device",
            "skeleton_ae_checkpoint_load",
            "skeleton_ae_stats_load",
            "skeleton_ae_registry_load",
            "skeleton_ae_cpu_z_cache_build",
            "normalization_stats_motion_z",
            "skeleton_ae_row_mapping",
            "first_batch_collation",
            "index_only_preflight",
            "index_only_summary.json",
            "summary_event",
            "rows_from_index_cache",
            "rows_from_index_cache_path",
            "wait_for_rows_from_index_cache",
            "ROWS_FROM_INDEX_CACHE_WAIT_TIMEOUT_SEC",
            "rows_from_index_cache_wait",
            "wait_before_read",
            "write_rows_from_index_cache",
            "tmp_path.replace(cache_path)",
            "cache/rows_from_index",
            "rows_from_index_stat",
            "rows_from_index_glob",
            "rows_from_index_progress",
            "rows_from_index_read",
            "rows_from_index_parse",
            "rows_from_index_filter",
            "rows_from_index_sample",
            "rows_from_index_row_count",
            "model_to_device",
            "model_to_device_cuda_synchronize",
            "model_init_seed",
            "model_construct",
            "model_parameter_checksum",
            "all_rank_parameter_checksums",
            "model_ddp_preflight",
            "ddp_wrap",
            "ddp_wrap_probe_minimal_mlp",
            "ddp_probe_suite",
            "ddp_probe_same_shape_sequential",
            "ddp_probe_fresh_concat_retargeter",
            "ddp_probe_single_linear_512",
            "ddp_probe_single_linear_1024",
            "ddp_probe_single_linear_1154",
            "_ddp_ctor",
            "_forward",
            "_backward",
            "_cuda_synchronize_pre_ddp",
            "_cuda_synchronize_post_forward",
            "_cuda_synchronize_post_backward",
            "first_forward",
            "logs/a0_stage_trace",
            "named_parameters",
            "named_buffers",
            "requires_grad",
            "contains_skeleton_encoder_params",
            "contains_frozen_encoder_parameter",
            "A0_DDP_PROBE",
            "A0_DDP_PROBE_ONLY",
            "A0_DDP_PROBE_INIT_SYNC",
            "A0_DDP_INIT_SYNC",
            "A0_DDP_PROBE_BACKWARD",
            "A0_DDP_PROBE_BUCKET_CAP_MB",
            "A0_DDP_PROBE_STATIC_GRAPH",
            "A0_DDP_PROBE_FIND_UNUSED_PARAMETERS",
            "A0_DDP_BROADCAST_BUFFERS",
            "dummy_loss",
            "loss.backward",
            "gradient_sha256",
            "parameter_sha256",
            "init_sync",
            "bucket_cap_mb",
            "static_graph",
            "find_unused_parameters",
            "broadcast_buffers",
            "device_ids",
            "output_device",
            "check_data_artifacts",
            "torch.cuda.synchronize",
        ):
            self.assertIn(token, text)
        self.assertIn("expected torchrun WORLD_SIZE", text)
        self.assertIn("skipped_count=int(skipped)", text)
        self.assertNotIn("skipped_count=len(skipped)", text)
        self.assertNotIn('stage_trace.log("index_only_preflight", "details", **summary)', text)

    def test_soma_prediction_recomposes_root_local_xy_for_visualization(self) -> None:
        text = (REPO_ROOT / "scripts" / "train_sonic_kin_skeleton_ae.py").read_text(encoding="utf-8")
        self.assertIn('input_data.get("format") == "soma_motionlib"', text)
        self.assertIn("pred_root[:, :2] += fallback_root_pos[:, :2]", text)
        self.assertIn('"root_pos": pred_root', text)
        self.assertIn('"root_euler": _rot6d_to_euler_xyz_batch(root_rot6d[:, 0])', text)


@unittest.skipIf(torch is None, "torch is required for A0 frozen AE tests")
class A0FrozenAEFeatureTests(unittest.TestCase):
    def test_feature_lookup_hard_fails_missing_and_ambiguous_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, stats, registry = _write_ae_artifacts(root, duplicate_source=True)
            config = _minimal_a0_config(root, checkpoint, stats, registry)
            lookup = sonic_train.build_skeleton_ae_feature_lookup(config, torch.device("cpu"))
            assert lookup is not None

            with self.assertRaisesRegex(ValueError, "missing_skeleton_geometry_count=1"):
                lookup.validate_and_annotate_rows([{"relative_path": "missing.pkl"}])

            with self.assertRaisesRegex(ValueError, "ambiguous_skeleton_geometry_count=1"):
                lookup.validate_and_annotate_rows(
                    [{"source_soma_proportional_path": "soma_proportional/bvh/shared.bvh"}]
                )

    def test_feature_lookup_caches_z64_and_optimizer_excludes_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, stats, registry = _write_ae_artifacts(root)
            config = _minimal_a0_config(root, checkpoint, stats, registry)
            lookup = sonic_train.build_skeleton_ae_feature_lookup(config, torch.device("cpu"))
            assert lookup is not None
            rows = [{"actor_uid": "A001", "relative_path": "clip.pkl"}]

            report = lookup.validate_and_annotate_rows(rows)
            embedding = lookup.embedding_for_row(rows[0])
            model = sonic_train.make_model(840, 64, 670, config)
            parameter_names = sonic_train.trainable_parameter_names(model)

            self.assertEqual(report["missing_skeleton_geometry_count"], 0)
            self.assertEqual(report["ambiguous_skeleton_geometry_count"], 0)
            self.assertEqual(tuple(embedding.shape), (64,))
            self.assertEqual(rows[0]["skeleton_ae_encoder_id"], "A001")
            self.assertFalse(any("encoder" in name for name in parameter_names))
            self.assertEqual(lookup.artifact_info["embedding_cache_device"], "cpu")

    def test_feature_lookup_keeps_registry_encoding_off_training_cuda_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, stats, registry = _write_ae_artifacts(root)
            config = _minimal_a0_config(root, checkpoint, stats, registry)
            config["skeleton_ae"]["cache_device"] = "cpu"

            lookup = sonic_train.build_skeleton_ae_feature_lookup(config, torch.device("cuda", 0))
            assert lookup is not None

            self.assertEqual(lookup.artifact_info["embedding_cache_device"], "cpu")
            self.assertEqual(lookup.artifact_info["training_device"], "cuda:0")

    def test_feature_lookup_rejects_cuda_registry_cache_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, stats, registry = _write_ae_artifacts(root)
            config = _minimal_a0_config(root, checkpoint, stats, registry)
            config["skeleton_ae"]["cache_device"] = "cuda"

            with self.assertRaisesRegex(ValueError, "registry cache must be built on CPU"):
                sonic_train.build_skeleton_ae_feature_lookup(config, torch.device("cpu"))

    def test_a0_dry_run_writes_manifest_dims_and_freeze_proof(self) -> None:
        try:
            import joblib  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("joblib is required for soma_motionlib dry-run fixture")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, stats, registry = _write_ae_artifacts(root)
            config_path = _write_dry_run_config(root, checkpoint, stats, registry)
            env = dict(os.environ)
            env["KIN_RUN_GROUP"] = "a0_dry_run_test"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "train_sonic_kin_skeleton_ae.py"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output_dir = root / "runs" / "a0_dry_run_test"
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "dry_run_summary.json").read_text(encoding="utf-8"))
            norm = torch.load(output_dir / "stats" / "normalization.pt", map_location="cpu", weights_only=False)

            self.assertEqual(manifest["feature_dims"]["motion"], 840)
            self.assertEqual(manifest["feature_dims"]["skeleton_embedding"], 64)
            self.assertEqual(manifest["feature_dims"]["model_input"], 904)
            self.assertEqual(manifest["feature_dims"]["target"], 670)
            self.assertTrue(manifest["skeleton_ae"]["skeleton_encoder_frozen"])
            self.assertFalse(manifest["optimizer"]["contains_skeleton_encoder_params"])
            self.assertFalse(manifest["ddp"]["init_sync"])
            self.assertEqual(manifest["ddp"]["init_sync_source"], "config.ddp.init_sync")
            self.assertEqual(summary["mapping_report"]["missing_skeleton_geometry_count"], 0)
            self.assertIn("skeleton_embedding_mean", norm)
            self.assertIn("skeleton_embedding_std", norm)
            self.assertTrue((output_dir / "cache" / "skeleton_embedding_cache.pt").exists())

    def test_index_only_preflight_writes_rows_cache_and_trace(self) -> None:
        try:
            import joblib  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("joblib is required for soma_motionlib index-only fixture")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, stats, registry = _write_ae_artifacts(root)
            config_path = _write_dry_run_config(root, checkpoint, stats, registry)
            env = dict(os.environ)
            env["KIN_RUN_GROUP"] = "a0_index_only_test"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "train_sonic_kin_skeleton_ae.py"),
                    "--config",
                    str(config_path),
                    "--index-only",
                    "--stage-trace",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output_dir = root / "runs" / "a0_index_only_test"
            summary = json.loads((output_dir / "index_only_summary.json").read_text(encoding="utf-8"))
            cache_path = output_dir / "cache" / "rows_from_index" / "rows_from_index_cache.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            trace_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (output_dir / "logs" / "a0_stage_trace").glob("*.jsonl")
            )

            self.assertEqual(summary["event"], "index_only_preflight")
            self.assertEqual(summary["row_count"], 3)
            self.assertEqual(summary["skipped_count"], 0)
            self.assertEqual(summary["rows_cache"], str(cache_path))
            self.assertEqual(cache["row_count"], 3)
            self.assertEqual(cache["skipped_count"], 0)
            self.assertFalse((output_dir / "manifest.json").exists())
            for token in (
                "rows_from_index_stat",
                "rows_from_index_glob",
                "rows_from_index_read",
                "rows_from_index_parse",
                "rows_from_index_filter",
                "rows_from_index_sample",
                "rows_from_index_row_count",
            ):
                self.assertIn(token, trace_text)


def _write_ae_artifacts(root: Path, *, duplicate_source: bool = False) -> tuple[Path, Path, Path]:
    checkpoint = root / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = SkeletonGeometryAE()
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": SKELETON_GEOMETRY_AE_ARCHITECTURE,
            "step": 0,
        },
        checkpoint,
    )
    stats = root / "stats" / "skeleton_geometry_normalization.pt"
    stats.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "skeleton_mean": torch.zeros(104),
            "skeleton_std": torch.ones(104),
        },
        stats,
    )
    registry = root / "registry" / "skeleton_ae_registry.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "actor_uid": "A001",
            "encoder_id": "A001",
            "split": "train",
            "source_soma_proportional_path": "soma_proportional/bvh/A001.bvh",
            "geometry_shape": "[104]",
            "geometry_json": json.dumps([0.01] * 104),
        },
        {
            "actor_uid": "A002",
            "encoder_id": "A002",
            "split": "validation",
            "source_soma_proportional_path": "soma_proportional/bvh/A002.bvh",
            "geometry_shape": "[104]",
            "geometry_json": json.dumps([0.02] * 104),
        },
    ]
    if duplicate_source:
        rows.extend(
            [
                {
                    "actor_uid": "A010",
                    "encoder_id": "A010",
                    "split": "train",
                    "source_soma_proportional_path": "soma_proportional/bvh/shared.bvh",
                    "geometry_shape": "[104]",
                    "geometry_json": json.dumps([0.10] * 104),
                },
                {
                    "actor_uid": "A011",
                    "encoder_id": "A011",
                    "split": "train",
                    "source_soma_proportional_path": "soma_proportional/bvh/shared.bvh",
                    "geometry_shape": "[104]",
                    "geometry_json": json.dumps([0.11] * 104),
                },
            ]
        )
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return checkpoint, stats, registry


def _minimal_a0_config(root: Path, checkpoint: Path, stats: Path, registry: Path) -> dict:
    return {
        "variant": {"type": "concat", "name": "A0_test"},
        "model": {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0},
        "skeleton_ae": {
            "enabled": True,
            "checkpoint": str(checkpoint),
            "normalization": str(stats),
            "registry_csv": str(registry),
            "freeze_encoder": True,
            "cache_device": "cpu",
        },
        "runtime": {
            "write_root": str(root / "runs"),
            "device": "cpu",
            "required_gpu_count": 0,
            "require_committed_code": False,
            "require_latest_code": False,
        },
        "ddp": {"init_sync": False},
    }


def _write_dry_run_config(root: Path, checkpoint: Path, stats: Path, registry: Path) -> Path:
    import joblib

    robot_dir = root / "robot"
    soma_dir = root / "soma"
    robot_dir.mkdir()
    soma_dir.mkdir()
    for index, actor in enumerate(("A001", "A002", "A001"), start=1):
        name = f"clip_{index:02d}_{actor}.pkl"
        frames = 12
        dof = np.linspace(0.0, 1.0, frames * 29, dtype=np.float32).reshape(frames, 29)
        robot = {
            "dof": dof,
            "root_rot": np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)),
            "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
            "fps": 50.0,
        }
        soma_frames = 29
        soma_joints = np.zeros((soma_frames, 26, 3), dtype=np.float32)
        soma_joints[..., 0] = np.arange(26, dtype=np.float32)
        soma_joints[..., 1] = index * 0.01
        soma = {
            "soma_joints": soma_joints,
            "soma_root_quat": np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (soma_frames, 1)),
            "fps": 120.0,
            "source_bvh": f"soma_proportional/bvh/{actor}.bvh",
        }
        joblib.dump({Path(name).stem: robot}, robot_dir / name)
        joblib.dump({Path(name).stem: soma}, soma_dir / name)

    config = {
        "schema_version": "test",
        "owner": "OnlineRetarget",
        "training_lane": "soma_motionlib_kin_only",
        "purpose": "A0 dry-run fixture",
        "source_repo": str(root),
        "source_rev": "test",
        "input_data": {
            "format": "soma_motionlib",
            "robot_motion_dir": str(robot_dir),
            "soma_motion_dir": str(soma_dir),
            "source_fps": 120.0,
            "target_fps": 50.0,
            "max_duration_delta_sec": 0.1,
            "max_clips": 3,
            "robot_root_rot_format": "xyzw",
        },
        "skeleton_ae": {
            "enabled": True,
            "checkpoint": str(checkpoint),
            "normalization": str(stats),
            "registry_csv": str(registry),
            "freeze_encoder": True,
            "cache_device": "cpu",
        },
        "output_dir": str(root / "runs" / "{run_group}"),
        "validation_command": "",
        "expected_artifacts": [],
        "variant": {"name": "A0_dry_run_test", "type": "concat"},
        "features": {
            "future_window_frames": 10,
            "future_step": 1,
            "include_root_pos_target": True,
        },
        "split": {"validation_ratio": 0.5, "hash_salt": "test"},
        "normalization": {"max_frames": 1000, "frames_per_chunk": 64},
        "model": {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0},
        "training": {
            "seed": 123,
            "max_steps": 1,
            "required_gpu_count": 0,
            "batch_frames": 64,
            "frame_stride": 1,
            "loader_chunk_frames": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "precision": "fp32",
            "num_workers": 0,
            "log_every": 1,
            "validate_every": 1,
            "validation_batches": 2,
            "checkpoint_every": 1,
            "keep_last_checkpoints": 1,
            "command_loss_weight": 1.0,
            "root_pos_loss_weight": 0.25,
            "root_rot_loss_weight": 0.5,
        },
        "visual_validation": {"enabled": False},
        "runtime": {
            "required_gpu_count": 0,
            "write_root": str(root / "runs"),
            "forbid_write_roots": [],
            "require_committed_code": False,
            "require_latest_code": False,
            "device": "cpu",
        },
        "ddp": {"init_sync": False},
        "wandb": {"enabled": False},
    }
    config_path = root / "a0_dry_run_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


if __name__ == "__main__":
    unittest.main()
