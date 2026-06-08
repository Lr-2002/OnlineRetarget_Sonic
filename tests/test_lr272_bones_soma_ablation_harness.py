import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "lr272_bones_soma_ablation_harness.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("lr272_bones_soma_ablation_harness", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Lr272BonesSomaAblationHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script_module()

    def test_candidate_matrix_covers_required_routes(self):
        candidates = self.script.build_candidates()
        routes = {candidate.route for candidate in candidates}
        self.assertIn("A_root_world_adapter", routes)
        self.assertIn("B_summarizer_preprocess", routes)
        self.assertIn("C_dof_convention", routes)
        self.assertIn("baseline", routes)
        self.assertTrue(any(candidate.candidate_id == "a_root_front_train_split_calibrated" for candidate in candidates))
        stopped = next(candidate for candidate in candidates if candidate.candidate_id == "a_root_front_train_split_calibrated")
        self.assertFalse(stopped.enabled)
        self.assertEqual(stopped.validation["status"], "stopped_negative_mixed10")
        self.assertTrue(any(candidate.candidate_id == "b_per_clip_skeleton_preroll_ramp" for candidate in candidates))
        self.assertTrue(any(candidate.candidate_id == "c_hip_pitch_sign_flip_probe" for candidate in candidates))
        lower_body = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == "c_lower_body_fk_signature_dof_map_train_split_v1"
        )
        self.assertTrue(lower_body.enabled)
        self.assertEqual(lower_body.root_world["xy_scale_mode"], "identity")
        self.assertEqual(lower_body.summarizer["raw_action_contract"], "current_soma_retarget_action")
        self.assertEqual(lower_body.validation["allowed_stages"], ("smoke1",))
        self.assertFalse(lower_body.dof_convention["train_split_fk_signature_map"]["allow_shoulder_elbow"])

    def test_stage_selection_prioritizes_worst_key_and_caps_rows(self):
        worst = "230413__dance_hiphop_camel_walk_360_R_fast_002__A317"
        rows = [{"motion_key": f"clip_{idx:02d}", "move_g1_path": f"g1/{idx:02d}.csv"} for idx in range(12)]
        rows.insert(7, {"motion_key": worst, "move_g1_path": "g1/worst.csv"})

        smoke = self.script.select_stage_rows(rows, self.script.STAGES[0], [worst])
        mixed = self.script.select_stage_rows(rows, self.script.STAGES[1], [worst])
        walk = self.script.select_stage_rows(rows, self.script.STAGES[2], [worst])

        self.assertEqual([row["motion_key"] for row in smoke], [worst])
        self.assertEqual(mixed[0]["motion_key"], worst)
        self.assertEqual(len(mixed), 10)
        self.assertEqual(len(walk), 13)

    def test_stage_selection_uses_lr271_key_not_quality_action(self):
        worst = "230413__dance_hiphop_camel_walk_360_R_fast_002__A317"
        rows = [
            {
                "lr271_key": worst if idx == 4 else f"230101__walk_{idx:03d}__A{idx:03d}",
                "pair_key_contract": "lr271_key = date__filename",
                "source_bvh": f"/data/soma/{idx:03d}.bvh",
                "official_bones_g1_csv_member": f"g1/csv/230101/{idx:03d}.csv",
                "merged_quality_action": "keep",
            }
            for idx in range(12)
        ]

        mixed = self.script.select_stage_rows(rows, self.script.STAGES[1], [worst])
        walk = self.script.select_stage_rows(rows, self.script.STAGES[2], [worst])

        self.assertEqual(self.script.row_key(rows[0]), rows[0]["lr271_key"])
        self.assertEqual(mixed[0]["lr271_key"], worst)
        self.assertEqual(len(mixed), 10)
        self.assertEqual(len(walk), 12)

    def test_build_campaign_writes_configs_commands_and_stage_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairing_csv = root / "pairing.csv"
            _write_pairing_csv(pairing_csv)
            output_dir = root / "campaign"

            manifest = self.script.build_campaign(
                pairing_csv=pairing_csv,
                output_dir=output_dir,
                repo_root=root / "repo",
                g1_tar=root / "g1.tar",
                soma_bvh_tar=root / "soma_proportional.tar",
                baseline_commit="b3ef2708",
                run_name="unit",
                worst_keys=[self.script.DEFAULT_WORST_KEYS[0]],
                retarget_template=(
                    "python runner.py --config {config} --stage-csv {stage_csv} "
                    "--out {output_dir} --candidate {candidate_id}"
                ),
                metric_template="python metrics.py --config {config} --stage {stage}",
                visual_template="python vis.py --config {config} --stage {stage}",
            )

            self.assertEqual(manifest["candidate_count"], len(self.script.build_candidates()))
            self.assertEqual(manifest["stage_rows"], {"mixed10": 10, "smoke1": 1, "walk100": 12})
            self.assertTrue((output_dir / "candidate_matrix.csv").exists())
            self.assertTrue((output_dir / "commands.jsonl").exists())
            self.assertTrue((output_dir / "commands.sh").exists())

            config = json.loads(
                (output_dir / "configs" / "a_root_xy_scale_per_clip_bestfit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["provenance"]["baseline"]["commit_expected"], "b3ef2708")
            self.assertEqual(config["main_reference"]["target"], "official BONES G1 CSV from g1.tar")
            self.assertEqual(config["candidate"]["root_world"]["xy_scale_mode"], "per_clip_bestfit_xy")

            commands = [
                json.loads(line)
                for line in (output_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            expected_commands = sum(
                len(candidate.validation.get("allowed_stages") or self.script.STAGES)
                for candidate in self.script.build_candidates()
                if candidate.enabled
            )
            self.assertEqual(len(commands), expected_commands)
            self.assertIn("--candidate a_root_xy_scale_global_1p10", "\n".join(row["retarget_command"] for row in commands))
            self.assertNotIn("a_root_front_train_split_calibrated", "\n".join(row["retarget_command"] for row in commands))
            lower_body_rows = [
                row for row in commands if row["candidate_id"] == "c_lower_body_fk_signature_dof_map_train_split_v1"
            ]
            self.assertEqual([row["stage"] for row in lower_body_rows], ["smoke1"])
            self.assertTrue(all(row["baseline_commit"] == "b3ef2708" for row in commands))

    def test_build_campaign_generates_executable_default_runner_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairing_csv = root / "pairing.csv"
            _write_pairing_csv(pairing_csv)
            output_dir = root / "campaign"

            self.script.build_campaign(
                pairing_csv=pairing_csv,
                output_dir=output_dir,
                repo_root=root / "repo",
                g1_tar=root / "g1.tar",
                soma_bvh_tar=root / "soma_proportional.tar",
                baseline_commit="b3ef2708",
                run_name="unit",
                worst_keys=[],
                runner_python="/opt/run/python",
            )

            rows = [
                json.loads(line)
                for line in (output_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            baseline_smoke = next(
                row
                for row in rows
                if row["candidate_id"] == "baseline_b3ef2708_soma" and row["stage"] == "smoke1"
            )
            self.assertIn("lr272_bones_soma_candidate_runner.py", baseline_smoke["retarget_command"])
            self.assertIn("--mode retarget", baseline_smoke["retarget_command"])
            self.assertIn("--mode metric", baseline_smoke["metric_command"])
            self.assertIn("--mode visual", baseline_smoke["visual_command"])
            self.assertIn("--render-isaac", baseline_smoke["visual_command"])
            self.assertIn("/opt/run/python", baseline_smoke["retarget_command"])

    def test_print_run_resolves_stage_from_candidate_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairing_csv = root / "pairing.csv"
            _write_pairing_csv(pairing_csv)
            output_dir = root / "campaign"
            self.script.build_campaign(
                pairing_csv=pairing_csv,
                output_dir=output_dir,
                repo_root=root / "repo",
                g1_tar=root / "g1.tar",
                soma_bvh_tar=root / "soma_proportional.tar",
                baseline_commit="b3ef2708",
                run_name="unit",
                worst_keys=[],
            )

            result = self.script.print_run(
                output_dir / "configs" / "baseline_b3ef2708_soma.json",
                "smoke1",
                output_dir / "runs" / "smoke1" / "baseline_b3ef2708_soma",
            )

        self.assertEqual(result["candidate_id"], "baseline_b3ef2708_soma")
        self.assertEqual(result["stage"], "smoke1")
        self.assertTrue(result["stage_csv"].endswith("smoke1.csv"))


def _write_pairing_csv(path: Path) -> None:
    worst = "230413__dance_hiphop_camel_walk_360_R_fast_002__A317"
    rows = [
        {
            "motion_key": worst if idx == 3 else f"clip_{idx:02d}",
            "move_soma_proportional_path": f"soma/clip_{idx:02d}.bvh",
            "move_g1_path": f"g1/clip_{idx:02d}.csv",
            "actor_uid": f"A{idx:03d}",
        }
        for idx in range(12)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
