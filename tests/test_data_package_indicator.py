import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online_retarget.data_package_indicator import (
    SCHEMA,
    filter_rows_by_data_package_config,
    package_pair_id,
    package_rows_sha256,
    parse_package_indicator,
)


REAL_KIN_WALK_INDICATOR = Path(
    "/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/"
    "lr280_data_package_inventory_20260609T0420Z/indicators/soma_motionlib/kin/walk.txt"
)
REAL_ROBOT_MOTION_DIR = Path(
    "/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_motionlib/robot_filtered_v1"
)
REAL_SOMA_MOTION_DIR = Path(
    "/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_motionlib/soma_proportional_filtered_v1"
)
REAL_KIN_WALK_ROWS_JSONL = REAL_KIN_WALK_INDICATOR.with_suffix(".rows.jsonl")
REAL_KIN_WALK_ROW_COUNT = 11248
REAL_KIN_WALK_ROWS_SHA256 = "2fb36f38d023752e2d1113b1c3455dcb98d1c82318262bde6dfc9c3d34fd79cd"


class DataPackageIndicatorTests(unittest.TestCase):
    def test_parse_and_filter_kin_walk_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            indicator = _write_indicator(Path(tmp), _walk_rows()[:2])
            parsed = parse_package_indicator(indicator)

            selected, summary = filter_rows_by_data_package_config(
                _walk_rows(),
                _input_data(indicator),
            )

        self.assertEqual(parsed.schema, SCHEMA)
        self.assertEqual(parsed.spec, "kin")
        self.assertEqual(parsed.category, "walk")
        self.assertEqual(len(parsed.rows), 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual([row["relative_path"] for row in selected], ["clip_walk_a.pkl", "clip_walk_b.pkl"])
        self.assertEqual(summary["selected_row_count"], 2)
        self.assertEqual(summary["missing_row_count"], 0)
        self.assertEqual(summary["package_rows_sha256"], package_rows_sha256(selected))
        self.assertEqual(parsed.package_rows_sha256, package_rows_sha256(parsed.rows))

    def test_no_package_leaves_rows_and_digest_unchanged(self):
        rows = _walk_rows()
        before_digest = package_rows_sha256(rows)

        selected, summary = filter_rows_by_data_package_config(rows, {"max_clips": 1})

        self.assertIsNone(summary)
        self.assertEqual(len(selected), len(rows))
        self.assertEqual(package_rows_sha256(selected), before_digest)

    def test_package_rows_digest_matches_inventory_line_contract(self):
        self.assertEqual(
            package_rows_sha256([_walk_rows()[0]]),
            "335e9fe7c34da47f57bfb11cff03a32edadbffa4288fb72e0ec2a3a18a2e34db",
        )

    def test_max_clips_is_applied_after_package_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            indicator = _write_indicator(Path(tmp), _walk_rows()[:2])
            selected, summary = filter_rows_by_data_package_config(
                _walk_rows(),
                _input_data(indicator, max_clips=1),
            )

        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["indicator_row_count"], 2)
        self.assertEqual(summary["matched_row_count"], 2)
        self.assertEqual(summary["selected_row_count"], 1)
        self.assertEqual(summary["rejected_row_count"], 1)
        self.assertTrue(summary["max_clips_applied"])

    def test_indicator_header_must_match_config_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            indicator = _write_indicator(Path(tmp), _walk_rows()[:1], spec="phy")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                filter_rows_by_data_package_config(_walk_rows(), _input_data(indicator))

    def test_duplicate_pair_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            indicator = _write_indicator(Path(tmp), [_walk_rows()[0], _walk_rows()[0]])
            with self.assertRaisesRegex(ValueError, "duplicate pair_id"):
                parse_package_indicator(indicator)

    def test_missing_paired_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            indicator = _write_indicator(Path(tmp), [_walk_rows()[2]])
            with self.assertRaisesRegex(ValueError, "missing base row"):
                filter_rows_by_data_package_config(_walk_rows()[:2], _input_data(indicator))

    def test_source_bvh_mismatch_is_rejected(self):
        row = dict(_walk_rows()[0])
        row["data_package_pair_id"] = package_pair_id(
            "clip_walk_a.pkl",
            "clip_walk_a.pkl",
            "soma_proportional/bvh/walk_a.bvh",
        )
        row["source_bvh"] = "soma_proportional/bvh/other_walk.bvh"
        with tempfile.TemporaryDirectory() as tmp:
            indicator = _write_indicator(Path(tmp), [_walk_rows()[0]])
            with self.assertRaisesRegex(ValueError, "source_bvh mismatch"):
                filter_rows_by_data_package_config([row], _input_data(indicator))

    def test_raw_phy_walk_real_indicator_is_not_a_paired_fixture(self):
        raw_phy = Path(
            "/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/"
            "lr280_data_package_inventory_20260609T0420Z/indicators/raw_sonic/phy/"
            "walk.relative_paths.tsv"
        )
        self.assertNotEqual(raw_phy.name, "walk.txt")

    def test_real_kin_walk_indicator_shape_when_available(self):
        if not REAL_KIN_WALK_INDICATOR.exists():
            self.skipTest(f"real kin/walk indicator is not available locally: {REAL_KIN_WALK_INDICATOR}")

        indicator = parse_package_indicator(REAL_KIN_WALK_INDICATOR)

        self.assertEqual(indicator.spec, "kin")
        self.assertEqual(indicator.category, "walk")
        self.assertEqual(len(indicator.rows), REAL_KIN_WALK_ROW_COUNT)
        self.assertEqual(indicator.package_rows_sha256, REAL_KIN_WALK_ROWS_SHA256)

    def test_real_kin_walk_package_selects_expected_paired_rows_when_available(self):
        if not REAL_KIN_WALK_INDICATOR.exists():
            self.skipTest(f"real kin/walk indicator is not available locally: {REAL_KIN_WALK_INDICATOR}")
        if (
            not REAL_KIN_WALK_ROWS_JSONL.exists()
            and (not REAL_ROBOT_MOTION_DIR.exists() or not REAL_SOMA_MOTION_DIR.exists())
        ):
            self.skipTest("real paired soma_motionlib dirs/rows artifact are not available locally")

        rows = _load_real_soma_motionlib_rows(
            REAL_ROBOT_MOTION_DIR,
            REAL_SOMA_MOTION_DIR,
            rows_jsonl=REAL_KIN_WALK_ROWS_JSONL,
        )
        unchanged, unchanged_summary = filter_rows_by_data_package_config(rows, {"max_clips": 20000})
        selected, package_summary = filter_rows_by_data_package_config(
            rows,
            {
                "format": "soma_motionlib",
                "max_clips": 20000,
                "data_package": {
                    "spec": "kin",
                    "category": "walk",
                    "indicator": str(REAL_KIN_WALK_INDICATOR),
                    "missing_policy": "error",
                },
            },
        )

        self.assertIsNone(unchanged_summary)
        self.assertEqual(len(unchanged), len(rows))
        self.assertEqual(package_rows_sha256(unchanged), package_rows_sha256(rows))
        self.assertEqual(len(selected), REAL_KIN_WALK_ROW_COUNT)
        self.assertEqual(package_summary["selected_row_count"], REAL_KIN_WALK_ROW_COUNT)
        self.assertEqual(package_summary["package_rows_sha256"], REAL_KIN_WALK_ROWS_SHA256)


def _walk_rows():
    return [
        {
            "relative_path": "clip_walk_a.pkl",
            "robot_relative_path": "clip_walk_a.pkl",
            "soma_relative_path": "clip_walk_a.pkl",
            "source_bvh": "soma_proportional/bvh/walk_a.bvh",
        },
        {
            "relative_path": "clip_walk_b.pkl",
            "robot_relative_path": "clip_walk_b.pkl",
            "soma_relative_path": "clip_walk_b.pkl",
            "source_bvh": "soma_proportional/bvh/walk_b.bvh",
        },
        {
            "relative_path": "clip_run_c.pkl",
            "robot_relative_path": "clip_run_c.pkl",
            "soma_relative_path": "clip_run_c.pkl",
            "source_bvh": "soma_proportional/bvh/run_c.bvh",
        },
    ]


def _write_indicator(root: Path, rows, *, spec: str = "kin", category: str = "walk") -> Path:
    path = root / "walk.txt"
    lines = [
        f"# schema={SCHEMA}",
        f"# spec={spec}",
        f"# category={category}",
        "\t".join(("pair_id", "relative_path", "robot_relative_path", "soma_relative_path", "source_bvh")),
    ]
    for row in rows:
        pair_id = package_pair_id(row["robot_relative_path"], row["soma_relative_path"], row["source_bvh"])
        lines.append(
            "\t".join(
                (
                    pair_id,
                    row["relative_path"],
                    row["robot_relative_path"],
                    row["soma_relative_path"],
                    row["source_bvh"],
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _input_data(indicator: Path, *, max_clips: int = 0):
    return {
        "format": "soma_motionlib",
        "max_clips": max_clips,
        "data_package": {
            "spec": "kin",
            "category": "walk",
            "indicator": str(indicator),
            "missing_policy": "error",
        },
    }


def _load_real_soma_motionlib_rows(robot_dir: Path, soma_dir: Path, *, rows_jsonl: Path | None = None):
    if rows_jsonl is not None and rows_jsonl.exists():
        return _load_real_soma_motionlib_rows_jsonl(rows_jsonl)
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest(
            "joblib is required to rebuild real motionlib rows when the accepted rows JSONL is unavailable"
        ) from exc
    import numpy as np

    rows = []
    for robot_path in sorted(robot_dir.glob("*.pkl")):
        if robot_path.name == "metadata.pkl":
            continue
        soma_path = soma_dir / robot_path.name
        if not soma_path.exists():
            continue
        robot = _single_joblib_entry(joblib.load(robot_path), robot_path)
        soma = _single_joblib_entry(joblib.load(soma_path), soma_path)
        if not {"dof", "root_rot"}.issubset(robot) or not {"soma_joints", "soma_root_quat"}.issubset(soma):
            continue
        robot_frames = int(np.asarray(robot["dof"]).shape[0])
        soma_frames = int(np.asarray(soma["soma_joints"]).shape[0])
        target_fps = float(robot.get("fps") or 50.0)
        source_fps = float(soma.get("fps") or 120.0)
        if robot_frames <= 1 or soma_frames <= 1 or target_fps <= 0 or source_fps <= 0:
            continue
        if target_fps >= source_fps:
            continue
        if abs(soma_frames / source_fps - robot_frames / target_fps) > 0.05:
            continue
        rows.append(
            {
                "relative_path": robot_path.name,
                "robot_relative_path": robot_path.name,
                "soma_relative_path": soma_path.name,
                "source_bvh": str(soma.get("source_bvh", "")),
            }
        )
    return rows


def _load_real_soma_motionlib_rows_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _single_joblib_entry(loaded, path: Path):
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"motionlib file must contain a non-empty mapping: {path}")
    key = path.stem
    if key in loaded:
        return loaded[key]
    return loaded[next(iter(loaded))]
