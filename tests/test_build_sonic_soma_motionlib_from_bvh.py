import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_sonic_soma_motionlib_from_bvh.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_sonic_soma_motionlib", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildSonicSomaMotionlibFromBvhTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script_module()

    def test_date_motion_key_maps_to_dated_bvh_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            candidates = self.script.bvh_candidates_for_motion_key(
                "230710__walk_ff_stop_360_R_001__A418",
                root,
            )

        self.assertEqual(
            candidates,
            (root / "230710" / "walk_ff_stop_360_R_001__A418.bvh",),
        )

    def test_resolve_bvh_uses_direct_date_candidate_before_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "230710" / "walk__A001.bvh"
            expected.parent.mkdir()
            expected.write_text("HIERARCHY\n", encoding="utf-8")

            resolved = self.script.resolve_bvh_for_motion_key(
                "230710__walk__A001",
                root,
                {},
            )

        self.assertEqual(resolved, expected)

    def test_non_date_motion_key_falls_back_to_bvh_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            indexed = root / "nested" / "clip_without_date__A001.bvh"
            indexed.parent.mkdir()
            indexed.write_text("HIERARCHY\n", encoding="utf-8")
            bvh_index = self.script.build_bvh_index(root, ["clip_without_date__A001"])

            resolved = self.script.resolve_bvh_for_motion_key(
                "clip_without_date__A001",
                root,
                bvh_index,
            )

        self.assertEqual(resolved, indexed)

    def test_load_robot_motion_keys_from_per_motion_pkls_without_joblib(self):
        with tempfile.TemporaryDirectory() as tmp:
            motion_dir = Path(tmp)
            (motion_dir / "metadata.pkl").write_bytes(b"ignored when per-file fallback is enough")
            (motion_dir / "230710__walk__A001.pkl").write_bytes(b"not-read")
            (motion_dir / "230710__turn__A002.pkl").write_bytes(b"not-read")
            (motion_dir / "metadata.pkl").unlink()

            keys = self.script.load_robot_motion_keys(motion_dir)

        self.assertEqual(keys, ["230710__turn__A002", "230710__walk__A001"])


if __name__ == "__main__":
    unittest.main()
