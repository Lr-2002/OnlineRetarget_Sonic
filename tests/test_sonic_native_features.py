import unittest
from pathlib import Path

from online_retarget.sonic_native_features import (
    FeatureContractError,
    SonicNativeFeatureContract,
    assert_matching_contracts,
    pack_inference_features,
    pack_source_motion_with_morphology,
    pack_training_pair,
)
from online_retarget.sonic_morphology import morphology_from_registry_row


ROOT = Path(__file__).resolve().parents[1]
PROPORTIONAL_CONFIG = ROOT / "configs" / "sonic_kin_only_soma_encoder_proportional.json"


class SonicNativeFeatureContractTests(unittest.TestCase):
    def test_contract_from_formal_config_has_stable_source_and_target_roles(self):
        contract = SonicNativeFeatureContract.from_config_path(PROPORTIONAL_CONFIG)

        self.assertIn("soma_joints_multi_future_local_nonflat", contract.source_keys)
        self.assertIn("soma_root_ori_b_multi_future", contract.source_keys)
        self.assertNotIn("body_pos_w", contract.source_keys)
        self.assertIn("body_pos_w", contract.target_label_keys)
        self.assertEqual(contract.target_fps, 50.0)
        self.assertEqual(len(contract.digest), 16)

    def test_training_and_inference_pack_the_same_source_contract(self):
        contract = SonicNativeFeatureContract.from_config_path(PROPORTIONAL_CONFIG)
        source = _source_payload(contract)
        target = _target_payload(contract)

        training = pack_training_pair(source, target, contract)
        inference = pack_inference_features(source, contract)

        self.assertEqual(training.source, inference.source)
        self.assertEqual(training.contract_digest, inference.contract_digest)
        assert_matching_contracts(contract, contract)

    def test_source_motion_can_be_merged_with_morphology_features(self):
        contract = SonicNativeFeatureContract.from_config_path(PROPORTIONAL_CONFIG)
        morphology = morphology_from_registry_row(_morphology_row()).as_source_features()
        source_motion = {
            "soma_joints_multi_future_local_nonflat": "source:joints",
            "soma_root_ori_b_multi_future": "source:root",
        }

        packed = pack_source_motion_with_morphology(source_motion, morphology, contract)

        self.assertEqual(packed.source["actor_uid"], "A001")
        self.assertEqual(packed.source["skeleton_id"], "A001")
        self.assertIn("bone_lengths", packed.source)

    def test_source_payload_rejects_target_only_body_pose(self):
        contract = SonicNativeFeatureContract.from_config_path(PROPORTIONAL_CONFIG)
        source = _source_payload(contract)
        source["body_pos_w"] = "target-state"

        with self.assertRaisesRegex(FeatureContractError, "target-only fields"):
            pack_inference_features(source, contract)

    def test_missing_required_source_feature_is_rejected(self):
        contract = SonicNativeFeatureContract.from_config_path(PROPORTIONAL_CONFIG)
        source = _source_payload(contract)
        source.pop("soma_root_ori_b_multi_future")

        with self.assertRaisesRegex(FeatureContractError, "missing source features"):
            pack_inference_features(source, contract)

    def test_contract_mismatch_is_rejected(self):
        contract = SonicNativeFeatureContract.from_config_path(PROPORTIONAL_CONFIG)
        changed = SonicNativeFeatureContract(
            source_keys=contract.source_keys + ("extra_source_feature",),
            target_label_keys=contract.target_label_keys,
            target_fps=contract.target_fps,
            optional_source_keys=contract.optional_source_keys,
            variant=contract.variant,
        )

        with self.assertRaisesRegex(FeatureContractError, "mismatch"):
            assert_matching_contracts(contract, changed)


def _source_payload(contract: SonicNativeFeatureContract) -> dict[str, object]:
    return {key: f"source:{key}" for key in contract.required_source_keys}


def _target_payload(contract: SonicNativeFeatureContract) -> dict[str, object]:
    return {key: f"target:{key}" for key in contract.target_label_keys}


def _morphology_row() -> dict[str, object]:
    return {
        "actor_uid": "A001",
        "encoder_id": "A001",
        "actor_height_cm": 170.0,
        "actor_foot_cm": 27.0,
        "actor_collarbone_height_cm": 140.0,
        "actor_collarbone_span_cm": 38.0,
        "actor_elbow_span_cm": 95.0,
        "actor_wrist_span_cm": 120.0,
        "actor_shoulder_span_cm": 160.0,
        "actor_hips_height_cm": 90.0,
        "actor_hips_bones_span_cm": 32.0,
        "actor_knee_height_cm": 50.0,
        "actor_ankle_height_cm": 9.0,
        "shape_path": "soma_shapes/soma_proportion_fit_mhr_params/A001.npz",
    }
