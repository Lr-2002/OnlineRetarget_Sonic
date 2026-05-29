from __future__ import annotations

import sys
import types
import unittest

from online_retarget.sonic_tokenizer_compat import install_tokenizer_cfg_compat


class TokenizerCompatTests(unittest.TestCase):
    def test_patch_allows_local_terms_rejected_by_upstream_signature(self) -> None:
        module = types.ModuleType("_online_retarget_fake_tokenizer_cfg")

        class TokenizerCfg:
            def __init__(self, existing: str = "base") -> None:
                self.existing = existing

        module.TokenizerCfg = TokenizerCfg
        sys.modules[module.__name__] = module
        try:
            install_tokenizer_cfg_compat()
            cfg = TokenizerCfg(
                existing="kept",
                root_pos_w_mf="position-term",
                root_rot_w_mf="rotation-term",
            )
        finally:
            sys.modules.pop(module.__name__, None)

        self.assertEqual(cfg.existing, "kept")
        self.assertEqual(cfg.root_pos_w_mf, "position-term")
        self.assertEqual(cfg.root_rot_w_mf, "rotation-term")

    def test_patch_handles_nested_tokenizer_cfg_classes(self) -> None:
        module = types.ModuleType("_online_retarget_nested_tokenizer_cfg")

        class ObservationsCfg:
            class TokenizerCfg:
                def __init__(self, existing: str = "nested") -> None:
                    self.existing = existing

        module.ObservationsCfg = ObservationsCfg
        sys.modules[module.__name__] = module
        try:
            install_tokenizer_cfg_compat()
            cfg = ObservationsCfg.TokenizerCfg(
                existing="kept",
                soma_morphology="morphology-term",
                root_pos_w_mf="position-term",
            )
        finally:
            sys.modules.pop(module.__name__, None)

        self.assertEqual(cfg.existing, "kept")
        self.assertEqual(cfg.soma_morphology, "morphology-term")
        self.assertEqual(cfg.root_pos_w_mf, "position-term")


if __name__ == "__main__":
    unittest.main()
