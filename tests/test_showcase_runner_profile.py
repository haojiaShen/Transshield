from __future__ import annotations

import os
import unittest
from unittest import mock

from showcase_api.config import load_showcase_config


class ShowcaseRunnerProfileTests(unittest.TestCase):
    def load_profile(self, **overrides):
        names = {
            "TRANSSHIELD_SHOWCASE_SPU_ATTENTION_POLICY",
            "TRANSSHIELD_SHOWCASE_SPU_SECURE_PRUNING_NETWORK",
            "TRANSSHIELD_SHOWCASE_SPU_FINAL_BLOCK_CLS_ONLY",
            "TRANSSHIELD_SHOWCASE_SPU_UNIFORM_ATTENTION_VALUE_FUSION",
        }
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in names:
                os.environ.pop(name, None)
            os.environ.update(overrides)
            return load_showcase_config().runner_profile

    def test_validated_uniform_profile_enables_both_graph_optimizations(self):
        profile = self.load_profile()
        self.assertEqual(profile.spu_attention_policy, "uniform")
        self.assertTrue(profile.spu_final_block_cls_only)
        self.assertTrue(profile.spu_uniform_attention_value_fusion)
        self.assertEqual(profile.spu_secure_pruning_network, "unpadded_selection")

    def test_nonuniform_profile_disables_uniform_only_defaults(self):
        profile = self.load_profile(
            TRANSSHIELD_SHOWCASE_SPU_ATTENTION_POLICY="standard",
        )
        self.assertFalse(profile.spu_final_block_cls_only)
        self.assertFalse(profile.spu_uniform_attention_value_fusion)

    def test_secure_pruning_network_can_select_unpadded_graph(self):
        profile = self.load_profile(
            TRANSSHIELD_SHOWCASE_SPU_SECURE_PRUNING_NETWORK="unpadded_selection",
        )
        self.assertEqual(profile.spu_secure_pruning_network, "unpadded_selection")

    def test_explicit_rollback_overrides_uniform_defaults(self):
        profile = self.load_profile(
            TRANSSHIELD_SHOWCASE_SPU_FINAL_BLOCK_CLS_ONLY="0",
            TRANSSHIELD_SHOWCASE_SPU_UNIFORM_ATTENTION_VALUE_FUSION="off",
        )
        self.assertFalse(profile.spu_final_block_cls_only)
        self.assertFalse(profile.spu_uniform_attention_value_fusion)


if __name__ == "__main__":
    unittest.main()
