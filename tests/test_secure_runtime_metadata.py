from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SecureRuntimeMetadataTest(unittest.TestCase):
    def test_secret_params_does_not_claim_no_host_materialization(self):
        runner_source = (
            REPO_ROOT
            / "integrations"
            / "transshield_runtime"
            / "e2e_secure_vit"
            / "transshield_e2e_secure_vit.py"
        ).read_text(encoding="utf-8")
        loader_source = (
            REPO_ROOT
            / "integrations"
            / "transshield_runtime"
            / "e2e_secure_vit"
            / "static_vit_params.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"host_model_params_materialized": True', runner_source)
        self.assertIn('"model_params_secret_inside_spu": bool(args.spu_params_mode == "secret")', runner_source)
        self.assertIn("torch.load(state_dict_path, map_location=\"cpu\"", loader_source)
        self.assertNotIn("host_model_params_materialized = false", loader_source)

    def test_secure_predictor_loader_does_not_claim_pruning_is_bypassed(self):
        from integrations.transshield_runtime.e2e_secure_vit.static_vit_params import (
            secure_pruning_unsupported_items,
        )

        remaining = secure_pruning_unsupported_items(
            [
                "intermediate feature reveal",
                "runtime pruning predictor path",
                "dynamic masking-pruning inside secure forward",
            ]
        )
        self.assertEqual(remaining, ["intermediate feature reveal"])


if __name__ == "__main__":
    unittest.main()
