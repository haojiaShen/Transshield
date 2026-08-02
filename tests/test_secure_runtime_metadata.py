from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SecureRuntimeMetadataTest(unittest.TestCase):
    def test_public_fixed_square_scale_requires_one_uniform_constant(self):
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.static_vit_params import (
            resolve_uniform_fixed_square_scale,
        )

        block = [np.asarray(0.0, dtype=np.float32) for _ in range(14)]
        block[10] = np.asarray(0.25, dtype=np.float32)
        params = [None, None, None, None, (tuple(block), tuple(block))]
        predictor = [np.asarray(0.0, dtype=np.float32) for _ in range(13)]
        for index in (4, 7, 10):
            predictor[index] = np.asarray(0.25, dtype=np.float32)

        self.assertEqual(
            resolve_uniform_fixed_square_scale(tuple(params), (tuple(predictor),)),
            0.25,
        )
        predictor[10] = np.asarray(0.5, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "identical architecture constant"):
            resolve_uniform_fixed_square_scale(tuple(params), (tuple(predictor),))

    def test_square_scale_fusion_preserves_regular_and_decomposed_linears(self):
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.static_vit_params import (
            fold_input_scale_into_linear_weight,
        )

        rng = np.random.default_rng(20260802)
        hidden = rng.normal(size=(5, 3)).astype(np.float32)
        alpha = np.asarray(0.25, dtype=np.float32)

        weight = rng.normal(size=(4, 3)).astype(np.float32)
        bias = rng.normal(size=(4,)).astype(np.float32)
        fused_weight = fold_input_scale_into_linear_weight(weight, alpha)
        original = (hidden * alpha) @ weight.T + bias
        fused = hidden @ fused_weight.T + bias
        np.testing.assert_allclose(fused, original, rtol=1e-6, atol=1e-6)

        down_weight = rng.normal(size=(2, 3)).astype(np.float32)
        up_weight = rng.normal(size=(4, 2)).astype(np.float32)
        fused_down, fused_up = fold_input_scale_into_linear_weight(
            (down_weight, up_weight),
            alpha,
        )
        original = ((hidden * alpha) @ down_weight.T) @ up_weight.T + bias
        fused = (hidden @ fused_down.T) @ fused_up.T + bias
        np.testing.assert_allclose(fused, original, rtol=1e-6, atol=1e-6)

    def test_predictor_fusion_scales_each_following_weight_once(self):
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.static_vit_params import (
            fold_predictor_square_activation_scales,
        )

        untouched = [np.asarray(index, dtype=np.float32) for index in range(13)]
        untouched[4] = np.asarray(0.25, dtype=np.float32)
        untouched[5] = np.ones((2, 2), dtype=np.float32)
        untouched[7] = np.asarray(0.5, dtype=np.float32)
        untouched[8] = np.ones((2, 2), dtype=np.float32)
        untouched[10] = np.asarray(0.125, dtype=np.float32)
        untouched[11] = np.ones((2, 2), dtype=np.float32)

        fused = fold_predictor_square_activation_scales(tuple(untouched))
        self.assertEqual(float(fused[4]), 1.0)
        self.assertEqual(float(fused[7]), 1.0)
        self.assertEqual(float(fused[10]), 1.0)
        np.testing.assert_array_equal(fused[5], np.full((2, 2), 0.25, dtype=np.float32))
        np.testing.assert_array_equal(fused[8], np.full((2, 2), 0.5, dtype=np.float32))
        np.testing.assert_array_equal(fused[11], np.full((2, 2), 0.125, dtype=np.float32))

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
