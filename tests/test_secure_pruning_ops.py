import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "full JAX/SPU runtime is intentionally absent from lightweight CI")
class SecurePruningOpsTests(unittest.TestCase):
    def test_exact_topk_mask_uses_lowest_index_for_boundary_ties(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            exact_topk_keep_mask,
        )

        score = jnp.asarray([[0.9, 0.7, 0.7, 0.7, 0.1]], dtype=jnp.float32)
        active = jnp.asarray([[[1], [1], [1], [1], [0]]], dtype=jnp.float32)
        keep = np.asarray(exact_topk_keep_mask(score, active, 3)).reshape(-1)
        np.testing.assert_array_equal(keep, np.asarray([True, True, True, False, False]))

    def test_compact_topk_moves_matching_token_payload(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            compact_topk_tokens,
        )

        score = jnp.asarray([[0.5, 0.9, 0.9, -0.2, 0.7]], dtype=jnp.float32)
        indices = jnp.asarray([[0, 1, 2, 3, 4]], dtype=jnp.int32)
        tokens = indices.astype(jnp.float32)[:, :, None]
        compact_tokens, compact_indices = compact_topk_tokens(
            score,
            tokens,
            indices,
            3,
            fxp_fraction_bits=16,
            original_token_count=5,
        )
        np.testing.assert_array_equal(np.asarray(compact_indices), np.asarray([[1, 2, 4]]))
        np.testing.assert_allclose(np.asarray(compact_tokens[:, :, 0]), np.asarray([[1.0, 2.0, 4.0]]))

    def test_compact_topk_keeps_original_index_tie_policy_after_reordering(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            compact_topk_tokens,
        )

        score = jnp.asarray([[0.25, 0.25, 0.25]], dtype=jnp.float32)
        indices = jnp.asarray([[4, 1, 3]], dtype=jnp.int32)
        tokens = indices.astype(jnp.float32)[:, :, None]
        _, compact_indices = compact_topk_tokens(
            score,
            tokens,
            indices,
            2,
            fxp_fraction_bits=16,
            original_token_count=5,
        )
        np.testing.assert_array_equal(np.asarray(compact_indices), np.asarray([[1, 3]]))

    def test_packed_topk_key_preserves_score_then_original_index_order(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            pack_topk_key,
        )

        score = jnp.asarray([[0.25, 0.5, 0.5, 0.125]], dtype=jnp.float32)
        indices = jnp.asarray([[4, 3, 1, 0]], dtype=jnp.int32)
        packed = np.asarray(
            pack_topk_key(
                score,
                indices,
                fxp_fraction_bits=16,
                original_token_count=5,
            )
        )
        order = np.argsort(-packed, axis=1)
        np.testing.assert_array_equal(order, np.asarray([[2, 1, 0, 3]]))

    def test_logical_uniform_mean_matches_explicit_zero_token_outputs(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            logical_uniform_mean,
        )

        physical = jnp.asarray([[[[2.0], [4.0], [8.0]]]], dtype=jnp.float32)
        dropped_zero_value = jnp.asarray([[[[1.5]]]], dtype=jnp.float32)
        compact_mean = logical_uniform_mean(physical, dropped_zero_value, 5)
        explicit = jnp.concatenate(
            [physical, jnp.broadcast_to(dropped_zero_value, (1, 1, 2, 1))],
            axis=2,
        ).mean(axis=2, keepdims=True)
        np.testing.assert_allclose(np.asarray(compact_mean), np.asarray(explicit))


if __name__ == "__main__":
    unittest.main()
