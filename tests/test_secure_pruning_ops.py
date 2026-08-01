import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


class SecurePruningScheduleTests(unittest.TestCase):
    def test_bitonic_schedule_compares_each_pair_once_and_restores_token_order(self):
        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            _bitonic_pair_schedule,
        )

        for token_count in (2, 4, 8, 256):
            stage_size = 2
            while stage_size <= token_count:
                stride = stage_size // 2
                while stride >= 1:
                    left, right, restore, _ = _bitonic_pair_schedule(
                        token_count,
                        stage_size,
                        stride,
                    )
                    self.assertEqual(len(left), token_count // 2)
                    self.assertEqual(len(right), token_count // 2)
                    paired_order = left + right
                    self.assertEqual(sorted(paired_order), list(range(token_count)))
                    self.assertEqual(
                        [paired_order[offset] for offset in restore],
                        list(range(token_count)),
                    )
                    stride //= 2
                stage_size *= 2

    def test_static_loader_trims_stages_outside_prefix_depth(self):
        from integrations.transshield_runtime.e2e_secure_vit.static_vit_params import (
            pruning_schedule_for_depth,
        )

        self.assertEqual(pruning_schedule_for_depth(0.7, 1), ([], []))
        locations, ratios = pruning_schedule_for_depth(0.7, 7)
        self.assertEqual(locations, [3, 6])
        self.assertEqual(len(ratios), 2)
        self.assertAlmostEqual(ratios[0], 0.7)
        self.assertAlmostEqual(ratios[1], 0.49)
        with self.assertRaisesRegex(ValueError, "base_rate"):
            pruning_schedule_for_depth(1.1, 10)

    def test_normalizes_valid_cumulative_schedule(self):
        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            normalize_pruning_schedule,
        )

        locations, counts = normalize_pruning_schedule([3, 6, 9], [137, 96, 67], depth=10)
        self.assertEqual(locations, (3, 6, 9))
        self.assertEqual(counts, (137, 96, 67))

    def test_rejects_mismatched_or_increasing_schedule(self):
        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            normalize_pruning_schedule,
        )

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            normalize_pruning_schedule([3, 6], [137], depth=10)
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            normalize_pruning_schedule([3, 6], [96, 137], depth=10)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            normalize_pruning_schedule([3], [197], depth=10, max_token_count=196)

    def test_rejects_duplicate_or_out_of_depth_locations(self):
        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            normalize_pruning_schedule,
        )

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            normalize_pruning_schedule([3, 3], [137, 96], depth=10)
        with self.assertRaisesRegex(ValueError, "executed depth"):
            normalize_pruning_schedule([3, 10], [137, 96], depth=10)


@unittest.skipUnless(JAX_AVAILABLE, "full JAX/SPU runtime is intentionally absent from lightweight CI")
class SecurePruningOpsTests(unittest.TestCase):
    def test_bitonic_sort_desc_matches_reference_with_ties(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            bitonic_sort_desc,
        )

        values = np.asarray(
            [
                [0.5, -1.0, 0.5, 3.0, 2.0, 2.0, -4.0, 1.0],
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        actual = np.asarray(bitonic_sort_desc(jnp.asarray(values)))
        expected = np.sort(values, axis=1)[:, ::-1]
        np.testing.assert_allclose(actual, expected)

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
