import importlib.util
import math
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


class SecurePruningScheduleTests(unittest.TestCase):
    def test_runner_defaults_to_validated_unpadded_selection_network(self):
        from integrations.transshield_runtime.e2e_secure_vit.transshield_e2e_secure_vit import (
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "run",
                "--runtime",
                "cpu",
                "--input-pt",
                "input.pt",
                "--output-pt",
                "output.pt",
                "--output-json",
                "output.json",
            ]
        )
        self.assertEqual(args.spu_secure_pruning_network, "unpadded_selection")
        self.assertEqual(args.spu_square_activation_scale_fusion, "none")
        self.assertFalse(args.spu_public_fixed_square_scale)

        fused_args = build_parser().parse_args(
            [
                "run",
                "--runtime",
                "cpu",
                "--input-pt",
                "input.pt",
                "--output-pt",
                "output.pt",
                "--output-json",
                "output.json",
                "--spu-square-activation-scale-fusion",
                "mlp",
            ]
        )
        self.assertEqual(fused_args.spu_square_activation_scale_fusion, "mlp")

        public_args = build_parser().parse_args(
            [
                "run",
                "--runtime",
                "cpu",
                "--input-pt",
                "input.pt",
                "--output-pt",
                "output.pt",
                "--output-json",
                "output.json",
                "--spu-public-fixed-square-scale",
            ]
        )
        self.assertTrue(public_args.spu_public_fixed_square_scale)

    def test_runtime_pruning_ratio_override_is_cumulative_and_validated(self):
        from integrations.transshield_runtime.e2e_secure_vit.cpu_static_vit import (
            resolve_runtime_pruning_token_ratios,
        )

        self.assertEqual(
            resolve_runtime_pruning_token_ratios((0.7, 0.49, 0.343), 0.0),
            (0.7, 0.49, 0.343),
        )
        ratios = resolve_runtime_pruning_token_ratios((0.7, 0.49, 0.343), 0.655)
        self.assertEqual([int(196 * value) for value in ratios], [128, 84, 55])
        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "token_ratio_base_override"):
                    resolve_runtime_pruning_token_ratios((0.7, 0.49, 0.343), invalid)

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

    def test_selection_schedule_removes_comparators_outside_output_cone(self):
        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            _bitonic_full_schedule,
            _bitonic_selection_schedule,
        )

        cases = (
            (256, tuple(range(137)), 4299),
            (256, tuple(range(96)), 4080),
            (128, tuple(range(67)), 1664),
            (128, (66,), 1471),
        )
        for token_count, outputs, expected_comparators in cases:
            with self.subTest(token_count=token_count, outputs=len(outputs)):
                full_count = sum(len(layer[0]) for layer in _bitonic_full_schedule(token_count))
                selected_count = sum(
                    len(layer[0])
                    for layer in _bitonic_selection_schedule(token_count, outputs)
                )
                self.assertEqual(selected_count, expected_comparators)
                self.assertLess(selected_count, full_count)

    def test_unpadded_network_is_exact_for_all_binary_inputs_up_to_ten_wires(self):
        from itertools import product

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            _bitonic_arbitrary_full_schedule,
        )

        for token_count in range(2, 11):
            schedule = _bitonic_arbitrary_full_schedule(token_count)
            for source in product((0, 1), repeat=token_count):
                values = list(source)
                for left, right, _, descending, _ in schedule:
                    for left_index, right_index, pair_descending in zip(
                        left,
                        right,
                        descending,
                    ):
                        left_value = values[left_index]
                        right_value = values[right_index]
                        high = max(left_value, right_value)
                        low = min(left_value, right_value)
                        values[left_index] = high if pair_descending else low
                        values[right_index] = low if pair_descending else high
                self.assertEqual(values, sorted(source, reverse=True))

    def test_unpadded_selection_halves_formal_pruning_comparator_count(self):
        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            _bitonic_unpadded_selection_schedule,
        )

        stages = ((196, tuple(range(137)), 2921), (137, tuple(range(96)), 1799), (96, (66,), 831))
        actual_total = 0
        for token_count, outputs, expected in stages:
            count = sum(
                len(layer[0])
                for layer in _bitonic_unpadded_selection_schedule(token_count, outputs)
            )
            self.assertEqual(count, expected)
            actual_total += count
        self.assertEqual(actual_total, 5551)
        self.assertLess(actual_total, 11008 // 2 + 100)

    def test_odd_even_candidate_reduces_low_latency_comparator_count(self):
        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            pruning_network_comparator_count,
        )

        stages = ((196, 128, False), (128, 84, False), (84, 55, True))
        bitonic_total = 0
        odd_even_total = 0
        for token_count, keep_count, threshold_only in stages:
            bitonic_total += pruning_network_comparator_count(
                token_count,
                keep_count,
                pruning_network="unpadded_selection",
                threshold_only=threshold_only,
            )
            odd_even_total += pruning_network_comparator_count(
                token_count,
                keep_count,
                pruning_network="odd_even_selection",
                threshold_only=threshold_only,
            )
        self.assertEqual(bitonic_total, 5287)
        self.assertEqual(odd_even_total, 4822)
        self.assertLess(odd_even_total, bitonic_total)

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

    def test_bitonic_selection_matches_requested_full_sort_outputs(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            bitonic_select_desc,
        )

        rng = np.random.default_rng(20260801)
        values = rng.integers(-4, 5, size=(5, 256)).astype(np.float32)
        outputs = (0, 31, 66, 95, 136, 255)
        actual = np.asarray(bitonic_select_desc(jnp.asarray(values), outputs))
        expected = np.sort(values, axis=1)[:, ::-1][:, outputs]
        np.testing.assert_array_equal(actual, expected)

    def test_unpadded_bitonic_selection_matches_reference(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            bitonic_unpadded_select_desc,
        )

        rng = np.random.default_rng(20260804)
        for token_count, outputs in (
            (196, (0, 31, 95, 136, 195)),
            (137, (0, 47, 95, 136)),
            (96, (0, 32, 66, 95)),
        ):
            values = rng.integers(-4, 5, size=(3, token_count)).astype(np.float32)
            actual = np.asarray(
                bitonic_unpadded_select_desc(jnp.asarray(values), outputs)
            )
            expected = np.sort(values, axis=1)[:, ::-1][:, outputs]
            np.testing.assert_array_equal(actual, expected)

    def test_unpadded_odd_even_selection_matches_reference(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            odd_even_unpadded_select_desc,
        )

        rng = np.random.default_rng(20260805)
        for token_count, outputs in (
            (196, (0, 31, 127, 195)),
            (128, (0, 47, 83, 127)),
            (84, (0, 27, 54, 83)),
        ):
            values = rng.integers(-4, 5, size=(3, token_count)).astype(np.float32)
            actual = np.asarray(
                odd_even_unpadded_select_desc(jnp.asarray(values), outputs)
            )
            expected = np.sort(values, axis=1)[:, ::-1][:, outputs]
            np.testing.assert_array_equal(actual, expected)

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

    def test_unique_packed_topk_mask_matches_generic_tie_path(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            exact_topk_keep_mask,
            pack_topk_key,
        )

        score = jnp.asarray([[0.5, 0.25, 0.5, -0.125, 0.25]], dtype=jnp.float32)
        indices = jnp.asarray([[4, 1, 3, 0, 2]], dtype=jnp.int32)
        packed = pack_topk_key(
            score,
            indices,
            fxp_fraction_bits=16,
            original_token_count=5,
        )
        active = jnp.ones((1, 5, 1), dtype=jnp.float32)
        generic = np.asarray(exact_topk_keep_mask(packed, active, 3))
        unique = np.asarray(exact_topk_keep_mask(packed, active, 3, unique_keys=True))
        np.testing.assert_array_equal(unique, generic)

    def test_selection_topk_mask_matches_full_sort_path(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            exact_topk_keep_mask,
        )

        rng = np.random.default_rng(20260802)
        score = jnp.asarray(rng.integers(-3, 4, size=(4, 197)).astype(np.float32))
        active = jnp.ones((4, 197, 1), dtype=jnp.float32)
        full_sort = np.asarray(exact_topk_keep_mask(score, active, 67))
        selection = np.asarray(
            exact_topk_keep_mask(
                score,
                active,
                67,
                pruning_network="selection",
            )
        )
        np.testing.assert_array_equal(selection, full_sort)

        unpadded = np.asarray(
            exact_topk_keep_mask(
                score,
                active,
                67,
                pruning_network="unpadded_selection",
            )
        )
        np.testing.assert_array_equal(unpadded, full_sort)

        odd_even = np.asarray(
            exact_topk_keep_mask(
                score,
                active,
                67,
                pruning_network="odd_even_selection",
            )
        )
        np.testing.assert_array_equal(odd_even, full_sort)

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

    def test_selection_compaction_matches_full_sort_tokens_and_indices(self):
        import jax.numpy as jnp
        import numpy as np

        from integrations.transshield_runtime.e2e_secure_vit.secure_pruning_ops import (
            compact_topk_tokens,
        )

        rng = np.random.default_rng(20260803)
        # Keep packed scores inside a range where float32 also preserves the
        # 2**-16 original-index tie unit used by the FM64/fxp16 SPU graph.
        score = jnp.asarray(
            (rng.integers(-8, 9, size=(3, 137)) / 64.0).astype(np.float32)
        )
        indices = jnp.broadcast_to(jnp.arange(137, dtype=jnp.int32), (3, 137))
        tokens = jnp.asarray(rng.normal(size=(3, 137, 7)).astype(np.float32))
        full_tokens, full_indices = compact_topk_tokens(
            score,
            tokens,
            indices,
            96,
            fxp_fraction_bits=16,
            original_token_count=196,
        )
        selected_tokens, selected_indices = compact_topk_tokens(
            score,
            tokens,
            indices,
            96,
            fxp_fraction_bits=16,
            original_token_count=196,
            pruning_network="selection",
        )
        np.testing.assert_array_equal(np.asarray(selected_indices), np.asarray(full_indices))
        np.testing.assert_array_equal(np.asarray(selected_tokens), np.asarray(full_tokens))

        unpadded_tokens, unpadded_indices = compact_topk_tokens(
            score,
            tokens,
            indices,
            96,
            fxp_fraction_bits=16,
            original_token_count=196,
            pruning_network="unpadded_selection",
        )
        np.testing.assert_array_equal(np.asarray(unpadded_indices), np.asarray(full_indices))
        np.testing.assert_array_equal(np.asarray(unpadded_tokens), np.asarray(full_tokens))

        odd_even_tokens, odd_even_indices = compact_topk_tokens(
            score,
            tokens,
            indices,
            96,
            fxp_fraction_bits=16,
            original_token_count=196,
            pruning_network="odd_even_selection",
        )
        np.testing.assert_array_equal(np.asarray(odd_even_indices), np.asarray(full_indices))
        np.testing.assert_array_equal(np.asarray(odd_even_tokens), np.asarray(full_tokens))

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
