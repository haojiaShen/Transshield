import importlib.util
import tempfile
import unittest


FULL_SPU_AVAILABLE = (
    importlib.util.find_spec("jax") is not None
    and importlib.util.find_spec("spu") is not None
)


class CompilerOptionsDigestTests(unittest.TestCase):
    def test_protobuf_and_pybind_style_options_are_deterministic(self):
        from integrations.transshield_runtime.e2e_secure_vit.spu_compile_cache import (
            _compiler_options_digest,
        )

        class ProtobufOptions:
            def __init__(self, payload):
                self.payload = payload

            def SerializeToString(self):
                return self.payload

        class PybindOptions:
            def __init__(self, enable_pretty_print, xla_pp_kind):
                self.enable_pretty_print = enable_pretty_print
                self.xla_pp_kind = xla_pp_kind

            def helper(self):
                return "ignored"

        self.assertEqual(
            _compiler_options_digest(ProtobufOptions(b"same")),
            _compiler_options_digest(ProtobufOptions(b"same")),
        )
        self.assertNotEqual(
            _compiler_options_digest(ProtobufOptions(b"same")),
            _compiler_options_digest(ProtobufOptions(b"different")),
        )
        self.assertEqual(
            _compiler_options_digest(PybindOptions(True, 2)),
            _compiler_options_digest(PybindOptions(True, 2)),
        )
        self.assertNotEqual(
            _compiler_options_digest(PybindOptions(True, 2)),
            _compiler_options_digest(PybindOptions(False, 2)),
        )


@unittest.skipUnless(FULL_SPU_AVAILABLE, "full JAX/SPU runtime is intentionally absent from lightweight CI")
class SpuCompileCacheTests(unittest.TestCase):
    def test_second_identical_compile_uses_persistent_entry(self):
        import numpy as np
        from spu import spu_pb2
        from spu.utils import frontend

        from integrations.transshield_runtime.e2e_secure_vit.spu_compile_cache import (
            get_spu_compile_cache_stats,
            install_spu_compile_cache,
            reset_spu_compile_cache_stats,
        )

        def add_one(value):
            return value + 1.0

        with tempfile.TemporaryDirectory(prefix="transshield_spu_compile_cache_test_") as cache_dir:
            install_spu_compile_cache(cache_dir, namespace="unit-test-v1")
            reset_spu_compile_cache_stats()
            compile_args = (
                frontend.Kind.JAX,
                add_one,
                (np.zeros((2,), dtype=np.float32),),
                {},
                ["input"],
                [spu_pb2.Visibility.VIS_SECRET],
                lambda output: ["output"],
            )
            frontend.compile(*compile_args)
            frontend.compile(*compile_args)
            stats = get_spu_compile_cache_stats()
            self.assertEqual(stats["misses"], 1)
            self.assertEqual(stats["hits"], 1)
            self.assertEqual(stats["errors"], 0)

    def test_cache_hit_refreshes_runtime_input_and_output_names(self):
        import numpy as np
        from spu import spu_pb2
        from spu.utils import frontend

        from integrations.transshield_runtime.e2e_secure_vit.spu_compile_cache import (
            get_spu_compile_cache_stats,
            install_spu_compile_cache,
            reset_spu_compile_cache_stats,
        )

        def add_one(value):
            return value + 1.0

        with tempfile.TemporaryDirectory(prefix="transshield_spu_compile_cache_names_test_") as cache_dir:
            install_spu_compile_cache(cache_dir, namespace="unit-test-runtime-names-v1")
            reset_spu_compile_cache_stats()
            common = (
                frontend.Kind.JAX,
                add_one,
                (np.zeros((2,), dtype=np.float32),),
                {},
            )
            frontend.compile(
                *common,
                ["input-first"],
                [spu_pb2.Visibility.VIS_SECRET],
                lambda output: ["output-first"],
            )
            executable, _ = frontend.compile(
                *common,
                ["input-second"],
                [spu_pb2.Visibility.VIS_SECRET],
                lambda output: ["output-second"],
            )
            stats = get_spu_compile_cache_stats()
            self.assertEqual(stats["misses"], 1)
            self.assertEqual(stats["hits"], 1)
            self.assertEqual(list(executable.input_names), ["input-second"])
            self.assertEqual(list(executable.output_names), ["output-second"])


if __name__ == "__main__":
    unittest.main()
