import importlib.util
import tempfile
import unittest


FULL_SPU_AVAILABLE = (
    importlib.util.find_spec("jax") is not None
    and importlib.util.find_spec("spu") is not None
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


if __name__ == "__main__":
    unittest.main()
