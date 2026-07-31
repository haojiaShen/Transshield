from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from showcase_api.bundle_preflight import inspect_bundle, require_bundle_ready


class BundlePreflightTest(unittest.TestCase):
    def test_missing_model_state_is_not_ready(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir)
            status = inspect_bundle(bundle_dir)
            self.assertFalse(status["ready"])
            self.assertFalse(status["model_state_present"])
            with self.assertRaisesRegex(RuntimeError, "model state is missing"):
                require_bundle_ready(bundle_dir)

    def test_manifest_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir)
            model_path = bundle_dir / "modified_plaintext_model_state_dict.pth"
            model_path.write_bytes(b"small-test-state")
            expected = hashlib.sha256(model_path.read_bytes()).hexdigest()
            (bundle_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "export": {
                            "model_state_dict_path": "/server/path/modified_plaintext_model_state_dict.pth",
                            "model_state_dict_sha256": expected,
                        }
                    }
                ),
                encoding="utf-8",
            )

            status = require_bundle_ready(bundle_dir, verify_hash=True)
            self.assertTrue(status["ready"])
            self.assertTrue(status["model_sha256_matches"])

            model_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                require_bundle_ready(bundle_dir, verify_hash=True)


if __name__ == "__main__":
    unittest.main()
