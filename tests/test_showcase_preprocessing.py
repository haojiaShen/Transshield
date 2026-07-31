from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from showcase_api.preprocessing import (
    decode_rgb_for_formal_evaluation,
    eval_resize_shorter_side,
    normalize_rgb_hwc,
    resize_for_formal_evaluation,
)


class ShowcasePreprocessingTest(unittest.TestCase):
    def test_formal_medical_geometry_uses_resize_then_center_crop(self):
        width, height = 400, 300
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(width, dtype=np.uint16)[None, :] % 256
        pixels[:, :, 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
        pixels[:, :, 2] = 127
        image = Image.fromarray(pixels)

        output = resize_for_formal_evaluation(image, input_size=224, crop_pct=0.875)

        self.assertEqual(eval_resize_shorter_side(224, 0.875), 256)
        self.assertEqual(output.size, (224, 224))
        # A crop-first implementation would preserve almost the full horizontal
        # source range. Resize-first + center-crop intentionally removes both sides.
        output_array = np.asarray(output)
        self.assertGreater(int(output_array[112, 0, 0]), 20)
        self.assertLess(int(output_array[112, -1, 0]), 235)

    def test_decode_reports_original_size_and_float_rgb(self):
        image = Image.new("RGB", (320, 240), color=(64, 128, 192))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        rgb, source_size = decode_rgb_for_formal_evaluation(
            buffer.getvalue(),
            input_size=224,
            crop_pct=0.875,
            max_image_dimension=8192,
        )

        self.assertEqual(source_size, (320, 240))
        self.assertEqual(rgb.shape, (224, 224, 3))
        self.assertEqual(rgb.dtype, np.float32)
        np.testing.assert_allclose(rgb[0, 0], np.array([64, 128, 192], dtype=np.float32) / 255.0)

    def test_formal_normalization_is_not_clipped_by_default(self):
        rgb = np.zeros((2, 2, 3), dtype=np.float32)
        normalized = normalize_rgb_hwc(
            rgb,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        clipped = normalize_rgb_hwc(
            rgb,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            clip_abs=2.0,
        )

        self.assertLess(float(normalized[0, 0, 0, 0]), -2.0)
        self.assertEqual(float(clipped[0, 0, 0, 0]), -2.0)


if __name__ == "__main__":
    unittest.main()
