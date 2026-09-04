import unittest

import cv2
import numpy as np

from algorithms_limb import (
    detect_faint_lunar_disk,
    refine_lunar_limb,
    refine_lunar_limb_elliptical,
)


def synthetic_partial_moon(center, axes, visible_from_x, size=800):
    image = np.zeros((size, size), np.uint8)
    cv2.ellipse(
        image,
        (int(round(center[0])), int(round(center[1]))),
        (int(round(axes[0])), int(round(axes[1]))),
        0.0,
        0.0,
        360.0,
        210,
        -1,
        cv2.LINE_AA,
    )
    image[:, :int(round(visible_from_x))] = 0
    return image


class ConstrainedEllipseLimbTests(unittest.TestCase):
    def test_well_covered_partial_disk_learns_ellipse_center(self):
        center = (410.0, 391.0)
        image = synthetic_partial_moon(center, (250.0, 242.0), center[0] - 55.0)
        result = refine_lunar_limb_elliptical(image, (407.0, 395.0, 247.0))

        self.assertIsNotNone(result)
        fitted, _, coverage, _, shape, updated = result
        self.assertTrue(updated)
        self.assertIsNotNone(shape)
        self.assertGreaterEqual(coverage, 0.42)
        self.assertLess(abs(fitted[0] - center[0]), 1.0)
        self.assertLess(abs(fitted[1] - center[1]), 1.0)

    def test_short_crescent_falls_back_without_cross_frame_state(self):
        axes = (250.0, 242.0)
        center = (414.0, 388.0)
        crescent = synthetic_partial_moon(center, axes, center[0] + 105.0)
        result = refine_lunar_limb_elliptical(
            crescent, (411.0, 392.0, 247.0), search_px=32,
        )

        self.assertIsNotNone(result)
        _, _, _, _, shape, updated = result
        self.assertFalse(updated)
        self.assertIsNone(shape)

    def test_short_crescent_center_can_use_reference_radius(self):
        axes = (250.0, 242.0)
        center = (414.0, 388.0)
        crescent = synthetic_partial_moon(center, axes, center[0] + 105.0)

        result = refine_lunar_limb(
            crescent,
            (411.0, 392.0, 247.0),
            fixed_radius=246.0,
            search_px=32,
        )

        self.assertIsNotNone(result)
        fitted, _, coverage, _ = result
        self.assertLess(coverage, 0.42)
        self.assertAlmostEqual(fitted[2], 246.0)
        self.assertLess(abs(fitted[0] - center[0]), 2.0)
        self.assertLess(abs(fitted[1] - center[1]), 2.0)


class FaintLimbFallbackTests(unittest.TestCase):
    @staticmethod
    def faint_scene(with_disk):
        rng = np.random.default_rng(4)
        height = width = 500
        yy, xx = np.mgrid[:height, :width]
        background = (
            105.0 + 0.025 * xx + 0.012 * yy
            + 0.000015 * np.square(xx - width / 2.0)
        )
        image = np.dstack((background + 4.0, background + 1.0, background - 2.0))
        if with_disk:
            disk = np.square(xx - 252.0) + np.square(yy - 244.0) <= 150.0 ** 2
            image[disk] += np.array([0.5, 1.0, 4.0])
        image += rng.normal(0.0, 1.2, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)

    def test_background_corrected_ring_integral_finds_faint_disk(self):
        result = detect_faint_lunar_disk(
            self.faint_scene(with_disk=True), 150.0, max_side=400,
        )

        self.assertIsNotNone(result)
        circle, confidence, _ = result
        self.assertLess(abs(circle[0] - 252.0), 1.0)
        self.assertLess(abs(circle[1] - 244.0), 1.0)
        self.assertGreater(confidence, 10.0)

    def test_background_noise_without_disk_is_rejected(self):
        result = detect_faint_lunar_disk(
            self.faint_scene(with_disk=False), 150.0, max_side=400,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
