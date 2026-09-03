import unittest

import cv2
import numpy as np

from algorithms_limb import refine_lunar_limb, refine_lunar_limb_elliptical


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


if __name__ == "__main__":
    unittest.main()
