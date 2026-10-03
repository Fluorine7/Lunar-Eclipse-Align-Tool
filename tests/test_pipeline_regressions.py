"""Regression checks for frame identity, fallback routing and determinism."""
import contextlib
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import algorithms_circle
import pipeline
from algorithms_independent import Analysis, Geometry
from algorithms_limb import detect_faint_lunar_disk


class PipelineRegressions(unittest.TestCase):
    def run_pipeline(self, collision=False, rescue=True, ellipse_target=False):
        reference = np.full((64, 64), 20, np.uint8)
        target = np.full((64, 64), 80, np.uint8)
        filename = "same.tif" if collision else "target.tif"
        with contextlib.ExitStack() as stack:
            def mock(name, **kwargs):
                return stack.enter_context(patch.object(pipeline, name, **kwargs))
            stack.enter_context(patch.object(pipeline.os.path, "exists", return_value=False))
            stack.enter_context(patch.object(pipeline.os.path, "isfile", return_value=True))
            stack.enter_context(patch.object(pipeline.os, "listdir", return_value=[filename]))
            mock("ensure_dir_exists", return_value=True)
            mock("log")
            fit = Geometry((32., 32.), 20., (20., 20.), 0., 'circle', .2, .8, 200, .2)
            reference_result = Analysis(fit, 'reference', {})
            mock('_reference', return_value=('/outside/same.tif', reference, reference_result))
            mock('choose_analysis_workers', return_value=1)
            mock('_write_reports')
            mock('imread_unicode', return_value=target)
            target_fit = Geometry((32., 32.), 20., (20., 20.), 0., 'circle', .2, .8, 200, .2,
                                  ['模型圆心分歧'] if ellipse_target else []) if rescue else None
            detect = mock('analyze_lunar_frame', return_value=Analysis(target_fit, 'independent', {}, '无可靠月缘'))
            saved = []
            mock("imwrite_with_exif", side_effect=lambda src, dst, image: saved.append(image.copy()) or True)
            pipeline.align_moon_images_incremental(
                "/input", "/output", (10, 30, 50, 20),
                reference_image_path="/outside/same.tif",
            )
            return detect.call_count, detect.call_count, saved, target

    def test_external_reference_with_same_name_does_not_replace_target(self):
        detected, rescued, saved, target = self.run_pipeline(collision=True)
        self.assertEqual(detected, 1)
        self.assertEqual(rescued, 1)
        self.assertEqual(len(saved), 1)
        np.testing.assert_array_equal(saved[0], target)

    def test_independently_validated_frame_is_written(self):
        _, rescued, saved, _ = self.run_pipeline()
        self.assertEqual(rescued, 1)
        self.assertEqual(len(saved), 1)

    def test_failed_rescue_is_not_written(self):
        _, rescued, saved, _ = self.run_pipeline(rescue=False)
        self.assertEqual(rescued, 1)
        self.assertEqual(saved, [])

    def test_uncertain_ellipse_is_not_written_as_success(self):
        _, checked, saved, target = self.run_pipeline(ellipse_target=True)
        self.assertEqual(checked, 1)
        self.assertEqual(saved, [])

    def test_detection_does_not_depend_on_elapsed_time(self):
        image = np.zeros((240, 240), np.uint8)
        cv2.circle(image, (122, 117), 65, 200, -1)
        with patch.object(time, "time", return_value=0):
            fast = algorithms_circle.detect_circle_phd2_enhanced(image, 55, 75, 50, 20)
        with patch.object(time, "time", side_effect=iter(range(0, 10000, 10)).__next__):
            slow = algorithms_circle.detect_circle_phd2_enhanced(image, 55, 75, 50, 20)
        self.assertIsNotNone(fast[0])
        np.testing.assert_array_equal(fast[0], slow[0])
        self.assertEqual(fast[3], slow[3])

    def test_neutral_rgb_has_same_annular_center_as_grayscale(self):
        image = np.zeros((500, 500), np.uint8)
        cv2.circle(image, (252, 244), 150, 100, -1)
        gray = detect_faint_lunar_disk(image, 150)
        rgb = detect_faint_lunar_disk(np.repeat(image[:, :, None], 3, axis=2), 150)
        self.assertIsNotNone(gray)
        self.assertIsNotNone(rgb)
        np.testing.assert_allclose(gray[0], rgb[0], atol=0.01)

    def test_neutral_rgb_noise_is_rejected(self):
        rng = np.random.default_rng(18)
        for _ in range(5):
            gray = rng.normal(100, 2, (500, 500)).astype(np.float32)
            image = np.repeat(gray[:, :, None], 3, axis=2)
            self.assertIsNone(detect_faint_lunar_disk(image, 150))

    def test_significant_conflicting_colour_peak_is_rejected(self):
        image = np.full((500, 500, 3), 100, np.float32)
        disk = np.zeros((500, 500), np.uint8)
        cv2.circle(disk, (240, 244), 120, 1, -1)
        image += disk[:, :, None] * 60
        colour = np.zeros((500, 500), np.uint8)
        cv2.circle(colour, (270, 244), 120, 1, -1)
        # Preserve luminance while adding an independently displaced colour disk.
        image[:, :, 2] += colour * 10
        image[:, :, 1] -= colour * (10 * 0.299 / 0.587)
        self.assertIsNone(detect_faint_lunar_disk(image, 120))


if __name__ == "__main__":
    unittest.main()
