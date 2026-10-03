import contextlib
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import algorithms_independent as ai
from algorithms_circle import detect_circle_phd2_enhanced
from algorithms_limb import _annular_edge_kernel
import pipeline


def scene(center=(310, 285), axes=(170, 165), angle=23, size=600, dtype=np.uint8):
    image = np.full((size, size), 5, dtype)
    cv2.ellipse(image, center, axes, angle, 0, 360, 210 if dtype == np.uint8 else 40000, -1, cv2.LINE_AA)
    return image


class IndependentGeometryTests(unittest.TestCase):
    def test_radius_window_rejects_old_ransac_leak(self):
        image = np.zeros((240, 240), np.uint8)
        cv2.circle(image, (120, 120), 100, 200, -1)
        result = detect_circle_phd2_enhanced(image, 55, 75, 50, 30)[0]
        self.assertTrue(result is None or 55 <= result[2] <= 75)

    def test_previous_frame_is_explicitly_rejected(self):
        with self.assertRaises(ValueError):
            detect_circle_phd2_enhanced(scene(), 150, 190, 50, 30, prev_circle=(10, 10, 170))

    def test_entrypoint_has_no_reference_or_previous_frame_argument(self):
        self.assertEqual(list(inspect.signature(ai.analyze_lunar_frame).parameters),
                         ['image', 'hough_params', 'strong_denoise'])

    def test_free_radius_recovers_full_limb_and_flags_short_arc_ambiguity(self):
        image = np.zeros((800, 800), np.uint8)
        cv2.circle(image, (410, 391), 250, 210, -1, cv2.LINE_AA)
        full = ai.fit_lunar_geometry(image, (407, 395, 246), (235, 265))
        self.assertIsNotNone(full)
        self.assertLess(np.linalg.norm(np.asarray(full.center) - (410, 391)), 1.)
        self.assertGreater(full.radius, 249)
        self.assertEqual(full.reasons, [])
        image[:, :560] = 0
        fit = ai.fit_lunar_geometry(image, (407, 395, 246), (235, 265))
        self.assertIsNotNone(fit)
        # A short arc no longer silently assumes a circular shape. Its precise
        # center is unidentifiable here; test free scale AND explicit rejection.
        self.assertGreater(fit.radius, 249)
        self.assertTrue(fit.reasons, 'A short arc still has unknown-shape ambiguity.')
        self.assertGreater(fit.shape_sensitivity, 1.)

    def test_ellipse_orientation_and_center_are_free(self):
        fit = ai.fit_lunar_geometry(scene(), (306, 289, 168), (150, 190))
        self.assertIsNotNone(fit)
        self.assertEqual(fit.model, 'unified-ellipse')
        self.assertLess(np.linalg.norm(np.asarray(fit.center) - (310, 285)), 1.)
        self.assertTrue(all(150 <= axis <= 190 for axis in fit.axes))
        self.assertEqual(fit.reasons, [])

    def test_translation_and_radius_changes_are_independent(self):
        for center, axes in [((210, 200), (155, 152)), ((380, 360), (185, 178))]:
            fit = ai.fit_lunar_geometry(scene(center, axes), (*center, np.mean(axes)), (145, 195))
            self.assertIsNotNone(fit)
            self.assertLess(np.linalg.norm(np.asarray(fit.center) - center), 1.)
            self.assertLess(abs(fit.radius - np.sqrt(axes[0] * axes[1])), 1.)

    def test_out_of_bounds_seed_is_not_clamped_into_a_result(self):
        self.assertIsNone(ai.fit_lunar_geometry(scene(), (310, 285, 170), (100, 130)))

    def test_sixteen_bit_analysis_does_not_modify_input(self):
        image = scene(dtype=np.uint16)
        original = image.copy()
        fit = ai.fit_lunar_geometry(image, (310, 285, 168), (150, 190))
        self.assertIsNotNone(fit)
        np.testing.assert_array_equal(image, original)
        gray, _ = ai._local_analysis(image, (310, 285, 168), (150, 190))
        self.assertEqual(gray.dtype, np.float32)

    def test_flat_frame_is_not_given_a_position(self):
        result = ai.analyze_lunar_frame(np.full((300, 300), 100, np.uint8), (60, 100, 50, 30))
        self.assertIsNone(result.geometry)

    def test_failed_coarse_detection_can_be_independently_rescued(self):
        image = scene()
        with patch.object(ai, 'detect_circle_phd2_enhanced', return_value=(None, None, 0, 'failed', 'dark')):
            analysis = ai.analyze_lunar_frame(image, (150, 190, 50, 25))
        self.assertIsNotNone(analysis.geometry)
        self.assertLess(np.linalg.norm(np.array(analysis.geometry.center) - (310, 285)), 1.)
        self.assertIn('环积分', analysis.method)

    def test_failed_block_fit_cannot_be_silently_ignored(self):
        original = ai._fit
        call_count = 0
        def fit(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Two seed fits, unified fit, final fit and two prior-sensitivity
            # checks precede the leave-block-out checks.
            return None if call_count == 7 else original(*args, **kwargs)
        with patch.object(ai, '_fit', side_effect=fit):
            result = ai.fit_lunar_geometry(scene(), (310, 285, 168), (150, 190))
        self.assertIsNotNone(result)
        self.assertTrue(any('未收敛' in reason for reason in result.reasons))

    def test_fft_cache_matches_zero_padded_spatial_correlation(self):
        rng = np.random.default_rng(32)
        image = rng.normal(size=(180, 230)).astype(np.float32)
        kernel = _annular_edge_kernel(40., 1.4)
        fh, fw = [cv2.getOptimalDFTSize(n + kernel.shape[0] - 1) for n in image.shape]
        spectrum, offset = ai._kernel_spectrum(40., 1.4, fh, fw)
        self.assertFalse(spectrum.flags.writeable)
        padded = np.zeros((fh, fw), np.float32)
        padded[:180, :230] = image
        actual = cv2.idft(cv2.mulSpectrums(cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT), spectrum, 0),
                         flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)[offset:offset+180, offset:offset+230]
        expected = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)

    def test_single_reversed_and_parallel_are_identical(self):
        images = {str(i): scene(center, axes, angle) for i, (center, axes, angle) in enumerate([
            ((220, 240), (160, 156), 10), ((370, 350), (180, 175), 66), ((300, 290), (169, 167), 130)])}
        def analyze(order, workers):
            records = {}
            with patch.object(pipeline, 'imread_unicode', side_effect=lambda path, *args: images[path].copy()):
                for path, loaded, error in pipeline.iter_frame_analyses(order, (145, 195, 50, 25), workers):
                    self.assertIsNone(error)
                    self.assertIsNotNone(loaded[1].geometry)
                    records[path] = asdict(loaded[1].geometry)
            return records
        baseline = analyze(list(images), 1)
        self.assertEqual(baseline, analyze(list(reversed(images)), 2))
        self.assertEqual(baseline, analyze(['1', '2', '0'], 4))
        for name in images:
            ai._kernel_spectrum.cache_clear()
            self.assertEqual(baseline[name], analyze([name], 1)[name])

    def test_annular_noise_does_not_supply_geometry(self):
        rng = np.random.default_rng(77)
        image = rng.normal(100, 2, (400, 400)).astype(np.float32)
        self.assertEqual(ai.annular_hypotheses(image, (105, 135)), [])

    def test_annular_search_returns_variable_radius_hypotheses(self):
        image = np.zeros((500, 500), np.uint8)
        cv2.circle(image, (252, 244), 150, 100, -1)
        seeds = ai.annular_hypotheses(image, (140, 160))
        self.assertTrue(seeds)
        self.assertTrue(all(140 <= seed[2] <= 160 for seed in seeds))
        fit = ai.fit_lunar_geometry(image, seeds[0], (140, 160))
        self.assertIsNotNone(fit)
        self.assertLess(np.linalg.norm(np.array(fit.center) - (252, 244)), 1.)

    def test_significant_conflicting_colour_is_not_a_hypothesis(self):
        image = np.full((500, 500, 3), 100, np.float32)
        disk = np.zeros((500, 500), np.uint8)
        cv2.circle(disk, (220, 244), 120, 1, -1)
        image += disk[:, :, None] * 60
        colour = np.zeros((500, 500), np.uint8)
        cv2.circle(colour, (285, 244), 120, 1, -1)
        image[:, :, 2] += colour * 10
        image[:, :, 1] -= colour * (10 * .299 / .587)
        self.assertEqual(ai.annular_hypotheses(image, (115, 125)), [])


class PipelineSafetyTests(unittest.TestCase):
    def test_reports_skipped_frames_and_continues_with_uint16_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / 'input', root / 'output'
            source.mkdir()
            cv2.imwrite(str(source / 'ref.tif'), scene(dtype=np.uint16))
            cv2.imwrite(str(source / 'target.tif'), scene((300, 300), dtype=np.uint16))
            cv2.imwrite(str(source / 'blank.tif'), np.zeros((600, 600), np.uint16))
            (source / 'broken.tif').write_bytes(b'invalid image')
            (source / '._ref.tif').write_bytes(b'macOS sidecar')
            with patch.object(pipeline, 'log'):
                report = pipeline.align_moon_images_incremental(str(source), str(output), (150, 190, 50, 25),
                                                               reference_image_path=str(source / 'ref.tif'), workers=2)
            self.assertIsNotNone(report)
            records = {r['file']: r for r in report['frames']}
            self.assertEqual(records['blank.tif']['status'], 'skipped')
            self.assertEqual(records['broken.tif']['status'], 'failed')
            self.assertEqual(records['target.tif']['status'], 'saved')
            self.assertNotIn('._ref.tif', records)
            self.assertFalse((output / 'aligned_blank.tif').exists())
            self.assertEqual(cv2.imread(str(output / 'aligned_target.tif'), -1).dtype, np.uint16)
            disk_report = json.loads((output / 'alignment-report.json').read_text())
            self.assertEqual(len(disk_report['frames']), 4)
            self.assertIn('blank.tif', (output / 'alignment-review.txt').read_text())
            # Refuse silently replacing old results.
            before = (output / 'aligned_target.tif').read_bytes()
            with patch.object(pipeline, 'log'):
                again = pipeline.align_moon_images_incremental(str(source), str(output), (150, 190, 50, 25),
                                                              reference_image_path=str(source / 'ref.tif'))
            self.assertIsNone(again)
            self.assertEqual((output / 'aligned_target.tif').read_bytes(), before)

    def test_worker_failure_does_not_cancel_remaining_frames(self):
        with patch.object(pipeline, '_load_and_analyze', side_effect=[ValueError('bad frame'), ('image', 'ok')]):
            results = list(pipeline.iter_frame_analyses(['bad', 'good'], (10, 30, 50, 20), 1))
        self.assertIsInstance(results[0][2], ValueError)
        self.assertEqual(results[1][1], ('image', 'ok'))

    def test_global_opencv_thread_setting_is_unchanged(self):
        before = cv2.getNumThreads()
        self.assertIn(pipeline.choose_analysis_workers(4_000_000), (1, 2))
        self.assertEqual(cv2.getNumThreads(), before)

    def test_invalid_worker_count_is_rejected(self):
        for requested in (0, 5, True, 1.5):
            with self.assertRaises(ValueError):
                pipeline.choose_analysis_workers(1_000_000, requested)


if __name__ == '__main__':
    unittest.main()
