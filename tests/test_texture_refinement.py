"""Known-transform tests; no assumptions about adjacent frame positions."""
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import algorithms_refine as ar
import pipeline
from algorithms_independent import Analysis, Geometry


def scene(seed=59):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.normal(100, 40, (640, 640)).astype(np.float32), (0, 0), 1.5)
    y, x = np.ogrid[:640, :640]
    image[(x-320)**2 + (y-320)**2 > 260**2] = 0
    return image


def move(image, dx=2.35, dy=-1.7):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (640, 640))


class TextureRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = scene()

    def refine(self, target, reference=None, **kwargs):
        info = {}
        result = ar.refine_alignment_multi_roi(
            self.ref if reference is None else reference, target, 320, 320, 260,
            roi_size=64, base_shift=(0., 0.), diagnostics=info, **kwargs)
        return result[0][:, 2], info

    def assert_accepted_shift(self, target, displacement=(2.35, -1.7), tolerance=.15, **kwargs):
        correction, info = self.refine(target, **kwargs)
        self.assertTrue(info['accepted'], info)
        self.assertLess(np.linalg.norm(correction + displacement), tolerance, info)
        return correction, info

    def test_integer_and_subpixel_correction_have_inverse_sign(self):
        for displacement in [(3., -2.), (-2., 3.), (2.35, -1.7), (0., 0.)]:
            with self.subTest(displacement=displacement):
                self.assert_accepted_shift(move(self.ref, *displacement), displacement)

    def test_global_gain_and_offset(self):
        baseline, _ = self.assert_accepted_shift(move(self.ref))
        for gain, offset in [(.03, 37.), (.3, 32.), (1.7, -15.)]:
            actual, _ = self.assert_accepted_shift(move(self.ref) * gain + offset)
            np.testing.assert_allclose(actual, baseline, atol=.003)

    def test_moderate_blur_noise_and_slow_illumination_gradient(self):
        shifted = move(self.ref)
        x = np.arange(640, dtype=np.float32)[None, :]
        rng = np.random.default_rng(33)
        for target in [cv2.GaussianBlur(shifted, (0, 0), 2.),
                       shifted + rng.normal(0, .3, shifted.shape).astype(np.float32),
                       shifted * (.25 + .75*x/640) + 10]:
            self.assert_accepted_shift(target)

    def test_no_shared_texture_and_single_sided_texture_are_rejected(self):
        shifted = move(self.ref)
        x = np.arange(640)[None, :]
        for target in [scene(101), np.full_like(self.ref, 100), np.zeros_like(self.ref),
                       np.where(x < 320, shifted, 10).astype(np.float32)]:
            correction, info = self.refine(target)
            self.assertFalse(info['accepted'], info)
            np.testing.assert_array_equal(correction, [0, 0])

    def test_nonrigid_warp_is_not_forced_into_translation(self):
        x, y = np.meshgrid(np.arange(640, dtype=np.float32), np.arange(640, dtype=np.float32))
        warped = cv2.remap(self.ref, x - 3*np.tanh((x-320)/80), y, cv2.INTER_LINEAR)
        correction, info = self.refine(warped)
        self.assertFalse(info['accepted'], info)
        self.assertIn('不一致', info['reason'])
        np.testing.assert_array_equal(correction, [0, 0])

    def test_large_shift_and_low_snr_are_rejected(self):
        rng = np.random.default_rng(12)
        for target in [move(self.ref, 8, 0), rng.normal(100, 40, self.ref.shape).astype(np.float32)]:
            correction, info = self.refine(target)
            self.assertFalse(info['accepted'], info)
            np.testing.assert_array_equal(correction, [0, 0])

    def test_no_elapsed_time_branch_and_no_order_state(self):
        inputs = [move(self.ref, *d) for d in [(3, -2), (-2.35, 1.7), (0, 0)]]
        baseline = [self.refine(t, time_budget_sec=0) for t in inputs]
        with ThreadPoolExecutor(max_workers=3) as pool:
            parallel = list(pool.map(self.refine, reversed(inputs)))
        for (a, ai), (b, bi) in zip(baseline, reversed(parallel)):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(ai, bi)

    def test_uint16_and_input_immutability(self):
        reference = np.round(self.ref * 300).astype(np.uint16)
        target = move(reference)
        originals = reference.copy(), target.copy()
        self.assert_accepted_shift(target, reference=reference)
        np.testing.assert_array_equal(reference, originals[0])
        np.testing.assert_array_equal(target, originals[1])
        features = ar._prepare(reference)
        self.assertEqual(features[0].dtype, np.float32)
        self.assertLess(features[0].min(), 0)
        self.assertGreater(features[0].max(), 0)

    def test_bad_phase_proposal_cannot_override_good_template(self):
        target = move(self.ref)
        baseline, _ = self.refine(target, use_phasecorr=False)
        for phase in [((.95, -.95), .99), ((float('nan'), 0), .99)]:
            with patch.object(ar.cv2, 'phaseCorrelate', return_value=phase):
                actual, info = self.refine(target)
            self.assertTrue(info['accepted'])
            np.testing.assert_allclose(actual, baseline, atol=.001)

    def test_pipeline_composes_correction_with_coarse_shift_once(self):
        target = move(self.ref, 32.35, -21.7)
        info = {}
        total = pipeline._refine_translation(self.ref, target, (320, 320), 260,
                                             np.array([-30., 20.]), diagnostics=info)
        self.assertTrue(info['accepted'], info)
        np.testing.assert_allclose(total, [-32.35, 21.7], atol=.15)

    def test_pipeline_retains_nonzero_coarse_shift_on_refusal(self):
        coarse = np.array([13., -17.])
        info = {}
        actual = pipeline._refine_translation(self.ref, np.zeros_like(self.ref), (320, 320), 260,
                                              coarse, diagnostics=info)
        np.testing.assert_array_equal(actual, coarse)
        self.assertFalse(info['accepted'])

    def test_invalid_inputs_fail_safely(self):
        correction, info = self.refine(self.ref[:300])
        np.testing.assert_array_equal(correction, [0, 0])
        self.assertFalse(info['accepted'])
        with self.assertRaises(ValueError):
            self.refine(np.full_like(self.ref, np.nan))

    def test_ui_has_one_switch_and_no_dead_method_selector(self):
        import ui
        source = inspect.getsource(ui.UniversalLunarAlignApp)
        self.assertNotIn('self.method_combo', source)
        self.assertNotIn('self.alignment_method', source)
        self.assertIn('启用实验性月面纹理微调', source)

    def test_batch_logs_refusal_keeps_valid_frame_and_preserves_output_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'input', root / 'output'
            source.mkdir()
            reference = np.round(self.ref * 300).astype(np.uint16)
            target = move(reference, 22.35, -21.7)
            cv2.imwrite(str(source / 'ref.tif'), reference)
            cv2.imwrite(str(source / 'target.tif'), target)
            cv2.imwrite(str(source / 'flat.tif'), np.full_like(reference, 10000))
            fit = Geometry((320., 320.), 260., (260., 260.), 0., 'circle', .2, .8, 200, .2)
            target_fit = Geometry((340., 300.), 260., (260., 260.), 0., 'circle', .2, .8, 200, .2)
            # Known geometry isolates the optional refinement from limb tests.
            with patch.object(pipeline, '_reference', return_value=(str(source / 'ref.tif'), reference, Analysis(fit, 'test', {}))), \
                 patch.object(pipeline, 'analyze_lunar_frame', return_value=Analysis(target_fit, 'test', {})), \
                 patch.object(pipeline, 'log') as log:
                report = pipeline.align_moon_images_incremental(
                    str(source), str(output), (240, 280, 50, 30), use_advanced_alignment=True,
                    alignment_method='feature', workers=1)
            self.assertIsNotNone(report)
            records = {r['file']: r for r in report['frames']}
            self.assertTrue(records['target.tif']['texture_refinement']['accepted'])
            np.testing.assert_allclose(records['target.tif']['shift'], [-22.35, 21.7], atol=.15)
            self.assertEqual(records['flat.tif']['status'], 'saved')
            self.assertFalse(records['flat.tif']['texture_refinement']['accepted'])
            np.testing.assert_array_equal(records['flat.tif']['shift'], [-20, 20])
            written = cv2.imread(str(output / 'aligned_target.tif'), -1)
            self.assertEqual(written.dtype, np.uint16)
            expected = cv2.warpAffine(target, np.float32([[1, 0, records['target.tif']['shift'][0]],
                                                        [0, 1, records['target.tif']['shift'][1]]]),
                                      (640, 640), flags=cv2.INTER_LANCZOS4)
            np.testing.assert_array_equal(written, expected)
            disk_report = json.loads((output / 'alignment-report.json').read_text())
            self.assertEqual(disk_report['texture_algorithm'], 'texture-consensus-v2')
            self.assertTrue(any('保留月缘结果' in str(call) for call in log.call_args_list))
            self.assertTrue(any('旧实验选项' in str(call) for call in log.call_args_list))


if __name__ == '__main__':
    unittest.main()
