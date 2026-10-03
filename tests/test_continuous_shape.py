"""Circle/ellipse continuity without temporal or fixed-orientation priors."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import algorithms_independent as ai


def edge_points(center, a, b, theta, count=360):
    t = np.linspace(0, 2*np.pi, count, endpoint=False)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.column_stack((a*np.cos(t), b*np.sin(t))) @ rotation.T + center


def smooth_disk(delta, theta=.73, center=(310.3, 285.7)):
    y, x = np.mgrid[:600, :600].astype(np.float32)
    dx, dy = x-center[0], y-center[1]
    u = np.cos(theta)*dx + np.sin(theta)*dy
    v = -np.sin(theta)*dx + np.cos(theta)*dy
    distance = (np.sqrt((u/(170+delta))**2 + (v/(170-delta))**2)-1)*170
    return (5 + 100*(1-np.tanh(distance/1.4))).astype(np.float32)


class ContinuousShapeTests(unittest.TestCase):
    bounds = (150., 190.)

    def fit_points(self, points, initial=None, penalty=ai.SHAPE_REGULARIZATION):
        if initial is None:
            initial = ai._shape_seed((306, 289), (168, 168), 0., self.bounds)
        return ai._fit(points, np.ones(len(points)), initial, self.bounds,
                       (np.array([280., 255.]), np.array([340., 315.])), regularization=penalty)

    def test_exact_shape_encoding_and_axis_order_invariance(self):
        for a, b, theta in [(180, 160, .47), (170, 170, 1.8), (175, 166, 3.1)]:
            p = ai._shape_seed((310, 285), (a, b), theta, self.bounds)
            swapped = ai._shape_seed((310, 285), (b, a), theta+np.pi/2, self.bounds)
            np.testing.assert_allclose(p, swapped, atol=1e-12)
            residual = ai._residual(p, edge_points((310, 285), a, b, theta), self.bounds)
            self.assertLess(np.max(np.abs(residual)), 1e-10)

    def test_both_axes_are_bounded_in_any_orientation(self):
        rng = np.random.default_rng(721)
        for m, u, v in rng.uniform(-12, 12, (100, 3)):
            mean, s1, s2, a, b = ai._shape_components([0, 0, m, u, v], self.bounds)
            self.assertTrue(150 <= b <= a <= 190)
            eigenvalues = np.linalg.eigvalsh([[mean+s1, s2], [s2, mean-s1]])
            np.testing.assert_allclose(eigenvalues, [b, a], atol=1e-11)

    def test_circle_has_two_identifiable_shape_directions(self):
        points = edge_points((310, 285), 170, 170, 0)
        p = ai._shape_seed((310, 285), (170, 170), 0, self.bounds)
        columns = []
        for axis in range(5):
            step = np.zeros(5)
            step[axis] = 1e-5
            columns.append((ai._residual(p+step, points, self.bounds)-ai._residual(p-step, points, self.bounds))/2e-5)
        jacobian = np.column_stack(columns)
        self.assertTrue(np.isfinite(jacobian).all())
        self.assertEqual(np.linalg.matrix_rank(jacobian), 5)
        self.assertLess(np.linalg.cond(jacobian), 50)

    def test_known_edges_cross_circle_without_center_jump(self):
        center = np.array([310.3, 285.7])
        previous = None
        for delta in np.linspace(-5, 5, 21):
            points = edge_points(center, 170+delta, 170-delta, .73)
            result = self.fit_points(points)
            self.assertIsNotNone(result)
            self.assertLess(np.linalg.norm(result[:2]-center), 1e-5)
            mean, s1, s2, a, b = ai._shape_components(result, self.bounds)
            expected = delta * np.array([np.cos(1.46), np.sin(1.46)])
            np.testing.assert_allclose([s1, s2], expected, atol=.025)
            if previous is not None:
                self.assertLess(np.linalg.norm(result[:2]-previous), 1e-5)
            previous = result[:2]

    def test_full_image_transition_has_one_model_and_free_direction(self):
        fits = []
        for delta in [-5, -2, -.5, 0, .5, 2, 5]:
            fit = ai.fit_lunar_geometry(smooth_disk(delta), (306, 289, 168), self.bounds)
            self.assertIsNotNone(fit)
            self.assertEqual(fit.model, 'unified-ellipse')
            self.assertEqual(fit.reasons, [])
            self.assertLess(np.linalg.norm(np.array(fit.center)-(310.3, 285.7)), .05)
            fits.append(fit)
        self.assertLess(np.linalg.norm(fits[3].shape_vector), .0001)
        self.assertGreater(fits[-1].shape_vector[1], .02)
        self.assertLess(fits[0].shape_vector[1], -.02)

    def test_shape_prior_does_not_prefer_horizontal_orientation(self):
        center = np.array([310., 285.])
        rng = np.random.default_rng(33)
        points = edge_points(center, 174, 166, .2) + rng.normal(0, .15, (360, 2))
        p = self.fit_points(points)
        angle = .71
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = (points-center) @ rotation.T + center
        q = self.fit_points(rotated)
        np.testing.assert_allclose(q[:2], rotation @ (p[:2]-center)+center, atol=1e-4)
        pshape = ai._shape_components(p, self.bounds)
        qshape = ai._shape_components(q, self.bounds)
        np.testing.assert_allclose(pshape[3:], qshape[3:], atol=1e-4)

    def test_shape_sensitivity_failure_is_not_silent(self):
        original = ai._fit
        count = 0
        def fit(*args, **kwargs):
            nonlocal count
            count += 1
            return None if count == 5 else original(*args, **kwargs)
        with patch.object(ai, '_fit', side_effect=fit):
            result = ai.fit_lunar_geometry(smooth_disk(3), (306, 289, 168), self.bounds)
        self.assertIsNotNone(result)
        self.assertIn('形状约束敏感性检查未收敛', result.reasons)

    def test_overflattened_shape_is_not_mistaken_for_a_valid_circle(self):
        image = np.full((600, 600), 5, np.uint8)
        cv2.ellipse(image, (310, 285), (180, 160), 41, 0, 360, 210, -1, cv2.LINE_AA)
        fit = ai.fit_lunar_geometry(image, (306, 289, 170), self.bounds)
        self.assertIsNotNone(fit)
        self.assertIn('椭圆轴比超限', fit.reasons)


if __name__ == '__main__':
    unittest.main()
