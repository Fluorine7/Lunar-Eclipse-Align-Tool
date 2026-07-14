"""High-precision lunar limb fitting for translation-only eclipse alignment.

Hough circles are used only as an initial guess.  The final center is fitted
from radial edge samples close to the expected physical lunar limb, which is
far less sensitive to eclipse illumination and exposure changes than lunar
surface texture matching.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def refine_lunar_limb(
    image: np.ndarray,
    initial_circle: tuple[float, float, float] | np.ndarray,
    *,
    fixed_radius: float | None = None,
    samples: int = 720,
    search_px: int = 24,
) -> tuple[tuple[float, float, float], float, float, int] | None:
    """Fit the outer lunar limb and return ``(circle, rms, coverage, count)``.

    ``fixed_radius`` should be supplied for non-reference frames from the
    reference-frame fit. It prevents a short illuminated crescent from trading
    radius error for a drifting center.
    """
    if image is None or image.size == 0:
        return None
    cx, cy, guessed_radius = map(float, initial_circle[:3])
    radius = float(fixed_radius if fixed_radius is not None else guessed_radius)
    if radius <= 8:
        return None

    gray = _normalized_gray(image)
    # Blur only enough to suppress sensor noise; do not blur the physical limb.
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    height, width = gray.shape
    offsets = np.arange(-search_px, search_px + 1, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float32)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    # Sample narrow radial profiles. This avoids creating several full-size
    # gradient images for 16-bit camera TIFFs.
    radii = radius + offsets[None, :]
    map_x = (cx + cos_a[:, None] * radii).astype(np.float32)
    map_y = (cy + sin_a[:, None] * radii).astype(np.float32)
    profiles = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = (map_x >= 1) & (map_x < width - 2) & (map_y >= 1) & (map_y < height - 2)

    # Going from inside to outside, the lunar limb is normally a negative
    # derivative. Use that sign to reject bright halos outside the Moon.
    derivative = -np.gradient(profiles, axis=1)
    distance_weight = np.exp(-0.5 * (offsets / max(5.0, search_px * 0.45)) ** 2)
    score = derivative * distance_weight[None, :]
    score[~valid] = -np.inf
    peak_idx = np.argmax(score, axis=1)
    strength = score[np.arange(samples), peak_idx]

    finite_strength = strength[np.isfinite(strength) & (strength > 0)]
    if finite_strength.size < 24:
        return None
    # Keep only the best radial edges, but retain enough arc coverage for
    # partially eclipsed images. The robust fit below rejects remaining noise.
    threshold = max(float(np.percentile(finite_strength, 40)), 2.0)
    keep = np.isfinite(strength) & (strength >= threshold)
    if int(np.count_nonzero(keep)) < 24:
        return None

    peak_offset = offsets[peak_idx].astype(np.float64)
    # Quadratic interpolation of the derivative maximum gives sub-pixel radial
    # positions without requiring a full-resolution Hough transform.
    for i in np.flatnonzero(keep):
        j = int(peak_idx[i])
        if 0 < j < len(offsets) - 1:
            left, center, right = score[i, j - 1], score[i, j], score[i, j + 1]
            denom = left - 2.0 * center + right
            if np.isfinite(denom) and abs(denom) > 1e-6:
                peak_offset[i] += float(np.clip(0.5 * (left - right) / denom, -0.75, 0.75))

    edge_radii = radius + peak_offset[keep]
    points = np.column_stack((
        cx + cos_a[keep] * edge_radii,
        cy + sin_a[keep] * edge_radii,
    )).astype(np.float64)
    weights = strength[keep].astype(np.float64)
    fitted = _robust_circle_fit(points, weights, (cx, cy, radius), fixed_radius is not None)
    if fitted is None:
        return None
    fit_cx, fit_cy, fit_r, residuals = fitted

    # A frame with all accepted samples in one tiny arc is not sufficiently
    # constrained, even if its local residual happens to be low.
    selected_angles = angles[keep]
    bins = np.zeros(72, dtype=bool)
    bins[(selected_angles / (2.0 * np.pi) * len(bins)).astype(int) % len(bins)] = True
    coverage = float(np.count_nonzero(bins) / len(bins))
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    if coverage < 0.10 or rms > 3.5:
        return None
    return (float(fit_cx), float(fit_cy), float(fit_r)), rms, coverage, int(points.shape[0])


def _normalized_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    values = gray.astype(np.float32)
    low, high = np.percentile(values, (1.0, 99.8))
    if high <= low:
        return np.zeros(gray.shape, np.uint8)
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _robust_circle_fit(
    points: np.ndarray,
    base_weights: np.ndarray,
    initial: tuple[float, float, float],
    radius_fixed: bool,
) -> tuple[float, float, float, np.ndarray] | None:
    cx, cy, radius = initial
    weights = np.maximum(base_weights / (np.median(base_weights) + 1e-6), 0.05)
    for _ in range(12):
        dx, dy = points[:, 0] - cx, points[:, 1] - cy
        distance = np.hypot(dx, dy)
        if np.any(distance < 1e-6):
            return None
        residual = distance - radius
        # Huber IRLS: deterministic and stable unlike the old random RANSAC.
        scale = max(0.5, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        huber = np.minimum(1.0, 2.5 * scale / (np.abs(residual) + 1e-6))
        w = weights * huber
        if radius_fixed:
            jacobian = np.column_stack((-dx / distance, -dy / distance))
        else:
            jacobian = np.column_stack((-dx / distance, -dy / distance, -np.ones_like(distance)))
        lhs = jacobian.T @ (w[:, None] * jacobian)
        rhs = jacobian.T @ (w * residual)
        try:
            delta = -np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return None
        cx += float(delta[0])
        cy += float(delta[1])
        if not radius_fixed:
            radius += float(delta[2])
        if radius <= 8:
            return None
        if float(np.max(np.abs(delta))) < 0.005:
            break
    residual = np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius
    return cx, cy, radius, residual
