"""High-precision lunar limb fitting for translation-only eclipse alignment.

Hough circles are used only as an initial guess.  The final center is fitted
from radial edge samples close to the expected physical lunar limb, which is
far less sensitive to eclipse illumination and exposure changes than lunar
surface texture matching.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np


def detect_faint_lunar_disk(
    image: np.ndarray,
    fixed_radius: float,
    *,
    max_side: int = 640,
) -> tuple[tuple[float, float, float], float, str] | None:
    """Locate an extremely faint disk with a fixed-radius annular matched filter.

    This is a low-contrast fallback, not a display operation.  It works on a
    temporary analysis image: remove a smooth sky background, denoise before
    contrast amplification, then integrate the signed inner-to-outer contrast
    around the whole expected limb.  The original image is never modified.

    Colour input is checked independently in luminance and red-minus-blue.  A
    candidate is accepted only when both channels agree, which rejects most
    noise peaks and sky gradients.  Monochrome input uses a stricter peak gate.
    """
    if image is None or image.size == 0 or not np.isfinite(fixed_radius):
        return None
    height, width = image.shape[:2]
    radius = float(fixed_radius)
    if radius <= 8 or 2.0 * radius >= min(height, width):
        return None

    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image.copy()
    small = small.astype(np.float32)

    if small.ndim == 3:
        blue, green, red = cv2.split(small[:, :, :3])
        channels = [
            ("亮度", 0.114 * blue + 0.587 * green + 0.299 * red),
            ("红蓝色差", red - blue),
        ]
    else:
        channels = [("灰度", small)]

    scaled_radius = radius * scale
    band = max(1.4, scaled_radius * 0.012)
    kernel = _annular_edge_kernel(scaled_radius, band)
    candidates = []
    for channel_name, channel in channels:
        residual = _remove_smooth_background(channel)
        # Denoise before clipping/stretching.  Keeping the data in float avoids
        # an extra 8-bit quantisation step on already faint edges.
        residual = cv2.GaussianBlur(residual, (0, 0), max(0.8, band * 0.45))
        low, high = np.percentile(residual, (1.0, 99.0))
        if not np.isfinite(low + high) or high - low < 1e-4:
            continue
        residual = np.clip(residual, low, high)

        response = cv2.filter2D(
            residual, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT,
        )
        margin = int(math.ceil(scaled_radius + 4.0 * band)) + 1
        if response.shape[0] <= 2 * margin or response.shape[1] <= 2 * margin:
            continue
        valid = response[margin:-margin, margin:-margin]
        _, peak, _, location = cv2.minMaxLoc(valid)
        px = int(location[0] + margin)
        py = int(location[1] + margin)
        sub_x = _quadratic_peak_offset(response[py, px - 1:px + 2])
        sub_y = _quadratic_peak_offset(response[py - 1:py + 2, px])

        median = float(np.median(valid))
        mad = 1.4826 * float(np.median(np.abs(valid - median)))
        peak_z = (float(peak) - median) / max(mad, 1e-6)
        candidates.append((channel_name, px + sub_x, py + sub_y, peak_z))

    if not candidates:
        return None
    # Colour is supporting evidence only when it contains a significant disk.
    # For neutral RGB data, require the stricter luminance-only gate below.
    # Never substitute a colour-only peak when luminance has failed.
    if len(channels) > 1:
        luminance = next((c for c in candidates if c[0] == "亮度"), None)
        colour = next((c for c in candidates if c[0] == "红蓝色差"), None)
        if luminance is None:
            return None
        if colour is None or colour[3] < 7.0:
            channels = channels[:1]
            candidates = [luminance]
    if len(channels) > 1:
        if len(candidates) < 2:
            return None
        first, second = candidates[:2]
        disagreement = math.hypot(first[1] - second[1], first[2] - second[2])
        if disagreement > max(3.0, 0.035 * scaled_radius):
            return None
        confidence = min(first[3], second[3])
        if confidence < 7.0:
            return None
        weights = np.maximum([first[3], second[3]], 1.0)
        fit_cx = float(np.average([first[1], second[1]], weights=weights))
        fit_cy = float(np.average([first[2], second[2]], weights=weights))
        detail = f"双通道一致, peakZ={confidence:.1f}"
    else:
        only = candidates[0]
        confidence = only[3]
        if confidence < 10.0:
            return None
        fit_cx, fit_cy = float(only[1]), float(only[2])
        detail = f"单通道, peakZ={confidence:.1f}"

    return (
        (fit_cx / scale, fit_cy / scale, radius),
        float(confidence),
        detail,
    )


@lru_cache(maxsize=8)
def _background_design(height: int, width: int) -> np.ndarray:
    """Immutable coordinate template only; never cache fitted image data."""
    yy, xx = np.mgrid[0:height, 0:width]
    x = (xx.ravel() - .5 * (width - 1)) / max(width, 1)
    y = (yy.ravel() - .5 * (height - 1)) / max(height, 1)
    design = np.column_stack((np.ones(x.size), x, y, x*x, x*y, y*y))
    design.setflags(write=False)
    return design


def _remove_smooth_background(channel: np.ndarray) -> np.ndarray:
    """Robustly subtract a quadratic sky/background surface."""
    height, width = channel.shape
    fit_scale = min(1.0, 240.0 / max(height, width))
    if fit_scale < 1.0:
        fit_image = cv2.resize(
            channel, None, fx=fit_scale, fy=fit_scale, interpolation=cv2.INTER_AREA,
        )
    else:
        fit_image = channel
    fit_h, fit_w = fit_image.shape
    design = _background_design(fit_h, fit_w)
    values = fit_image.reshape(-1).astype(np.float64)
    keep = np.ones(values.size, dtype=bool)
    coefficients = None
    for _ in range(5):
        if int(np.count_nonzero(keep)) < 32:
            return channel.astype(np.float32) - float(np.median(channel))
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        error = values - design @ coefficients
        center = float(np.median(error[keep]))
        scale = 1.4826 * float(np.median(np.abs(error[keep] - center)))
        if scale < 1e-6:
            break
        new_keep = np.abs(error - center) <= 2.5 * scale
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    full_y, full_x = np.ogrid[0:height, 0:width]
    full_x = (full_x - 0.5 * (width - 1)) / max(width, 1)
    full_y = (full_y - 0.5 * (height - 1)) / max(height, 1)
    background = (
        coefficients[0] + coefficients[1] * full_x + coefficients[2] * full_y
        + coefficients[3] * full_x * full_x
        + coefficients[4] * full_x * full_y
        + coefficients[5] * full_y * full_y
    )
    return channel.astype(np.float32) - background.astype(np.float32)


def _annular_edge_kernel(radius: float, band: float) -> np.ndarray:
    margin = int(math.ceil(4.0 * band))
    size = int(2 * math.ceil(radius + margin) + 1)
    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.hypot(xx - size // 2, yy - size // 2)
    inner = np.exp(-0.5 * ((distance - (radius - band)) / band) ** 2)
    outer = np.exp(-0.5 * ((distance - (radius + band)) / band) ** 2)
    kernel = inner - outer
    kernel[np.abs(distance - radius) > 4.0 * band] = 0.0
    kernel -= float(np.mean(kernel))
    kernel /= max(float(np.linalg.norm(kernel)), 1e-6)
    return kernel.astype(np.float32)


def _quadratic_peak_offset(values: np.ndarray) -> float:
    if values.size != 3 or not np.all(np.isfinite(values)):
        return 0.0
    left, center, right = map(float, values)
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))


def prepare_lunar_limb_image(image: np.ndarray) -> np.ndarray:
    """Normalize and lightly denoise one frame once for all limb fits."""
    return cv2.GaussianBlur(_normalized_gray(image), (0, 0), 1.2)


def refine_lunar_limb(
    image: np.ndarray,
    initial_circle: tuple[float, float, float] | np.ndarray,
    *,
    fixed_radius: float | None = None,
    samples: int = 720,
    search_px: int = 24,
    prepared_gray: np.ndarray | None = None,
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

    # Reuse the same normalized analysis image when the caller tries multiple
    # models. This avoids repeated full-frame percentile scans and blurs.
    gray = prepared_gray if prepared_gray is not None else prepare_lunar_limb_image(image)
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


def refine_lunar_limb_elliptical(
    image: np.ndarray,
    initial_circle: tuple[float, float, float] | np.ndarray,
    *,
    samples: int = 720,
    search_px: int = 24,
    prepared_gray: np.ndarray | None = None,
) -> tuple[
    tuple[float, float, float], float, float, int,
    tuple[float, float, float] | None, bool,
] | None:
    """Independently refine one frame with a tightly gated ellipse.

    The ellipse is accepted only when this frame alone contains enough limb
    directions to constrain its center and shape. Short crescents and noisy
    frames fall back to their own robust circle result. No state from another
    frame is accepted or returned as a prior.
    """
    analysis_gray = (
        prepared_gray if prepared_gray is not None else prepare_lunar_limb_image(image)
    )
    circle_result = refine_lunar_limb(
        image, initial_circle, samples=samples, search_px=search_px,
        prepared_gray=analysis_gray,
    )
    if circle_result is None:
        return None
    circle, circle_rms, _, _ = circle_result

    sampled = _sample_limb_points(
        image, circle, samples=samples, search_px=search_px,
        prepared_gray=analysis_gray,
    )
    if sampled is None:
        return (*circle_result, None, False)
    points, strengths, coverage, quadrants = sampled

    candidate = _robust_ellipse_shape(points, strengths)
    candidate_ok = False
    candidate_shape = None
    if candidate is not None:
        ecx, ecy, axis_a, axis_b, candidate_shape, ellipse_rms = candidate
        ratio = min(axis_a, axis_b) / max(axis_a, axis_b)
        radius = float(circle[2])
        candidate_ok = (
            coverage >= 0.42
            and quadrants >= 3
            and ratio >= 0.94
            and 0.90 * radius <= axis_a <= 1.10 * radius
            and 0.90 * radius <= axis_b <= 1.10 * radius
            and math.hypot(ecx - circle[0], ecy - circle[1]) <= max(8.0, 0.06 * radius)
            and ellipse_rms <= 2.0
        )

    if not candidate_ok:
        return (*circle_result, None, False)
    fitted_shape = candidate_shape

    fixed_fit = _fit_center_for_shape(
        points, strengths, (float(circle[0]), float(circle[1])), fitted_shape,
    )
    if fixed_fit is None:
        return (*circle_result, None, False)
    fit_cx, fit_cy, ellipse_rms = fixed_fit
    center_delta = math.hypot(fit_cx - circle[0], fit_cy - circle[1])

    rms_limit = max(1.25, 1.8 * float(circle_rms))
    if ellipse_rms > rms_limit or center_delta > max(8.0, 0.06 * float(circle[2])):
        return (*circle_result, None, False)

    q00, q01, q11 = fitted_shape
    determinant = q00 * q11 - q01 * q01
    if determinant <= 0:
        return (*circle_result, None, False)
    equivalent_radius = determinant ** -0.25
    return (
        (float(fit_cx), float(fit_cy), float(equivalent_radius)),
        float(ellipse_rms), float(coverage), int(points.shape[0]),
        tuple(map(float, fitted_shape)), True,
    )


def _sample_limb_points(
    image: np.ndarray,
    circle: tuple[float, float, float] | np.ndarray,
    *,
    samples: int,
    search_px: int,
    prepared_gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    """Return signed outer-limb samples near ``circle``."""
    cx, cy, radius = map(float, circle[:3])
    gray = prepared_gray if prepared_gray is not None else prepare_lunar_limb_image(image)
    height, width = gray.shape
    offsets = np.arange(-search_px, search_px + 1, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float32)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    radii = radius + offsets[None, :]
    map_x = (cx + cos_a[:, None] * radii).astype(np.float32)
    map_y = (cy + sin_a[:, None] * radii).astype(np.float32)
    profiles = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = (map_x >= 1) & (map_x < width - 2) & (map_y >= 1) & (map_y < height - 2)

    derivative = -np.gradient(profiles, axis=1)
    distance_weight = np.exp(-0.5 * (offsets / max(5.0, search_px * 0.45)) ** 2)
    score = derivative * distance_weight[None, :]
    score[~valid] = -np.inf
    peak_idx = np.argmax(score, axis=1)
    strength = score[np.arange(samples), peak_idx]
    finite = strength[np.isfinite(strength) & (strength > 0)]
    if finite.size < 24:
        return None
    keep = np.isfinite(strength) & (strength >= max(float(np.percentile(finite, 40)), 2.0))
    if int(np.count_nonzero(keep)) < 24:
        return None

    peak_offset = offsets[peak_idx].astype(np.float64)
    for i in np.flatnonzero(keep):
        j = int(peak_idx[i])
        if 0 < j < len(offsets) - 1:
            left, center, right = score[i, j - 1], score[i, j], score[i, j + 1]
            denom = left - 2.0 * center + right
            if np.isfinite(denom) and abs(denom) > 1e-6:
                peak_offset[i] += float(np.clip(0.5 * (left - right) / denom, -0.75, 0.75))

    selected_angles = angles[keep]
    edge_radii = radius + peak_offset[keep]
    points = np.column_stack((
        cx + cos_a[keep] * edge_radii,
        cy + sin_a[keep] * edge_radii,
    )).astype(np.float64)
    bins = np.zeros(72, dtype=bool)
    bins[(selected_angles / (2.0 * np.pi) * len(bins)).astype(int) % len(bins)] = True
    coverage = float(np.count_nonzero(bins) / len(bins))
    quadrants = int(np.count_nonzero(np.bincount(
        (selected_angles / (0.5 * np.pi)).astype(int) % 4, minlength=4,
    )))
    return points, strength[keep].astype(np.float64), coverage, quadrants


def _ellipse_matrix(ellipse) -> tuple[float, float, float]:
    (_, _), (diameter_a, diameter_b), angle_deg = ellipse
    axis_a, axis_b = float(diameter_a) * 0.5, float(diameter_b) * 0.5
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    ia2, ib2 = 1.0 / (axis_a * axis_a), 1.0 / (axis_b * axis_b)
    return (
        c * c * ia2 + s * s * ib2,
        c * s * (ia2 - ib2),
        s * s * ia2 + c * c * ib2,
    )


def _shape_residuals(points: np.ndarray, center, shape) -> np.ndarray:
    q00, q01, q11 = shape
    dx, dy = points[:, 0] - center[0], points[:, 1] - center[1]
    level = np.sqrt(np.maximum(q00 * dx * dx + 2.0 * q01 * dx * dy + q11 * dy * dy, 0.0))
    determinant = q00 * q11 - q01 * q01
    radius = determinant ** -0.25
    return (level - 1.0) * radius


def _robust_ellipse_shape(points: np.ndarray, base_weights: np.ndarray):
    if points.shape[0] < 20:
        return None
    selected = np.arange(points.shape[0])
    ellipse = None
    residuals = None
    for _ in range(6):
        if selected.size < 20:
            return None
        try:
            ellipse = cv2.fitEllipseAMS(points[selected].astype(np.float32))
        except cv2.error:
            return None
        shape = _ellipse_matrix(ellipse)
        residuals = np.abs(_shape_residuals(points, ellipse[0], shape))
        median = float(np.median(residuals[selected]))
        mad = 1.4826 * float(np.median(np.abs(residuals[selected] - median)))
        new_selected = np.flatnonzero(residuals <= median + 3.0 * max(0.35, mad))
        if new_selected.size == selected.size:
            break
        selected = new_selected
    (cx, cy), (diameter_a, diameter_b), _ = ellipse
    inlier_residuals = _shape_residuals(points[selected], (cx, cy), shape)
    rms = float(np.sqrt(np.average(
        np.square(inlier_residuals), weights=np.maximum(base_weights[selected], 1e-6),
    )))
    return float(cx), float(cy), float(diameter_a) * 0.5, float(diameter_b) * 0.5, shape, rms


def _fit_center_for_shape(points: np.ndarray, base_weights: np.ndarray, initial_center, shape):
    cx, cy = map(float, initial_center)
    q00, q01, q11 = map(float, shape)
    determinant = q00 * q11 - q01 * q01
    if determinant <= 0:
        return None
    radius = determinant ** -0.25
    normalized_weights = np.maximum(base_weights / (np.median(base_weights) + 1e-6), 0.05)
    for _ in range(12):
        dx, dy = points[:, 0] - cx, points[:, 1] - cy
        qx, qy = q00 * dx + q01 * dy, q01 * dx + q11 * dy
        level = np.sqrt(np.maximum(dx * qx + dy * qy, 1e-12))
        residual = (level - 1.0) * radius
        scale = max(0.35, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        huber = np.minimum(1.0, 2.5 * scale / (np.abs(residual) + 1e-6))
        weights = normalized_weights * huber
        jacobian = np.column_stack((-radius * qx / level, -radius * qy / level))
        lhs = jacobian.T @ (weights[:, None] * jacobian)
        rhs = jacobian.T @ (weights * residual)
        try:
            delta = -np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return None
        cx += float(delta[0])
        cy += float(delta[1])
        if float(np.max(np.abs(delta))) < 0.005:
            break
    residual = _shape_residuals(points, (cx, cy), shape)
    rms = float(np.sqrt(np.average(np.square(residual), weights=normalized_weights)))
    return cx, cy, rms


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
