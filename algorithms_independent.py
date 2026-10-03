"""Stateless, bounded lunar geometry. No reference/previous-frame inputs.

The only cross-call caches contain immutable coordinate/kernel templates.
Annular correlation supplies hypotheses, never final fixed-radius solutions.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import math
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from algorithms_circle import detect_circle_phd2_enhanced
from algorithms_limb import (
    _annular_edge_kernel, _remove_smooth_background,
    _quadratic_peak_offset,
)

SHAPE_REGULARIZATION = .002


@dataclass
class Geometry:
    center: tuple[float, float]
    radius: float
    axes: tuple[float, float]
    angle: float
    model: str
    rms: float
    coverage: float
    count: int
    spread: float
    reasons: list[str] = field(default_factory=list)
    shape_vector: tuple[float, float] = (0., 0.)
    shape_sensitivity: float = 0.

    @property
    def circle(self):
        return (*self.center, self.radius)


@dataclass
class Analysis:
    geometry: Geometry | None
    method: str
    timings: dict[str, float]
    reason: str = ""


def validate_radius_bounds(bounds):
    lo, hi = map(float, bounds)
    if not (np.isfinite(lo + hi) and 8 < lo < hi):
        raise ValueError("半径范围必须满足 8 < 最小半径 < 最大半径，且均为有限数值。")
    return lo, hi


def _shape_components(params, bounds):
    """Decode a bounded symmetric shape matrix without an orientation angle.

    params[2:] = (m, u, v). The eigenvalues of [[m+u,v],[v,m-u]]
    are m +/- hypot(u,v); applying a logistic spectral map puts BOTH
    semi-axes strictly inside the requested radius interval. At u=v=0
    this is a circle, with finite derivatives in both shape directions.
    T = [[mean+s1, s2], [s2, mean-s1]] maps a unit circle to the ellipse.
    """
    lo, hi = bounds
    m, u, v = params[2:]
    d = math.hypot(u, v)
    a, b = lo + (hi-lo) * expit(np.array([m+d, m-d]))
    if d < 1e-6:
        p = expit(m)
        factor = (hi-lo) * p * (1-p)
    else:
        factor = (a-b) / (2*d)
    return (a+b)/2, factor*u, factor*v, a, b


def _shape_seed(center, axes, angle, bounds):
    """Encode a same-frame initial ellipse; angle is never optimized directly."""
    lo, hi = bounds
    fractions = np.clip((np.asarray(axes)-lo) / (hi-lo), 1e-6, 1-1e-6)
    z = np.log(fractions / (1-fractions))
    m, d = (z[0]+z[1])/2, (z[0]-z[1])/2
    return np.array([*center, m, d*np.cos(2*angle), d*np.sin(2*angle)], dtype=float)


def _residual(params, points, bounds=None):
    dx, dy = points[:, 0] - params[0], points[:, 1] - params[1]
    if len(params) == 3:
        # Circle fits are only used to initialize/resample this frame's edges.
        return np.hypot(dx, dy) - params[2]
    mean, s1, s2, a, b = _shape_components(params, bounds)
    q00, q01, q11 = (mean-s1)/(a*b), -s2/(a*b), (mean+s1)/(a*b)
    x, y = q00*dx + q01*dy, q01*dx + q11*dy
    level = x*x + y*y - 1
    gradient = 2 * np.hypot(q00*x + q01*y, q01*x + q11*y)
    return level / np.maximum(gradient, 1e-8)


def _fit(points, strengths, initial, bounds, center_bounds, regularization=SHAPE_REGULARIZATION):
    """Optimize all model parameters, including radius, on this frame's points."""
    lo, hi = bounds
    lower = [*center_bounds[0], lo]
    upper = [*center_bounds[1], hi]
    scales = [1., 1., 1.]
    if len(initial) == 5:
        lower, upper = [*center_bounds[0], -16., -16., -16.], [*center_bounds[1], 16., 16., 16.]
        p = expit(initial[2])
        step = 1 / max((hi-lo)*p*(1-p), 1.)
        scales = [1., 1., step, step, step]
    weights = np.sqrt(np.clip(strengths / max(float(np.median(strengths)), 1e-6), .2, 3.))
    initial = np.clip(initial, np.asarray(lower) + 1e-6, np.asarray(upper) - 1e-6)
    def residual(p):
        data = weights * _residual(p, points, bounds)
        if len(p) == 5:
            _, s1, s2, _, _ = _shape_components(p, bounds)
            # Isotropic weak shrinkage, measured in pixels, not an angle prior.
            return np.r_[data, np.sqrt(len(points)*regularization)*np.array([s1, s2])]
        return data

    def robust_data_quadratic_prior(z):
        root = np.sqrt(1+z)
        rho = np.array([2*(root-1), 1/root, -.5/(root**3)])
        rho[:, -2:] = np.array([z[-2:], np.ones(2), np.zeros(2)])
        return rho

    result = least_squares(
        residual, initial,
        bounds=(lower, upper), loss=robust_data_quadratic_prior if len(initial) == 5 else 'soft_l1', f_scale=.7,
        x_scale=scales, max_nfev=70, ftol=1e-6, xtol=1e-6, gtol=1e-6,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return None
    return result.x


def _coverage(points, center):
    angles = np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2 * np.pi)
    bins = np.unique((angles * (72 / (2 * np.pi))).astype(int) % 72)
    return len(bins) / 72, angles


def _local_analysis(image, seed, bounds):
    # Crop based on THIS image's coarse hypothesis, never another image.
    h, w = image.shape[:2]
    margin = max(bounds[1], seed[2]) * 1.15 + 64
    x0, y0 = max(0, int(seed[0] - margin)), max(0, int(seed[1] - margin))
    x1, y1 = min(w, int(math.ceil(seed[0] + margin))), min(h, int(math.ceil(seed[1] + margin)))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = gray.astype(np.float32)
    low, high = np.percentile(gray, (1, 99.8))
    if not np.isfinite(low + high) or high <= low:
        return None
    # Keep sub-8-bit contrast for uint16/float input. Output pixels are untouched.
    gray = np.clip((gray - np.float32(low)) * np.float32(255 / (high - low)), 0, 255)
    return cv2.GaussianBlur(gray, (0, 0), 1.2), np.array([x0, y0], dtype=float)


def _sample_edges(gray, circle, search_px):
    """Signed edges with per-profile noise thresholds, not global brightness.

    Spatial angular smoothing is within ONE image. A faint dark-side limb is
    not discarded merely because the opposite side is orders of magnitude
    brighter. Among significant local peaks prefer the outer limb.
    """
    cx, cy, radius = circle
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False, dtype=np.float32)
    offsets = np.arange(-search_px, search_px + 1, dtype=np.float32)
    distances = radius + offsets
    mx = (cx + np.cos(angles)[:, None] * distances).astype(np.float32)
    my = (cy + np.sin(angles)[:, None] * distances).astype(np.float32)
    profiles = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    padded = np.concatenate((profiles[-4:], profiles, profiles[:4]))
    profiles = cv2.GaussianBlur(padded, (1, 7), sigmaX=0, sigmaY=1.1)[4:-4]
    gradient = -np.gradient(profiles, axis=1)
    median = np.median(gradient, axis=1, keepdims=True)
    noise = 1.4826 * np.median(np.abs(gradient - median), axis=1, keepdims=True)
    threshold = np.maximum(.12, 4. * noise)
    peaks = (gradient[:, 1:-1] >= gradient[:, :-2]) & (gradient[:, 1:-1] > gradient[:, 2:])
    peaks &= gradient[:, 1:-1] >= threshold
    valid = (mx > 2) & (mx < gray.shape[1] - 3) & (my > 2) & (my < gray.shape[0] - 3)
    peaks &= valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
    # A weak isolated bump far outside a strong edge is not useful. The low
    # ratio still permits a weaker physical rim outside an eclipse boundary.
    peaks &= gradient[:, 1:-1] >= .12 * np.max(gradient, axis=1, keepdims=True)
    keep = np.any(peaks, axis=1)
    if np.count_nonzero(keep) < 30:
        return None
    index = peaks.shape[1] - 1 - np.argmax(peaks[:, ::-1], axis=1) + 1
    rows = np.flatnonzero(keep)
    index = index[keep]
    left, mid, right = gradient[rows, index-1], gradient[rows, index], gradient[rows, index+1]
    denominator = left - 2 * mid + right
    sub = np.divide(.5 * (left - right), denominator, out=np.zeros_like(mid), where=np.abs(denominator) > 1e-6)
    radii = radius + offsets[index] + np.clip(sub, -.75, .75)
    points = np.column_stack((cx + np.cos(angles[keep]) * radii, cy + np.sin(angles[keep]) * radii)).astype(float)
    strengths = np.clip(mid / np.maximum(noise[rows, 0], .03), 1, 10).astype(float)
    return points, strengths


def fit_lunar_geometry(image, seed, radius_bounds):
    bounds = validate_radius_bounds(radius_bounds)
    if seed is None or not np.all(np.isfinite(seed)) or not bounds[0] <= seed[2] <= bounds[1]:
        return None
    local = _local_analysis(image, seed, bounds)
    if local is None:
        return None
    gray, origin = local
    initial = np.array(seed, dtype=float)
    initial[:2] -= origin
    move = max(16., .10 * initial[2])
    center_bounds = (initial[:2] - move, initial[:2] + move)
    current = initial.copy()
    # Resample after recentering, but never move the original search bounds.
    for _ in range(2):
        sampled = _sample_edges(gray, current, search_px=max(24, int(.06 * current[2])))
        if sampled is None:
            return None
        points, strengths = sampled
        fit = _fit(points, strengths, current, bounds, center_bounds)
        if fit is None:
            return None
        current = fit

    # One final model for every frame, including circles. No residual threshold
    # selects between circle and ellipse; the coarse circle is ONLY a seed.
    params = _shape_seed(current[:2], (current[2], current[2]), 0., bounds)
    params = _fit(points, strengths, params, bounds, center_bounds)
    if params is None:
        return None
    errors = _residual(params, points, bounds)
    scale = max(.35, 1.4826 * np.median(np.abs(errors - np.median(errors))))
    inliers = np.abs(errors) <= max(1.2, 3 * scale)
    points, strengths = points[inliers], strengths[inliers]
    if len(points) < 30:
        return None
    final = _fit(points, strengths, params, bounds, center_bounds)
    if final is None:
        return None
    params = final
    errors = _residual(params, points, bounds)
    coverage, angles = _coverage(points, params[:2])
    rms = float(np.sqrt(np.mean(errors ** 2)))
    reasons = []
    mean, s1, s2, a, b = _shape_components(params, bounds)
    axes = (a, b)
    if min(axes) / max(axes) < .94:
        reasons.append('椭圆轴比超限')
    if coverage < .42:
        reasons.append(f'有效月缘覆盖不足 ({coverage:.0%})，中心/半径或形状存在歧义')
    if rms > 2.0:
        reasons.append(f'月缘残差偏大 ({rms:.2f}px)')
    if min(min(axes) - bounds[0], bounds[1] - max(axes)) < .25:
        reasons.append('拟合触及半径范围边界')
    if np.any(params[:2] - center_bounds[0] < .25) or np.any(center_bounds[1] - params[:2] < .25):
        reasons.append('精修触及中心搜索边界')

    # A shape prior must not manufacture a stable center on an ambiguous arc.
    # These comparisons are within THIS image, never between adjacent frames.
    sensitivity_centers = []
    for penalty in (0., 4*SHAPE_REGULARIZATION):
        alternative = _fit(points, strengths, params, bounds, center_bounds, regularization=penalty)
        if alternative is not None:
            sensitivity_centers.append(alternative[:2])
    sensitivity = max((float(np.linalg.norm(c-params[:2])) for c in sensitivity_centers), default=float('inf'))
    if len(sensitivity_centers) != 2:
        reasons.append('形状约束敏感性检查未收敛')
    if sensitivity > 1.:
        reasons.append(f'圆心依赖形状约束 ({sensitivity:.2f}px)，形状仍有歧义')

    # Leave out an angular block and refit ALL parameters WITHOUT the shape
    # prior; otherwise regularization could conceal short-arc degeneracy.
    centers = []
    attempted = 0
    sectors = (angles * (6 / (2 * np.pi))).astype(int) % 6
    for sector in range(6):
        keep = sectors != sector
        if np.count_nonzero(~keep) < 8 or np.count_nonzero(keep) < 30:
            continue
        attempted += 1
        subset = _fit(points[keep], strengths[keep], params, bounds, center_bounds, regularization=0.)
        if subset is not None:
            centers.append(subset[:2])
    spread = max((float(np.linalg.norm(c - params[:2])) for c in centers), default=float('inf'))
    if len(centers) < 3:
        reasons.append('可用独立弧段不足，无法验证圆心稳定性')
    elif spread > 2.0:
        reasons.append(f'分弧重拟合圆心不稳定 ({spread:.2f}px)')
    if len(centers) < attempted:
        reasons.append('部分分弧重拟合未收敛，不能完整验证圆心')
    absolute = params[:2] + origin
    if not (0 <= absolute[0] < image.shape[1] and 0 <= absolute[1] < image.shape[0]):
        return None
    # Orientation is only a derived value for drawing/export. At a circle it
    # is arbitrary and is NEVER fed back into the fit or another frame.
    angle = .5 * math.atan2(s2, s1) % math.pi if math.hypot(s1, s2) > 1e-9 else 0.
    return Geometry(tuple(map(float, absolute)), float(np.sqrt(a*b)),
                    tuple(map(float, axes)), angle, 'unified-ellipse', rms, coverage, len(points), spread, reasons,
                    (float(s1/mean), float(s2/mean)), sensitivity)


@lru_cache(maxsize=12)
def _kernel_spectrum(radius, band, height, width):
    kernel = _annular_edge_kernel(radius, band)
    padded = np.zeros((height, width), np.float32)
    padded[:kernel.shape[0], :kernel.shape[1]] = kernel
    spectrum = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    spectrum.setflags(write=False)
    return spectrum, kernel.shape[0] // 2


def annular_hypotheses(image, radius_bounds, max_side=384):
    """Search a radius interval; return only same-frame starting hypotheses.

    Reuse each channel FFT across radii. Cached spectra never contain image
    pixels, detected centers, fitted radii or information from another frame.
    """
    lo, hi = validate_radius_bounds(radius_bounds)
    h, w = image.shape[:2]
    hi = min(hi, .46 * min(h, w))
    if hi <= lo:
        return []
    # Exact isotropic scale; rounding affects coordinates by < half a pixel
    # at this coarse stage, which is followed by original-pixel fitting.
    scale = min(1., max_side / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    if small.ndim == 3:
        b, g, r = cv2.split(small[:, :, :3])
        channels = [.114 * b + .587 * g + .299 * r, r - b]
    else:
        channels = [small]
    radii = np.linspace(lo * scale, hi * scale, 7)
    band = max(1.4, float(np.mean(radii)) * .012)
    max_kernel = _annular_edge_kernel(float(radii[-1]), band).shape[0]
    fh, fw = [cv2.getOptimalDFTSize(n + max_kernel - 1) for n in small.shape[:2]]
    all_channels = []
    for channel in channels:
        residual = _remove_smooth_background(channel)
        residual = cv2.GaussianBlur(residual, (0, 0), max(.8, band * .45))
        low, high = np.percentile(residual, (1, 99))
        if not np.isfinite(low + high) or high - low < 1e-4:
            all_channels.append([])
            continue
        padded = np.zeros((fh, fw), np.float32)
        padded[:small.shape[0], :small.shape[1]] = np.clip(residual, low, high)
        spectrum = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        candidates = []
        for radius in radii:
            kernel_fft, offset = _kernel_spectrum(float(radius), band, fh, fw)
            response = cv2.idft(cv2.mulSpectrums(spectrum, kernel_fft, 0),
                                flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            response = response[offset:offset + small.shape[0], offset:offset + small.shape[1]]
            margin = int(math.ceil(radius + 4 * band)) + 1
            if min(response.shape) <= 2 * margin:
                continue
            valid = response[margin:-margin, margin:-margin]
            _, peak, _, loc = cv2.minMaxLoc(valid)
            x, y = loc[0] + margin, loc[1] + margin
            z = (peak - float(np.median(valid))) / max(1e-6, 1.4826 * float(np.median(np.abs(valid - np.median(valid)))))
            center = (x + _quadratic_peak_offset(response[y, x-1:x+2]),
                      y + _quadratic_peak_offset(response[y-1:y+2, x]))
            candidates.append((z, center[0] / scale, center[1] / scale, radius / scale))
        all_channels.append(candidates)
    if not all_channels[0]:
        return []
    luminance = sorted(all_channels[0], reverse=True)
    colour = sorted(all_channels[1], reverse=True) if len(all_channels) > 1 else []
    if colour and colour[0][0] >= 7:
        if luminance[0][0] < 7 or math.hypot(luminance[0][1] - colour[0][1], luminance[0][2] - colour[0][2]) > max(3., .035 * np.mean([lo, hi])):
            return []
        threshold = 7
    else:
        threshold = 10
    selected = []
    for candidate in luminance:
        if candidate[0] < threshold:
            continue
        if all(abs(candidate[3] - old[2]) > (hi - lo) / 7 for old in selected):
            selected.append(candidate[1:])
        if len(selected) == 2:
            break
    return selected


def analyze_lunar_frame(image, hough_params, strong_denoise=False):
    """The production entry point. Deliberately accepts no reference geometry."""
    lo, hi, p1, p2 = hough_params
    bounds = validate_radius_bounds((lo, hi))
    timings = {}
    start = time.perf_counter()
    scale = min(1., 1600. / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    detected, _, _, method, _ = detect_circle_phd2_enhanced(
        small, max(1, int(lo * scale)), max(2, int(math.ceil(hi * scale))), p1, p2,
        strong_denoise=strong_denoise,
    )
    seed = np.asarray(detected, dtype=float) / scale if detected is not None else None
    timings['coarse'] = time.perf_counter() - start
    start = time.perf_counter()
    fit = fit_lunar_geometry(image, seed, bounds)
    timings['limb'] = time.perf_counter() - start
    # Only difficult frames pay for multi-radius whole-frame annular search.
    # An annular hypothesis must pass the same free-geometry checks as Hough.
    if fit is None or fit.reasons:
        start = time.perf_counter()
        seeds = annular_hypotheses(image, bounds)
        alternatives = [fit_lunar_geometry(image, candidate, bounds) for candidate in seeds]
        alternatives = [candidate for candidate in alternatives if candidate is not None]
        candidates = ([fit] if fit is not None else []) + alternatives
        if candidates:
            candidates.sort(key=lambda x: (bool(x.reasons), x.spread, -x.coverage, x.rms))
            fit = candidates[0]
            if any(c is fit for c in alternatives):
                method = '多半径环积分初定位 + 自由几何精修'
            # Conflicting otherwise valid same-frame solutions are not silently
            # resolved by a fixed-radius or brightness preference.
            trusted = [c for c in candidates if not c.reasons]
            if len(trusted) > 1 and any(np.linalg.norm(np.asarray(c.center) - fit.center) > 3 for c in trusted):
                fit.reasons.append('同帧多初值解的圆心分歧超过 3px')
        timings['rescue'] = time.perf_counter() - start
    return Analysis(fit, method, timings, '' if fit is not None else '本帧没有通过几何拟合的月缘候选')
