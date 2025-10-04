from enum import Enum

import logging
from math import ceil, floor

import numpy as np
import cv2
from numpy._typing._array_like import NDArray

from lunar_eclipse_align.utils.image import Image
from lunar_eclipse_align.utils.tools import (
    clip,
    ring_mask,
    touches_border,
)
from skimage.measure import CircleModel, ransac
from lunar_eclipse_align.utils.data_types import (
    HoughParams,
    PointArray,
    Circle,
    VectorArray,
)
from lunar_eclipse_align.utils.constants import (
    RESIDUAL,
    THUMB_SIZE,
)


class BrightnessMode(Enum):
    AUTO = "auto"
    BRIGHT = "bright"
    NORMAL = "normal"
    DARK = "dark"


# ============== 预处理 & 质量评估 ==============


def adaptive_preprocessing(
    img: Image, brightness_mode: BrightnessMode
) -> tuple[NDArray, BrightnessMode]:
    """将图像转换为适合圆检测的灰度，并做适度增强。返回 (processed_gray, brightness_mode)"""

    gray = img.gray_8bit
    mean_brightness = float(np.mean(gray))
    if brightness_mode == BrightnessMode.AUTO:
        if mean_brightness > 140:
            brightness_mode = BrightnessMode.BRIGHT
        elif mean_brightness < 70:
            brightness_mode = BrightnessMode.DARK
        else:
            brightness_mode = BrightnessMode.NORMAL

    if brightness_mode == BrightnessMode.BRIGHT:
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    elif brightness_mode == BrightnessMode.DARK:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return filtered, brightness_mode


def evaluate_circle_quality(gray: NDArray, circle: Circle) -> float:
    """对检测到的圆做一个稳定的质量打分，越大越好（0~100）。"""

    w, h = gray.shape[1], gray.shape[0]

    angles = np.linspace(0, 2 * np.pi, 48)
    directions = VectorArray(
        np.column_stack((np.cos(angles), np.sin(angles))), safe=False
    )
    inners: PointArray = directions * (circle.radius - RESIDUAL) + circle.center
    outers: PointArray = directions * (circle.radius + RESIDUAL) + circle.center
    mask = (
        (0 <= inners.x)
        & (inners.x < w)
        & (0 <= inners.y)
        & (inners.y < h)
        & (0 <= outers.x)
        & (outers.x < w)
        & (0 <= outers.y)
        & (outers.y < h)
    )
    if not np.sum(mask):
        return 0.0
    inners = inners.filter(mask)
    outers = outers.filter(mask)

    inner_vals = gray[np.round(inners.y).astype(int), np.round(inners.x).astype(int)]
    outer_vals = gray[np.round(outers.y).astype(int), np.round(outers.x).astype(int)]
    edge_strengths = np.abs(outer_vals - inner_vals)

    avg_edge = float(np.mean(edge_strengths))
    consistency = 1.0 / (1.0 + np.std(edge_strengths) / max(1.0, avg_edge))
    score = avg_edge * consistency
    return float(score)


# ============== 高光裁剪和星点抑制辅助 ==============
def clip_highlights(gray: NDArray, pct: float = 99.8):
    """Clip very bright highlights (glare/bloom) to a percentile to help Hough/RANSAC."""
    gray_float = gray.astype(np.float32)
    cap = np.percentile(gray_float, pct)
    if cap <= 0:
        return gray
    gray_float = np.minimum(gray_float, cap)
    gray_float = gray_float / (cap + 1e-6) * 255.0
    return gray_float.astype(np.uint8)


def remove_stars_small(gray: NDArray):
    """Suppress point-like stars/noise while preserving lunar rim."""
    # Top-hat to remove small bright dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    g = cv2.subtract(gray, tophat)
    # Gentle median to clean salt-pepper
    g = cv2.medianBlur(g, 3)
    return g


# ============== 稳健外缘 RANSAC（用于血月/缺口） ==============


def edge_points_outer_rim(
    gray: NDArray, prev_circle: Circle | None = None
) -> PointArray | None:
    # Canny 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return None

    pts = PointArray(np.stack([xs, ys], axis=1), safe=False)
    if prev_circle is None:
        return pts
    vectors = pts - prev_circle.center
    distance_cond = (prev_circle.radius * 0.85 < vectors.norms()) & (
        vectors.norms() < prev_circle.radius * 1.15
    )

    valid_pts = pts.filter(distance_cond)
    valid_vectors = vectors.filter(distance_cond)
    valid_x = valid_vectors.x.astype(np.int32)
    valid_y = valid_vectors.y.astype(np.int32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    valid_gx = gx[valid_y, valid_x]
    valid_gy = gy[valid_y, valid_x]
    gradients = VectorArray(np.stack([valid_gx, valid_gy], axis=1), safe=False)

    cond = gradients.normalize() * valid_vectors.normalize() > 0.2
    keep = valid_pts.filter(cond)
    if len(keep) < 30:
        return None
    return keep


# ============== 辅助：粗估 & 环形 ROI（抑制星点/加速霍夫） ==============


def rough_center_radius(gray: NDArray, min_r: float, max_r: float) -> Circle | None:
    g: NDArray = cv2.GaussianBlur(gray, (0, 0), 2.0)
    # Use adaptive + Otsu fallback to handle glare/crescent
    thr = max(10, round(np.mean(g) + 0.3 * np.std(g)))
    _, bw1 = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    _, bw2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_adap = cv2.adaptiveThreshold(
        g.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -5
    )
    bw = cv2.max(bw1, cv2.max(bw2, bw_adap))
    bw = cv2.morphologyEx(
        src=bw,
        op=cv2.MORPH_OPEN,
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    if min_r * 0.6 <= r <= max_r * 1.6:
        return Circle(cx, cy, r)
    return None


# ============== UI 调参可视化：分析区域掩膜（仅供显示） ==============
def build_analysis_mask(
    img: Image,
    brightness_min: float = 3 / 255.0,
) -> NDArray[np.bool]:
    """
    仅供 UI 调参窗口显示"分析区域"用：
    - uint8 归一化 -> 轻度去噪
    - Otsu 阈值 与 亮度下限并联
    - 形态学开运算清点
    - 仅保留最大连通域
    返回 bool(H,W)。不影响主流程检测。
    """
    gray = cv2.GaussianBlur(img.gray_8bit, (3, 3), 0)

    # Otsu 自动阈值
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 亮度下限阈值
    brightness_threshold = max(1, round(brightness_min * 255.0))
    _, brightness_mask = cv2.threshold(
        gray, brightness_threshold, 255, cv2.THRESH_BINARY
    )

    # 两个掩码取交集
    combined_mask = cv2.bitwise_and(otsu_mask, brightness_mask)

    # 形态学开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 找到所有轮廓
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(cleaned_mask, dtype=bool)

    # 保留最大连通域
    largest_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(cleaned_mask)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return final_mask.astype(bool)


# ============== 主检测（供 pipeline 调用） ==============


def initial_process(
    img: Image, strong_denoise: bool = False
) -> tuple[NDArray, BrightnessMode]:
    """
    设置检测环境，包括预处理和降噪

    Args:
        image: 输入图像
        strong_denoise: 是否使用强力降噪

    Returns:
        tuple: (processed, processed_det, proc_for_hough, brightness_mode, H, W)
    """

    processed, brightness_mode = adaptive_preprocessing(img, BrightnessMode.AUTO)

    # 可选：强力降噪（仅影响检测，不影响最终成片）
    if strong_denoise:

        # fast NLM 能在强噪场景下保持边缘
        processed = cv2.fastNlMeansDenoising(
            processed, None, h=10, templateWindowSize=7, searchWindowSize=21
        )
        # 轻度中值进一步压盐胡椒
        processed = cv2.medianBlur(processed, 3)

    processed = clip_highlights(processed, pct=99.8)
    # Use a detection-optimized copy to make Hough/RANSAC more stable on glare/bloom frames
    processed = remove_stars_small(processed)

    return processed, brightness_mode


def hough_on_thumb_detect(gray: NDArray, params: HoughParams) -> Circle | None:
    logging.debug(f"缩放霍夫(thumb)")
    height, width = gray.shape[:2]
    max_side = max(height, width)
    circles = []
    scale = 1.0
    if max_side > THUMB_SIZE:
        scale = THUMB_SIZE / max_side
        height, width = round(height * scale), round(width * scale)

        gray = cv2.resize(
            gray,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        params = HoughParams(
            minRadius=max(1, floor(params.minRadius * scale)),
            maxRadius=max(2, ceil(params.maxRadius * scale)),
            param1=params.param1,
            param2=max(params.param2 - 5, 10),
        )

    circles = cv2.HoughCircles(
        gray,
        minDist=height // 2,
        **params,
    )
    if circles is None:
        return None

    circle = Circle.from_ndarray(circles[0][0]).scale(1 / scale)
    return circle


# ------------------ ROI构建 ------------------
def build_detection_roi(
    gray: NDArray,
    params: HoughParams,
    prev_circle: Circle | None,
) -> NDArray:
    height, width = gray.shape[:2]
    # —— 粗估中心半径，构建环形 ROI —— #
    est = rough_center_radius(gray, params.minRadius, params.maxRadius)
    if est:
        inner = min((params.minRadius / est.radius) * 0.9, 0.85)
        outer = max(1.15, (params.maxRadius / est.radius) * 1.05)
        ring = ring_mask(width, height, est, inner=inner, outer=outer) * np.uint8(255)
        gray = cv2.bitwise_and(gray, gray, mask=ring)

    # —— 若给出上一帧圆心半径，合并一个"历史先验"环形 ROI —— #
    if prev_circle:
        inner = clip(0.70, (params.minRadius / prev_circle.radius) * 0.9, 0.85)
        outer = clip(1.15, (params.maxRadius / prev_circle.radius) * 1.05, 1.30)
        ring_prev = ring_mask(
            width, height, prev_circle, inner=inner, outer=outer
        ) * np.uint8(255)
        gray = cv2.bitwise_and(gray, gray, mask=ring_prev)

    return gray


# ------------------ 超时检测函数 ------------------
def timeout_fallback_detection(gray: NDArray, params: HoughParams) -> Circle | None:
    logging.debug("超时降级(thumb)")
    height, width = gray.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    small_height, small_width = round(height * scale), round(width * scale)
    small = cv2.resize(
        gray,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )
    params = HoughParams(
        minRadius=max(1, floor(params.minRadius * scale)),
        maxRadius=max(2, ceil(params.maxRadius * scale)),
        param1=max(params.param1, 20),
        param2=max(params.param2 - 5, 8),
    )

    circles = cv2.HoughCircles(
        small,
        minDist=small_height // 2,
        **params,
    )
    if circles is None:
        return None

    circle = Circle.from_ndarray(circles[0][0]).scale(1 / scale)
    return circle


# ------------------ 稳健RANSAC检测 ------------------


def detect_circle_robust(
    gray: NDArray, prev_circle: Circle | None = None
) -> Circle | None:
    logging.debug("稳健RANSAC")
    pts = edge_points_outer_rim(gray, prev_circle)
    if not pts:
        return None
    model: CircleModel | None
    inliers: list[bool] | None
    model, inliers = ransac(
        data=pts._arr,
        model_class=CircleModel,
        min_samples=3,
        residual_threshold=RESIDUAL,
        max_trials=120,
    )
    if model is None or inliers is None:
        return None
    if np.sum(inliers) < 40:
        return None
    cx, cy, r = model.params  # CircleModel.params 返回 (xc, yc, r)
    cand = Circle(float(cx), float(cy), float(r))
    if prev_circle is None:
        return cand
    vectors = pts - cand.center
    arctans = np.arctan2(vectors.y, vectors.x)
    span: np.float64 = np.ptp(arctans)
    if span < (2 * np.pi / 3.0):  # <120°
        cand = Circle(cand.x, cand.y, prev_circle.radius)
    return cand


# ------------------ 标准霍夫检测 ------------------
def standard_hough_detect(gray: NDArray, params: HoughParams) -> Circle | None:
    logging.debug("标准霍夫")
    height, width = gray.shape[:2]
    circles = cv2.HoughCircles(
        gray,
        minDist=height // 2,
        **params,
    )
    if circles is None:
        return None
    circle = Circle.from_ndarray(circles[0][0])
    return circle


# ------------------ 自适应霍夫检测 ------------------
def adaptive_hough_detect(
    gray: NDArray,
    params: HoughParams,
    brightness_mode: BrightnessMode,
) -> Circle | None:
    params = params.copy()
    height, width = gray.shape[:2]
    if brightness_mode == BrightnessMode.BRIGHT:
        params.param1 += 20
        params.param2 = max(params.param2 - 5, 10)
    elif brightness_mode == BrightnessMode.DARK:
        params.param1 = max(params.param1 - 15, 20)
        params.param2 = max(params.param2 - 10, 5)
    else:
        params.param2 = max(params.param2 - 8, 8)
    logging.debug(f"自适应霍夫(P1={params.param1},P2={params.param2})")

    circles = cv2.HoughCircles(
        gray,
        minDist=height // 2,
        **params,
    )
    if circles is None:
        return None

    circle = Circle.from_ndarray(circles[0][0])
    return circle


# ------------------ 轮廓检测 ------------------
def contour_detect(gray: NDArray, hough: HoughParams) -> Circle | None:
    mean_val = float(np.mean(gray))
    tv = max(50, round(mean_val * 0.7))
    logging.debug(f"轮廓检测(T={tv})")

    _, binary = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = hough.minRadius**2 * np.pi * 0.3
    max_area = hough.maxRadius**2 * np.pi * 2.0
    contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    circles = [
        Circle(c[0][0], c[0][1], c[1])
        for cnt in contours
        if (c := cv2.minEnclosingCircle(cnt))
    ]

    for c in circles:
        if hough.minRadius <= c.radius <= hough.maxRadius:
            return c

    return None


# ------------------ Padding 降级检测 ------------------
def padding_fallback_detect(
    gray: NDArray,
    params: HoughParams,
) -> Circle | None:
    logging.debug("Padding降级")
    pad = max(32, round(params.maxRadius * 1.2))
    padded_gray = cv2.copyMakeBorder(
        gray,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    # Use constant black padding to avoid mirrored ghosts influencing Hough
    est = rough_center_radius(
        padded_gray, floor(params.minRadius * 1.1), ceil(params.maxRadius * 1.1)
    )
    padded_gray = padded_gray
    if est is not None:
        ring = ring_mask(
            padded_gray.shape[1],  # width
            padded_gray.shape[0],  # height
            est,
            inner=0.70,
            outer=1.15,
        ) * np.uint8(255)
        padded_gray = cv2.bitwise_and(padded_gray, padded_gray, mask=ring)

    height, width = padded_gray.shape[:2]
    params = HoughParams(
        minRadius=max(1, floor(params.minRadius * 1.1)),
        maxRadius=max(2, ceil(params.maxRadius * 1.1)),
        param1=max(params.param1, 20),
        param2=max(params.param2 - 5, 8),
    )
    circles = cv2.HoughCircles(
        padded_gray,
        minDist=height // 2,
        **params,
    )
    if circles is None:
        return detect_circle_robust(padded_gray)
    circle = Circle.from_ndarray(circles[0][0])
    return circle


# ------------------ 最终圆验证 ------------------
def final_detect(
    gray: NDArray,
    params: HoughParams,
) -> Circle | None:
    logging.debug("最终验证")
    # 在严格窗口内再做一次快速霍夫重试
    height, width = gray.shape

    params = HoughParams(
        minRadius=max(1, params.minRadius),
        maxRadius=params.maxRadius,
        param1=max(params.param1, 20),
        param2=max(params.param2 - 5, 8),
    )
    circles = cv2.HoughCircles(
        gray,
        minDist=max(16, min(height, width) // 4),
        **params,
    )
    if circles is None:
        return None
    circle = Circle.from_ndarray(circles[0][0])
    if not (params.minRadius <= circle.radius <= params.maxRadius):
        return None

    return circle


def detect_circle(
    img: Image,
    params: HoughParams,
    strong_denoise=False,
    prev_circle: Circle | None = None,
) -> Circle | None:
    """
    主检测函数 - 重构后的协调器版本

    将原本的单体大函数拆分为多个职责明确的辅助函数
    """
    logging.debug(
        f"开始检测：{params.minRadius=:.2f} {params.maxRadius=:.2f} {params.param1=:.2f} {params.param2=:.2f}"
    )
    best_circle: Circle | None = None
    best_quality: float = 0.0
    # 1. 设置检测环境
    gray, brightness_mode = initial_process(img, strong_denoise)

    def update_best(circle: Circle | None):
        nonlocal best_circle, best_quality, gray
        if not circle:
            return

        quality = evaluate_circle_quality(gray, circle)
        # quality = evaluate_circle_quality(img.normalized_gray, circle)
        if not best_circle or quality > best_quality:
            best_circle = circle
            best_quality = quality

    # 2. 构建检测ROI
    masked_gray = build_detection_roi(
        gray,
        params,
        prev_circle,
    )
    # update_best(hough_on_thumb_detect(masked_gray, params))
    # 3. 尝试稳健RANSAC检测
    update_best(detect_circle_robust(gray))

    # 5. 尝试标准霍夫检测
    update_best(standard_hough_detect(masked_gray, params))

    # 6. 尝试自适应霍夫检测
    if not best_circle or best_quality < 15:
        update_best(adaptive_hough_detect(masked_gray, params, brightness_mode))

    # —— 轮廓备选 —— #
    if not best_circle or best_quality < 10:
        update_best(contour_detect(gray, params))

    if (
        not best_circle
        or (best_quality < 10)
        or touches_border(img.width, img.height, best_circle)
    ):
        update_best(padding_fallback_detect(gray, params))

    if best_circle and not (params.minRadius <= best_circle.radius <= params.maxRadius):
        # —— 最终半径窗口一致性检查（严格遵守 UI 设定） —— #
        update_best(final_detect(gray, params))

    if not best_circle:
        logging.error(f"  ✗ 圆检测失败")
        return None
    logging.info(f"  ○  圆检测成功")
    logging.debug(f"{str(best_circle)=}, {best_quality=:.2f}, {brightness_mode=}")

    return best_circle


def detect_circle_quick(
    img: Image, params: HoughParams, strong_denoise=False
) -> Circle | None:

    gray, brightness_mode = initial_process(img, strong_denoise)

    circle = hough_on_thumb_detect(gray, params)
    return circle
