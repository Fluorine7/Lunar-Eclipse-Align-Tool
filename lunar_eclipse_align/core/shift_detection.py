import itertools
import logging
import cv2
import numpy as np
import time
from lunar_eclipse_align.utils.image import Image
from lunar_eclipse_align.utils.tools import (
    clip,
    soft_disk_mask,
)
from numpy.typing import NDArray
from lunar_eclipse_align.utils.data_types import Point, Vector, Circle, ROI, VectorArray
from lunar_eclipse_align.utils.constants import (
    MAX_REFINE_DELTA_PX,
    MAX_REFINE_DELTA_PX,
    MIN_INLIERS,
    MIN_MEAN_ZNCC,
)

# ---------------- 工具函数 ----------------


def clahe_and_bandpass(gray: NDArray) -> NDArray[np.uint8]:

    # Local contrast to suppress global illumination, emphasize small-scale details
    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # High-pass only (remove mid/low frequencies): unsharp mask
    hp = cv2.subtract(
        cla, cv2.GaussianBlur(cla, (0, 0), 6.0)
    )  # sigma ~6 -> keep only high freq

    # Edge emphasis via Laplacian (pure high-frequency operator)
    lap = cv2.Laplacian(cla, cv2.CV_32F, ksize=3)

    # Combine and normalize
    combo = 0.5 * np.abs(hp.astype(np.float32)) + 0.5 * np.abs(lap)
    cv2.normalize(combo, combo, 0, 255, cv2.NORM_MINMAX)
    return combo.astype(np.uint8)


def match_roi_zncc_local(
    roi_refF: NDArray,
    tgtF: NDArray,
    roi: ROI,
    search: int = 12,
    mask_patch: NDArray | None = None,
) -> tuple[Vector[float], float] | None:
    """Match ref_patch around (x,y) in tgt_img within +/-search window.
    Uses TM_CCORR_NORMED with optional template mask (same size as ref_patch).
    Returns (dx, dy, score).
    """
    h, w = roi.height, roi.width
    H, W = tgtF.shape[:2]

    # target search window centered at the ref patch center
    c: Point[int] = roi.start_point + Vector(w // 2, h // 2)
    pos0: Point[int] = Point(
        max(0, c.x - w // 2 - search), max(0, c.y - h // 2 - search)
    )
    pos1: Point[int] = Point(
        min(W, pos0.x + w + 2 * search), min(H, pos0.y + h + 2 * search)
    )

    crop = tgtF[pos0.y : pos1.y, pos0.x : pos1.x]

    if crop.shape[0] < h or crop.shape[1] < w:
        return None

    tpl = roi_refF.astype(np.float32)
    win = crop.astype(np.float32)

    # Variance guard: if the (masked) template has almost no texture, skip
    if mask_patch is not None and mask_patch.shape == roi_refF.shape:
        m = mask_patch.astype(np.float32) / 255.0
        area = m.sum()
        if area < 16:  # too small effective area
            return None
        tpl_eff = tpl[m > 0.5]
        if tpl_eff.size == 0 or np.std(tpl_eff) < 1e-3:
            return None
        res = cv2.matchTemplate(
            win, tpl, cv2.TM_CCORR_NORMED, mask=mask_patch.astype(np.uint8)
        )
    else:
        if np.std(tpl) < 1e-3:
            return None
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCORR_NORMED)

    # Handle NaN/Inf from OpenCV edge cases
    if not np.isfinite(res).all():
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)

    minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
    maxloc = Vector(maxloc[0], maxloc[1])
    d = pos0 + maxloc + Vector(w // 2, h // 2) - c

    return d.as_type(float), maxv


def initial_process(
    img: Image, ref_img: Image, circle: Circle
) -> tuple[NDArray, NDArray]:
    """
    准备对齐数据：创建遮罩、预处理图像、调整参数

    返回: (refF, tgtF, energy, adjusted_n_rois, adjusted_roi_size, adjusted_search)
    """

    logging.debug(f"[Refine] 启用自适应ROI与亮度门控: brightness∈[30,220]")

    # 软盘遮罩，仅保留月盘内纹理
    mask: NDArray[np.float32] = soft_disk_mask(
        ref_img.height, ref_img.width, circle, inner=0.97 * 0.9, outer=0.97
    )

    # 归一化转为8-bit，用于能量图和ROI选择

    # 梯度/DoG 预处理（抗亮度变化），用于匹配阶段
    refF = clahe_and_bandpass(ref_img.gray_8bit)
    tgtF = clahe_and_bandpass(img.gray_8bit)

    # 只保留盘内
    refF = (refF.astype(np.float32) * mask).astype(np.uint8)
    tgtF = (tgtF.astype(np.float32) * mask).astype(np.uint8)

    return refF, tgtF


def roi_matches_phasecorr(
    tgtF: NDArray,
    refF_patch: NDArray,
    roi: ROI,
    search: int,
    mask_patch: NDArray | None = None,
) -> tuple[Vector, float] | None:
    ref_h, ref_w = refF_patch.shape
    h, w = tgtF.shape
    roi_center: Point[int] = ROI(roi.x, roi.y, ref_w, ref_h).center.as_type(int)
    p: Point[int] = roi_center - Vector(ref_w // 2, ref_h // 2) - Vector(search, search)
    e: Point[int] = Point(min(w, p.x + ref_w), min(h, p.y + ref_h))

    tgtF_patch = tgtF[p.y : e.y, p.x : e.x]
    if tgtF_patch.shape != refF_patch.shape:
        return None
    refF_patch = refF_patch.astype(np.float32)
    tgtF_patch = tgtF_patch.astype(np.float32)

    if mask_patch is not None and mask_patch.shape == refF_patch.shape:
        mean = mask_patch.astype(np.float32) / 255.0
        refF_patch = refF_patch * mean
        tgtF_patch = tgtF_patch * mean
        # 纹理/有效面积检查
        eff = refF_patch[mean > 0.5]
        if eff.size == 0 or np.std(eff) < 1e-3:
            return None

    (dx2, dy2), resp = cv2.phaseCorrelate(refF_patch, tgtF_patch)
    if resp is None or not np.isfinite(resp):
        return None
    if abs(dx2) >= 2.5 or abs(dy2) >= 2.5:
        return None
    return Vector(dx2, dy2), float(resp)


def collect_roi_matches(
    rois: list[ROI],
    refF: NDArray,
    tgtF: NDArray,
    ref_img: Image,
    search: int,
    time_budget_sec: float,
) -> tuple[list[Point], list[Vector], list[float], list[float]]:
    """
    收集ROI匹配结果

    返回: (centers, d_list, weights, zncc_list)
    """
    # 收集 ROI 局部位移向量（dx_i, dy_i）及其中心（xi, yi）
    centers: list[Point[float]] = []
    zncc_list: list[float] = []
    d_list: list[Vector[float]] = []
    weights: list[float] = []
    t_start = time.perf_counter()

    for roi in rois:
        # 时间预算：超时则提前结束，防止个别难帧拖慢
        if time.perf_counter() - t_start > time_budget_sec:
            break

        refF_patch = refF[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

        # —— 过滤过暗/低纹理 ROI（避免落在阴影/背景）——
        mean = float(refF_patch.mean())
        std = float(refF_patch.std())
        if mean < 8.0 or std < 6.0:  # 可按画面调节
            logging.debug(
                f"[Refine] 丢弃ROI: 过暗/低纹理 mean={mean:.1f}, std={std:.1f}"
            )
            continue

        min_bright, max_bright = 30, 220
        ref_block8 = ref_img.gray_8bit[
            roi.y : roi.y + roi.height, roi.x : roi.x + roi.width
        ]
        mask_patch = (min_bright <= ref_block8 <= max_bright) * np.uint8(255)
        if mask_patch.size > 0:
            mask_patch = cv2.erode(mask_patch, np.ones((3, 3), np.uint8), iterations=1)

        rt = match_roi_zncc_local(
            refF_patch, tgtF, roi, search=search, mask_patch=mask_patch
        )
        if rt is None:
            logging.debug(f"[Refine] 丢弃ROI: 匹配失败")
            continue
        vd, zncc = rt
        if zncc < 0.28:
            logging.debug(f"[Refine] 丢弃ROI: ZNCC过低 zncc={zncc:.2f}")
            continue

        # 丢弃命中搜索边界的解，通常是含糊/假峰
        if (abs(vd.x) >= (search - 0.5)) or (abs(vd.y) >= (search - 0.5)):
            logging.debug(
                f"[Refine] 丢弃ROI: 命中搜索边界 dx={vd.x:.2f}, dy={vd.y:.2f}"
            )
            continue

        # 可选：相位相关做亚像素微调（仅在纹理足够时启用）
        if zncc < 0.60:
            rt = roi_matches_phasecorr(tgtF, refF_patch, roi, search, mask_patch)
            if rt:
                vd = vd + rt[0]
                zncc = max(zncc, rt[1])  # 取更好的响应值

        centers.append(roi.center)
        d_list.append(vd)

        # 结合对比度作为权重，弱纹理权重低

        weights.append(max(1e-3, zncc) * (0.5 + 0.5 * clip(0.0, std / 20.0, 1.0)))
        zncc_list.append(zncc)
        logging.debug(
            f"[Refine] 采纳ROI: zncc={zncc:.2f}, dx={vd.x:.2f}, dy={vd.y:.2f}"
        )

    return centers, d_list, weights, zncc_list


def robust_estimate(
    d_list: list[Vector], weights: list[float]
) -> tuple[Vector, int, float] | None:
    """
    使用Tukey双权IRLS进行稳健估计

    返回: (estimated_shift, inlier_count, score)
    """
    w_arr = np.asarray(weights, dtype=np.float64)
    d_arr = VectorArray(np.array(d_list), safe=False)

    # —— 仅估计平移（无旋转），使用Tukey双权IRLS（单帧内稳健，无跨帧约束）——
    med = np.median(d_arr._arr, axis=0)
    med = Vector.from_ndarray(med, float)
    resid = (d_arr - med).norms()

    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    sigma = 1.4826 * mad + 1e-6
    c = 4.685 * sigma  # Tukey截止

    # 初始权（来自匹配质量）
    base_w = np.clip(w_arr, 1e-6, None)

    # 一次IRLS即可（经验上已足够稳健）
    r = (d_arr - med).norms()
    tukey = np.zeros_like(r)
    m = r < c
    rr = r[m] / (c + 1e-6)
    tukey[m] = (1 - rr**2) ** 2

    ww = base_w * tukey

    if ww.sum() < 1e-6 or (tukey > 0).sum() < 3:
        logging.debug(f"[Refine] IRLS失败，回退霍夫)")
        return None

    t = Vector.from_ndarray(np.average(d_arr._arr, axis=0, weights=ww), float)
    cnt = int((tukey > 0).sum())
    n = len(d_list)
    score = float((cnt / float(n)) * (ww.max() / (base_w.max() + 1e-6)))

    return t, cnt, score


def too_close(a: ROI, b: ROI) -> bool:
    min_w = min(a.width, b.width) // 2
    min_h = min(a.height, b.height) // 2
    return (abs(a.x - b.x) < min_w) and (abs(a.y - b.y) < min_h)


def select_rois(
    refF: NDArray,
    disk_mask: NDArray[np.float32],
    r: float,
    k: int = 16,
    box: int = 128,
    border: int = 10,
    avoid_edge_ratio: float = 0.06,
    ref_gray: NDArray | None = None,
    brightness_range: tuple[int, int] = (30, 220),
) -> list[ROI]:
    """Pick top-k ROI boxes inside the lunar disk by local energy, spaced apart.
    Supports adaptive ROI size based on local brightness if ref_img is provided.
    """
    h, w = refF.shape
    energy = cv2.GaussianBlur(refF, (0, 0), 1.2)
    rois: list[ROI] = []
    # keep away from limb/edge a little bit
    margin = max(border, int(r * avoid_edge_ratio))
    step = max(24, box // 2)  # stride to reduce overlap
    integ = cv2.integral(energy.astype(np.float32))

    def calc_score(roi: ROI) -> float:
        x, y, w, h = roi.x, roi.y, roi.width, roi.height
        sum_rect = float(
            integ[y + h, x + w] - integ[y, x + w] - integ[y + h, x] + integ[y, x]
        )
        return sum_rect / (roi.width * roi.height + 1e-6)

    for y, x in itertools.product(
        range(margin, h - margin - box, step), range(margin, w - margin - box, step)
    ):
        submask = disk_mask[y : y + box, x : x + box]
        if submask.mean() < 0.6:
            continue
        # Determine adaptive box size based on local brightness if ref_img provided

        if ref_gray is None:
            roi = ROI(x, y, box, box)
            roi.score = calc_score(roi)
            rois.append(roi)
            continue

        roi_gray = ref_gray[y : y + box, x : x + box]

        min_bright, max_bright = brightness_range
        fraction = np.mean(min_bright <= roi_gray <= max_bright)
        if fraction < 0.4:
            continue
        # Compute mean brightness in the ROI region
        roi_brightness = np.mean(roi_gray)
        # Map brightness to ROI size: brighter -> larger ROI, darker -> smaller ROI
        min_box = max(64, int(box * 0.5))
        max_box = box

        # Clamp brightness
        brightness = np.clip(roi_brightness, min_bright, max_bright)
        scale = (brightness - min_bright) / (max_bright - min_bright)

        local_box = int(min_box + scale * (max_box - min_box))
        # Adjust local_box to be multiple of 8 for consistency
        local_box = max((local_box // 8) * 8, 64)

        if y + local_box > h - margin or x + local_box > w - margin:
            continue
        submask = disk_mask[y : y + local_box, x : x + local_box]
        if submask.mean() < 0.6:
            continue
        roi = ROI(x, y, local_box, local_box)
        roi.score = calc_score(roi)
        rois.append(roi)

    rois.sort(key=lambda t: t.score, reverse=True)

    pickeds: list[ROI] = []
    for roi in rois:
        if any(too_close(roi, picked) for picked in pickeds):
            continue
        pickeds.append(roi)
        if len(pickeds) >= k:
            break
    return pickeds


def make_multi_roi_shift(
    img: Image,
    ref_img: Image,
    circle: Circle,
    n_rois: int | None = None,
    roi_size: int | None = None,
    search: int | None = None,
    time_budget_sec=1.2,
) -> Vector[float] | None:

    # 自适应/裁剪 ROI 数量与大小，避免过多卷积
    n_rois = clip(8, n_rois if n_rois else int(circle.radius // 70), 24)
    roi_size = clip(64, roi_size if roi_size else int(circle.radius * 0.18), 128)
    search = clip(6, search if search else int(circle.radius * 0.05), 18)

    logging.debug(
        f"[Refine] HxW={ref_img.height}x{ref_img.width}, r≈{circle.radius:.1f}, n_rois={n_rois}, roi_init={roi_size}, search={search}"
    )
    # 1. 准备对齐数据

    refF, tgtF = initial_process(img, ref_img, circle)
    # 能量图直接用高频响应（仅保留高频，避免中频干扰）

    # 2. 选择ROI
    mask = soft_disk_mask(
        ref_img.height, ref_img.width, circle, inner=0.97 * 0.9, outer=0.97
    )
    rois: list[ROI] = select_rois(
        refF,
        mask,
        circle.radius,
        k=n_rois,
        box=roi_size,
        ref_gray=ref_img.gray_8bit,
    )
    logging.debug(f"[Refine] ROI候选数={len(rois)}")

    if not rois:
        return None
    sizes = [roi.width for roi in rois]
    logging.debug(
        f"[Refine] 自适应ROI统计: 平均={np.mean(sizes):.0f}, 最小={np.min(sizes)}, 最大={np.max(sizes)}"
    )

    # 3. 收集ROI匹配结果
    centers, d_list, weights, zncc_list = collect_roi_matches(
        rois, refF, tgtF, ref_img, search, time_budget_sec
    )

    num_centers = len(centers)
    logging.debug(f"[Refine] 参与拟合的ROI数={num_centers}")

    if num_centers < 3:
        # Not enough ROIs; if baseline (Hough) is provided, fall back to it
        logging.debug(f"[Refine] ROI不足")
        return None

    zncc_arr = (
        np.asarray(zncc_list, dtype=np.float64)
        if len(zncc_list)
        else np.array([], dtype=np.float64)
    )
    mean_zncc = float(zncc_arr.mean()) if zncc_arr.size else 0.0

    # 4. 稳健估计
    if not (rt := robust_estimate(d_list, weights)):
        logging.debug(f"[Refine] 估计失败")
        return None

    shift, cnt, score = rt

    # 5. 质量门控

    if cnt < MIN_INLIERS:
        logging.debug(f"[Refine] 触发门控，inliers={cnt}<{MIN_INLIERS}")
        return None

    if mean_zncc < MIN_MEAN_ZNCC:
        logging.debug(
            f"[Refine] 触发门控，mean_zncc={mean_zncc:.2f}<{MIN_MEAN_ZNCC:.2f}"
        )
        return None

    logging.debug(
        f"[Refine] 内点={cnt}/{num_centers}, 平移=({shift.x:.2f},{shift.y:.2f}), meanZNCC={mean_zncc:.2f}, score={score:.3f}"
    )
    logging.info(f"[Refine] score={score:.3f}, inliers={int(cnt)}, roi_init≈{roi_size}")
    return shift


def advanced_detect_shift(
    img: Image, ref_img: Image, ref_circle: Circle
) -> Vector[float] | None:

    shift = make_multi_roi_shift(
        img,
        ref_img,
        ref_circle,
    )
    if not shift:
        return None
    # if (shift - base_shift).norm() > MAX_REFINE_DELTA_PX:
    #     return None

    residual = shift.norm()
    logging.info(f"    [Refine] 残差=Δ{residual:.2f}px")
    if residual > MAX_REFINE_DELTA_PX:
        logging.warning(
            f"    [Refine] 残差过大(Δ={residual:.2f}px > {MAX_REFINE_DELTA_PX:.1f}px)，放弃精配准并保持霍夫平移"
        )
        return None

    logging.info(f"Multi-ROI refine (仅平移)")
    return shift


def masked_phase_corr(
    img: Image, ref_img: Image, circle: Circle, inner: float = 0.90, outer: float = 0.98
) -> Vector[float] | None:
    W, H = ref_img.widthXheight

    mask = soft_disk_mask(H, W, circle, inner=inner, outer=outer)

    rg = ref_img.gray_8bit * mask
    tg = img.gray_8bit * mask

    (dx, dy), response = cv2.phaseCorrelate(tg, rg)
    logging.debug(f"    [PhaseCorr] response={response:.3f}, dx={dx:.2f}, dy={dy:.2f}")
    if response < 0.5:
        return None
    return Vector(dx, dy)


def detect_mask_phase_shift(
    img: Image,
    ref_img: Image,
    ref_circle: Circle,
) -> Vector[float] | None:
    logging.debug(f"Masked PhaseCorr")
    shift = masked_phase_corr(
        img,
        ref_img,
        ref_circle,
    )

    return shift
