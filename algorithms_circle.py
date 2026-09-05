import math
import numpy as np
import cv2

from utils_common import (
    imread_unicode, safe_join, log, force_garbage_collection
)

# ============== 预处理 & 质量评估 ==============

def adaptive_preprocessing(image, brightness_mode="auto"):
    """将图像转换为适合圆检测的灰度，并做适度增强。返回 (processed_gray, brightness_mode)"""
    if image.ndim > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    mean_brightness = float(np.mean(gray))
    if brightness_mode == "auto":
        if mean_brightness > 140:
            brightness_mode = "bright"
        elif mean_brightness < 70:
            brightness_mode = "dark"
        else:
            brightness_mode = "normal"

    if brightness_mode == "bright":
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    elif brightness_mode == "dark":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return filtered, brightness_mode


def evaluate_circle_quality(image_gray, circle):
    """对检测到的圆做一个稳定的质量打分，越大越好（0~100）。"""
    try:
        cx, cy, radius = int(circle[0]), int(circle[1]), int(circle[2])
        h, w = image_gray.shape[:2]

        angles = np.linspace(0, 2 * np.pi, 48)
        edge_strengths = []
        for angle in angles:
            ix = int(cx + (radius - 2) * np.cos(angle))
            iy = int(cy + (radius - 2) * np.sin(angle))
            ox = int(cx + (radius + 2) * np.cos(angle))
            oy = int(cy + (radius + 2) * np.sin(angle))
            if 0 <= ix < w and 0 <= iy < h and 0 <= ox < w and 0 <= oy < h:
                inner_val = float(image_gray[iy, ix])
                outer_val = float(image_gray[oy, ox])
                edge_strengths.append(abs(outer_val - inner_val))

        if not edge_strengths:
            return 0.0

        avg_edge = float(np.mean(edge_strengths))
        consistency = 1.0 / (1.0 + np.std(edge_strengths) / max(1.0, avg_edge))
        score = avg_edge * consistency
        return float(min(100.0, score))
    except Exception:
        return 0.0

# ============== 高光裁剪和星点抑制辅助 ==============
def _clip_highlights(gray, pct=99.8):
    """Clip very bright highlights (glare/bloom) to a percentile to help Hough/RANSAC."""
    g = gray.astype(np.float32)
    cap = np.percentile(g, pct)
    if cap <= 0:
        return gray
    g = np.minimum(g, cap)
    g = g / (cap + 1e-6) * 255.0
    return g.astype(np.uint8)

def _remove_stars_small(gray):
    """Suppress point-like stars/noise while preserving lunar rim."""
    # Top-hat to remove small bright dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    g = cv2.subtract(gray, tophat)
    # Gentle median to clean salt-pepper
    g = cv2.medianBlur(g, 3)
    return g

# ============== 稳健外缘 RANSAC（用于血月/缺口） ==============

def _fit_circle_least_squares(points):
    if len(points) < 3:
        return None
    A = np.c_[2*points[:,0], 2*points[:,1], np.ones(points.shape[0])]
    b = points[:,0]**2 + points[:,1]**2
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = x[0], x[1]
        r = math.sqrt(x[2] + cx*cx + cy*cy)
        return (float(cx), float(cy), float(r))
    except Exception:
        return None

def _fit_circle_ransac(points, iterations=120, threshold=2.0, min_inliers=40):
    if len(points) < 3:
        return None
    best_circle = None
    best_inliers = 0
    N = len(points)
    # A fixed local generator makes one frame reproducible regardless of which
    # files were processed before it. Never consume NumPy's process-global RNG.
    rng = np.random.default_rng(0)
    for _ in range(iterations):
        try:
            idx = rng.choice(N, 3, replace=False)
        except ValueError:
            return None
        tri = points[idx]
        cand = _fit_circle_least_squares(tri)
        if cand is None:
            continue
        cx, cy, r = cand
        d = np.sqrt((points[:,0]-cx)**2 + (points[:,1]-cy)**2)
        inliers = np.sum(np.abs(d - r) < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            mask = (np.abs(d - r) < threshold)
            best_circle = _fit_circle_least_squares(points[mask])
    if best_circle is not None and best_inliers >= min_inliers:
        return best_circle
    return None

def _edge_points_outer_rim(gray, prev_circle=None):
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    if prev_circle is not None:
        cx, cy, r = prev_circle
        d = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
        ring_mask = (d > r*0.85) & (d < r*1.15)
        pts = pts[ring_mask]
        if len(pts) == 0:
            return None

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        keep = []
        for x, y in pts:
            vx, vy = x - cx, y - cy
            nrm = math.hypot(vx, vy) + 1e-6
            nx, ny = vx/nrm, vy/nrm
            gxx = gx[int(y), int(x)]
            gyy = gy[int(y), int(x)]
            gn = math.hypot(gxx, gyy) + 1e-6
            gx_n, gy_n = gxx/gn, gyy/gn
            if (gx_n*nx + gy_n*ny) > 0.2:
                keep.append([x, y])
        if len(keep) >= 30:
            pts = np.asarray(keep, dtype=np.float32)
        elif len(pts) < 30:
            return None

    return pts

def detect_circle_robust(gray, prev_circle=None):
    pts = _edge_points_outer_rim(gray, prev_circle)
    if pts is None or len(pts) < 30:
        return prev_circle
    cand = _fit_circle_ransac(pts)
    if cand is None:
        return prev_circle
    cx, cy, r = cand
    vec = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    span = np.ptp(vec)
    if prev_circle is not None and span < (2*np.pi/3.0):  # <120°
        cx_prev, cy_prev, r_prev = prev_circle
        cand = (cx, cy, r_prev)
    return (float(cand[0]), float(cand[1]), float(cand[2]))

# ============== 遮罩相位相关（亚像素平移微调） ==============

def masked_phase_corr(ref_gray, tgt_gray, cx, cy, r):
    H, W = ref_gray.shape
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    mask = (dist <= r*0.98).astype(np.float32)
    band = (dist >= r*0.90) & (dist <= r*0.98)
    t = (dist[band] - r*0.90) / (r*0.08 + 1e-6)
    mask[band] = 0.5*(1 + np.cos(np.pi*(1 - t)))

    rg = (ref_gray * mask).astype(np.float32)
    tg = (tgt_gray * mask).astype(np.float32)

    (dx, dy), _ = cv2.phaseCorrelate(rg, tg)
    return float(dx), float(dy)

# ============== 辅助：粗估 & 环形 ROI（抑制星点/加速霍夫） ==============

def _rough_center_radius(gray, min_r, max_r):
    g = cv2.GaussianBlur(gray, (0, 0), 2.0)
    # Use adaptive + Otsu fallback to handle glare/crescent
    thr = max(10, int(np.mean(g) + 0.3 * np.std(g)))
    _, bw1 = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    _, bw2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_adap = cv2.adaptiveThreshold(g.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 51, -5)
    bw = cv2.max(bw1, cv2.max(bw2, bw_adap))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    if r < min_r * 0.6 or r > max_r * 1.6:
        return None
    return float(cx), float(cy), float(r)

def _ring_mask(h, w, cx, cy, r, inner=0.70, outer=1.15):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    m = ((dist >= r * inner) & (dist <= r * outer)).astype(np.uint8) * 255
    return m

# ============== 边缘亚像素细化（统一在返回前做一次） ==============
def _refine_circle_subpixel(det_gray, circle, search_px=3, samples=180):
    """在已检测到的圆附近做一次亚像素细化。
    det_gray: 用于检测的灰度图；circle: (cx,cy,r)
    返回 (cx,cy,r) 或原 circle（失败）。"""
    try:
        import numpy as _np
        import cv2 as _cv2
        h, w = det_gray.shape[:2]
        cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
        if r <= 3:
            return circle
        g = _cv2.GaussianBlur(det_gray, (3,3), 0)
        gx = _cv2.Sobel(g, _cv2.CV_32F, 1, 0, ksize=3)
        gy = _cv2.Sobel(g, _cv2.CV_32F, 0, 1, ksize=3)
        thetas = _np.linspace(0, 2*_np.pi, samples, endpoint=False)
        pts = []
        for th in thetas:
            nx, ny = _np.cos(th), _np.sin(th)
            best_s, best_val = 0.0, -1e9
            for s in range(-int(search_px), int(search_px)+1):
                x = int(round(cx + (r + s) * nx))
                y = int(round(cy + (r + s) * ny))
                if 0 <= x < w and 0 <= y < h:
                    val = float(gx[y,x]*nx + gy[y,x]*ny)
                    if val > best_val:
                        best_val = val
                        best_s = s
            x = cx + (r + best_s) * nx
            y = cy + (r + best_s) * ny
            if 0 <= x < w and 0 <= y < h:
                pts.append([x, y])
        if len(pts) < max(48, samples//3):
            return circle
        P = _np.asarray(pts, dtype=_np.float32)
        A = _np.c_[2*P[:,0], 2*P[:,1], _np.ones(P.shape[0])]
        b = P[:,0]**2 + P[:,1]**2
        x, *_ = _np.linalg.lstsq(A, b, rcond=None)
        cx2, cy2 = float(x[0]), float(x[1])
        r2 = float(_np.sqrt(max(1e-6, x[2] + cx2*cx2 + cy2*cy2)))
        if _np.hypot(cx2-cx, cy2-cy) > max(2.5, 0.01*r) or abs(r2-r) > max(2.5, 0.01*r):
            return circle
        return (cx2, cy2, r2)
    except Exception:
        return circle

# ============== UI 调参可视化：分析区域掩膜（仅供显示） ==============
def build_analysis_mask(img_gray, brightness_min=3/255.0, min_radius=None, max_radius=None):
    """
    仅供 UI 调参窗口显示“分析区域”用：
    - uint8 归一化 -> 轻度去噪
    - Otsu 阈值 与 亮度下限并联
    - 形态学开运算清点
    - 仅保留最大连通域
    返回 bool(H,W)。不影响主流程检测。
    """
    try:
        g = img_gray
        if g is None:
            return np.zeros((1,1), dtype=bool)
        if g.dtype != np.uint8:
            g = cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        g = cv2.GaussianBlur(g, (3,3), 0)
        _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        floor_t = max(1, int(round(float(brightness_min) * 255.0)))
        _, floor = cv2.threshold(g, floor_t, 255, cv2.THRESH_BINARY)
        m = cv2.bitwise_and(otsu, floor)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.zeros_like(m, dtype=bool)
        c = max(cnts, key=cv2.contourArea)
        keep = np.zeros_like(m)
        cv2.drawContours(keep, [c], -1, 255, thickness=cv2.FILLED)
        return keep.astype(bool)
    except Exception:
        # 兜底：整图 False
        shape = (1,1) if img_gray is None else img_gray.shape[:2]
        return np.zeros(shape, dtype=bool)

# 兼容 UI 中的优先调用名
build_analysis_mask_ui = build_analysis_mask

# ============== 主检测（供 pipeline 调用） ==============

def detect_circle_phd2_enhanced(image, min_radius, max_radius, param1, param2, strong_denoise=False, prev_circle=None):
    """
    返回: (best_circle [cx, cy, r], processed_gray, quality, method_str, brightness_mode)
    失败: (None, processed_or_input_gray, 0, 'error', 'unknown')
    """
    try:
        processed, brightness_mode = adaptive_preprocessing(image, "auto")
        # 可选：强力降噪（仅影响检测，不影响最终成片）
        if strong_denoise:
            try:
                # fast NLM 能在强噪场景下保持边缘
                processed = cv2.fastNlMeansDenoising(processed, None, h=10, templateWindowSize=7, searchWindowSize=21)
                # 轻度中值进一步压盐胡椒
                processed = cv2.medianBlur(processed, 3)
            except Exception:
                pass
        best_circle, best_score, detection_method = None, 0.0, "none"

        # Use a detection-optimized copy to make Hough/RANSAC more stable on glare/bloom frames
        processed_det = _remove_stars_small(_clip_highlights(processed, pct=99.8))
        H, W = processed_det.shape
        proc_for_hough = processed_det

        # —— 粗估中心半径，构建环形 ROI —— #
        est = _rough_center_radius(processed_det, min_radius, max_radius)
        if est is not None:
            cx0, cy0, r0 = est
            ring = _ring_mask(H, W, cx0, cy0, r0, inner=0.70, outer=1.15)
            proc_for_hough = cv2.bitwise_and(processed_det, processed_det, mask=ring)
        # —— 若给出上一帧圆心半径，合并一个“历史先验”环形 ROI —— #
        if prev_circle is not None and all(np.isfinite(prev_circle)):
            try:
                pcx, pcy, pr = float(prev_circle[0]), float(prev_circle[1]), float(prev_circle[2])
                inner = max(0.70, min(0.85, (min_radius/max(pr,1e-6)) * 0.9))
                outer = min(1.30, max(1.15, (max_radius/max(pr,1e-6)) * 1.05))
                ring_prev = _ring_mask(H, W, pcx, pcy, pr, inner=inner, outer=outer)
                proc_for_hough = cv2.bitwise_and(proc_for_hough, proc_for_hough, mask=ring_prev)
            except Exception:
                pass
        else:
            max_side = max(H, W)
            if max_side > 1800:
                scale = 1800.0 / max_side
                small = cv2.resize(processed_det, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
                s_min = max(1, int(min_radius * scale))
                s_max = max(s_min + 1, int(max_radius * scale))
                try:
                    sc = cv2.HoughCircles(
                        small, cv2.HOUGH_GRADIENT,
                        dp=1.2, minDist=small.shape[0] // 2,
                        param1=param1, param2=max(param2 - 5, 10),
                        minRadius=s_min, maxRadius=s_max,
                    )
                    if sc is not None:
                        c = sc[0][0]
                        circle = np.array([c[0] / scale, c[1] / scale, c[2] / scale], dtype=np.float32)
                        q = evaluate_circle_quality(processed, circle) * 1.02
                        if q > best_score:
                            best_score, best_circle = q, circle
                            detection_method = "缩放霍夫(thumb)"
                except Exception:
                    pass

        # —— 稳健外缘 RANSAC —— #
        try:
            robust = detect_circle_robust(processed_det, None)
            if robust is not None:
                q = evaluate_circle_quality(processed, robust) * 1.05
                if q > best_score:
                    best_score = q
                    best_circle = np.array(robust, dtype=np.float32)
                    detection_method = "稳健外缘RANSAC"
        except Exception:
            pass

        # —— 标准霍夫（在 ROI 上） —— #
        try:
            height, _ = processed_det.shape
            circles = cv2.HoughCircles(
                proc_for_hough, cv2.HOUGH_GRADIENT,
                dp=1, minDist=height,
                param1=param1, param2=param2,
                minRadius=min_radius, maxRadius=max_radius,
            )
            if circles is not None:
                for c in circles[0]:
                    q = evaluate_circle_quality(processed, c)
                    if q > best_score:
                        best_score, best_circle = q, c
                        detection_method = f"标准霍夫(P1={param1},P2={param2})"
        except Exception:
            pass

        # —— 自适应参数霍夫（在 ROI 上） —— #
        if best_score < 15:
            try:
                if brightness_mode == "bright":
                    ap1, ap2 = param1 + 20, max(param2 - 5, 10)
                elif brightness_mode == "dark":
                    ap1, ap2 = max(param1 - 15, 20), max(param2 - 10, 5)
                else:
                    ap1, ap2 = param1, max(param2 - 8, 8)

                circles2 = cv2.HoughCircles(
                    proc_for_hough, cv2.HOUGH_GRADIENT,
                    dp=1.2, minDist=height // 2,
                    param1=ap1, param2=ap2,
                    minRadius=min_radius, maxRadius=max_radius,
                )
                if circles2 is not None:
                    for c in circles2[0]:
                        q = evaluate_circle_quality(processed, c)
                        if q > best_score:
                            best_score, best_circle = q, c
                            detection_method = f"自适应霍夫(P1={ap1},P2={ap2})"
            except Exception:
                pass

        # —— 轮廓备选 —— #
        if best_score < 10:
            try:
                mean_val = float(np.mean(processed_det))
                tv = max(50, int(mean_val * 0.7))
                _, binary = cv2.threshold(processed_det, tv, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if min_radius ** 2 * np.pi * 0.3 <= area <= max_radius ** 2 * np.pi * 2.0:
                        (cx, cy), r = cv2.minEnclosingCircle(cnt)
                        if min_radius <= r <= max_radius:
                            c = np.array([cx, cy, r])
                            q = evaluate_circle_quality(processed, c) * 0.7
                            if q > best_score:
                                best_score, best_circle = q, c
                                detection_method = f"轮廓检测(T={tv})"
            except Exception:
                pass

        # —— padding-based fallback —— #
        def _touches_border(circle, w, h, margin=5):
            if circle is None:
                return True
            cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
            return (cx - r < margin) or (cy - r < margin) or (cx + r > w - margin) or (cy + r > h - margin)

        need_pad = (best_circle is None) or (best_score < 10) or _touches_border(best_circle, W, H)
        if need_pad:
            pad = int(max(32, round(max_radius * 1.2)))
            processed_pad = cv2.copyMakeBorder(
                processed_det, pad, pad, pad, pad,
                borderType=cv2.BORDER_CONSTANT, value=0
            )
            # Use constant black padding to avoid mirrored ghosts influencing Hough
            est_p = _rough_center_radius(processed_pad, int(min_radius*1.1), int(max_radius*1.1))
            if est_p is not None:
                ring_p = _ring_mask(processed_pad.shape[0], processed_pad.shape[1], est_p[0], est_p[1], est_p[2], inner=0.70, outer=1.15)
                proc_pad_for_hough = cv2.bitwise_and(processed_pad, processed_pad, mask=ring_p)
            else:
                proc_pad_for_hough = processed_pad
            circles_p = cv2.HoughCircles(
                proc_pad_for_hough, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=processed_pad.shape[0]//2,
                param1=max(param1, 20), param2=max(param2-5, 8),
                minRadius=int(min_radius*1.1), maxRadius=int(max_radius*1.1)
            )
            if circles_p is None:
                robust_p = detect_circle_robust(processed_pad, None)
                if robust_p is not None:
                    circles_list = [np.array(robust_p, dtype=np.float32)]
                else:
                    circles_list = []
            else:
                circles_list = [c for c in circles_p[0]]
            for c_p in circles_list:
                # Build a matching padded version of the original processed (for scoring)
                scored_pad = processed_pad  # using detection pad for speed; acceptable because only ranking
                q_pad = evaluate_circle_quality(scored_pad, c_p)
                cxp, cyp, rp = float(c_p[0]), float(c_p[1]), float(c_p[2])
                hh, ww = processed_pad.shape[:2]
                yy, xx = np.ogrid[:hh, :ww]
                mask = ((xx - cxp)**2 + (yy - cyp)**2 <= rp**2)
                # 对应原图区域
                crop = mask[pad:pad+H, pad:pad+W]
                visible_ratio = float(np.count_nonzero(crop)) / (np.pi*rp*rp + 1e-6)
                q_adj = max(10.0, float(q_pad) * np.sqrt(max(0.05, min(1.0, visible_ratio))))
                if q_adj > best_score:
                    # 映射回原图坐标
                    best_circle = np.array([cxp - pad, cyp - pad, rp], dtype=np.float32)
                    best_score = q_adj
                    detection_method = f"{detection_method}+pad0({pad})" if detection_method else f"pad0({pad})"

        # —— 最终半径窗口一致性检查（严格遵守 UI 设定） —— #
        try:
            if best_circle is not None:
                rr = float(best_circle[2])
                if not (float(min_radius) <= rr <= float(max_radius)):
                    # 在严格窗口内再做一次快速霍夫重试
                    height, width = processed_det.shape
                    _minDist_coreS = max(16, min(height, width) // 4)
                    scS = cv2.HoughCircles(
                        processed_det, cv2.HOUGH_GRADIENT,
                        dp=1.2, minDist=_minDist_coreS,
                        param1=max(param1, 20), param2=max(param2-5, 8),
                        minRadius=int(max(1, min_radius)), maxRadius=int(max_radius)
                    )
                    if scS is not None:
                        bestS = None; bestQS = -1.0
                        for c in scS[0]:
                            if float(min_radius) <= float(c[2]) <= float(max_radius):
                                q = float(evaluate_circle_quality(processed, c))
                                if q > bestQS:
                                    bestQS = q; bestS = c
                        if bestS is not None:
                            best_circle = bestS
                            best_score = bestQS
                            detection_method = f"{detection_method}|strict-window"
        except Exception:
            pass

        return best_circle, processed, best_score, detection_method, brightness_mode

    except Exception:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return None, gray, 0, "error", "unknown"
