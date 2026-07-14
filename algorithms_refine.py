import cv2
import numpy as np
import math
import time

# ---------------- 工具函数 ----------------

def _soft_disk_mask(h, w, cx, cy, r, inner=0.0, outer=0.98):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    m = np.zeros((h, w), np.float32)
    r_in = r * max(0.0, inner)
    r_out = r * min(1.0, outer)
    core = dist <= (r_out * 0.90)
    m[core] = 1.0
    band = (dist > (r_out * 0.90)) & (dist <= r_out)
    if np.any(band):
        t = (dist[band] - r_out * 0.90) / (r_out * 0.10 + 1e-6)
        m[band] = 0.5 * (1 + np.cos(np.pi * t))
    if r_in > 1:
        m[dist < r_in] = 0
    return m


def _clahe_and_bandpass(gray):
    g = gray
    # Ensure 8-bit input
    if g.dtype != np.uint8:
        g = cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Local contrast to suppress global illumination, emphasize small-scale details
    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)

    # High-pass only (remove mid/low frequencies): unsharp mask
    hp = cv2.subtract(cla, cv2.GaussianBlur(cla, (0,0), 6.0))  # sigma ~6 -> keep only high freq

    # Edge emphasis via Laplacian (pure high-frequency operator)
    lap = cv2.Laplacian(cla, cv2.CV_32F, ksize=3)

    # Combine and normalize
    combo = 0.5 * np.abs(hp.astype(np.float32)) + 0.5 * np.abs(lap)
    combo = cv2.normalize(combo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return combo


def _match_roi_zncc_local(ref_patch, tgt_img, x, y, search=12, mask_patch=None):
    """Match ref_patch around (x,y) in tgt_img within +/-search window.
    Uses TM_CCORR_NORMED with optional template mask (same size as ref_patch).
    Returns (dx, dy, score).
    """
    h, w = ref_patch.shape
    H, W = tgt_img.shape

    # target search window centered at the ref patch center
    cx = int(x + w/2)
    cy = int(y + h/2)
    x0 = max(0, cx - w//2 - search)
    y0 = max(0, cy - h//2 - search)
    x1 = min(W, x0 + w + 2*search)
    y1 = min(H, y0 + h + 2*search)
    crop = tgt_img[y0:y1, x0:x1]

    if crop.shape[0] < h or crop.shape[1] < w:
        return 0.0, 0.0, -1.0

    tpl = ref_patch.astype(np.float32)
    win = crop.astype(np.float32)

    # Variance guard: if the (masked) template has almost no texture, skip
    if mask_patch is not None and mask_patch.shape == ref_patch.shape:
        m = (mask_patch.astype(np.float32) / 255.0)
        area = m.sum()
        if area < 16:  # too small effective area
            return 0.0, 0.0, -1.0
        tpl_eff = tpl[m > 0.5]
        if tpl_eff.size == 0 or np.std(tpl_eff) < 1e-3:
            return 0.0, 0.0, -1.0
        method = cv2.TM_CCORR_NORMED
        res = cv2.matchTemplate(win, tpl, method, mask=mask_patch.astype(np.uint8))
    else:
        if np.std(tpl) < 1e-3:
            return 0.0, 0.0, -1.0
        method = cv2.TM_CCORR_NORMED
        res = cv2.matchTemplate(win, tpl, method)

    # Handle NaN/Inf from OpenCV edge cases
    if not np.isfinite(res).all():
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)

    minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
    dx = (x0 + maxloc[0]) - (cx - w//2)
    dy = (y0 + maxloc[1]) - (cy - h//2)
    return float(dx), float(dy), float(maxv)


def _solve_abtx_ty(u, v, dx, dy, w=None):
    """解线性最小二乘：
    dpx = (cosθ-1)*u - sinθ*v + tx = a*u - b*v + tx
    dpy = sinθ*u + (cosθ-1)*v + ty = b*u + a*v + ty
    未强制单位圆约束，先解 a=cosθ-1, b=sinθ；随后归一化恢复 (cosθ, sinθ)。
    支持权重 w。
    返回：theta, tx, ty, cos_t, sin_t
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    dx = np.asarray(dx, dtype=np.float64).ravel()
    dy = np.asarray(dy, dtype=np.float64).ravel()

    n = u.size
    A = np.zeros((2*n, 4), dtype=np.float64)
    b = np.zeros((2*n,), dtype=np.float64)

    # dpx 行
    A[0::2, 0] = u       # a
    A[0::2, 1] = -v      # b
    A[0::2, 2] = 1.0     # tx
    A[0::2, 3] = 0.0     # ty
    b[0::2] = dx

    # dpy 行
    A[1::2, 0] = v       # a
    A[1::2, 1] =  u      # b
    A[1::2, 2] = 0.0     # tx
    A[1::2, 3] = 1.0     # ty
    b[1::2] = dy

    if w is not None:
        w = np.asarray(w, dtype=np.float64).ravel()
        w = np.clip(w, 1e-6, None)
        W = np.sqrt(np.repeat(w, 2))
        A = A * W[:, None]
        b = b * W

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, tx, ty = x

    cos_t = a + 1.0
    sin_t = b_
    # 归一化到单位圆，避免轻微缩放/剪切
    s = math.hypot(cos_t, sin_t)
    if s <= 1e-12:
        cos_t, sin_t = 1.0, 0.0
    else:
        cos_t /= s
        sin_t /= s

    theta = math.atan2(sin_t, cos_t)
    return float(theta), float(tx), float(ty), float(cos_t), float(sin_t)

from utils_common import log as _uilog

# simple debug bridge
def _dbg(msg, debug_cb=None):
    try:
        if debug_cb is None:
            print(msg)
        else:
            debug_cb(msg)
    except Exception:
        pass


def _select_rois(energy_map, disk_mask, r, k=16, box=128, border=10, avoid_edge_ratio=0.06, ref_img=None, brightness_range=(30,220)):
    """Pick top-k ROI boxes inside the lunar disk by local energy, spaced apart.
    Supports adaptive ROI size based on local brightness if ref_img is provided.
    """
    h, w = energy_map.shape
    rois = []
    # keep away from limb/edge a little bit
    margin = max(border, int(r * avoid_edge_ratio))
    step = max(24, box // 2)  # stride to reduce overlap
    integ = cv2.integral(energy_map.astype(np.float32))

    def sum_rect(x0, y0, bw, bh):
        # integral image sum over rectangle [x0,x0+bw) x [y0,y0+bh)
        return float(integ[y0+bh, x0+bw] - integ[y0, x0+bw] - integ[y0+bh, x0] + integ[y0, x0])

    for y in range(margin, h - margin - box, step):
        for x in range(margin, w - margin - box, step):
            submask = disk_mask[y:y+box, x:x+box]
            if submask.mean() < 0.6:
                continue
            # Determine adaptive box size based on local brightness if ref_img provided
            local_box = box
            if ref_img is not None:
                roi_block = ref_img[y:y+box, x:x+box]
                bmin, bmax = brightness_range
                frac_in = np.mean((roi_block >= bmin) & (roi_block <= bmax))
                if frac_in < 0.4:
                    continue
                # Compute mean brightness in the ROI region
                roi_brightness = np.mean(roi_block)
                # Map brightness to ROI size: brighter -> larger ROI, darker -> smaller ROI
                min_box = max(64, int(box * 0.5))
                max_box = box
                bmin, bmax = brightness_range
                # Clamp brightness
                b = np.clip(roi_brightness, bmin, bmax)
                scale = (b - bmin) / (bmax - bmin)
                local_box = int(min_box + scale * (max_box - min_box))
                # Adjust local_box to be multiple of 8 for consistency
                local_box = (local_box // 8) * 8
                if local_box < 64:
                    local_box = 64
                if y + local_box > h - margin or x + local_box > w - margin:
                    continue
                submask = disk_mask[y:y+local_box, x:x+local_box]
                if submask.mean() < 0.6:
                    continue
                score = sum_rect(x, y, local_box, local_box) / (local_box * local_box + 1e-6)
                rois.append((score, x, y, local_box, local_box))
            else:
                score = sum_rect(x, y, box, box) / (box * box + 1e-6)
                rois.append((score, x, y, box, box))

    rois.sort(key=lambda t: t[0], reverse=True)

    picked = []
    for sc, x, y, bw, bh in rois:
        too_close = False
        for _sc2, x2, y2, bw2, bh2 in picked:
            min_sep_x = min(bw, bw2) // 2
            min_sep_y = min(bh, bh2) // 2
            if abs(x - x2) < min_sep_x and abs(y - y2) < min_sep_y:
                too_close = True
                break
        if not too_close:
            picked.append((sc, x, y, bw, bh))
        if len(picked) >= k:
            break
    return picked

def refine_alignment_multi_roi(
    ref_gray, tgt_gray, cx, cy, r,
    n_rois=16, roi_size=128, search=12,
    base_shift=None, max_refine_delta_px=6.0, min_inliers=6, min_mean_zncc=0.55,
    use_phasecorr=True, use_ecc=False,
    time_budget_sec=1.2,
    debug_cb=None
):
    """
    输入：已做圆心粗配准的 ref_gray / tgt_gray，以及参考圆心(cx,cy)与半径 r
    输出：(M2x3, score, n_inliers, theta_deg)
    特性：
      • 仅估计残余平移（无旋转、无缩放）
      • ROI 只在局部窗口内做匹配，避免全图卷积导致卡顿
      • 内置时间预算，个别困难帧自动提前结束并回退上层策略
      • 过滤过暗/低纹理 ROI，提升稳健性
      • 当 refine 结果偏离过大或质量较低时，回退到霍夫基线 (base_shift)
    失败：返回 (None, 0.0, 0, 0.0)
    """
    H, W = ref_gray.shape
    _dbg(f"[Refine] HxW={H}x{W}, r≈{r:.1f}, n_rois={n_rois}, roi_init={roi_size}, search={search}", debug_cb)
    _dbg("[Refine] 使用盘内纹理 ROI；不使用绝对亮度门控", debug_cb)

    # 软盘遮罩，仅保留月盘内纹理
    mask = _soft_disk_mask(H, W, cx, cy, r, inner=0.0, outer=0.97)

    # 归一化转为8-bit，用于能量图和ROI选择
    ref_gray_8u = ref_gray
    if ref_gray.dtype != np.uint8:
        ref_gray_8u = cv2.normalize(ref_gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tgt_gray_8u = tgt_gray
    if tgt_gray.dtype != np.uint8:
        tgt_gray_8u = cv2.normalize(tgt_gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 梯度/DoG 预处理（抗亮度变化），用于匹配阶段
    refF = _clahe_and_bandpass(ref_gray)
    tgtF = _clahe_and_bandpass(tgt_gray)

    # 只保留盘内
    refF = (refF.astype(np.float32) * mask).astype(np.uint8)
    tgtF = (tgtF.astype(np.float32) * mask).astype(np.uint8)

    # 能量图直接用高频响应（仅保留高频，避免中频干扰）
    energy = cv2.GaussianBlur(refF, (0, 0), 1.2)

    # 自适应/裁剪 ROI 数量与大小，避免过多卷积
    n_rois = int(np.clip(n_rois if n_rois else (r/70), 8, 24))
    roi_size = int(np.clip(roi_size if roi_size else (r*0.12), 64, 128))
    search = int(np.clip(search if search else (r*0.05), 6, 18))

    # 月食阶段亮度可跨很大范围；只依据盘内高频纹理选块，不以绝对灰度排除 ROI。
    rois = _select_rois(energy, mask, r, k=n_rois, box=roi_size)
    _dbg(f"[Refine] ROI候选数={len(rois)}", debug_cb)
    sizes = [bw for (_sc, _x, _y, bw, bh) in rois]
    if sizes:
        _dbg(f"[Refine] 自适应ROI统计: 平均={np.mean(sizes):.0f}, 最小={np.min(sizes)}, 最大={np.max(sizes)}", debug_cb)
    if not rois:
        return None, 0.0, 0, 0.0

    # 收集 ROI 局部位移向量（dx_i, dy_i）及其中心（xi, yi）
    centers = []
    zncc_list = []
    dx_list, dy_list, weights = [], [], []

    t_start = time.perf_counter()

    for sc, x, y, bw, bh in rois:
        # 时间预算：超时则提前结束，防止个别难帧拖慢
        if time.perf_counter() - t_start > time_budget_sec:
            break

        ref_patch = refF[y:y + bh, x:x + bw]

        # —— 过滤过暗/低纹理 ROI（避免落在阴影/背景）——
        m = float(ref_patch.mean()); s = float(ref_patch.std())
        if m < 8.0 or s < 6.0:   # 可按画面调节
            _dbg(f"[Refine] 丢弃ROI: 过暗/低纹理 mean={m:.1f}, std={s:.1f}", debug_cb)
            continue

        # 几何月盘遮罩，而非绝对亮度遮罩：阴影中的有效地形不应被误删。
        mask_patch = (mask[y:y + bh, x:x + bw] > 0.5).astype(np.uint8) * 255
        if mask_patch.size > 0:
            mask_patch = cv2.erode(mask_patch, np.ones((3,3), np.uint8), iterations=1)

        dx, dy, zncc = _match_roi_zncc_local(ref_patch, tgtF, x, y, search=search, mask_patch=mask_patch)
        if zncc < 0.28:
            _dbg(f"[Refine] 丢弃ROI: ZNCC过低 zncc={zncc:.2f}", debug_cb)
            continue

        # 丢弃命中搜索边界的解，通常是含糊/假峰
        if (abs(dx) >= (search - 0.5)) or (abs(dy) >= (search - 0.5)):
            _dbg(f"[Refine] 丢弃ROI: 命中搜索边界 dx={dx:.2f}, dy={dy:.2f}", debug_cb)
            continue

        # 可选：相位相关做亚像素微调（仅在纹理足够时启用）
        if use_phasecorr:
            h, w = ref_patch.shape
            # ZNCC 已给出整数峰值；在该峰值位置取同尺寸目标块，再仅估计
            # 亚像素残差，不能从原始位置重复估计整段位移。
            xp = int(round(x + dx))
            yp = int(round(y + dy))
            xe = min(tgtF.shape[1], xp + w)
            ye = min(tgtF.shape[0], yp + h)
            tgt_patch = tgtF[yp:ye, xp:xe]
            if tgt_patch.shape == ref_patch.shape:
                rp = ref_patch.astype(np.float32)
                tp = tgt_patch.astype(np.float32)
                if mask_patch is not None and mask_patch.shape == ref_patch.shape:
                    m = (mask_patch.astype(np.float32) / 255.0)
                    rp = rp * m
                    tp = tp * m
                    # 纹理/有效面积检查
                    eff = rp[m > 0.5]
                    if eff.size == 0 or np.std(eff) < 1e-3:
                        tgt_patch = None
                if tgt_patch is not None:
                    (dx2, dy2), resp = cv2.phaseCorrelate(rp, tp)
                    if (zncc >= 0.60) and (resp is not None) and np.isfinite(resp) and (resp > 0.20) and (abs(dx2) < 2.5) and (abs(dy2) < 2.5):
                        dx += dx2; dy += dy2
                        zncc = max(zncc, float(resp))

        centers.append((x + bw / 2.0, y + bh / 2.0))
        dx_list.append(float(dx))
        dy_list.append(float(dy))
        # 结合对比度作为权重，弱纹理权重低
        weights.append(float(max(1e-3, zncc) * (0.5 + 0.5*np.clip(s/20.0, 0.0, 1.0))))
        zncc_list.append(float(zncc))
        _dbg(f"[Refine] 采纳ROI: zncc={zncc:.2f}, dx={dx:.2f}, dy={dy:.2f}", debug_cb)

    n = len(centers)
    _dbg(f"[Refine] 参与拟合的ROI数={n}", debug_cb)
    if n < 3:
        # Not enough ROIs; if baseline (Hough) is provided, fall back to it
        if base_shift is not None:
            tx_h, ty_h = float(base_shift[0]), float(base_shift[1])
            M_h = np.array([[1.0, 0.0, tx_h],
                            [0.0, 1.0, ty_h]], dtype=np.float32)
            _dbg(f"[Refine] ROI不足，回退粗对齐残差: shift=({tx_h:.2f},{ty_h:.2f})", debug_cb)
            return M_h, 0.0, 0, 0.0
        return None, 0.0, 0, 0.0

    centers = np.asarray(centers, dtype=np.float64)
    dx_arr = np.asarray(dx_list, dtype=np.float64)
    dy_arr = np.asarray(dy_list, dtype=np.float64)
    w_arr  = np.asarray(weights, dtype=np.float64)

    zncc_arr = np.asarray(zncc_list, dtype=np.float64) if len(zncc_list) else np.array([], dtype=np.float64)
    mean_zncc = float(zncc_arr.mean()) if zncc_arr.size else 0.0

    # —— 仅估计平移（无旋转），使用Tukey双权IRLS（单帧内稳健，无跨帧约束）——
    medx = np.median(dx_arr)
    medy = np.median(dy_arr)
    resid = np.hypot(dx_arr - medx, dy_arr - medy)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    sigma = 1.4826 * mad + 1e-6
    c = 4.685 * sigma  # Tukey截止

    # 初始权（来自匹配质量）
    base_w = np.clip(w_arr, 1e-6, None)

    # 一次IRLS即可（经验上已足够稳健）
    r = np.hypot(dx_arr - medx, dy_arr - medy)
    tukey = np.zeros_like(r)
    m = r < c
    rr = r[m] / (c + 1e-6)
    tukey[m] = (1 - rr**2)**2

    ww = base_w * tukey
    if ww.sum() < 1e-6 or (tukey > 0).sum() < 3:
        if base_shift is not None:
            tx_h, ty_h = float(base_shift[0]), float(base_shift[1])
            M_h = np.array([[1.0, 0.0, tx_h],
                            [0.0, 1.0, ty_h]], dtype=np.float32)
            _dbg(f"[Refine] IRLS失败，回退粗对齐残差: shift=({tx_h:.2f},{ty_h:.2f})", debug_cb)
            return M_h, 0.0, 0, 0.0
        return None, 0.0, 0, 0.0

    tx = float(np.average(dx_arr, weights=ww))
    ty = float(np.average(dy_arr, weights=ww))

    cnt = int((tukey > 0).sum())
    score = float((cnt / float(n)) * (ww.max() / (base_w.max() + 1e-6)))

    # --- Quality gate & baseline fallback (no inter-frame constraint) ---
    if base_shift is not None:
        dx_h, dy_h = float(base_shift[0]), float(base_shift[1])
        delta = math.hypot(tx - dx_h, ty - dy_h)
        bad = False
        reasons = []
        if cnt < int(min_inliers):
            bad = True
            reasons.append(f"inliers={cnt}<{int(min_inliers)}")
        if mean_zncc < float(min_mean_zncc):
            bad = True
            reasons.append(f"mean_zncc={mean_zncc:.2f}<{float(min_mean_zncc):.2f}")
        if delta > float(max_refine_delta_px):
            bad = True
            reasons.append(f"|refine-hough|={delta:.2f}px>{float(max_refine_delta_px):.2f}")
        if bad:
            # Fall back to the residual baseline (normally identity after coarse alignment).
            tx, ty = dx_h, dy_h
            _dbg("[Refine] 触发门控，回退粗对齐残差: " + "; ".join(reasons) + f" -> ({tx:.2f},{ty:.2f})", debug_cb)

    # 最终 2x3 仿射矩阵（仅平移）
    M = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty]], dtype=np.float32)

    theta_deg = 0.0
    _dbg(f"[Refine] 内点={cnt}/{n}, 平移=({tx:.2f},{ty:.2f}), meanZNCC={mean_zncc:.2f}, score={score:.3f}", debug_cb)
    return M, score, int(cnt), theta_deg