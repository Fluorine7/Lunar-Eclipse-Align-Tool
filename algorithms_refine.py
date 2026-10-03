"""Pair-local texture refinement AFTER independent limb alignment.

Measured displacement is target relative to reference; the returned matrix is
its INVERSE, applied to the target. No previous frame, clock deadline or mutable
image cache participates. All filters operate on analysis copies only.
"""
import math

import cv2
import numpy as np


def _prepare(gray):
    if gray.ndim != 2 or not np.isfinite(gray).all():
        raise ValueError('纹理微调需要有限值的二维灰度图')
    raw = gray.astype(np.float32)
    low, high = np.percentile(raw, (1, 99.5))
    if high <= low:
        return None
    valid = np.ones(gray.shape, bool)
    if np.issubdtype(gray.dtype, np.integer):
        valid &= (gray > 0) & (gray < np.iinfo(gray.dtype).max)
    raw = (raw - float(low)) / float(high - low)  # No clipping or per-tile CLAHE.
    fine = cv2.GaussianBlur(raw, (0, 0), 1.2) - cv2.GaussianBlur(raw, (0, 0), 6.)
    coarse = cv2.GaussianBlur(raw, (0, 0), 2.4) - cv2.GaussianBlur(raw, (0, 0), 9.)
    noise = raw - cv2.GaussianBlur(raw, (0, 0), .7)
    return fine, coarse, noise, valid


def _zncc(a, b):
    a = a.astype(np.float64) - float(np.mean(a))
    b = b.astype(np.float64) - float(np.mean(b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / norm) if norm > 1e-12 else -1.


def _patch(image, x, y, width, height):
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        return None
    return cv2.getRectSubPix(image, (width, height), (x + (width - 1) / 2, y + (height - 1) / 2))


def _match_roi_zncc_local(ref_patch, tgt_img, x, y, search=12, mask_patch=None):
    """True zero-mean normalized correlation; return displacement, NOT warp.

    Only complete interior patches are used. A nontrivial mask is refused so a
    common artificial mask edge cannot become the matched feature.
    """
    h, w = ref_patch.shape
    if mask_patch is not None and not np.all(mask_patch):
        return 0., 0., -1.
    x0, y0 = max(0, int(x) - search), max(0, int(y) - search)
    x1 = min(tgt_img.shape[1], int(x) + w + search)
    y1 = min(tgt_img.shape[0], int(y) + h + search)
    crop = tgt_img[y0:y1, x0:x1]
    if crop.shape[0] < h or crop.shape[1] < w or ref_patch.std() < 1e-7:
        return 0., 0., -1.
    res = cv2.matchTemplate(crop, ref_patch, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=-1., posinf=-1., neginf=-1.)
    _, peak, _, (px, py) = cv2.minMaxLoc(res)
    if px == 0 or py == 0 or px == res.shape[1] - 1 or py == res.shape[0] - 1:
        return 0., 0., -1.
    other = res.copy()
    other[max(0, py-2):py+3, max(0, px-2):px+3] = -1
    if peak - float(other.max()) < .025:
        return 0., 0., -1.
    def quadratic(v0, v1, v2):
        curvature = v0 - 2 * v1 + v2
        return float(np.clip(.5 * (v0 - v2) / curvature, -.5, .5)) if curvature < -1e-7 else 0.
    dx = x0 + px - x + quadratic(res[py, px-1], res[py, px], res[py, px+1])
    dy = y0 + py - y + quadratic(res[py-1, px], res[py, px], res[py+1, px])
    return float(dx), float(dy), float(peak)


def _texture_ok(features, x, y, box):
    fine, _, noise, valid = features
    x, y = int(round(x)), int(round(y))
    region = fine[y:y+box, x:x+box]
    if region.shape != (box, box) or valid[y:y+box, x:x+box].mean() < .95:
        return False
    high = noise[y:y+box, x:x+box]
    noise_sigma = 1.4826 * np.median(np.abs(high - np.median(high)))
    return float(region.std()) > max(.002, .9 * float(noise_sigma))


def _select_rois(features, cx, cy, radius, box, count, search):
    fine = features[0]
    h, w = fine.shape
    buckets = [[] for _ in range(8)]
    step = max(24, box * 3 // 4)
    # Round-robin angular sectors: one illuminated side must not monopolize
    # the evidence. No reference-centered mask is multiplied into the pixels.
    for y in range(search + 2, h - box - search - 1, step):
        for x in range(search + 2, w - box - search - 1, step):
            corners = [(x-cx, y-cy), (x+box-cx, y-cy),
                       (x-cx, y+box-cy), (x+box-cx, y+box-cy)]
            if max(math.hypot(a, b) for a, b in corners) > .88 * radius:
                continue
            if not _texture_ok(features, x, y, box):
                continue
            sector = int((math.atan2(y+box/2-cy, x+box/2-cx) + math.pi) * 4 / math.pi) % 8
            buckets[sector].append((float(fine[y:y+box, x:x+box].std()), x, y))
    for bucket in buckets:
        bucket.sort(reverse=True)
    selected = []
    while any(buckets) and len(selected) < count:
        for bucket in buckets:
            while bucket:
                _, x, y = bucket.pop(0)
                if all(abs(x-x2) >= box * .75 or abs(y-y2) >= box * .75 for x2, y2 in selected):
                    selected.append((x, y))
                    break
            if len(selected) >= count:
                break
    return selected


def _consensus(vectors, centers, scores, cx, cy, radius, min_inliers):
    vectors, centers, scores = map(np.asarray, (vectors, centers, scores))
    if len(vectors) < min_inliers:
        return None, '可靠纹理块不足', None
    middle = np.median(vectors, axis=0)
    residuals = np.linalg.norm(vectors - middle, axis=1)
    keep = residuals <= .65
    if keep.sum() < min_inliers or keep.mean() < .75:
        return None, '局部位移不一致，可能有形变或误匹配', keep
    locations = centers[keep]
    quadrants = (locations[:, 0] >= cx).astype(int) + 2 * (locations[:, 1] >= cy).astype(int)
    if (len(np.unique(quadrants)) < 3 or np.ptp(locations[:, 0]) < radius * .6
            or np.ptp(locations[:, 1]) < radius * .6):
        return None, '有效纹理集中在一侧，空间覆盖不足', keep
    displacement = np.average(vectors[keep], axis=0, weights=np.maximum(scores[keep], .01)**2)
    return displacement, '', keep


def refine_alignment_multi_roi(
    ref_gray, tgt_gray, cx, cy, r, n_rois=16, roi_size=128, search=12,
    base_shift=None, max_refine_delta_px=6.0, min_inliers=6, min_mean_zncc=.65,
    use_phasecorr=True, use_ecc=False, time_budget_sec=None, debug_cb=None,
    diagnostics=None,
):
    """Unified two-scale ZNCC + checked phase refinement, with safe rejection.

    Inputs are already limb-aligned. Return a target-to-reference correction
    matrix, mean accepted ZNCC, inlier count, and zero rotation. Rejection returns
    the residual baseline and zero quality. Legacy time_budget_sec is ignored:
    bounded ROI count, not machine speed, determines the evidence used.
    """
    info = diagnostics if diagnostics is not None else {}
    info.clear()
    info.update(accepted=False, reason='', candidates=0, matched=0, inliers=0,
                phase_used=0, correction=[0., 0.], algorithm='texture-consensus-v2')
    baseline = np.asarray(base_shift if base_shift is not None else (0., 0.), dtype=float)
    if baseline.shape != (2,) or not np.isfinite(baseline).all():
        raise ValueError('无效的微调回退位移')

    def finish(correction, reason='', score=0., count=0):
        info.update(accepted=not bool(reason), reason=reason,
                    correction=np.asarray(correction).tolist())
        if debug_cb is not None:
            debug_cb('[纹理微调] ' + (f'采用 correction=({correction[0]:.3f}, {correction[1]:.3f})px'
                                     if not reason else '保留月缘结果：' + reason))
        matrix = np.float32([[1, 0, correction[0]], [0, 1, correction[1]]])
        return matrix, float(score), int(count), 0.

    if use_ecc:
        raise ValueError('不支持 ECC；实验分支已统一为双尺度纹理一致性微调')
    if ref_gray.shape != tgt_gray.shape or ref_gray.ndim != 2:
        return finish(baseline, '参考图和目标图尺寸不一致')
    if not np.isfinite([cx, cy, r]).all() or r <= 0:
        return finish(baseline, '月盘几何参数无效')
    ref, tgt = _prepare(ref_gray), _prepare(tgt_gray)
    if ref is None or tgt is None:
        return finish(baseline, '图像无有效动态范围')
    box = int(np.clip(roi_size, 48, 128))
    search = int(np.clip(search, 6, 18))
    rois = _select_rois(ref, cx, cy, r, box, int(np.clip(n_rois, 8, 24)), search)
    info['candidates'] = len(rois)
    vectors, centers, scores, coarse_vectors = [], [], [], []
    window = cv2.createHanningWindow((box, box), cv2.CV_32F)
    for x, y in rois:
        ref_patch = ref[0][y:y+box, x:x+box]
        dx, dy, score = _match_roi_zncc_local(ref_patch, tgt[0], x, y, search)
        if score < min_mean_zncc or not _texture_ok(tgt, x+dx, y+dy, box):
            continue
        dcx, dcy, coarse_score = _match_roi_zncc_local(ref[1][y:y+box, x:x+box], tgt[1], x, y, search)
        if coarse_score < min_mean_zncc or math.hypot(dx-dcx, dy-dcy) > .6:
            continue
        # Phase response is a separate gate, NEVER interchangeable with ZNCC.
        if use_phasecorr:
            ix, iy = int(round(dx)), int(round(dy))
            tp = _patch(tgt[0], x+ix, y+iy, box, box)
            if tp is not None:
                rp = np.ascontiguousarray(ref_patch - ref_patch.mean())
                tp = np.ascontiguousarray(tp - tp.mean())
                (px, py), response = cv2.phaseCorrelate(rp, tp, window.copy())
                if (np.isfinite([px, py, response]).all() and response >= .3
                        and abs(px) <= 1. and abs(py) <= 1.):
                    phase_shift = np.array([ix+px, iy+py])
                    if np.linalg.norm(phase_shift - [dcx, dcy]) <= .6:
                        phase_patch = _patch(tgt[0], x+phase_shift[0], y+phase_shift[1], box, box)
                        template_patch = _patch(tgt[0], x+dx, y+dy, box, box)
                        if (phase_patch is not None and template_patch is not None
                                and _zncc(ref_patch, phase_patch) >= _zncc(ref_patch, template_patch)):
                            dx, dy = phase_shift
                            info['phase_used'] += 1
        matched = _patch(tgt[0], x+dx, y+dy, box, box)
        if matched is None:
            continue
        # Reciprocity: the matched target patch must return to this location.
        back_x, back_y, back_score = _match_roi_zncc_local(matched, ref[0], x, y, search)
        if back_score < min_mean_zncc or math.hypot(back_x, back_y) > .5:
            continue
        score = _zncc(ref_patch, matched)
        if score < min_mean_zncc:
            continue
        vectors.append((dx, dy))
        coarse_vectors.append((dcx, dcy))
        centers.append((x+box/2, y+box/2))
        scores.append(score)

    info['matched'] = len(vectors)
    displacement, reason, keep = _consensus(vectors, centers, scores, cx, cy, r, min_inliers)
    info['inliers'] = int(keep.sum()) if keep is not None else 0
    if reason:
        return finish(baseline, reason)
    coarse_shift, coarse_reason, _ = _consensus(coarse_vectors, centers, scores, cx, cy, r, min_inliers)
    if coarse_reason or np.linalg.norm(coarse_shift-displacement) > .35:
        return finish(baseline, '双尺度位移不一致')
    correction = -displacement  # Target displacement is NOT a target warp.
    if np.linalg.norm(correction-baseline) > max_refine_delta_px:
        return finish(baseline, '微调超过允许范围')
    info['spread_px'] = float(np.max(np.linalg.norm(np.asarray(vectors)[keep]-displacement, axis=1)))
    info['mean_zncc'] = float(np.mean(np.asarray(scores)[keep]))
    return finish(correction, score=info['mean_zncc'], count=info['inliers'])
