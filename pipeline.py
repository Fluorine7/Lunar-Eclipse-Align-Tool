import os, math, time
import cv2, numpy as np

from utils_common import (
    log, normalize_path, ensure_dir_exists, safe_join,
    imread_unicode, imwrite_unicode, imwrite_with_exif, get_memory_usage_mb,
    force_garbage_collection, MemoryManager, SUPPORTED_EXTS, VERSION
)

from algorithms_circle import detect_circle_phd2_enhanced, masked_phase_corr
from algorithms_limb import (
    detect_faint_lunar_disk,
    refine_lunar_limb,
    refine_lunar_limb_elliptical,
)

# refine 返回可能是 (M, score, nin) 也可能是 (M, theta_deg, score, nin)
from algorithms_refine import refine_alignment_multi_roi  # 兼容旧/新签名

# 兼容不同版本 refine 返回值
# 可能: (M, score, nin) / (M, theta_deg, score, nin) / (M, score, nin, theta_deg)
# 返回统一: (M, theta_deg, score, nin)
def _unpack_refine_result(res):
    M = None; theta_deg = 0.0; score = 0.0; nin = 0
    if not isinstance(res, tuple):
        return M, theta_deg, score, nin
    if len(res) < 3:
        return M, theta_deg, score, nin
    M = res[0]
    tail = list(res[1:])
    # 提取 nin: 优先取 int；没有的话从尾部取近似整数
    nin_idx = None
    for i, v in enumerate(tail):
        if isinstance(v, (int, np.integer)):
            nin = int(v); nin_idx = i; break
    if nin_idx is None:
        # 没有明确的 int，就尝试把接近整数的最后一个当作 nin
        for i in reversed(range(len(tail))):
            v = tail[i]
            if isinstance(v, (float, np.floating)) and abs(v - round(v)) < 1e-6 and v >= 0:
                nin = int(round(v)); nin_idx = i; break
    if nin_idx is not None:
        tail.pop(nin_idx)
    # 现在 tail 应该有两个浮点: 角度 和 分数
    # 分数通常在 [0,1.5]，角度通常在 [-180, 180]
    cand = [float(x) for x in tail[:2]] + ([0.0] if len(tail)==1 else [])
    if len(cand) >= 2:
        a, b = cand[0], cand[1]
        # 试着判别谁是 score
        def is_score(x):
            return -0.05 <= x <= 1.5
        if is_score(a) and not is_score(b):
            score, theta_deg = a, b
        elif is_score(b) and not is_score(a):
            score, theta_deg = b, a
        else:
            # 都像/都不像分数，按常见顺序 (score, theta)
            score, theta_deg = a, b
    elif len(cand) == 1:
        # 只有一个值，优先当 score
        val = cand[0]
        if -0.05 <= val <= 1.5:
            score = val
        else:
            theta_deg = val
    return M, float(theta_deg), float(score), int(nin)

# Helper to extract actual ROI used from refine_alignment_multi_roi result, fallback to default
def _extract_roi_used(res, default_roi):
    """
    Try to get the actual ROI size used by refine_alignment_multi_roi from its return tuple.
    Backward compatible:
      - Old signatures: (M, score, nin) or (M, theta, score, nin) -> fall back to default_roi
      - New signature we added: (M, theta, score, nin, avg_roi[, ...]) -> use that avg_roi
    """
    roi_used = int(default_roi)
    try:
        if isinstance(res, tuple) and len(res) >= 5:
            candidate = res[4]
            if isinstance(candidate, (int, float, np.integer, np.floating)) and candidate > 0:
                roi_used = int(round(float(candidate)))
    except Exception:
        pass
    return roi_used

# ------------------ 调试图保存 ------------------
def save_debug_image(processed_img, target_center, reference_center,
                     shift_x, shift_y, confidence, method,
                     debug_output_folder, filename, reference_filename):
    try:
        if processed_img is None:
            return
        if processed_img.ndim == 2:
            debug_image = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        else:
            debug_image = processed_img.copy()
        cv2.circle(debug_image, (int(target_center[0]), int(target_center[1])), 5, (0,0,255), -1)
        cv2.circle(debug_image, (int(reference_center[0]), int(reference_center[1])), 15, (0,255,255), 3)
        cv2.line(debug_image, (int(target_center[0]), int(target_center[1])),
                 (int(reference_center[0]), int(reference_center[1])), (0,255,255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [
            f"Method: {method[:35]}",
            f"Shift: ({shift_x:.1f}, {shift_y:.1f})",
            f"Confidence: {confidence:.3f}",
            f"Reference: {reference_filename}",
            f"Mode: Incremental Processing"
        ]
        for j, t in enumerate(texts):
            cv2.putText(debug_image, t, (10, 25 + j*25), font, 0.6, (255,255,255), 2)
        debug_path = safe_join(debug_output_folder, f"debug_{filename}")
        imwrite_unicode(debug_path, debug_image)
    except Exception as e:
        print(f"调试图像生成失败: {e}")

# ------------------ 缩略图辅助 ------------------
def _detect_circle_on_thumb(img, min_r, max_r, p1, p2, max_side=1600, strong_denoise=False):
    """
    在缩略图上做一次圆检测，返回 (ok, (cx,cy,r), scale, quality, method)
    成功时 (True, circle_fullres, scale, quality, method)；失败时 (False, None, scale, 0, '')
    """
    H, W = img.shape[:2]
    max_wh = max(H, W)
    scale = 1.0
    if max_wh > max_side:
        scale = max_side / float(max_wh)
    small = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img

    s_min = max(1, int(min_r * scale))
    s_max = max(s_min + 1, int(max_r * scale))

    t0 = time.time()
    circle_s, _, quality_s, method_s, _ = detect_circle_phd2_enhanced(small, s_min, s_max, p1, p2, strong_denoise=strong_denoise)
    dt = time.time() - t0

    if circle_s is None:
        return False, None, scale, 0.0, ""

    cx = float(circle_s[0] / scale)
    cy = float(circle_s[1] / scale)
    r  = float(circle_s[2] / scale)
    return True, (cx, cy, r), scale, float(quality_s), f"{method_s}(thumb,{small.shape[1]}x{small.shape[0]}, {dt:.2f}s)"

# ------------------ 主流程 ------------------
def align_moon_images_incremental(input_folder, output_folder, hough_params,
                                 log_box=None, debug_mode=False, debug_image_basename="",
                                 completion_callback=None, progress_callback=None,
                                 reference_image_path=None, use_advanced_alignment=False,
                                 alignment_method='auto', strong_denoise=False):
    memory_manager = MemoryManager()
    try:
        input_folder = normalize_path(input_folder)
        output_folder = normalize_path(output_folder)
        if not ensure_dir_exists(output_folder):
            raise Exception(f"无法创建输出文件夹: {output_folder}")
        debug_output_folder = safe_join(output_folder, "debug")
        if debug_mode and not ensure_dir_exists(debug_output_folder):
            raise Exception(f"无法创建调试文件夹: {debug_output_folder}")

        try:
            image_files = sorted([f for f in os.listdir(input_folder)
                                  if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS])
        except Exception as e:
            raise Exception(f"读取输入文件夹失败: {e}")
        if not image_files:
            raise Exception(f"在 '{input_folder}' 中未找到支持的图片文件")

        min_rad, max_rad, param1, param2 = hough_params
        total_files = len(image_files)

        log("=" * 60, log_box)
        log(f"月食圆面对齐工具 V{VERSION} - 增量处理版", log_box)
        log(f"处理模式: 增量处理 (边检测边保存)", log_box)
        log(f"文件总数: {total_files}", log_box)
        log(f"实验性月面纹理微调: {'启用' if use_advanced_alignment else '关闭（常规流程）'}", log_box)
        log("=" * 60, log_box)

        # 参考图像
        log("阶段 1/2: 确定参考图像...", log_box)
        reference_image = None; reference_center = None
        reference_filename = None; best_quality = 0.0
        reference_radius = None

        # ---------- 用户指定参考图 ----------
        if reference_image_path and os.path.exists(reference_image_path):
            ref_filename = os.path.basename(reference_image_path)
            log(f"加载用户指定的参考图像: {ref_filename}", log_box)

            t_ref0 = time.time()
            ref_img = imread_unicode(reference_image_path, cv2.IMREAD_UNCHANGED)
            if ref_img is not None:
                H, W = ref_img.shape[:2]
                log(f"参考图尺寸: {W}x{H}", log_box)

                # 先在缩略图做，映射回原图
                ok, circle, scale, q, meth = _detect_circle_on_thumb(
                    ref_img, min_rad, max_rad, param1, param2, max_side=1600, strong_denoise=strong_denoise
                )
                if ok:
                    reference_image = ref_img.copy()
                    reference_center = (circle[0], circle[1])
                    reference_filename = ref_filename
                    best_quality = q
                    reference_radius = circle[2]
                    log(f"✓ 参考图像检测成功: 质量={q:.1f}, 方法={meth}, 半径≈{reference_radius:.1f}px", log_box)
                else:
                    log("缩略图检测失败，回退到原图做一次圆检测（可能较慢）...", log_box)
                    t1 = time.time()
                    circle_full, _, qf, mf, _ = detect_circle_phd2_enhanced(
                        ref_img, min_rad, max_rad, param1, param2, strong_denoise=strong_denoise
                    )
                    dt1 = time.time() - t1
                    if circle_full is not None:
                        reference_image = ref_img.copy()
                        reference_center = (circle_full[0], circle_full[1])
                        reference_filename = ref_filename
                        best_quality = float(qf)
                        reference_radius = float(circle_full[2])
                        log(f"✓ 参考图像检测成功: 质量={best_quality:.1f}, 方法={mf}, 半径≈{reference_radius:.1f}px, 用时 {dt1:.2f}s", log_box)
                    else:
                        log("✗ 参考图像检测失败，将自动选择", log_box)
            else:
                log("✗ 参考图像读取失败，将自动选择", log_box)

        # ---------- 自动扫描前 N 张 ----------
        if reference_image is None:
            scan_count = min(10, total_files)
            log(f"自动选择参考图像 (扫描前{scan_count}张)...", log_box)
            for i, filename in enumerate(image_files[:scan_count]):
                if progress_callback:
                    progress_callback(int((i / scan_count) * 20), f"扫描参考图像: {filename}")
                input_path = safe_join(input_folder, filename)
                img0 = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
                if img0 is None:
                    continue

                ok, circle, scale, q, meth = _detect_circle_on_thumb(
                    img0, min_rad, max_rad, param1, param2, max_side=1600, strong_denoise=strong_denoise
                )
                if ok and q > best_quality:
                    if reference_image is not None: del reference_image
                    reference_image = img0.copy()
                    reference_center = (circle[0], circle[1])
                    reference_filename = filename
                    best_quality = q
                    reference_radius = circle[2]
                    log(f"  候选参考图像: {filename}, 质量={q:.1f}, 方法={meth}", log_box)

                del img0
                force_garbage_collection()

        if reference_image is None:
            raise Exception("无法找到有效的参考图像，请检查图像质量和参数设置")

        # Refine the reference limb. Its radius supplies the missing scale
        # constraint when one target contains only a short illuminated arc.
        # Every target center is still solved independently and output is never
        # scaled or linked to another target frame.
        ref_limb = refine_lunar_limb_elliptical(
            reference_image,
            (float(reference_center[0]), float(reference_center[1]), float(reference_radius)),
        )
        if ref_limb is not None:
            ref_circle, ref_rms, ref_coverage, ref_points, ref_shape, ref_shape_updated = ref_limb
            reference_center = (ref_circle[0], ref_circle[1])
            reference_radius = ref_circle[2]
            ref_model = "受约束椭圆" if ref_shape is not None else "稳健圆"
            log(
                f"✓ 外缘精定位参考图({ref_model}): center=({ref_circle[0]:.2f}, {ref_circle[1]:.2f}), "
                f"r={ref_circle[2]:.2f}, RMS={ref_rms:.2f}px, coverage={ref_coverage:.0%}, n={ref_points}",
                log_box,
            )
        else:
            log("⚠ 参考图外缘精定位失败，保留霍夫圆结果。", log_box)

        log(f"🎯 最终参考图像: {reference_filename}, 质量评分={best_quality:.1f}", log_box)

        # 处理所有图像
        log(f"\n阶段 2/2: 增量处理所有图像...", log_box)
        success_count = 0; failed_files = []
        brightness_stats = {"bright": 0, "normal": 0, "dark": 0}
        method_stats = {}

        # 为速度统计
        t_all0 = time.time()

        for i, filename in enumerate(image_files):
            if progress_callback:
                progress_callback(20 + int((i / total_files) * 80), f"处理: {filename}")
            try:
                input_path = safe_join(input_folder, filename)

                # 参考图：直接另存
                if filename == reference_filename:
                    output_path = safe_join(output_folder, f"aligned_{filename}")
                    if imwrite_with_exif(input_path, output_path, reference_image):
                        success_count += 1
                        log(f"  🎯 {filename}: [参考图像] 已保存", log_box)
                        if debug_mode and filename == debug_image_basename:
                            save_debug_image(reference_image, reference_center, reference_center,
                                             0, 0, 1.0, "Reference Image",
                                             safe_join(output_folder, "debug"), filename, reference_filename)
                    else:
                        log(f"  ✗ {filename}: 保存失败", log_box); failed_files.append(filename)
                    continue

                # 读取目标
                t_read = time.time()
                target_image = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
                if target_image is None:
                    log(f"  ✗ {filename}: 读取失败", log_box); failed_files.append(filename); continue
                dt_read = time.time() - t_read

                # 圆检测
                t_det = time.time()
                circle, processed, quality, method, brightness = detect_circle_phd2_enhanced(
                    target_image, min_rad, max_rad, param1, param2,
                    strong_denoise=strong_denoise, prev_circle=None
                )
                dt_det = time.time() - t_det

                if circle is None:
                    log(f"  ✗ {filename}: 圆检测失败(耗时 {dt_det:.2f}s)", log_box)
                    failed_files.append(filename); del target_image; continue

                limb_used = False
                faint_limb_used = False
                # Every target is solved independently. No previous-frame
                # center, radius, or ellipse shape is allowed into this fit.
                limb = refine_lunar_limb_elliptical(target_image, circle)
                if limb is not None:
                    limb_circle, limb_rms, limb_coverage, limb_points, limb_shape, shape_updated = limb
                    radius_constrained = False
                    # A short visible arc cannot independently determine both
                    # radius and center. Use the fixed reference radius as the
                    # missing batch-level scale constraint; this is still a
                    # target-to-reference solve and never reads another target.
                    if (
                        limb_shape is None
                        and limb_coverage < 0.42
                        and reference_radius is not None
                    ):
                        fixed_limb = refine_lunar_limb(
                            target_image,
                            limb_circle,
                            fixed_radius=float(reference_radius),
                            search_px=32,
                        )
                        if fixed_limb is not None:
                            fixed_circle, fixed_rms, fixed_coverage, fixed_points = fixed_limb
                            if fixed_rms <= 3.5:
                                limb_circle = fixed_circle
                                limb_rms = fixed_rms
                                limb_coverage = fixed_coverage
                                limb_points = fixed_points
                                radius_constrained = True
                    circle = np.asarray(limb_circle, dtype=np.float32)
                    ellipse_used = limb_shape is not None and shape_updated
                    if ellipse_used:
                        limb_model = "单帧受约束椭圆"
                    elif radius_constrained:
                        limb_model = "参考半径约束圆"
                    else:
                        limb_model = "单帧稳健圆"
                    method = f"{method} + 外缘精定位({limb_model})"
                    quality = max(float(quality), 100.0 / (1.0 + limb_rms))
                    limb_used = True
                    log(
                        f"    [Limb:{limb_model}] center=({limb_circle[0]:.2f}, {limb_circle[1]:.2f}), "
                        f"r={limb_circle[2]:.2f}px, RMS={limb_rms:.2f}px, "
                        f"coverage={limb_coverage:.0%}, n={limb_points}, "
                        f"shape={'independent' if shape_updated else 'circle-fallback'}",
                        log_box,
                    )
                elif reference_radius is not None:
                    # If normal edge sampling cannot obtain a trustworthy limb,
                    # try a fixed-radius annular matched filter on a temporary
                    # enhanced analysis copy.  It never modifies the output and
                    # uses no previous target frame.
                    faint_limb = detect_faint_lunar_disk(
                        target_image, float(reference_radius),
                    )
                    if faint_limb is not None:
                        faint_circle, faint_confidence, faint_detail = faint_limb
                        circle = np.asarray(faint_circle, dtype=np.float32)
                        quality = max(float(quality), min(95.0, 35.0 + 3.0 * faint_confidence))
                        method = f"{method} + 低信噪比环积分(参考半径)"
                        limb_used = True
                        faint_limb_used = True
                        log(
                            f"    [FaintLimb:参考半径环积分] center=({circle[0]:.2f}, {circle[1]:.2f}), "
                            f"r={circle[2]:.2f}px, {faint_detail}",
                            log_box,
                        )
                    else:
                        log("    [Limb] 外缘精定位及低信噪比回退均不可靠，保留霍夫圆结果。", log_box)
                else:
                    log("    [Limb] 外缘精定位不可靠，保留霍夫圆结果。", log_box)

                brightness_stats[brightness] += 1
                method_stats[method] = method_stats.get(method, 0) + 1

                target_center = (circle[0], circle[1])

                # 初始：圆心平移到参考
                shift_x = reference_center[0] - target_center[0]
                shift_y = reference_center[1] - target_center[1]
                confidence = max(0.30, min(0.98, quality / 100.0))
                if faint_limb_used:
                    align_method = "低信噪比参考半径环积分对齐"
                else:
                    align_method = "外缘圆心对齐" if limb_used else "霍夫圆心对齐"
                theta_deg = 0.0

                rows, cols = target_image.shape[:2]
                M = np.float32([[1,0,shift_x],[0,1,shift_y]])
                aligned = cv2.warpAffine(target_image, M, (cols, rows),
                                         flags=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # 实验性月面纹理微调（仅残余平移；常规流程默认关闭）
                try:
                    if reference_radius is not None and use_advanced_alignment:
                        ref_gray = reference_image if reference_image.ndim==2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
                        tgt_gray2 = aligned if aligned.ndim==2 else cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

                        roi_size = max(64, min(160, int(reference_radius*0.18)))
                        max_refine_delta_px = 6.0
                        t_refine = time.time()
                        res = refine_alignment_multi_roi(
                            ref_gray, tgt_gray2,
                            float(reference_center[0]), float(reference_center[1]),
                            float(reference_radius),
                            n_rois=16, roi_size=roi_size, search=12,
                            use_phasecorr=True, use_ecc=False,
                            # `aligned` 已经应用了霍夫圆心平移；ROI 输出和
                            # 回退基线都必须处于残余坐标系，不能再次应用 shift。
                            base_shift=(0.0, 0.0),
                            max_refine_delta_px=max_refine_delta_px,
                            debug_cb=(lambda msg: log(f"    {msg}", log_box)) if debug_mode else (lambda msg: None),
                        )
                        dt_refine = time.time() - t_refine
                        M2, theta_deg, score, nin = _unpack_refine_result(res)
                        roi_used = _extract_roi_used(res, roi_size)
                        log(f"    [Refine] score={score:.3f}, inliers={nin}, roi_init≈{roi_used}, t={dt_refine:.2f}s", log_box)
                        residual = None
                        if M2 is not None:
                            tx = float(M2[0,2])
                            ty = float(M2[1,2])
                            residual = (tx**2 + ty**2) ** 0.5
                            log(f"    [Refine] 残差=Δ{residual:.2f}px", log_box)
                            if residual > max_refine_delta_px:
                                M2 = None
                                log(f"    [Refine] 残差过大(Δ={residual:.2f}px > {max_refine_delta_px:.1f}px)，放弃精配准并保持霍夫平移", log_box)
                        if M2 is not None:
                            aligned = cv2.warpAffine(
                                aligned, M2, (cols, rows),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0
                            )
                            confidence = max(confidence, float(score))
                            align_method = f"外缘圆心对齐 + 实验性纹理微调（仅平移, inliers={nin}, roi_init≈{roi_used}, Δ={residual:.2f}px, gate≤{max_refine_delta_px:.0f}px, {dt_refine:.2f}s)"
                        else:
                            # Eclipse illumination changes make a rejected ROI
                            # solution safer than a blind phase-correlation
                            # fallback. Keep the independently validated limb
                            # transform instead of introducing frame-to-frame
                            # jitter from lunar-surface texture.
                            log("    [纹理微调] 无有效解，保留外缘圆心对齐。", log_box)
                except Exception as e:
                    log(f"    [Refine异常] {filename}: {e}", log_box)

                # 保存
                out_path = safe_join(output_folder, f"aligned_{filename}")
                if imwrite_with_exif(input_path, out_path, aligned):
                    success_count += 1
                    log(f"  ✓ {filename}: 偏移=({shift_x:.1f},{shift_y:.1f}), "
                        f"质量={quality:.1f}, 置信度={confidence:.3f}, 圆检耗时={dt_det:.2f}s, 读取={dt_read:.2f}s | {align_method}", log_box)

                    if debug_mode and filename == debug_image_basename and processed is not None:
                        save_debug_image(processed, target_center, reference_center,
                                         shift_x, shift_y, confidence, align_method,
                                         debug_output_folder, filename, reference_filename)
                else:
                    log(f"  ✗ {filename}: 变换成功但保存失败", log_box)
                    failed_files.append(filename)

                del target_image, aligned
                if 'processed' in locals(): del processed
                force_garbage_collection()

            except Exception as e:
                log(f"  ✗ {filename}: 处理异常 - {e}", log_box)
                failed_files.append(filename)
                for v in ['target_image','aligned','processed']:
                    if v in locals(): del locals()[v]
                force_garbage_collection()

        if progress_callback: progress_callback(100, "处理完成")
        del reference_image; force_garbage_collection()

        log("=" * 60, log_box)
        log(f"增量对齐完成! 成功对齐 {success_count}/{total_files} 张图像", log_box)
        log(f"使用参考图像: {reference_filename}", log_box)
        log(f"对齐流程: {'受约束椭圆/稳健圆外缘 + 低信噪比环积分回退 + 实验性纹理微调（仅平移）' if use_advanced_alignment else '霍夫初定位 + 受约束椭圆/稳健圆外缘 + 低信噪比环积分回退（仅平移）'}", log_box)
        if failed_files:
            head = ', '.join(failed_files[:5]) + ("..." if len(failed_files)>5 else "")
            log(f"失败文件({len(failed_files)}): {head}", log_box)
        if method_stats:
            log("圆检测方法统计: " + ', '.join([f"{k}={v}" for k,v in method_stats.items()]), log_box)
        log(f"当前内存使用: {get_memory_usage_mb():.1f} MB", log_box)
        if completion_callback:
            completion_callback(True, f"增量处理完成！成功对齐 {success_count}/{total_files} 张图像")

    except Exception as e:
        import traceback
        err = f"增量处理过程中发生错误: {e}\n{traceback.format_exc()}"
        log(err, log_box)
        if completion_callback:
            completion_callback(False, err)
    finally:
        force_garbage_collection()
