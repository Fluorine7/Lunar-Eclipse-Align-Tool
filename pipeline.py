"""Bounded-memory batch alignment using independent per-image geometry."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
import os
import time

import cv2
import numpy as np

from algorithms_independent import analyze_lunar_frame, validate_radius_bounds, SHAPE_REGULARIZATION
from algorithms_refine import refine_alignment_multi_roi
from utils_common import (
    log, normalize_path, ensure_dir_exists, safe_join, imread_unicode,
    imwrite_unicode, imwrite_with_exif, SUPPORTED_EXTS, VERSION,
)


def choose_analysis_workers(image_bytes, requested=None):
    """Small bounded pool; do not alter process-global OpenCV thread settings."""
    if requested is not None and (isinstance(requested, bool) or int(requested) != requested or not 1 <= requested <= 4):
        raise ValueError('分析并行数必须为 1–4，或留空自动选择。')
    try:
        import psutil
        available = psutil.virtual_memory().available
    except (ImportError, OSError):
        available = 1024 ** 3
    per_worker = max(256 * 1024 ** 2, int(image_bytes) * 10)
    memory_limit = max(1, int(max(0, available * .4 - image_bytes * 4) // per_worker))
    cpu_limit = max(1, (os.cpu_count() or 1) // 2)
    return max(1, min(int(requested) if requested is not None else 2, cpu_limit, memory_limit))


def _load_and_analyze(path, params, strong_denoise):
    start = time.perf_counter()
    image = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    read_seconds = time.perf_counter() - start
    if image is None:
        raise ValueError('图像读取失败')
    analysis = analyze_lunar_frame(image, params, strong_denoise)
    analysis.timings['read'] = read_seconds
    return image, analysis


def iter_frame_analyses(paths, params, workers=1, strong_denoise=False):
    """At most workers in-flight frames; deliver results/errors in order.

    Workers receive their own path and common settings, never reference/other
    target geometry. Unlike Executor.map, the submission queue is bounded.
    """
    paths = iter(paths)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='lunar-analysis') as pool:
        pending = deque()
        for _ in range(workers):
            path = next(paths, None)
            if path is not None:
                pending.append((path, pool.submit(_load_and_analyze, path, params, strong_denoise)))
        while pending:
            path, future = pending.popleft()
            try:
                result, error = future.result(), None
            except Exception as exc:
                result, error = None, exc
            yield path, result, error
            following = next(paths, None)
            if following is not None:
                pending.append((following, pool.submit(_load_and_analyze, following, params, strong_denoise)))


def _report_value(value):
    if isinstance(value, dict):
        return {k: _report_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _write_reports(output_folder, report):
    path = safe_join(output_folder, 'alignment-report.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as stream:
        json.dump(_report_value(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(path + '.tmp', path)
    path = safe_join(output_folder, 'alignment-review.txt')
    with open(path + '.tmp', 'w', encoding='utf-8') as stream:
        stream.write('复查清单 / Review list\n不可靠帧已跳过，未写入本次对齐输出。\n')
        for record in report['frames']:
            if record['status'] != 'saved':
                stream.write(f"{record['file']}: {record['status']} — {record.get('reason', '')}\n")
    os.replace(path + '.tmp', path)


def _reference(input_folder, filenames, specified, params, strong_denoise, log_box):
    if specified:
        if not os.path.isfile(specified):
            raise ValueError(f'参考图不存在：{specified}')
        image, result = _load_and_analyze(specified, params, strong_denoise)
        if result.geometry is None or result.geometry.reasons:
            detail = '; '.join(result.geometry.reasons) if result.geometry else result.reason
            raise ValueError(f'参考图月缘不可靠，请换一张月缘完整的参考图：{detail}')
        return specified, image, result
    # Sample across the list, not just its often equally obscured first files.
    indices = np.unique(np.linspace(0, len(filenames) - 1, min(12, len(filenames))).astype(int))
    best = None
    for count, i in enumerate(indices, 1):
        path = safe_join(input_folder, filenames[i])
        log(f'参考图检查 {count}/{len(indices)}: {filenames[i]}', log_box)
        try:
            image, result = _load_and_analyze(path, params, strong_denoise)
        except Exception as exc:
            log(f'  跳过参考候选：{exc}', log_box)
            continue
        fit = result.geometry
        if fit is not None and not fit.reasons:
            score = (fit.coverage, -fit.spread, -fit.rms)
            if best is None or score > best[0]:
                best = score, path, image, result
    if best is None:
        raise ValueError('未找到可靠参考图。请选择月缘较完整的参考图并确认半径范围。')
    return best[1:]


def _refine_translation(reference_gray, image, center, radius, shift, diagnostics=None):
    """Optional reference-texture refinement; render original pixels only once."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    moved_gray = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)
    result = refine_alignment_multi_roi(
        reference_gray, moved_gray, *center, radius,
        n_rois=16, roi_size=max(64, min(160, int(radius * .18))), search=12,
        use_phasecorr=True, use_ecc=False, base_shift=(0., 0.), max_refine_delta_px=6.,
        diagnostics=diagnostics,
    )
    matrix2 = result[0] if isinstance(result, tuple) and result else None
    if (matrix2 is not None and result[1] > 0 and result[2] >= 6
            and np.all(np.isfinite(matrix2)) and np.allclose(matrix2[:, :2], np.eye(2), atol=1e-6)):
        delta = matrix2[:, 2]
        if np.linalg.norm(delta) <= 6:
            return shift + delta
    return shift


def align_moon_images_incremental(input_folder, output_folder, hough_params,
                                 log_box=None, debug_mode=False, debug_image_basename='',
                                 completion_callback=None, progress_callback=None,
                                 reference_image_path=None, use_advanced_alignment=False,
                                 alignment_method='auto', strong_denoise=False, workers=None):
    """Align by translation. Uncertain geometry is skipped and fully reported.

    alignment_method is a deprecated compatibility argument. All legacy names
    explicitly select the SAME unified experimental texture estimator.
    """
    report = None
    try:
        input_folder, output_folder = normalize_path(input_folder), normalize_path(output_folder)
        if os.path.normcase(os.path.realpath(input_folder)) == os.path.normcase(os.path.realpath(output_folder)):
            raise ValueError('输入和输出目录不能相同。')
        validate_radius_bounds(hough_params[:2])
        if not ensure_dir_exists(output_folder):
            raise ValueError(f'无法创建输出文件夹：{output_folder}')
        filenames = sorted(name for name in os.listdir(input_folder)
                           if not name.startswith('._') and os.path.splitext(name)[1].lower() in SUPPORTED_EXTS
                           and os.path.isfile(safe_join(input_folder, name)))
        if not filenames:
            raise ValueError('输入目录没有支持的图片。')
        collisions = [name for name in filenames if os.path.exists(safe_join(output_folder, f'aligned_{name}'))]
        if collisions:
            raise ValueError('输出目录已有同名对齐图，请选择新的输出目录，避免混合不同算法结果。')
        log(f'月食圆面对齐工具 V{VERSION} — 逐帧独立、半径范围内浮动', log_box)
        log('阶段 1/2：选择并验证参考图（只定义输出位置，不约束目标形状）', log_box)
        ref_path, reference_image, ref_analysis = _reference(
            input_folder, filenames, reference_image_path, hough_params, strong_denoise, log_box)
        reference = ref_analysis.geometry
        canonical_ref = os.path.normcase(os.path.realpath(ref_path))
        # Header-only inspection protects mixed-resolution collections from a
        # small reference causing an overly optimistic worker memory budget.
        from PIL import Image
        largest_bytes = reference_image.nbytes
        for name in filenames:
            try:
                with Image.open(safe_join(input_folder, name)) as header:
                    largest_bytes = max(largest_bytes, header.width * header.height * 8)
            except (OSError, ValueError):
                pass  # The actual reader will report this frame's error.
        worker_count = choose_analysis_workers(largest_bytes, workers)
        log(f'阶段 2/2：{len(filenames)} 张；分析并行={worker_count}，后台写入=1；有界队列', log_box)
        log('可疑或失败帧将跳过，并写入 alignment-review.txt；不会使用上一帧补位置。', log_box)
        if use_advanced_alignment:
            log('实验性纹理微调：统一使用双尺度 ZNCC + 相位残差 + 空间一致性检查；不可靠则保留月缘结果。', log_box)
            if alignment_method != 'auto':
                log(f'旧实验选项 {alignment_method!r} 已合并，使用统一纹理微调。', log_box)
        ref_gray = (cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY) if reference_image.ndim == 3 else reference_image) if use_advanced_alignment else None
        report = {'version': VERSION, 'detector': 'independent-continuous-shape-v2', 'reference': os.path.realpath(ref_path),
                  'reference_geometry': asdict(reference), 'parameters': list(hough_params),
                  'shape_regularization': SHAPE_REGULARIZATION,
                  'workers': worker_count, 'strong_denoise': strong_denoise,
                  'experimental_texture': use_advanced_alignment,
                  'texture_algorithm': 'texture-consensus-v2' if use_advanced_alignment else None,
                  'frames': []}
        paths = [safe_join(input_folder, name) for name in filenames
                 if os.path.normcase(os.path.realpath(safe_join(input_folder, name))) != canonical_ref]
        pending = deque()

        def finish_write():
            future, record = pending.popleft()
            try:
                if not future.result():
                    raise OSError('图像保存失败')
                record['status'] = 'saved'
                log(f"  ✓ {record['file']}: shift=({record['shift'][0]:.2f}, {record['shift'][1]:.2f})", log_box)
            except Exception as exc:
                record.update(status='failed', reason=f'写入失败：{exc}')
                log(f"  ✗ {record['file']}: {record['reason']}", log_box)

        start_all = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='lunar-writer') as writer:
            if any(os.path.normcase(os.path.realpath(safe_join(input_folder, name))) == canonical_ref for name in filenames):
                record = {'file': os.path.basename(ref_path), 'status': 'pending', 'shift': [0., 0.],
                          'geometry': asdict(reference), 'method': 'reference', 'timings': ref_analysis.timings}
                report['frames'].append(record)
                pending.append((writer.submit(imwrite_with_exif, ref_path,
                                               safe_join(output_folder, f'aligned_{os.path.basename(ref_path)}'), reference_image), record))
            for index, (path, loaded, error) in enumerate(iter_frame_analyses(paths, hough_params, worker_count, strong_denoise), 1):
                filename = os.path.basename(path)
                record = {'file': filename, 'status': 'skipped', 'reason': ''}
                report['frames'].append(record)
                if progress_callback:
                    progress_callback(10 + int(85 * index / max(1, len(paths))), f'分析 {index}/{len(paths)}: {filename}')
                if error is not None:
                    record.update(status='failed', reason=f'{type(error).__name__}: {error}')
                    log(f'  ✗ {filename}: {record["reason"]}', log_box)
                    continue
                image, analysis = loaded
                fit = analysis.geometry
                record.update(method=analysis.method, timings=analysis.timings,
                              geometry=asdict(fit) if fit else None)
                if fit is not None:
                    log(f'  [月缘] {filename}: {fit.model}, r={fit.radius:.2f}px, '
                        f'coverage={fit.coverage:.0%}, RMS={fit.rms:.2f}px, '
                        f'轴比={min(fit.axes)/max(fit.axes):.5f}, 形状敏感性={fit.shape_sensitivity:.2f}px, '
                        f'分弧偏移={fit.spread:.2f}px, '
                        f'分析={sum(v for k, v in analysis.timings.items() if k != "read"):.3f}s', log_box)
                if fit is None or fit.reasons:
                    record['reason'] = '; '.join(fit.reasons) if fit else analysis.reason
                    log(f'  ⚠ 需复查，跳过 {filename}: {record["reason"]}', log_box)
                    del image, loaded
                    continue
                shift = np.asarray(reference.center) - fit.center
                if use_advanced_alignment:
                    texture = {}
                    start_texture = time.perf_counter()
                    try:
                        shift = _refine_translation(ref_gray, image, reference.center, reference.radius, shift,
                                                    diagnostics=texture)
                    except Exception as exc:
                        texture.update(accepted=False, reason=f'{type(exc).__name__}: {exc}', correction=[0., 0.])
                    record['texture_refinement'] = texture
                    record['timings']['texture'] = time.perf_counter() - start_texture
                    if texture.get('accepted'):
                        dx, dy = texture['correction']
                        log(f'  [纹理] {filename}: 采用 ({dx:.3f}, {dy:.3f})px，'
                            f'一致纹理块 {texture["inliers"]}/{texture["matched"]}', log_box)
                    else:
                        log(f'  [纹理] {filename}: 保留月缘结果 — {texture.get("reason", "未通过校验")}', log_box)
                start_warp = time.perf_counter()
                matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
                aligned = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                record['timings']['warp'] = time.perf_counter() - start_warp
                record.update(shift=shift.tolist(), status='pending')
                if debug_mode and filename == debug_image_basename:
                    debug = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.ellipse(debug, (fit.center, (2 * fit.axes[0], 2 * fit.axes[1]), math.degrees(fit.angle)), (0, 255, 0), 2)
                    imwrite_unicode(safe_join(output_folder, 'debug', filename), debug)
                pending.append((writer.submit(imwrite_with_exif, path, safe_join(output_folder, f'aligned_{filename}'), aligned), record))
                if len(pending) >= 2:
                    finish_write()
                del image, loaded, aligned
            while pending:
                finish_write()
        report['seconds'] = time.perf_counter() - start_all
        report['frames'].sort(key=lambda record: record['file'])
        _write_reports(output_folder, report)
        saved = sum(record['status'] == 'saved' for record in report['frames'])
        review = len(report['frames']) - saved
        message = f'处理完成：保存 {saved}/{len(filenames)} 张，跳过/失败 {review} 张。'
        if review:
            message += '\n请查看 alignment-review.txt 和 alignment-report.json。'
        log(message, log_box)
        if progress_callback:
            progress_callback(100, '已完成，需复查' if review else '处理完成')
        if completion_callback:
            completion_callback(True, message)
        return report
    except Exception as exc:
        if report is not None:
            report['fatal_error'] = f'{type(exc).__name__}: {exc}'
            try:
                _write_reports(output_folder, report)
            except OSError:
                pass
        log(f'处理失败：{exc}', log_box)
        if completion_callback:
            completion_callback(False, str(exc))
        return None
