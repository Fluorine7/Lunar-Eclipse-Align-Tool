import logging
from pathlib import Path
import os, math, time
from typing import Iterable

import cv2, numpy as np

from lunar_eclipse_align.utils.image import Image, ImageFile
from lunar_eclipse_align.utils.tools import (
    Circle,
    get_memory_usage_mb,
)
from lunar_eclipse_align.utils.data_types import HoughParams, Point, Vector

from lunar_eclipse_align.core.circle_detection import (
    detect_circle,
    detect_circle_quick,
)

from lunar_eclipse_align.core.shift_detection import (
    advanced_detect_shift,
    detect_mask_phase_shift,
)


def auto_select_reference(
    input_files: Iterable[ImageFile],
    hough: HoughParams,
    strong_denoise: bool = False,
) -> ImageFile | None:

    reference_file: ImageFile | None = None

    logging.info(f"自动选择参考图像...")
    min_distance_to_center = math.inf
    best_image_file = None
    # 选择位置最靠近中心的
    for input_file in input_files:

        img = input_file.image
        if not img:
            continue

        circle = detect_circle_quick(img, hough, strong_denoise=strong_denoise)
        if not circle:
            continue
        center = Point(img.width / 2, img.height / 2)
        distance_to_center = (circle.center - center).norm()
        if not best_image_file or distance_to_center < min_distance_to_center:
            best_image_file = input_file
            reference_file = input_file
            min_distance_to_center = distance_to_center
    if not reference_file:
        logging.error("未能找到合适的参考图像")
        return None
    logging.info(f"🎯 最终参考图像: {reference_file.path.name}")
    return best_image_file


def get_user_reference_circle(
    reference_path: Path,
    input_files: dict[Path, ImageFile],
    hough: HoughParams,
    strong_denoise: bool,
) -> Circle | None:

    if reference_path not in input_files:
        logging.error(f"指定的参考图像 {reference_path} 不在输入目录中")
        return None
    reference_file = input_files[reference_path]
    if not reference_file.image:
        logging.error(f"无法加载指定的参考图像 {reference_path}")
        return None
    ref_circle = detect_circle(reference_file.image, hough, strong_denoise)
    if not ref_circle:
        logging.error(f"未能在指定的参考图像 {reference_path} 中检测到月食圆")
        return None
    return ref_circle


def get_reference_circle(
    reference_path: Path | None,
    input_files: dict[Path, ImageFile],
    hough: HoughParams,
    strong_denoise: bool,
) -> tuple[Circle, ImageFile] | None:
    logging.info("阶段 1/2: 确定参考图像...")
    if reference_path:
        user_ref_circle = get_user_reference_circle(
            reference_path, input_files, hough, strong_denoise
        )
        if user_ref_circle:
            return user_ref_circle, input_files[reference_path]

    reference_file = auto_select_reference(
        input_files.values(),
        hough,
        strong_denoise,
    )
    if not reference_file:
        return None
    assert reference_file.image
    ref_circle = detect_circle(
        reference_file.image,
        hough,
        strong_denoise=strong_denoise,
    )
    if not ref_circle:
        logging.error("未能在参考图像中检测到月食圆")
        return None
    return ref_circle, reference_file


def do_shift(img: Image, shift: Vector[float]) -> Image:
    M = np.array([[1, 0, shift.x], [0, 1, shift.y]], dtype=np.float32)

    shifted = cv2.warpAffine(
        img.rgb,
        M,
        img.widthXheight,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return Image(rgb=shifted)


def align(
    img: Image,
    circle: Circle,
    ref_img: Image,
    ref_circle: Circle,
    use_advanced_alignment: bool,
) -> Image:
    shift = ref_circle.center - circle.center
    logging.debug(f"初始对齐: shift=({shift.x:.1f},{shift.y:.1f})")
    img = do_shift(img, shift)

    if use_advanced_alignment and (
        shift := advanced_detect_shift(img, ref_img, ref_circle)
    ):
        img = do_shift(img, shift)

    if shift := detect_mask_phase_shift(img, ref_img, ref_circle):
        img = do_shift(img, shift)
    return img


# ------------------ 单图像处理 ------------------
def process_single_image(
    input_file: ImageFile,
    output_dir: Path,
    ref_image: Image,
    ref_circle: Circle,
    hough: HoughParams,
    last_circle: Circle | None,
    use_advanced_alignment: bool = False,
    strong_denoise: bool = False,
):

    start_time = time.time()
    input_image = input_file.image
    if input_image is None:
        return None
    circle = detect_circle(
        input_image,
        hough,
        strong_denoise=strong_denoise,
        prev_circle=last_circle,
    )
    logging.info(f"处理{input_file.path.name} 耗时 {time.time()-start_time:.2f}s")
    if not circle:
        return None

    output_image = align(
        input_image, circle, ref_image, ref_circle, use_advanced_alignment
    )
    output_image.icc = input_image.icc
    output_file = ImageFile(output_dir / f"{input_file.path.name}", mode="w")
    # 保存
    output_file.image = output_image
    output_file.save()

    return circle


# ------------------ 主流程 ------------------


def process_images(
    input_dir: Path,
    output_dir: Path,
    hough: HoughParams,
    reference_path: Path | None = None,
    use_advanced_alignment=False,
    strong_denoise=False,
):
    """
    月食图像增量对齐主函数 - 重构后的协调器版本

    将原本的单体大函数拆分为多个职责明确的小函数，提高可维护性和可测试性
    """

    # 1. 设置目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载图像文件
    input_files = ImageFile.load(input_dir)

    # 记录基本信息
    logging.info(f"处理模式: 增量处理 (边检测边保存)")
    logging.info(f"文件总数: {len(input_files)}")
    logging.info(f"多ROI精配准: {'启用' if use_advanced_alignment else '禁用'}")

    # 3. 选择参考图像
    if not (
        rt := get_reference_circle(reference_path, input_files, hough, strong_denoise)
    ):
        logging.error("未能确定参考图像，处理终止")
        return
    ref_circle, ref_file = rt
    assert ref_file.image
    logging.info(
        f"🎯 参考图像: {ref_file.path.name}, 圆心=({ref_circle.x:.1f},{ref_circle.y:.1f}), 半径={ref_circle.radius:.1f}px"
    )

    # 4. 处理所有图像
    logging.info(f"\n阶段 2/2: 顺序处理所有图像...")
    success_count = 0
    failed_files: list[ImageFile] = []

    # 以参考图圆作为先验，后续逐帧更新
    last_circle: Circle | None = ref_circle

    for input_file in input_files.values():
        if input_file == ref_file:
            # 参考图像直接复制到输出目录
            output_file = ImageFile(output_dir / f"{input_file.path.name}", mode="w")
            output_file.image = ref_file.image
            output_file.save()
            success_count += 1
            new_last_circle = ref_circle
            continue

        # 处理单个图像
        new_last_circle = process_single_image(
            input_file=input_file,
            output_dir=output_dir,
            ref_image=ref_file.image,
            ref_circle=ref_circle,
            hough=hough,
            last_circle=last_circle,
            use_advanced_alignment=use_advanced_alignment,
            strong_denoise=strong_denoise,
        )

        if new_last_circle:
            success_count += 1
            last_circle = new_last_circle
        else:
            failed_files.append(input_file)

    logging.info(f"增量对齐完成! 成功对齐 {success_count}/{len(input_files)} 张图像")

    logging.info(
        f"对齐算法: {'多ROI精配准（仅平移）' if use_advanced_alignment else 'PHD2圆心算法'}"
    )
    if failed_files:
        failed_file_names = [f.path.name for f in failed_files[:5]]
        head = ", ".join(failed_file_names) + ("..." if len(failed_files) > 5 else "")
        logging.info(f"失败文件({len(failed_files)}): {head}")

    logging.debug(f"当前内存使用: {get_memory_usage_mb():.1f} MB")
