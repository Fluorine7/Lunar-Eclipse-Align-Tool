"""
图像对齐功能测试
"""

from pathlib import Path
from lunar_eclipse_align.core.pipeline import align
from lunar_eclipse_align.core.circle_detection import detect_circle
from lunar_eclipse_align.utils.image import ImageFile
from tests.utils import load_images_config, load_params_config


# ============ 手动测试配置 ============
# 修改这里选择要测试的图像
MANUAL_TEST_REF_IMAGE = "left-top"  # 参考图像
MANUAL_TEST_TARGET_IMAGE = "right-bottom"  # 待对齐的目标图像
USE_ADVANCED_ALIGNMENT = False  # 是否使用高级对齐
MAX_TOLERANCE = 2.0  # 对齐后允许的最大圆心误差（像素）


def test_align_manually():
    """手动指定图像进行对齐测试 - 修改上面的常量来选择测试对象"""
    images_config = load_images_config()
    params_config = load_params_config()

    # 获取参考图像配置
    ref_img_data = images_config[MANUAL_TEST_REF_IMAGE]
    ref_param_name = ref_img_data.default_params
    ref_param_data = params_config[ref_param_name]

    # 获取目标图像配置
    target_img_data = images_config[MANUAL_TEST_TARGET_IMAGE]
    target_param_name = target_img_data.default_params
    target_param_data = params_config[target_param_name]

    # 打印测试配置
    print(f"\n[对齐测试配置]")
    print(f"  参考图像: {MANUAL_TEST_REF_IMAGE} (参数: {ref_param_name})")
    print(f"  目标图像: {MANUAL_TEST_TARGET_IMAGE} (参数: {target_param_name})")
    print(f"  高级对齐: {USE_ADVANCED_ALIGNMENT}")

    # 加载参考图像
    ref_img_path = Path(ref_img_data.path)
    ref_image = ImageFile(ref_img_path).image
    assert ref_image is not None, f"无法读取参考图像: {ref_img_path}"

    # 加载目标图像
    target_img_path = Path(target_img_data.path)
    target_image = ImageFile(target_img_path).image
    assert target_image is not None, f"无法读取目标图像: {target_img_path}"

    # 检测参考图像的圆
    ref_params = ref_param_data.to_hough_params()
    ref_circle = detect_circle(ref_image, ref_params)
    assert ref_circle is not None, f"参考图像圆检测失败: {MANUAL_TEST_REF_IMAGE}"

    # 检测目标图像的圆
    target_params = target_param_data.to_hough_params()
    target_circle = detect_circle(target_image, target_params)
    assert target_circle is not None, f"目标图像圆检测失败: {MANUAL_TEST_TARGET_IMAGE}"

    # 计算对齐前的圆心距离
    initial_distance = (target_circle.center - ref_circle.center).norm()

    # 执行对齐
    aligned_image = align(
        target_image, target_circle, ref_image, ref_circle, USE_ADVANCED_ALIGNMENT
    )

    # 对齐后重新检测圆
    aligned_circle = detect_circle(aligned_image, target_params)
    assert aligned_circle is not None, "对齐后圆检测失败"

    # 计算对齐后的圆心距离
    final_distance = (aligned_circle.center - ref_circle.center).norm()

    # 验证对齐效果：对齐后的距离应该明显小于对齐前
    # 允许的最大误差（像素）
    assert final_distance < MAX_TOLERANCE, (
        f"对齐后圆心距离 ({final_distance:.2f}) 超过容差 ({MAX_TOLERANCE})\n"
        f"对齐前距离: {initial_distance:.2f}\n"
        f"对齐改善: {initial_distance - final_distance:.2f}"
    )

    # 验证对齐有改善（除非本来就很接近）

    assert final_distance < initial_distance, (
        f"对齐后反而变差了！\n"
        f"对齐前: {initial_distance:.2f}\n"
        f"对齐后: {final_distance:.2f}"
    )
