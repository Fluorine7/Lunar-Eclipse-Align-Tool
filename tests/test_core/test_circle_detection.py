"""
圆检测参数化测试 - 使用配置文件
"""

from dataclasses import dataclass
import pytest
import yaml
from pathlib import Path
from lunar_eclipse_align.core.circle_detection import detect_circle
from lunar_eclipse_align.utils.data_types import HoughParams
from lunar_eclipse_align.utils.image import ImageFile
from tests.utils import (
    check_result,
    format_failure_message,
    load_images_config,
    load_params_config,
)


# ============ 统一的失败消息格式化函数 ============


# ============ 测试1: 所有图片使用各自默认参数 ============
@pytest.mark.parametrize("img_name", load_images_config().keys())
def test_all_images(img_name):
    """测试所有图像，每个图像使用其默认参数"""
    images_config = load_images_config()
    params_config = load_params_config()

    img_data = images_config[img_name]
    param_name = img_data.default_params
    param_data = params_config[param_name]

    # 打印测试配置
    print(f"\n[测试配置] 图像={img_name}, 参数={param_name}")

    # 加载图像
    img_path = Path(img_data.path)
    image = ImageFile(img_path).image
    assert image is not None, f"无法读取图像: {img_path}"

    # 创建参数
    params = param_data.to_hough_params()

    # 执行检测
    result = detect_circle(image, params)

    # 验证结果
    expected = img_data.expected_result
    tolerance = img_data.tolerance

    if result is None:
        pytest.fail(
            format_failure_message(
                img_name, img_data, param_name, param_data, params, expected
            )
        )

    in_tolerance, _, _, _ = check_result(result, expected, tolerance)

    assert in_tolerance, format_failure_message(
        img_name, img_data, param_name, param_data, params, expected, result, tolerance
    )


# ============ 测试2: 手动指定图像和参数 ============

# 在这里修改你想测试的图像和参数
MANUAL_TEST_IMAGE = "deficient_edge_bad"  # 修改这里选择图像
MANUAL_TEST_PARAM = "basic"  # 修改这里选择参数


def test_manually():
    """手动指定图像和参数进行测试 - 修改上面的常量来选择测试对象"""
    images_config = load_images_config()
    params_config = load_params_config()

    img_data = images_config[MANUAL_TEST_IMAGE]
    param_data = params_config[MANUAL_TEST_PARAM]

    # 打印测试配置
    print(f"\n[测试配置] 图像={MANUAL_TEST_IMAGE}, 参数={MANUAL_TEST_PARAM}")

    # 加载图像
    img_path = Path(img_data.path)
    image = ImageFile(img_path).image
    assert image is not None, f"无法读取图像: {img_path}"

    # 创建参数
    params = param_data.to_hough_params()

    # 执行检测
    result = detect_circle(image, params)

    # 验证结果
    expected = img_data.expected_result
    tolerance = img_data.tolerance

    if result is None:
        pytest.fail(
            format_failure_message(
                MANUAL_TEST_IMAGE,
                img_data,
                MANUAL_TEST_PARAM,
                param_data,
                params,
                expected,
            )
        )

    in_tolerance, _, _, _ = check_result(result, expected, tolerance)

    assert in_tolerance, format_failure_message(
        MANUAL_TEST_IMAGE,
        img_data,
        MANUAL_TEST_PARAM,
        param_data,
        params,
        expected,
        result,
        tolerance,
    )


# ============ 测试3: 探索特定图像的所有参数 ============

EXPLORE_IMAGE = "bright"  # 修改这里选择要探索的图像


@pytest.mark.parametrize("param_name", load_params_config().keys())
def test_all_params_for_image(param_name):
    """对指定图像测试所有参数配置 - 用于参数调优"""
    images_config = load_images_config()
    params_config = load_params_config()

    assert (
        EXPLORE_IMAGE in images_config
    ), f"图像 '{EXPLORE_IMAGE}' 不存在。可用的图像: {list(images_config.keys())}"

    img_data = images_config[EXPLORE_IMAGE]
    param_data = params_config[param_name]

    # 打印测试配置
    print(f"\n[探索] 图像={EXPLORE_IMAGE}, 参数={param_name}")

    # 加载图像
    img_path = Path(img_data.path)
    image = ImageFile(img_path).image
    assert image is not None, f"无法读取图像: {img_path}"

    # 创建参数
    params = param_data.to_hough_params()

    # 执行检测
    result = detect_circle(image, params)

    # 验证结果
    expected = img_data.expected_result
    tolerance = img_data.tolerance

    if result is None:
        pytest.fail(
            format_failure_message(
                EXPLORE_IMAGE, img_data, param_name, param_data, params, expected
            )
        )

    in_tolerance, _, _, _ = check_result(result, expected, tolerance)

    assert in_tolerance, format_failure_message(
        EXPLORE_IMAGE,
        img_data,
        param_name,
        param_data,
        params,
        expected,
        result,
        tolerance,
    )
