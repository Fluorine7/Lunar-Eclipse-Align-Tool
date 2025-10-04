"""
圆检测功能单元测试
"""

import pytest

from pathlib import Path
from lunar_eclipse_align.core.circle_detection import (
    detect_circle,
    detect_circle_robust,
    standard_hough_detect,
    adaptive_hough_detect,
    contour_detect,
    padding_fallback_detect,
    final_detect,
    initial_process,
    build_detection_roi,
)
from lunar_eclipse_align.utils.data_types import HoughParams
from lunar_eclipse_align.utils.image import Image, ImageFile


@pytest.fixture
def test_setup():
    """共享的测试设置，准备图像和参数"""
    # 统一使用 ImageFile 加载图像
    img_path = Path("images/basic/DSC_3549.tif")
    image = ImageFile(img_path).image
    assert image is not None, f"无法读取测试图像: {img_path}"

    # 使用统一的霍夫参数 (经过调试验证的参数)
    params = HoughParams(minRadius=150, maxRadius=315, param1=50, param2=30)

    # 预期检测结果 (原图坐标系)
    expected_result = {"x": 841.05, "y": 717.14, "radius": 241.76}

    # 准备预处理后的数据
    gray, brightness_mode = initial_process(image, False)
    masked_gray = build_detection_roi(gray, params, None)

    return {
        "image": image,
        "params": params,
        "expected_result": expected_result,
        "gray": gray,
        "masked_gray": masked_gray,
        "brightness_mode": brightness_mode,
    }


def test_detect_circle_basic(test_setup):
    """基本的圆检测测试，使用真实月食图像"""
    image = test_setup["image"]
    params = test_setup["params"]
    expected = test_setup["expected_result"]

    # 调用 detect_circle 函数
    result = detect_circle(image, params)

    # 验证返回结果
    assert result is not None, "detect_circle 应该返回圆检测结果"

    # 验证与预期结果的差异
    x_diff = abs(result.x - expected["x"])
    y_diff = abs(result.y - expected["y"])
    radius_diff = abs(result.radius - expected["radius"])

    # 使用assert输出完整结果信息
    assert (
        x_diff < 2 and y_diff < 2 and radius_diff < 3
    ), f"""
        检测结果: 圆心({result.x:.2f},{result.y:.2f}), 半径={result.radius:.2f}
        预期结果: 圆心({expected['x']:.2f},{expected['y']:.2f}), 半径={expected['radius']:.2f}
        差异: X={x_diff:.2f}, Y={y_diff:.2f}, 半径={radius_diff:.2f}
        """


def test_detect_circle_robust(test_setup):
    """测试稳健RANSAC圆检测"""
    gray = test_setup["gray"]

    # 调用 detect_circle_robust 函数
    result = detect_circle_robust(gray)

    # 如果检测到圆，验证其属性
    if result:

        print(
            f"RANSAC检测到的圆: 中心({result.x:.1f}, {result.y:.1f}), 半径{result.radius:.1f}"
        )
    else:
        print("RANSAC未检测到圆")


def test_standard_hough_detect(test_setup):
    """测试标准霍夫圆检测"""
    masked_gray = test_setup["masked_gray"]
    params = test_setup["params"]

    # 调用 standard_hough_detect 函数
    results = standard_hough_detect(masked_gray, params)

    # 验证返回结果是列表
    assert isinstance(results, list), "standard_hough_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, "x"), "圆对象应该有 x 属性"
        assert hasattr(circle, "y"), "圆对象应该有 y 属性"
        assert hasattr(circle, "radius"), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(
            f"标准霍夫检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}"
        )
    else:
        print("标准霍夫未检测到圆")


def test_adaptive_hough_detect(test_setup):
    """测试自适应霍夫圆检测"""
    masked_gray = test_setup["masked_gray"]
    params = test_setup["params"]
    brightness_mode = test_setup["brightness_mode"]

    # 调用 adaptive_hough_detect 函数
    results = adaptive_hough_detect(masked_gray, params, brightness_mode)

    # 验证返回结果是列表
    assert isinstance(results, list), "adaptive_hough_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, "x"), "圆对象应该有 x 属性"
        assert hasattr(circle, "y"), "圆对象应该有 y 属性"
        assert hasattr(circle, "radius"), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(
            f"自适应霍夫检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}"
        )
    else:
        print("自适应霍夫未检测到圆")


def test_contour_detect(test_setup):
    """测试轮廓圆检测"""
    gray = test_setup["gray"]
    params = test_setup["params"]

    # 调用 contour_detect 函数
    results = contour_detect(gray, params)

    # 验证返回结果是列表
    assert isinstance(results, list), "contour_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, "x"), "圆对象应该有 x 属性"
        assert hasattr(circle, "y"), "圆对象应该有 y 属性"
        assert hasattr(circle, "radius"), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(
            f"轮廓检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}"
        )
    else:
        print("轮廓检测未检测到圆")


def test_padding_fallback_detect(test_setup):
    """测试padding降级圆检测"""
    gray = test_setup["gray"]
    params = test_setup["params"]

    # 调用 padding_fallback_detect 函数
    results = padding_fallback_detect(gray, params)

    # 验证返回结果是列表
    assert isinstance(results, list), "padding_fallback_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, "x"), "圆对象应该有 x 属性"
        assert hasattr(circle, "y"), "圆对象应该有 y 属性"
        assert hasattr(circle, "radius"), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(
            f"padding降级检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}"
        )
    else:
        print("padding降级检测未检测到圆")


def test_final_detect(test_setup):
    """测试最终圆检测"""
    gray = test_setup["gray"]
    params = test_setup["params"]

    # 调用 final_detect 函数
    results = final_detect(gray, params)

    # 验证返回结果是列表
    assert isinstance(results, list), "final_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, "x"), "圆对象应该有 x 属性"
        assert hasattr(circle, "y"), "圆对象应该有 y 属性"
        assert hasattr(circle, "radius"), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(
            f"最终检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}"
        )
    else:
        print("最终检测未检测到圆")


def test_detect_circle_with_crop(test_setup):
    """测试指定裁剪区域和参数的圆检测"""
    image = test_setup["image"]
    params = test_setup["params"]
    expected = test_setup["expected_result"]

    # 裁剪区域 (根据调试日志)
    top, left = 369.45, 484.18
    bottom, right = 995.90, 1115.22

    # 裁剪图像
    crop_img = Image(rgb=image.rgb[int(top) : int(bottom), int(left) : int(right)])

    # 执行检测
    result = detect_circle(crop_img, params, strong_denoise=False)

    # 验证结果
    assert result is not None, "应该检测到圆"

    # 变换坐标到原图坐标系
    original_x = result.x + left
    original_y = result.y + top

    # 验证变换后的坐标与预期结果相近 (允许一定误差)
    x_diff = abs(original_x - expected["x"])
    y_diff = abs(original_y - expected["y"])
    radius_diff = abs(result.radius - expected["radius"])

    # 使用assert输出完整结果信息
    assert (
        x_diff < 2 and y_diff < 2 and radius_diff < 3
    ), f"""
        裁剪检测: 圆心({result.x:.2f},{result.y:.2f}), 半径={result.radius:.2f}
        变换到原图: 圆心({original_x:.2f},{original_y:.2f})
        预期原图: 圆心({expected['x']:.2f},{expected['y']:.2f}), 半径={expected['radius']:.2f}
        差异: X={x_diff:.2f}, Y={y_diff:.2f}, 半径={radius_diff:.2f}
        """
