from pathlib import Path

import yaml
from pydantic import BaseModel

from lunar_eclipse_align.utils.data_types import Circle, HoughParams


TEST_DATA_DIR = Path(__file__).parent / "data"


class ImageConfig(BaseModel):
    model_config = {"frozen": True}

    class Tolerance(BaseModel):
        model_config = {"frozen": True}
        center: float
        radius: float

    class ExpectedCircle(BaseModel):
        model_config = {"frozen": True}
        x: float
        y: float
        radius: float

    path: str
    default_params: str
    expected_result: ExpectedCircle
    description: str | None = None
    tolerance: Tolerance = Tolerance(center=2.0, radius=3.0)


class ParamsConfig(BaseModel):
    model_config = {"frozen": True}

    param1: int
    param2: int
    minRadius: int
    maxRadius: int
    description: str | None = None

    def to_hough_params(self) -> HoughParams:
        return HoughParams(
            param1=self.param1,
            param2=self.param2,
            minRadius=self.minRadius,
            maxRadius=self.maxRadius,
        )


def load_images_config():
    """加载图像配置"""
    with open(TEST_DATA_DIR / "images.yaml") as f:
        return {
            name: ImageConfig(**data)
            for name, data in yaml.safe_load(f)["images"].items()
        }


def load_params_config():
    """加载参数配置"""
    with open(TEST_DATA_DIR / "hough_params.yaml") as f:
        return {
            name: ParamsConfig(**data)
            for name, data in yaml.safe_load(f)["param_sets"].items()
        }


def format_failure_message(
    img_name,
    img_data,
    param_name,
    param_data,
    params,
    expected,
    result=None,
    tolerance=None,
):
    """
    统一的失败消息格式化函数

    Args:
        result: 检测结果，None 表示未检测到圆
        tolerance: 容差，仅当 result 不为 None 时需要
    """
    # 头部和基本信息
    if result is None:
        title = "检测失败: 未检测到圆"
    else:
        title = "检测结果超出容差范围"

    msg = (
        f"\n"
        f"{'='*60}\n"
        f"{title}\n"
        f"{'='*60}\n"
        f"图像: {img_name} ({img_data['description']})\n"
        f"参数: {param_name} ({param_data['description']})\n"
        f"  minRadius={params.minRadius}, maxRadius={params.maxRadius}\n"
        f"  param1={params.param1}, param2={params.param2}\n"
    )

    # 如果有检测结果，添加结果和差异分析
    if result is not None and tolerance is not None:
        x_diff = abs(result.x - expected.x)
        y_diff = abs(result.y - expected.y)
        radius_diff = abs(result.radius - expected.radius)

        msg += (
            f"\n"
            f"检测结果: 圆心({result.x:.2f}, {result.y:.2f}), 半径={result.radius:.2f}\n"
            f"预期结果: 圆心({expected['x']:.2f}, {expected['y']:.2f}), 半径={expected['radius']:.2f}\n"
            f"\n"
            f"差异分析:\n"
            f"  X 差异    = {x_diff:6.2f} (容差 < {tolerance['x']}){' ✓' if x_diff < tolerance['x'] else ' ✗'}\n"
            f"  Y 差异    = {y_diff:6.2f} (容差 < {tolerance['y']}){' ✓' if y_diff < tolerance['y'] else ' ✗'}\n"
            f"  半径差异  = {radius_diff:6.2f} (容差 < {tolerance['radius']}){' ✓' if radius_diff < tolerance['radius'] else ' ✗'}\n"
        )
    else:
        # 只有预期结果
        msg += f"预期结果: 圆心({expected['x']:.2f}, {expected['y']:.2f}), 半径={expected['radius']:.2f}\n"

    msg += f"{'='*60}"
    return msg


def check_result(
    result: Circle,
    expected: ImageConfig.ExpectedCircle,
    tolerance: ImageConfig.Tolerance,
):
    """检查结果是否在容差范围内"""

    x_diff = abs(result.x - expected.x)
    y_diff = abs(result.y - expected.y)
    radius_diff = abs(result.radius - expected.radius)

    in_tolerance = (
        x_diff < tolerance.center
        and y_diff < tolerance.center
        and radius_diff < tolerance.radius
    )

    return in_tolerance, x_diff, y_diff, radius_diff
