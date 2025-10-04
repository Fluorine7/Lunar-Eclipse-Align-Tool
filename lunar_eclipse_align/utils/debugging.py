import cv2
import numpy as np
from numpy.typing import NDArray
from lunar_eclipse_align.utils.constants import DEBUG
from lunar_eclipse_align.utils.data_types import Circle, PointArray


def debug_to_bgr(img_array: NDArray) -> NDArray:
    """将图像转换为BGR uint8格式以便显示（仅在DEBUG模式下可用）"""
    # 归一化到uint8
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    elif img_array.dtype in [np.float32, np.float64]:
        img_array = img_array.astype(np.uint8)

    if len(img_array.shape) == 2:  # 灰度图
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def debug_draw_circle(
    bgr_array: NDArray,
    circle: Circle,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> NDArray:
    """在图像上绘制圆（仅在DEBUG模式下可用）"""
    bgr_array = bgr_array.copy()
    center = (round(circle.x), round(circle.y))
    radius = round(circle.radius)
    cv2.circle(bgr_array, center, radius, color, thickness)
    cv2.circle(bgr_array, center, 2, color, -1)  # 红色中心点
    return bgr_array


def debug_show(img_array: NDArray, *circles):
    """使用OpenCV显示图像进行调试（仅在DEBUG模式下可用）"""
    bgr_array = debug_to_bgr(img_array)
    # 如果提供了圆，绘制圆
    for circle in circles:
        bgr_array = debug_draw_circle(bgr_array, circle)
    cv2.imshow("Debug View", bgr_array)
    cv2.waitKey(0)  # 等待任意按键
    cv2.destroyAllWindows()  # 关闭所有窗口


def debug_overlap(ref_img_array: NDArray, img_array: NDArray, circle: Circle | None):
    """使用OpenCV显示图像进行调试（仅在DEBUG模式下可用）"""
    if not DEBUG:
        return

    ref_bgr_array = debug_to_bgr(ref_img_array)
    bgr_array = debug_to_bgr(img_array)
    ref_bgr_array = ref_bgr_array * np.array(
        [0, 1, 0], dtype=np.uint8
    )  # 只保留蓝色通道
    bgr_array = bgr_array * np.array([1, 0, 0], dtype=np.uint8)  # 只保留绿色

    overlap_bgr = ref_bgr_array + bgr_array

    # 如果提供了圆，绘制圆
    if circle is not None:
        overlap_bgr = debug_draw_circle(overlap_bgr, circle)
    cv2.imshow("Debug Overlap View", overlap_bgr)
    cv2.waitKey(0)  # 等待任意按键
    cv2.destroyAllWindows()  # 关闭所有窗口


def debug_draw_edge_points(
    bgr_array: NDArray,
    edge_points: PointArray,
    color: tuple[int, int, int],
    thickness: int,
) -> NDArray:
    """
    在图像上绘制边缘点

    Args:
        img_array: 输入图像（灰度或RGB）
        edge_points: 边缘点集合 (PointArray 对象)
        color: 点的颜色 (B, G, R)，默认绿色
        thickness: 点的半径（像素），默认2

    Returns:
        绘制了边缘点的图像副本
    """
    bgr_array = bgr_array.copy()

    # 用小圆圈画每个点
    for point in edge_points:
        cv2.circle(bgr_array, (round(point.x), round(point.y)), thickness, color, -1)

    return bgr_array


def debug_show_edge(
    img_array: NDArray,
    *edges: PointArray,
):
    # 转换为BGR显示
    img_bgr = debug_to_bgr(img_array)
    # 绘制边缘点
    for edge in edges:
        img_bgr = debug_draw_edge_points(img_bgr, edge, color=(0, 0, 255), thickness=2)

    # 显示窗口
    cv2.imshow("Debug View", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
