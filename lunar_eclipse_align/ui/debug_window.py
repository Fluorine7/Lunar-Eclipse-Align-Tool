import logging
from math import floor
import os
from pathlib import Path
import threading
import queue
import cv2
import numpy as np
from typing import TYPE_CHECKING, Iterator

from lunar_eclipse_align.core.circle_detection import build_analysis_mask, detect_circle

from lunar_eclipse_align.utils.data_types import Circle
from lunar_eclipse_align.utils.image import Image, ImageFile


if TYPE_CHECKING:
    from lunar_eclipse_align.ui.main_window import UniversalLunarAlignApp
from numpy.typing import NDArray
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QGraphicsView,
    QGraphicsScene,
    QPushButton,
    QMessageBox,
    QLabel,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
)
from PySide6.QtCore import (
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QPixmap,
    QImage,
    QColor,
    QPen,
    QBrush,
    QIcon,
)

# 导入工具函数


class DebugWindow(QDialog):
    """调试窗口：可选择样张并实时调节参数"""

    def __init__(self, app_controller: "UniversalLunarAlignApp"):
        super().__init__(app_controller)
        self.app: "UniversalLunarAlignApp" = app_controller
        self.setWindowTitle("调试窗口（参数实时预览）")
        self.resize(980, 680)
        self.setMinimumSize(760, 520)

        # 图像数据
        self.current_scale = 1.0

        self.show_mask: bool = False
        self.filename_label: QLabel | None = None

        self._setup_ui()
        self._center_window()
        self.refresh()

    def _center_window(self):
        """居中显示窗口"""
        self.adjustSize()
        screen = self.screen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)

    def _setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # 第一行工具条
        toolbar1 = QWidget()
        toolbar1_layout = QHBoxLayout(toolbar1)
        toolbar1_layout.setContentsMargins(0, 0, 0, 0)

        # 左箭头按钮
        self.prev_button = QPushButton()
        self.prev_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_ArrowLeft)
        )
        self.prev_button.setFixedSize(40, 30)
        self.prev_button.setToolTip("上一张图片")
        self.prev_button.clicked.connect(self._on_prev_image)
        toolbar1_layout.addWidget(self.prev_button)

        # 右箭头按钮
        self.next_button = QPushButton()
        self.next_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_ArrowRight)
        )
        self.next_button.setFixedSize(40, 30)
        self.next_button.setToolTip("下一张图片")
        self.next_button.clicked.connect(self._on_next_image)
        toolbar1_layout.addWidget(self.next_button)

        # 文件名显示Label
        self.filename_label = QLabel("未加载图片")
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        toolbar1_layout.addWidget(self.filename_label)

        toolbar1_layout.addStretch()

        layout.addWidget(toolbar1)

        # 第二行工具条
        toolbar2 = QWidget()
        toolbar2_layout = QHBoxLayout(toolbar2)
        toolbar2_layout.setContentsMargins(0, 0, 0, 0)

        self.show_mask_check = QCheckBox("显示分析区域")
        self.show_mask_check.setChecked(self.show_mask)
        self.show_mask_check.stateChanged.connect(self.on_show_mask_changed)
        toolbar2_layout.addWidget(self.show_mask_check)

        layout.addWidget(toolbar2)

        # 图像显示区域
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(34, 34, 34)))
        layout.addWidget(self.graphics_view)

    def showEvent(self, event):
        """窗口显示时自动弹出文件选择（仅当没有图像时）"""
        super().showEvent(event)
        if self.current_dir and self.current_dir == self.app.input_path:
            return

        if self.app.input_path is None or not self.app.input_path.is_dir():
            QMessageBox.warning(
                self,
                "错误",
                "输入目录未设置！请先在主窗口中设置输入目录。",
            )
            event.ignore()
            QTimer.singleShot(0, self.close)  # 延迟关闭窗口，避免事件冲突
            return

        self.refresh()
        self.current_dir = self.app.input_path
        # pick one image in the input directory
        self.image_files = list(ImageFile.load(self.current_dir).items())
        self.current_index = 0
        QTimer.singleShot(100, self.debug_display_image)

    def on_show_mask_changed(self, state):
        """显示分析区域状态改变"""
        self.show_mask = state == Qt.CheckState.Checked.value
        self.refresh_pixmap()

    def _on_prev_image(self):
        """切换到上一张图片"""
        if not self.image_files:
            return
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.debug_display_image()

    def _on_next_image(self):
        """切换到下一张图片"""
        if not self.image_files:
            return
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.debug_display_image()

    def debug_display_image(self):
        if not (0 <= self.current_index < len(self.image_files)):
            return

        # 更新文件名显示
        current_path, imgfile = self.image_files[self.current_index]
        logging.info(f"Debug图片：{str(current_path)}")
        if self.filename_label:
            self.filename_label.setText(current_path.name)

        img = imgfile.image
        if img is None:
            return
        self.refresh_pixmap()
        circle = detect_circle(
            img,
            self.app.params,
            strong_denoise=self.app.enable_strong_denoise,
        )
        if circle:
            self._draw_circle(circle.scale(self.current_scale))

    def refresh(self):
        self.graphics_scene.clear()
        self.current_dir: Path | None = None
        self.current_index: int = 0
        self.image_files: list[tuple[Path, ImageFile]] = []
        self.current_img: Image | None = None
        self.current_path: Path | None = None
        self.drawed_center: QGraphicsEllipseItem | None = None
        self.drawed_circle: QGraphicsEllipseItem | None = None
        self.current_pixmap: QGraphicsPixmapItem | None = None
        if self.filename_label:
            self.filename_label.setText("未加载图片")

    def refresh_pixmap(self):
        """刷新显示"""
        if not (0 <= self.current_index < len(self.image_files)):
            return
        img = self.image_files[self.current_index][1].image
        if img is None:
            return
        if self.current_pixmap:
            self.graphics_scene.removeItem(self.current_pixmap)
            self.current_pixmap = None

        display_rgb = img.rgb_8bit
        # 如果显示分析区域
        if self.show_mask:

            # 构建分析掩膜
            mask: NDArray[np.bool] = build_analysis_mask(img, brightness_min=3 / 255.0)

            # 创建红色叠加
            red_overlay: NDArray[np.uint8] = np.full_like(
                display_rgb, [0, 0, 255], dtype=np.uint8
            )

            alpha = mask.astype(np.float32) * 0.35
            alpha = alpha[:, :, np.newaxis]

            display_rgb = (display_rgb * (1 - alpha) + red_overlay * alpha).astype(
                np.uint8
            )
        # 显示图像
        self.display_pixmap(display_rgb)

    def display_pixmap(self, rgb: NDArray):
        """显示图像"""
        # 转换为QPixmap
        H, W = rgb.shape[:2]
        q_img = QImage(rgb.data, W, H, rgb.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 计算缩放
        view_size = self.graphics_view.size()
        scale = min(
            view_size.width() / W,
            view_size.height() / H,
            1.0,
        )
        if scale < 1.0:
            pixmap = pixmap.scaled(
                floor(W * scale),
                floor(H * scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # 清除场景并添加新图像
        self.current_pixmap = self.graphics_scene.addPixmap(pixmap)
        self.current_scale = scale

    def _draw_circle(self, circle: Circle):
        """绘制检测到的圆"""
        if self.drawed_circle:
            self.graphics_scene.removeItem(self.drawed_circle)
            self.drawed_circle = None
        if self.drawed_center:
            self.graphics_scene.removeItem(self.drawed_center)
            self.drawed_center = None

        cx, cy, r = circle.x, circle.y, circle.radius
        # 画圆
        self.drawed_circle = self.graphics_scene.addEllipse(
            cx - r, cy - r, 2 * r, 2 * r, QPen(QColor(255, 77, 79), 2)
        )
        self.drawed_circle.setZValue(1)

        # 画圆心
        self.drawed_center = self.graphics_scene.addEllipse(
            cx - 3,
            cy - 3,
            6,
            6,
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(255, 77, 79)),
        )
        self.drawed_center.setZValue(1)
