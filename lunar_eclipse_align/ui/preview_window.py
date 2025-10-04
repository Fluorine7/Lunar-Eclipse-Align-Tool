# ui_windows_pyside6.py
"""
PySide6版本的窗口类
包含调试窗口、预览窗口、进度窗口等
"""
import logging
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING

from lunar_eclipse_align.core.circle_detection import detect_circle, detect_circle_quick

from lunar_eclipse_align.utils.image import Image, ImageFile
from lunar_eclipse_align.ui.select_rect import InteractiveGraphicsView
from lunar_eclipse_align.utils.constants import SUPPORTED_EXTS
from lunar_eclipse_align.utils.data_types import ZERO, Circle, Vector

if TYPE_CHECKING:
    from lunar_eclipse_align.ui.main_window import UniversalLunarAlignApp

from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSlider,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QGraphicsScene,
    QApplication,
)
from PySide6.QtCore import (
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QFont,
    QPixmap,
    QImage,
    QColor,
    QPen,
    QBrush,
)


class PreviewWindow(QDialog):
    """预览窗口"""

    def __init__(self, app_controller: "UniversalLunarAlignApp"):
        super().__init__(app_controller)
        self.app: "UniversalLunarAlignApp" = app_controller
        self.setWindowTitle("预览与半径估计")
        self.resize(1100, 650)
        self.setMinimumSize(900, 500)

        # 图像相关
        self.current_path: Path | None = None
        self.preview_scale: float = 1.0

        # 矩形选择相关
        self.detected_circle: Circle | None = None  # (cx, cy, r) 检测到的圆
        self.preview_img: Image | None = None
        self.params = self.app.params.copy()
        self._center_window()
        self._setup_ui()

    def showEvent(self, event):
        """窗口显示时自动弹出文件选择（仅当没有图像时）"""
        super().showEvent(event)
        if self.preview_img is None:
            QTimer.singleShot(100, self.choose_image)

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

        self.choose_image_btn = QPushButton("选择参考图像")
        self.choose_image_btn.clicked.connect(self.choose_image)
        toolbar1_layout.addWidget(self.choose_image_btn)

        toolbar1_layout.addWidget(QLabel("增减范围: "))
        self.delta_spin = QSpinBox()
        self.delta_spin.setRange(0, 100)
        self.delta_spin.setValue(5)
        toolbar1_layout.addWidget(self.delta_spin)
        toolbar1_layout.addWidget(QLabel("%"))

        self.est_label = QLabel("估计半径: —")
        toolbar1_layout.addWidget(self.est_label)

        toolbar1_layout.addStretch()

        layout.addWidget(toolbar1)

        # 第二行工具条
        toolbar2 = QWidget()
        toolbar2_layout = QHBoxLayout(toolbar2)
        toolbar2_layout.setContentsMargins(0, 0, 0, 0)

        self.apply_btn = QPushButton("应用参数和参考图像")
        self.apply_btn.clicked.connect(self.apply_setting)
        toolbar2_layout.addWidget(self.apply_btn)

        toolbar2_layout.addStretch()

        layout.addWidget(toolbar2)

        # 参数调整区域
        param_group = QWidget()
        param_layout = QVBoxLayout(param_group)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(8)

        # 边缘敏感度 (param1)
        edge_row = QWidget()
        edge_layout = QHBoxLayout(edge_row)
        edge_layout.setContentsMargins(0, 0, 0, 0)

        edge_label = QLabel("边缘敏感度:")
        edge_layout.addWidget(edge_label, 0)

        self.param1_slider = QSlider(Qt.Orientation.Horizontal)
        self.param1_slider.setRange(20, 150)
        self.param1_slider.setValue(self.params.param1)
        edge_layout.addWidget(self.param1_slider, 1)

        self.param1_value_label = QLabel(str(self.params.param1))
        self.param1_value_label.setMinimumWidth(30)
        edge_layout.addWidget(self.param1_value_label, 0)

        # 帮助图标
        edge_help_icon = QLabel("❓")
        edge_help_icon.setToolTip(
            "控制边缘检测的敏感程度\n"
            "• 数值越高：只检测强边缘，减少噪声干扰\n"
            "• 数值越低：检测更多边缘，可能包含噪声\n"
            "• 建议范围：20-150\n"
            "• 月食图像推荐：50-80（明亮）, 30-50（暗淡）"
        )
        edge_help_icon.setStyleSheet("color: #2196F3; font-size: 12px;")
        edge_layout.addWidget(edge_help_icon, 0)

        param_layout.addWidget(edge_row)

        # 圆心阈值 (param2)
        center_row = QWidget()
        center_layout = QHBoxLayout(center_row)
        center_layout.setContentsMargins(0, 0, 0, 0)

        center_label = QLabel("圆心阈值:")
        center_layout.addWidget(center_label, 0)

        self.param2_slider = QSlider(Qt.Orientation.Horizontal)
        self.param2_slider.setRange(10, 50)
        self.param2_slider.setValue(self.params.param2)
        center_layout.addWidget(self.param2_slider, 1)

        self.param2_value_label = QLabel(str(self.params.param2))
        self.param2_value_label.setMinimumWidth(30)
        center_layout.addWidget(self.param2_value_label, 0)

        center_help_icon = QLabel("❓")
        center_help_icon.setToolTip(
            "控制圆心检测的严格程度\n"
            "• 数值越高：要求更多证据确认圆心，结果更可靠\n"
            "• 数值越低：更容易检测到圆，但可能有误检\n"
            "• 建议范围：10-50\n"
            "• 月食图像推荐：25-35（清晰）, 15-25（模糊）"
        )
        center_help_icon.setStyleSheet("color: #2196F3; font-size: 12px;")
        center_layout.addWidget(center_help_icon, 0)

        param_layout.addWidget(center_row)

        layout.addWidget(param_group)

        # 图像显示区域 - 使用新的交互式视图
        self.graphics_view = InteractiveGraphicsView()
        self.graphics_view.rect_drawed.connect(self.detect_circle_n_draw)
        self.graphics_view.rect_resized.connect(self.detect_circle_n_draw)
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(51, 51, 51)))

        layout.addWidget(self.graphics_view)

        self.drawed_circle = None
        self.drawed_center = None

        # 设置参数滑块连接
        self._setup_param_connections()

    def _setup_param_connections(self):
        """设置参数滑块的连接"""
        # 从主窗口读取当前参数值初始化滑块
        self.param1_slider.setValue(self.params.param1)
        self.param2_slider.setValue(self.params.param2)
        self.param1_value_label.setText(str(self.params.param1))
        self.param2_value_label.setText(str(self.params.param2))

        # 连接信号
        self.param1_slider.valueChanged.connect(self._on_param_changed)
        self.param2_slider.valueChanged.connect(self._on_param_changed)
        self.param1_slider.sliderReleased.connect(self.detect_circle_n_draw)
        self.param2_slider.sliderReleased.connect(self.detect_circle_n_draw)

    def _on_param_changed(self):
        """参数变化时直接更新主窗口参数并刷新显示"""
        # 直接更新主窗口的参数对象
        self.params.param1 = self.param1_slider.value()
        self.params.param2 = self.param2_slider.value()

        # 更新本窗口显示
        self.param1_value_label.setText(str(self.param1_slider.value()))
        self.param2_value_label.setText(str(self.param2_slider.value()))

    def choose_image(self):
        """选择预览图像"""
        input_path = self.app.input_path
        initial_dir = (
            self.app.input_path if input_path and input_path.is_dir() else Path()
        )
        file_filter = f"支持的图像 ( {' '.join(SUPPORTED_EXTS)} );;所有文件 (*.*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图像", str(initial_dir), file_filter
        )

        if file_path:
            self.current_path = Path(file_path)
            img = ImageFile(self.current_path).image
            if img is None:
                QMessageBox.critical(self, "错误", "无法读取该图像。")
                return
            self.preview_img = img

            self.setWindowTitle(f"预览与半径估计 - {self.current_path.name}")
            self.refresh()
            self.display_image()

    def refresh(self):
        self.graphics_scene.clear()
        self.graphics_view.current_rect = None
        self.detected_circle = None  # 重置检测结果
        self.preview_scale = 1.0
        self.drawed_circle = None
        self.drawed_center = None

    def display_image(self):
        """显示图像"""
        if self.preview_img is None:
            # 显示提示
            self.graphics_scene.clear()
            text = self.graphics_scene.addText("请选择参考图像，在图上拖拽鼠标框选月亮")
            text.setDefaultTextColor(QColor("lightgray"))
            return

        # 转换为QPixmap
        W, H = self.preview_img.widthXheight
        q_img = QImage(
            self.preview_img.rgb_8bit.data,
            W,
            H,
            self.preview_img.rgb_8bit.strides[0],
            QImage.Format.Format_RGB888,
        )
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
        self.graphics_scene.addPixmap(pixmap)
        self.preview_scale = scale
        self.detect_circle_n_draw()

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

        # 画圆心
        self.drawed_center = self.graphics_scene.addEllipse(
            cx - 3,
            cy - 3,
            6,
            6,
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(255, 77, 79)),
        )

    def detect_circle_n_draw(self):
        """检测圆并绘制"""
        if self.preview_img is None:
            return
        img = self.preview_img
        delta = ZERO
        max_side = max(img.width, img.height)
        if self.graphics_view.current_rect:
            # 获取矩形区域（场景坐标）
            rect = self.graphics_view.current_rect.rect()

            # 转换为图像坐标
            scale = self.preview_scale

            x, y, width, height = rect.x(), rect.y(), rect.width(), rect.height()

            top, left, bottom, right = (
                y / scale,
                x / scale,
                (y + height) / scale,
                (x + width) / scale,
            )
            logging.debug(
                f"预览：矩形区域 {top=:.2f}, {left=:.2f}, {bottom=:.2f}, {right=:.2f}"
            )
            # 估计半径和中心
            img = Image(
                rgb=self.preview_img.rgb[
                    round(top) : round(bottom),
                    round(left) : round(right),
                ]
            )
            max_side = max(img.width, img.height)
            # 直接使用主窗口的参数，只临时修改半径范围用于检测

            delta = Vector(left, top)

        self.params.minRadius = max_side // 5  # 临时设置检测用的半径范围
        self.params.maxRadius = max_side // 2

        circle = detect_circle_quick(img, self.params, self.app.enable_strong_denoise)
        if not circle:
            logging.error("预览：圆检测失败")
            return
            # 回退到估计值

        self.detected_circle = circle.shift(delta)
        logging.info(f"预览：检测到圆")
        logging.debug(str(self.detected_circle))

        # 重新绘制以显示检测结果
        self._draw_circle(self.detected_circle.scale(self.preview_scale))

        # 更新标签

        self.est_label.setText(f"估计半径: {self.detected_circle.radius:.2f} px")

    def apply_setting(self):
        """应用检测到的半径"""
        if not self.current_path:
            QMessageBox.warning(self, "提示", "请先选择参考图像。")
            return
        if self.detected_circle is None:
            QMessageBox.warning(self, "提示", "请先检测半径。")
            return

        delta = self.delta_spin.value() / 100
        r = self.detected_circle.radius
        min_r = max(1, floor(r * (1 - delta)))
        max_r = max(2, ceil(r * (1 + delta)))
        self.params.minRadius = min_r
        self.params.maxRadius = max_r

        logging.info(
            f"应用预览参数: 最小半径 {min_r}， 最大半径 {max_r}， param1 {self.params.param1}, param2 {self.params.param2}"
        )
        logging.info(f"设置参考图像: {str(self.current_path)}")
        # 更新主窗口显示
        self.app.update_hough_params(self.params)

        # 设置参考图像
        self.app.update_reference_path(self.current_path)

        QMessageBox.information(
            self,
            "已应用",
            f"检测半径: {self.detected_circle.radius:.2f} px\n"
            f"设置范围: {min_r} - {max_r}\n"
            f"边缘敏感度: {self.params.param1}\n"
            f"圆心阈值: {self.params.param2}\n"
            f"参考图像: {self.current_path.name if self.current_path else '无'}",
        )


class ProgressWindow(QDialog):
    """进度窗口"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("处理进度")
        self.resize(400, 150)
        self.setFixedSize(400, 150)  # 固定大小

        self._setup_ui()
        self._center_window()

    def _center_window(self):
        """居中显示窗口"""
        screen = self.screen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)

    def _setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # 状态标签
        self.status_label = QLabel("准备开始...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # 百分比标签
        self.percent_label = QLabel("0%")
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.percent_label)

    def update_progress(self, progress, status):
        """更新进度"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        self.percent_label.setText(f"{progress}%")
        QApplication.processEvents()  # 强制更新UI
