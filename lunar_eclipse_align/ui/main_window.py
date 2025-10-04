# ui_pyside6.py
"""
月食圆面对齐工具 - PySide6版本
UI界面模块
"""
import os
from pathlib import Path


from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QTextBrowser,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QGroupBox,
    QDialog,
    QFrame,
)
from PySide6.QtCore import (
    Qt,
    Signal,
    QThread,
    QObject,
    QTimer,
)
from PySide6.QtGui import QPixmap


# 导入工具函数
from lunar_eclipse_align.core.pipeline import process_images


from lunar_eclipse_align.ui.debug_window import DebugWindow
from lunar_eclipse_align.ui.preview_window import PreviewWindow, ProgressWindow
from lunar_eclipse_align.utils.constants import SUPPORTED_EXTS, SYSTEM
from lunar_eclipse_align.utils.data_types import HoughParams
from lunar_eclipse_align.utils.logging import enable_gui_logging


# 定义信号用于线程间通信
class ProgressSignal(QObject):
    """进度信号类"""

    progress_updated = Signal(int, str)  # 进度百分比，状态文本


class AlignmentThread(QThread):
    """对齐处理线程"""

    finished = Signal(bool, str)  # 是否成功，消息

    def __init__(
        self,
        in_path: Path,
        out_path: Path,
        hough: HoughParams,
        ref_path: Path | None,
        use_advanced,
        method,
        strong_denoise,
    ):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.hough = hough
        self.ref_path = ref_path
        self.use_advanced = use_advanced
        self.method = method
        self.strong_denoise = strong_denoise

    def run(self):
        """执行对齐处理"""

        # 执行对齐处理
        process_images(
            self.in_path,
            self.out_path,
            self.hough,
            self.ref_path,
            self.use_advanced,
            self.strong_denoise,
        )

        self.finished.emit(True, "所有图像处理完成！")


class UniversalLunarAlignApp(QMainWindow):
    """月食圆面对齐工具主窗口"""

    def __init__(self):
        super().__init__()

        # 初始化变量

        self.progress_window = None
        self.alignment_thread = None
        self._about_photo = None
        self._qr_photo = None

        # 初始化UI变量
        self._init_variables()

        # 设置窗口
        self.setWindowTitle("月食圆面对齐工具")
        self.resize(920, 800)
        self.setMinimumSize(750, 700)

        # 创建UI
        self._setup_ui()

        # 设置初始日志信息
        self._set_initial_log_message()

        # 设置信号连接
        self._connect_signals()

        self.preview_window = PreviewWindow(self)
        self.debug_window = DebugWindow(self)

    def _init_variables(self):
        """初始化变量"""
        self.input_path: Path | None = None
        self.output_path: Path | None = None
        self.reference_path: Path | None = None

        # 参数设置
        self.params = HoughParams(minRadius=300, maxRadius=800, param1=50, param2=30)

        self.use_advanced_alignment = False
        self.alignment_method = "auto"
        self.enable_strong_denoise = False

    def _setup_ui(self):
        """设置UI界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # 创建各个UI区域
        self._create_path_section(main_layout)
        self._create_parameter_section(main_layout)
        self._create_debug_section(main_layout)
        self._create_action_section(main_layout)
        self._create_log_section(main_layout)

    def _create_path_section(self, parent_layout):
        """创建路径设置区域"""
        group = QGroupBox("1. 路径设置")
        layout = QGridLayout()
        group.setLayout(layout)

        # 输入文件夹
        layout.addWidget(QLabel("输入文件夹:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("选择包含月食图像的文件夹...")
        self.input_edit.textChanged.connect(self.set_input_path)

        layout.addWidget(self.input_edit, 0, 1)
        self.input_browse_btn = QPushButton("浏览...")
        self.input_browse_btn.clicked.connect(self.select_input_folder)
        layout.addWidget(self.input_browse_btn, 0, 2)

        # 输出文件夹
        layout.addWidget(QLabel("输出文件夹:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择处理后图像的保存文件夹...")
        self.output_edit.textChanged.connect(self.set_output_path)
        layout.addWidget(self.output_edit, 1, 1)
        self.output_browse_btn = QPushButton("浏览...")
        self.output_browse_btn.clicked.connect(self.select_output_folder)
        layout.addWidget(self.output_browse_btn, 1, 2)

        # # 参考图像
        layout.addWidget(QLabel("参考图像:"), 2, 0)
        self.ref_label = QLabel("（在预览窗口选择）")
        layout.addWidget(self.ref_label, 2, 1)

        # 帮助提示和强力降噪选项
        help_layout = QHBoxLayout()
        help_text = QLabel(
            "💡参考图像：作为对齐基准的图像。请在预览&半径估计窗口选择。"
        )
        help_text.setStyleSheet("color: gray; font-size: 10pt;")
        help_layout.addWidget(help_text)

        help_layout.addStretch()

        self.strong_denoise_check = QCheckBox("强力降噪(仅检测/对齐)")
        self.strong_denoise_check.setChecked(False)
        help_layout.addWidget(self.strong_denoise_check)

        layout.addLayout(help_layout, 3, 0, 1, 3)

        parent_layout.addWidget(group)

    def _create_parameter_section(self, parent_layout):
        """创建参数调节区域"""
        # 创建水平布局容器
        param_container = QWidget()
        param_layout = QHBoxLayout(param_container)
        param_layout.setContentsMargins(0, 0, 0, 0)

        # PHD2参数区域
        hough_group = QGroupBox("2. PHD2霍夫圆参数")
        hough_layout = QVBoxLayout(hough_group)

        # 帮助文本
        help_text = QLabel("• PHD2增强算法：三级检测策略，自适应图像亮度\n")
        help_text.setStyleSheet("font-size: 9pt;")
        hough_layout.addWidget(help_text)

        # 参数显示（只读）
        param_configs = [
            ("minRadius", "最小半径:"),
            ("maxRadius", "最大半径:"),
            ("param1", "边缘敏感度:"),
            ("param2", "圆心阈值:"),
        ]

        self.param_labels = {}
        for key, label_text in param_configs:
            # 创建参数行
            param_row = QWidget()
            param_row_layout = QHBoxLayout(param_row)
            param_row_layout.setContentsMargins(0, 0, 0, 0)

            # 标签
            param_label = QLabel(label_text)
            param_row_layout.addWidget(param_label, 1)

            # 显示值的标签
            value_label = QLabel()
            value_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            if key in ["minRadius", "maxRadius"]:
                value_label.setText(f"{self.params[key]} px")
            else:
                value_label.setText(str(self.params[key]))
            param_row_layout.addWidget(value_label, 0)

            # 保存标签引用
            self.param_labels[key] = value_label

            hough_layout.addWidget(param_row)
        param_layout.addWidget(hough_group, 2)

        # 多ROI精配准区域
        advanced_group = QGroupBox("3. 多ROI精配准")
        advanced_layout = QVBoxLayout(advanced_group)

        self.advanced_check = QCheckBox("启用多ROI精配准(仅支持赤道仪跟踪拍摄的素材)")
        self.advanced_check.setChecked(False)
        advanced_layout.addWidget(self.advanced_check)

        # 算法说明
        advanced_layout.addWidget(QLabel("算法说明:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["auto", "phase_corr", "template", "feature", "centroid"]
        )
        self.method_combo.setCurrentText("auto")
        self.method_combo.setEnabled(False)
        advanced_layout.addWidget(self.method_combo)

        # 算法帮助
        algo_help = QLabel(
            "• 在月盘内自动选择多块ROI进行 ZNCC/相位相关微调\n"
            "• 对亮度变化与阴影边界更鲁棒，失败时自动回退到圆心对齐\n"
            "• 建议在偏食/生光阶段启用，多数情况默认关闭即可"
        )
        algo_help.setStyleSheet("color: darkgreen; font-size: 8pt;")
        advanced_layout.addWidget(algo_help)

        warning = QLabel("⚠️ 实验性功能，不推荐开启")
        warning.setStyleSheet("color: orange; font-size: 9pt;")
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        advanced_layout.addWidget(warning)

        param_layout.addWidget(advanced_group, 1)

        parent_layout.addWidget(param_container)

    def _create_debug_section(self, parent_layout):
        """创建预览与调试区域"""
        group = QGroupBox("4. 预览与调试")
        layout = QHBoxLayout(group)

        self.preview_btn = QPushButton("选择参考并预览")
        self.preview_btn.clicked.connect(self.open_preview)
        layout.addWidget(self.preview_btn, 1)

        self.debug_btn = QPushButton("打开调试窗口（实时参数预览）")
        self.debug_btn.clicked.connect(self.open_debug)
        layout.addWidget(self.debug_btn, 1)

        parent_layout.addWidget(group)

    def _create_action_section(self, parent_layout):
        """创建操作区域"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(200, 10, 200, 10)

        # 开始对齐按钮
        self.start_btn = QPushButton("🚀 开始集成对齐")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_alignment)
        layout.addWidget(self.start_btn, 1)

        # 打赏作者按钮
        self.donate_btn = QPushButton("打赏作者")
        self.donate_btn.clicked.connect(self.show_about_author)
        layout.addWidget(self.donate_btn, 0)

        parent_layout.addWidget(widget)

    def _create_log_section(self, parent_layout):
        """创建日志显示区域"""
        # 日志浏览器
        self.log_browser = QTextBrowser()
        self.log_browser.setReadOnly(True)
        self.log_browser.setMaximumHeight(200)
        parent_layout.addWidget(self.log_browser)
        enable_gui_logging(self.log_browser)

    def _connect_signals(self):
        """连接信号"""
        self.advanced_check.stateChanged.connect(self.on_advanced_changed)

    def _set_initial_log_message(self):
        """设置初始日志信息"""
        welcome = (
            f"欢迎使用月食圆面对齐工具 By @正七价的氟离子\n"
            f"运行平台: {SYSTEM}\n"
            "================================================================\n\n"
            "算法说明：\n"
            "• PHD2增强算法：基于霍夫圆检测，适用于完整清晰的月球\n"
            "• 多ROI精配准：适用于偏食、生光等复杂阶段（实验性）\n"
            "• 回退机制：确保在任何情况下都有可用的对齐方案\n\n"
            "使用建议：\n"
            "• 使用预览工具准确估算半径范围\n"
            "• 参数2（累加器阈值）是最关键的调整参数\n"
            f"• 支持格式：{', '.join(SUPPORTED_EXTS)}\n"
        )

        self.log_browser.append(welcome)

    def update_hough_params(self, params: HoughParams):
        """从Preview Window接收参数更新"""
        self.params = params.copy()
        self.param_labels["param1"].setText(str(params.param1))
        self.param_labels["param2"].setText(str(params.param2))
        self.param_labels["minRadius"].setText(f"{params.minRadius} px")
        self.param_labels["maxRadius"].setText(f"{params.maxRadius} px")

    def update_reference_path(self, ref_path: Path):
        """更新参考路径"""
        self.reference_path = ref_path
        self.ref_label.setText(ref_path.name)

    def on_advanced_changed(self):
        """高级功能状态改变"""
        enabled = self.advanced_check.isChecked()
        self.method_combo.setEnabled(enabled)

    def set_input_path(self, text: str):
        """设置输入路径"""
        self.input_path = Path(text)

        output_dir = self.input_path.parent / f"{self.input_path.name}_aligned"
        self.output_edit.setText(str(output_dir))

    def set_output_path(self, text: str):
        """设置输出路径"""
        self.output_path = Path(text)

    def select_input_folder(self):
        """选择输入文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if not folder:
            return

        self.input_edit.setText(folder)

    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_path = Path(folder)
            self.output_edit.setText(self.output_path.name)

    def open_preview(self):
        """打开预览窗口（复用现有窗口实例）"""

        self.preview_window.show()
        self.preview_window.raise_()
        self.preview_window.activateWindow()

    def open_debug(self):
        """打开调试窗口"""
        self.debug_window.show()
        self.debug_window.raise_()
        self.debug_window.activateWindow()

    def _warning_dialog(self, title, message):
        """显示警告对话框"""
        return (
            QMessageBox.warning(
                self,
                title,
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

    def start_alignment(self):
        """开始对齐处理"""
        # 验证输入
        if not self.input_path or not os.path.isdir(self.input_path):
            QMessageBox.critical(self, "错误", "请选择有效的输入文件夹。")
            return

        if not self.output_path:
            QMessageBox.critical(self, "错误", "请指定输出文件夹。")
            return

        # 检查SciPy依赖
        use_advanced = self.advanced_check.isChecked()
        method = self.method_combo.currentText()

        # 检查参考图像
        ref_path = self.reference_path
        if ref_path and not os.path.exists(ref_path):
            if not self._warning_dialog(
                "警告",
                f"指定的参考图像不存在：\n{ref_path}\n\n是否继续（将自动选择参考图像）？",
            ):
                return
            else:
                ref_path = None

        # 准备霍夫参数

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.start_btn.setText(
            "集成对齐中 (多ROI + PHD2)..." if use_advanced else "PHD2对齐中..."
        )

        # 创建并启动处理线程
        self.alignment_thread = AlignmentThread(
            self.input_path,
            self.output_path,
            self.params,
            ref_path,
            use_advanced,
            method,
            self.strong_denoise_check.isChecked(),
        )

        # 连接信号
        self.alignment_thread.finished.connect(self.on_task_complete)

        # 启动线程
        self.alignment_thread.start()

    def show_progress_window(self):
        """显示进度窗口"""
        if self.progress_window is None or not self.progress_window.isVisible():
            self.progress_window = ProgressWindow(self)
        return self.progress_window

    def on_task_complete(self, success: bool, message: str):

        self.start_btn.setEnabled(True)
        self.start_btn.setText("🚀 开始集成对齐")

        def show_message():
            if success:
                QMessageBox.information(self, "处理完成", message)
            else:
                QMessageBox.critical(
                    self,
                    "处理失败",
                    f"处理过程中发生错误，详情请查看日志。\n\n{message}",
                )

        QTimer.singleShot(0, show_message)

    def show_about_author(self):
        """显示关于作者窗口"""
        dialog = QDialog(self)
        dialog.setWindowTitle("关于作者")
        dialog.setModal(True)

        # 主布局
        main_layout = QGridLayout(dialog)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 查找头像和二维码文件（从包内 resources 目录）
        base_dir = Path(__file__).parent.parent / "resources"
        avatar_path = None
        for name in ("avatar.jpg", "avatar.png", "avatar.jpeg"):
            p = base_dir / name
            if p.exists():
                avatar_path = p
                break

        qr_path = None
        for name in ("QRcode.jpg", "QRcode.png", "QRcode.jpeg"):
            p = base_dir / name
            if p.exists():
                qr_path = p
                break

        # 左侧：标题+头像+描述
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 标题行（标题+头像）
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("正七价的氟离子")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # 头像
        avatar_label = QLabel()
        if avatar_path:
            try:
                pixmap = QPixmap(str(avatar_path))
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        100, 100,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    avatar_label.setPixmap(scaled_pixmap)
            except Exception:
                pass
        header_layout.addWidget(avatar_label)

        left_layout.addWidget(header_widget)

        # 描述文本
        desc_label = QLabel(
            "在家带娃的奶妈，不会写程序的天文爱好者不是老司机。\n"
            "感谢使用《月食圆面对齐工具》，欢迎反馈与交流！\n"
            "如果您愿意，欢迎支持一点养娃的奶粉钱（右侧支付宝二维码）。"
        )
        desc_label.setWordWrap(True)
        desc_label.setMaximumWidth(440)
        left_layout.addWidget(desc_label)
        left_layout.addStretch()

        main_layout.addWidget(left_panel, 0, 0, 3, 1)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator, 0, 1, 3, 1)

        # 右侧：二维码
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        qr_label = QLabel()
        if qr_path:
            try:
                pixmap = QPixmap(str(qr_path))
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        240, 240,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    qr_label.setPixmap(scaled_pixmap)
            except Exception:
                pass
        right_layout.addWidget(qr_label)

        qr_text = QLabel("支付宝 · 打赏支持")
        qr_text.setStyleSheet("color: gray;")
        qr_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(qr_text)

        main_layout.addWidget(right_panel, 0, 2, 3, 1)

        # 底部按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        main_layout.addWidget(close_btn, 3, 0, 1, 3, Qt.AlignmentFlag.AlignRight)

        dialog.exec()
