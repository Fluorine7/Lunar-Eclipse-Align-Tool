#!/usr/bin/env python3
"""
月食圆面对齐工具 - PySide6版本
主程序入口
"""
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from lunar_eclipse_align import __version__ as VERSION
from lunar_eclipse_align.ui.main_window import UniversalLunarAlignApp
from lunar_eclipse_align.utils.logging import setup_logging


def main():
    """主程序入口"""

    # 创建应用实例
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName("月食圆面对齐工具")
    app.setApplicationVersion(VERSION)
    app.setOrganizationName("正七价的氟离子")

    # 设置应用样式（自动使用系统原生样式）
    app.setStyle("Fusion")  # 跨平台现代样式

    # 启用高DPI支持
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

    # 导入主窗口类（避免循环导入）

    # 创建主窗口
    window = UniversalLunarAlignApp()
    window.setWindowTitle(f"月食圆面对齐工具 V{VERSION} By @正七价的氟离子")
    window.resize(920, 800)
    window.setMinimumSize(750, 700)

    # 显示窗口
    window.show()

    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    setup_logging()
    main()
