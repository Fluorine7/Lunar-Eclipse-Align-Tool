# -*- coding: utf-8 -*-
# build.py
"""
PyInstaller 构建脚本 - PySide6 版本
用于将月食圆面对齐工具打包成独立可执行文件
"""
import os
import sys
import platform
import shutil
from pathlib import Path
from PyInstaller.__main__ import run

# 设置 UTF-8 编码输出（修复 Windows 控制台中文显示）
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

APP_NAME = "Lunar Eclipse Align Tool"
ENTRY = "lunar_eclipse_align/main.py"
from lunar_eclipse_align import __version__ as VERSION

# 调试模式：保留控制台窗口
DEBUG_BUILD = os.environ.get("DEBUG_BUILD", "0") == "1"


def sep():
    """PyInstaller --add-data 的路径分隔符：Windows 用 ; 其余用 :"""
    return ";" if platform.system() == "Windows" else ":"


def clean_build():
    """清理上次构建产物"""
    for d in ("build", "dist", f"{APP_NAME}.spec"):
        if os.path.exists(d):
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
    print("✓ 清理完成")


def collect_package_data():
    """收集包内的数据文件（resources 目录）"""
    add_data_args = []

    # 包内 resources 目录
    resources_dir = Path("lunar_eclipse_align/resources")
    if resources_dir.exists():
        # 收集所有资源文件
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.ico"]:
            for file in resources_dir.glob(ext):
                dest = "lunar_eclipse_align/resources"
                add_data_args += ["--add-data", f"{file}{sep()}{dest}"]
        print(f"✓ 找到 resources 目录: {len(list(resources_dir.iterdir()))} 个文件")
    else:
        print("⚠ 警告: resources 目录不存在")

    return add_data_args


def main():
    """主构建函数"""
    print(f"{'='*60}")
    print(f"  月食圆面对齐工具 - PyInstaller 构建脚本")
    print(f"  版本: {VERSION}")
    print(f"  平台: {platform.system()} {platform.machine()}")
    print(f"{'='*60}\n")

    # 清理旧构建
    clean_build()

    # 检查入口文件
    if not os.path.exists(ENTRY):
        print(f"❌ 错误: 找不到入口文件 {ENTRY}")
        sys.exit(1)
    print(f"✓ 入口文件: {ENTRY}")

    # 收集数据文件
    add_data_args = collect_package_data()

    # PyInstaller 参数
    args = [
        ENTRY,
        "--name",
        APP_NAME,
        "--onedir" if platform.system() == "Darwin" else "--onefile",
        "--noconfirm",
        "--clean",
        "--log-level",
        "WARN",
    ]

    # 非调试模式下隐藏控制台
    if not DEBUG_BUILD:
        args.insert(5, "--windowed")  # GUI 程序，不显示控制台
    else:
        print("⚠️  调试模式：保留控制台窗口")

    # 注：如果运行时报缺少模块，可按需添加 --hidden-import
    # 例如：
    # args += ["--hidden-import", "PySide6.QtCore"]
    # args += ["--hidden-import", "scipy.special"]

    # 添加数据文件参数
    args += add_data_args

    # 可选：添加图标（如果有）
    icon_file = None
    if platform.system() == "Windows":
        if os.path.exists("icon.ico"):
            icon_file = "icon.ico"
    elif platform.system() == "Darwin":  # macOS
        if os.path.exists("icon.icns"):
            icon_file = "icon.icns"

    if icon_file:
        args += ["--icon", icon_file]
        print(f"✓ 使用图标: {icon_file}")

    # 打印构建参数
    print(f"\n{'='*60}")
    print("PyInstaller 参数:")
    print(" ".join(args))
    print(f"{'='*60}\n")

    # 执行构建
    try:
        run(args)
        print(f"\n{'='*60}")
        print(f"✓ 构建成功！")
        print(f"  可执行文件位于: dist/{APP_NAME}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
