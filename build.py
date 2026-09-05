# build.py
import os, sys, platform, shutil, plistlib, subprocess
from PyInstaller.__main__ import run
from PyInstaller.utils.hooks import collect_data_files

APP_NAME = "Lunar_Eclipse_Align_Tool_V150b"
APP_VERSION = "1.5.0"
ENTRY = "main.py"
MACOS_MIN_VERSION = "14.0"
BUNDLE_IDENTIFIER = "com.fluorine.lunar-eclipse-align-tool"

def sep():
    # PyInstaller --add-data 的路径分隔符：Windows 用 ; 其余用 :
    return ";" if platform.system() == "Windows" else ":"

def main():
    is_macos = platform.system() == "Darwin"
    if is_macos:
        # The packaged Python and every binary wheel must support this target too.
        os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", MACOS_MIN_VERSION)

    # 清理上次构建
    for d in ("build", "dist", f"{APP_NAME}.spec"):
        if os.path.exists(d):
            shutil.rmtree(d) if os.path.isdir(d) else os.remove(d)

    # 收集第三方包静态资源（尤其是 ttkthemes 的主题）
    datas = []
    try:
        datas += collect_data_files("ttkthemes", include_py_files=False)
    except Exception:
        pass

    # 你的本地资源（头像 + 支付宝二维码）
    local_datas = [
        f"avatar.jpg{sep()}.",
        f"QRcode.jpg{sep()}.",
        f"LICENSE{sep()}.",
    ]

    # 将 collect_data_files 返回的 (src, dest) 转为 --add-data 形式
    add_data_args = []
    for src, dst in datas:
        add_data_args += ["--add-data", f"{src}{sep()}{dst or '.'}"]
    for s in local_datas:
        if os.path.exists(s.split(sep())[0]):  # 文件存在才添加
            add_data_args += ["--add-data", s]

    args = [
        ENTRY,
        "--name", APP_NAME,
        "--windowed",              # GUI 程序，隐藏控制台
        "--noconfirm",
        "--clean",
        "--log-level", "WARN",
        # 可选：自定义图标
        # "--icon", "your_icon.ico" if platform.system()=="Windows" else "your_icon.icns",
    ]

    if is_macos:
        # A normal .app bundle works better with macOS signing and Gatekeeper.
        args += [
            "--onedir",
            "--target-arch", platform.machine(),
            "--osx-bundle-identifier", BUNDLE_IDENTIFIER,
        ]
    else:
        args += ["--onefile"]

    args += add_data_args

    # 有些平台打包 tk 可能发散依赖，保守起见可加上隐藏导入（一般不用）
    # args += ["--hidden-import", "PIL._tkinter_finder"]

    target = f"macOS {MACOS_MIN_VERSION}+ ({platform.machine()})" if is_macos else platform.system()
    print(f"Building {APP_NAME} for {target}...")
    run(args)

    if is_macos:
        app_path = os.path.join("dist", f"{APP_NAME}.app")
        plist_path = os.path.join(app_path, "Contents", "Info.plist")
        with open(plist_path, "rb") as f:
            info = plistlib.load(f)
        info.update({
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION.replace(".", ""),
            "LSMinimumSystemVersion": MACOS_MIN_VERSION,
            "NSHumanReadableCopyright": "Copyright © 2025–2026 Fluorine Zhu",
        })
        with open(plist_path, "wb") as f:
            plistlib.dump(info, f)
        # Updating Info.plist invalidates PyInstaller's ad-hoc signature.
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", app_path],
            check=True,
        )

if __name__ == "__main__":
    main()
