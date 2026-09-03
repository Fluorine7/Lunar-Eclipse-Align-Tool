# build.py
import os, sys, platform, shutil
from PyInstaller.__main__ import run
from PyInstaller.utils.hooks import collect_data_files

APP_NAME = "Lunar_Eclipse_Align_Tool_V140b"
ENTRY = "main.py"

def sep():
    # PyInstaller --add-data 的路径分隔符：Windows 用 ; 其余用 :
    return ";" if platform.system() == "Windows" else ":"

def main():
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
        "--onefile",               # 关键：单文件打包
        "--windowed",              # GUI 程序，隐藏控制台
        "--noconfirm",
        "--clean",
        "--log-level", "WARN",
        # 可选：自定义图标
        # "--icon", "your_icon.ico" if platform.system()=="Windows" else "your_icon.icns",
    ] + add_data_args

    # 有些平台打包 tk 可能发散依赖，保守起见可加上隐藏导入（一般不用）
    # args += ["--hidden-import", "PIL._tkinter_finder"]

    print("PyInstaller args:\n", " ".join(args))
    run(args)

if __name__ == "__main__":
    main()
