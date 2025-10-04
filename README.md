# 月食圆面对齐工具 (Lunar Eclipse Align Tool)

一个用于月食照片批量对齐的工具，基于 PHD2 增强算法和多 ROI 精配准技术。

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

## 安装与使用

### Windows 用户

1. 前往 [Releases](../../releases) 页面
2. 下载最新版本的 `LunarEclipseAlignTool-Windows.exe`
3. 双击运行（首次运行可能需要允许 Windows Defender）

### macOS 用户

1. 前往 [Releases](../../releases) 页面
2. 下载最新版本的 `LunarEclipseAlignTool-macOS.dmg`
3. 打开 DMG，拖拽到 Applications 文件夹
4. **首次运行**：右键点击应用 → "打开"（绕过 Gatekeeper）

### Linux 用户

#### 系统配置

- Python >= 3.13
- PySide6 >= 6.9.3

建议使用conda配置：
```bash
conda create -n lunar-eclipse-align 
conda activate lunar-eclipse-align
conda install python>=3.13 pyside6>=6.9.3
```

#### 安装步骤

<!-- **方式 1：从源码安装** -->

```bash
# 克隆仓库
git clone https://github.com/yourusername/Lunar-Eclipse-Align-Tool.git
cd Lunar-Eclipse-Align-Tool

# 安装（开发模式）
pip install -e .

# 或安装包含测试依赖
pip install -e .[test,dev]
```

<!-- **方式 2：从 PyPI 安装**

```bash
# 使用 pip 安装
pip install lunar-eclipse-align

# 或使用 pipx（推荐，隔离环境）
pipx install lunar-eclipse-align
``` -->

#### 运行

```bash
# 命令行启动 GUI
lunar-eclipse-align

# 或直接运行模块
python -m lunar_eclipse_align.main
```

## 使用说明

### 基本流程

1. **选择输入文件夹**：包含待对齐的月食图像
2. **选择输出文件夹**：对齐后的图像保存位置
3. **打开预览窗口**：
   - 选择参考图像（对齐基准）
   - 框选月球区域估计半径
   - 调整检测参数（边缘敏感度、圆心阈值）
   - 应用参数到主窗口
4. **开始对齐**：自动批量处理所有图像

### 参数说明

#### 边缘敏感度 (param1)
- 控制边缘检测的敏感程度
- **数值越高**：只检测强边缘，减少噪声
- **数值越低**：检测更多边缘，可能包含噪声
- **推荐范围**：20-150
- **月食推荐**：50-80（明亮），30-50（暗淡）

#### 圆心阈值 (param2)
- 控制圆心检测的严格程度
- **数值越高**：要求更多证据，结果更可靠
- **数值越低**：更容易检测到圆，但可能误检
- **推荐范围**：10-50
- **月食推荐**：25-35（清晰），15-25（模糊）


## 支持的图像格式

- TIFF (.tif, .tiff)
- PNG (.png)
- JPEG (.jpg, .jpeg)
- 其他 OpenCV/Pillow 支持的格式

## 开发与构建

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/Lunar-Eclipse-Align-Tool.git
cd Lunar-Eclipse-Align-Tool

# 安装开发依赖
pip install -e .[dev,test]

# 运行测试
pytest

# 代码覆盖率
pytest --cov=lunar_eclipse_align
```

### 构建可执行文件

```bash
# Linux
python build.py

# Windows/macOS
# 推送 tag 到 GitHub，自动通过 GitHub Actions 构建
git tag v1.2.0
git push origin v1.2.0
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE)


