import os
import platform


SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"
SUPPORTED_EXTS = {"*.tif", "*.tiff", "*.bmp", "*.png", "*.jpg", "*.jpeg"}

# 内存管理
MAX_IMAGES_IN_MEMORY = 10
MEMORY_THRESHOLD_MB = 500
MAX_SCAN_COUNT = 10
THUMB_SIZE = 1600  # 图像最长边缩略尺寸
MAX_REFINE_DELTA_PX = 6.0
MIN_MEAN_ZNCC = 0.55
MIN_INLIERS = 6
RESIDUAL = 2.0
DEBUG = bool(os.environ.get("DEBUG", False))

MAX_LOADED_FILES = 5

MIN_KEEP_FILES = min(1, MAX_LOADED_FILES - 1)
