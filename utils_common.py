# utils_common.py
import os, sys, platform, gc
import numpy as np
import cv2
from PIL import Image
try:
    import piexif
except Exception:
    piexif = None
import tkinter as tk

# ----------------- 系统/常量 -----------------
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

VERSION = "1.5.0-beta"
DEFAULT_DEBUG_MODE = False
DEFAULT_DEBUG_IMAGE_PATH = ""
SUPPORTED_EXTS = {'.tif', '.tiff', '.bmp', '.png', '.jpg', '.jpeg'}

# 根据系统设置默认字体
if IS_WINDOWS:
    DEFAULT_FONT = ("Microsoft YaHei", 9)
    UI_FONT = ("Microsoft YaHei", 9)
elif IS_MACOS:
    DEFAULT_FONT = ("SF Pro Display", 13)
    UI_FONT = ("SF Pro Display", 13)
else:
    DEFAULT_FONT = ("DejaVu Sans", 9)
    UI_FONT = ("DejaVu Sans", 9)

# 内存管理
MAX_IMAGES_IN_MEMORY = 10
MEMORY_THRESHOLD_MB = 500

def get_memory_usage_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def force_garbage_collection():
    gc.collect()

class MemoryManager:
    def __init__(self, threshold_mb=MEMORY_THRESHOLD_MB):
        self.threshold_mb = threshold_mb
        self.image_cache = {}
        self.access_order = []
    def should_clear_memory(self):
        return get_memory_usage_mb() > self.threshold_mb or len(self.image_cache) > MAX_IMAGES_IN_MEMORY
    def clear_old_images(self, keep_count=5):
        if len(self.access_order) > keep_count:
            to_remove = self.access_order[:-keep_count]
            for key in to_remove:
                if key in self.image_cache:
                    del self.image_cache[key]
                self.access_order.remove(key)
        force_garbage_collection()

# 路径/I-O
def normalize_path(path):
    if not path:
        return path
    path = path.replace('\\', os.sep).replace('/', os.sep)
    return os.path.normpath(path)

def ensure_dir_exists(dir_path):
    try:
        dir_path = normalize_path(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False

def safe_join(*paths):
    return normalize_path(os.path.join(*paths))

def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        path = normalize_path(path)
        if not IS_WINDOWS or path.isascii():
            img = cv2.imread(path, flags)
            if img is not None:
                return img
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, flags)
            if img is not None:
                return img
        except Exception:
            pass
        return cv2.imread(path, flags)
    except Exception as e:
        print(f"图像读取失败 {path}: {e}")
        return None

def imwrite_unicode(path, image):
    try:
        path = normalize_path(path)
        parent_dir = os.path.dirname(path)
        if not ensure_dir_exists(parent_dir):
            return False
        ext = os.path.splitext(path)[1].lower() or ".tif"
        if not IS_WINDOWS or path.isascii():
            if ext in (".tif", ".tiff"):
                params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                return cv2.imwrite(path, image, params)
            return cv2.imwrite(path, image)
        else:
            if ext in (".tif", ".tiff"):
                params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                ok, buf = cv2.imencode(".tif", image, params)
            else:
                ok, buf = cv2.imencode(ext, image)
            if ok:
                buf.tofile(path)
                return True
            return False
    except Exception as e:
        print(f"图像保存失败 {path}: {e}")
        return False

def imwrite_with_exif(src_path, dst_path, img_bgr):
    """
    优先保留 src_path 的 EXIF/ICC，用 Pillow 写出 JPEG/TIFF。
    失败或不支持时回退到 imwrite_unicode。
    参数:
        src_path: 原始读取图像的文件路径(用于提取EXIF/ICC)
        dst_path: 目标输出路径
        img_bgr:  OpenCV(BGR) 图像
    返回: bool 是否写出成功
    """
    try:
        dst_path = normalize_path(dst_path)
        parent_dir = os.path.dirname(dst_path)
        if not ensure_dir_exists(parent_dir):
            return False

        ext = os.path.splitext(dst_path)[1].lower()
        # 仅在 JPEG/TIFF 尝试保EXIF，其余格式走原逻辑
        if ext not in (".jpg", ".jpeg", ".tif", ".tiff"):
            return imwrite_unicode(dst_path, img_bgr)

        exif_bytes = None
        icc = None
        # 读取源图的 EXIF 与 ICC
        try:
            with Image.open(src_path) as src_im:
                exif_bytes = src_im.info.get("exif", None)
                icc = src_im.info.get("icc_profile", None)
                if exif_bytes and piexif is not None:
                    try:
                        exif_dict = piexif.load(exif_bytes)
                        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
                        exif_bytes = piexif.dump(exif_dict)
                    except Exception:
                        pass
        except Exception:
            pass

        # Pillow cannot reliably encode RGB uint16 TIFF arrays. Use tifffile
        # for that path so Canon 16-bit TIFFs retain both their bit depth and
        # the embedded ICC profile (TIFF tag 34675).
        if ext in (".tif", ".tiff") and img_bgr.dtype == np.uint16:
            try:
                import tifffile
                if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
                    output, photometric = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), "rgb"
                elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
                    output, photometric = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA), "rgb"
                else:
                    output, photometric = img_bgr, "minisblack"
                extra_tags = [(34675, "B", len(icc), icc, False)] if icc else []
                tifffile.imwrite(dst_path, output, photometric=photometric,
                                 compression=None, metadata=None, extratags=extra_tags)
                return True
            except Exception:
                pass

        if Image is None:
            return imwrite_unicode(dst_path, img_bgr)
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = img_bgr
        im = Image.fromarray(rgb)
        save_kwargs = {}
        if exif_bytes is not None:
            save_kwargs["exif"] = exif_bytes
        if icc is not None:
            save_kwargs["icc_profile"] = icc
        if ext in (".jpg", ".jpeg"):
            save_kwargs.setdefault("quality", 95)
        else:
            save_kwargs.setdefault("compression", "tiff_deflate")
        im.save(dst_path, **save_kwargs)
        return True
    except Exception:
        # 任意异常回退
        return imwrite_unicode(dst_path, img_bgr)

def to_display_rgb(img):
    if img is None:
        return None
    try:
        img_float = img.astype(np.float32)
        img_u8 = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if img_u8.ndim == 2:
            return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        elif img_u8.shape[2] == 4:
            return cv2.cvtColor(img_u8, cv2.COLOR_BGRA2RGB)
        elif img_u8.shape[2] == 3:
            return cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(img_u8[:,:,:3], cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"图像转换失败: {e}")
        return None

# 统一日志
def log(msg, log_box=None):
    if log_box:
        try:
            log_box.master.after(0, lambda: (
                log_box.config(state="normal"),
                log_box.insert(tk.END, str(msg) + "\n"),
                log_box.see(tk.END),
                log_box.config(state="disabled")
            ))
        except Exception:
            pass
    if msg:
        print(msg)
