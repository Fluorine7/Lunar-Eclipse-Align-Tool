from abc import ABC, abstractmethod
import logging
from pathlib import Path
from time import time
import traceback
from typing import Any, Literal, Mapping, Type
from PIL import Image as PILImage

import cv2
import numpy as np
from numpy.typing import NDArray

import gc

from lunar_eclipse_align.utils.constants import (
    MAX_LOADED_FILES,
    MIN_KEEP_FILES,
    SUPPORTED_EXTS,
)
import tifffile

# exif的保留比较麻烦，现在的问题是exif和位深度很难两全
# 第三方库要么支持16bit，要么支持exif写入
# 想要完美保留exif，目前只能借助exiftool之类的外部工具


class ImageFileIO(ABC):
    @staticmethod
    @abstractmethod
    def write(image: "Image", path: Path) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def read(file_path: Path) -> "Image | None":
        raise NotImplementedError


class TiffIO(ImageFileIO):
    """使用tifffile库处理TIFF文件，完整保留位深度和ICC配置"""

    @staticmethod
    def read(file_path: Path) -> "Image | None":
        try:
            with tifffile.TiffFile(file_path) as tif:
                # 读取图像数据，保留原始位深度
                rgb = tif.asarray()

                # 获取ICC Profile
                icc = None
                if 34675 in tif.pages[0].tags:  # ICC Profile tag # type: ignore
                    icc_tag = tif.pages[0].tags[34675]  # type: ignore
                    icc = icc_tag.value

                return Image(rgb=rgb, icc=icc)
        except Exception as e:
            logging.error(f"无法加载TIFF图像 {file_path}: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def write(image: "Image", path: Path) -> bool:
        try:
            # 构建ICC tag
            extratags = []
            if image.icc and isinstance(image.icc, bytes):
                extratags.append((34675, 7, len(image.icc), image.icc, True))

            # 使用tifffile保存，保留原始位深度和ICC
            tifffile.imwrite(
                path,
                image.rgb,
                compression="deflate",  # 无损压缩
                photometric="rgb",
                extratags=extratags if extratags else None,
            )
            return True
        except Exception as e:
            logging.error(f"无法保存TIFF图像到 {path}: {e}")
            traceback.print_exc()
            return False


class PngIO(ImageFileIO):
    """使用imagecodecs处理PNG文件，完整保留位深度(uint8/uint16)和ICC配置"""

    @staticmethod
    def read(file_path: Path) -> "Image | None":
        try:
            import imagecodecs

            # 用imagecodecs读取像素数据（保留原始位深度）
            with open(file_path, "rb") as f:
                png_bytes = f.read()
            rgb = imagecodecs.png_decode(png_bytes)

            # 确保是RGB格式
            if len(rgb.shape) == 2:
                # 灰度转RGB
                rgb = np.stack([rgb] * 3, axis=-1)
            elif len(rgb.shape) == 3 and rgb.shape[2] == 4:
                # 移除alpha通道
                rgb = rgb[:, :, :3]

            # 用Pillow读取ICC
            pil_image = PILImage.open(file_path)
            icc = pil_image.info.get("icc_profile", None)
            pil_image.close()

            return Image(rgb=rgb, icc=icc)
        except Exception as e:
            logging.error(f"无法加载PNG图像 {file_path}: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def write(image: "Image", path: Path) -> bool:
        try:
            import imagecodecs

            # 使用imagecodecs编码PNG（支持16位RGB）
            png_bytes = imagecodecs.png_encode(image.rgb, level=6)

            # 如果有ICC，插入iCCP chunk
            if image.icc and isinstance(image.icc, bytes):
                png_bytes = PngIO._insert_icc_chunk(png_bytes, image.icc)

            # 写入文件
            with open(path, "wb") as f:
                f.write(png_bytes)

            return True
        except Exception as e:
            logging.error(f"无法保存PNG图像到 {path}: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def _insert_icc_chunk(png_data: bytes, icc: bytes) -> bytes:
        """在PNG数据中插入ICC配置chunk"""
        import struct
        import zlib

        # PNG文件签名
        if png_data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("不是有效的PNG数据")

        # 读取所有chunks
        chunks = []
        pos = 8
        while pos < len(png_data):
            length = struct.unpack(">I", png_data[pos : pos + 4])[0]
            chunk_type = png_data[pos + 4 : pos + 8]
            chunk_data = png_data[pos + 8 : pos + 8 + length]
            crc = png_data[pos + 8 + length : pos + 12 + length]
            chunks.append((chunk_type, chunk_data, crc))
            pos += 12 + length

        # 构建新的chunks列表，在IDAT之前插入iCCP
        new_chunks = []
        inserted = False

        for chunk_type, chunk_data, crc in chunks:
            # 在第一个IDAT之前插入iCCP chunk
            if chunk_type == b"IDAT" and not inserted:
                # iCCP chunk格式: profile_name\0compression_method + compressed_data
                profile_name = b"ICC Profile\x00"
                compression_method = b"\x00"  # zlib compression
                compressed_icc = zlib.compress(icc, level=9)
                iccp_chunk_data = profile_name + compression_method + compressed_icc

                # 计算CRC
                crc_value = zlib.crc32(b"iCCP" + iccp_chunk_data) & 0xFFFFFFFF
                iccp_crc = struct.pack(">I", crc_value)

                new_chunks.append((b"iCCP", iccp_chunk_data, iccp_crc))
                inserted = True

            new_chunks.append((chunk_type, chunk_data, crc))

        # 重新组装PNG文件
        result = b"\x89PNG\r\n\x1a\n"
        for chunk_type, chunk_data, crc in new_chunks:
            result += struct.pack(">I", len(chunk_data))
            result += chunk_type
            result += chunk_data
            result += crc

        return result


class JpegIO(ImageFileIO):
    """使用Pillow库处理JPEG文件，完整保留ICC配置（JPEG仅支持8位深度）"""

    @staticmethod
    def read(file_path: Path) -> "Image | None":
        try:
            # 使用Pillow读取JPEG
            pil_image = PILImage.open(file_path)

            # 提取ICC配置文件
            icc = pil_image.info.get("icc_profile", None)

            # 转换为numpy数组（JPEG总是8位）
            rgb = np.array(pil_image)

            # 确保是RGB格式
            if pil_image.mode == "RGB":
                pass  # 已经是正确格式
            elif pil_image.mode == "L":
                # 灰度转RGB
                rgb = np.stack([rgb] * 3, axis=-1)
            elif pil_image.mode == "RGBA":
                # 移除alpha通道（JPEG不支持透明度）
                rgb = rgb[:, :, :3]
            elif pil_image.mode == "CMYK":
                # CMYK转RGB
                pil_image = pil_image.convert("RGB")
                rgb = np.array(pil_image)

            pil_image.close()
            return Image(rgb=rgb, icc=icc)
        except Exception as e:
            logging.error(f"无法加载JPEG图像 {file_path}: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def write(image: "Image", path: Path) -> bool:
        try:
            rgb_data = image.rgb

            # JPEG只支持8位，需要转换16位图像
            if rgb_data.dtype == np.uint16:
                # 线性缩放到8位
                rgb_data = (rgb_data / 257).astype(np.uint8)  # 65535/255 = 257
            elif rgb_data.dtype != np.uint8:
                # 其他类型也转换为8位
                rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)

            # 创建PIL图像
            pil_image = PILImage.fromarray(rgb_data, mode="RGB")

            # 准备保存参数
            save_kwargs: dict[str, Any] = {
                "format": "JPEG",
                "quality": 95,  # 高质量
                "optimize": True,  # 优化编码
                "subsampling": 0,  # 4:4:4 色度子采样（最高质量）
            }

            # 添加ICC配置文件
            if image.icc:
                save_kwargs["icc_profile"] = image.icc

            # 保存JPEG文件
            pil_image.save(path, **save_kwargs)
            return True
        except Exception as e:
            logging.error(f"无法保存JPEG图像到 {path}: {e}")
            traceback.print_exc()
            return False


class FallbackIO(ImageFileIO):
    """用Pillow来读写其他格式的图像文件，位深度和ICC可能有限"""

    @staticmethod
    def read(file_path: Path) -> "Image | None":
        try:
            pil_image = PILImage.open(file_path)
            icc = pil_image.info.get("icc_profile", None)
            rgb = np.array(pil_image)

            # 确保是RGB格式
            if pil_image.mode == "RGB":
                pass
            elif pil_image.mode == "L":
                rgb = np.stack([rgb] * 3, axis=-1)
            elif pil_image.mode == "RGBA":
                rgb = rgb[:, :, :3]
            elif pil_image.mode == "CMYK":
                pil_image = pil_image.convert("RGB")
                rgb = np.array(pil_image)

            pil_image.close()
            return Image(rgb=rgb, icc=icc)
        except Exception as e:
            logging.error(f"无法加载图像 {file_path}: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def write(image: "Image", path: Path) -> bool:
        try:
            rgb_data = image.rgb

            # Pillow只支持8位和16位图像保存，转换为8位
            if rgb_data.dtype == np.uint16:
                rgb_data = (rgb_data / 257).astype(np.uint8)
            elif rgb_data.dtype != np.uint8:
                rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)

            pil_image = PILImage.fromarray(rgb_data, mode="RGB")

            save_kwargs = {}
            if image.icc and isinstance(image.icc, bytes):
                save_kwargs["icc_profile"] = image.icc

            pil_image.save(path, **save_kwargs)
            return True
        except Exception as e:
            logging.error(f"无法保存图像到 {path}: {e}")
            traceback.print_exc()
            return False


class Image:
    _rgb: NDArray
    _rgb_8bit: NDArray | None = None
    _bgr: NDArray | None = None
    icc: bytes | None
    _normalized_gray: NDArray | None = None
    _width: int
    _height: int
    FileIO: Mapping[str, Type[ImageFileIO]] = {
        ".tif": TiffIO,
        ".tiff": TiffIO,
        ".png": PngIO,
        ".jpg": JpegIO,
        ".jpeg": JpegIO,
    }

    def __init__(
        self,
        *,
        rgb: NDArray | None = None,
        bgr: NDArray | None = None,
        icc: bytes | None = None,
    ):

        if rgb is not None:
            self._rgb = rgb
        elif bgr is not None:
            self._rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        else:
            raise ValueError("必须提供 rgb 或 bgr 数据")
        self._height, self._width = self.rgb.shape[:2]
        self.icc = icc

    @staticmethod
    def from_file(file_path: Path) -> "Image | None":
        """从文件加载图像"""
        suffix = file_path.suffix.lower()
        if suffix in Image.FileIO:
            return Image.FileIO[suffix].read(file_path)
        else:
            return FallbackIO.read(file_path)

    def save(self, file_path: Path) -> bool:
        """保存图像到文件"""
        suffix = file_path.suffix.lower()
        if suffix in Image.FileIO:
            return Image.FileIO[suffix].write(self, file_path)
        else:
            return FallbackIO.write(self, file_path)

    @property
    def rgb(self) -> NDArray:
        """获取 RGB 格式的图像数据"""
        return self._rgb

    @property
    def rgb_8bit(self) -> NDArray:
        if self._rgb.dtype == np.uint8:
            return self._rgb
        if self._rgb_8bit is None:
            self._rgb_8bit = (self._rgb / 256).astype(np.uint8)
        return self._rgb_8bit

    @property
    def bgr(self) -> NDArray:
        """获取 BGR 格式的图像数据"""
        if self._bgr is None:
            self._bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        return self._bgr

    @property
    def gray_8bit(self) -> NDArray[np.uint8]:
        """获取灰度图像数据"""
        if self._normalized_gray is None:
            gray = cv2.cvtColor(self._rgb, cv2.COLOR_RGB2GRAY)
            self._normalized_gray = cv2.normalize(
                gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U  # type: ignore
            )
            assert self._normalized_gray is not None
        return self._normalized_gray

    @property
    def width(self) -> int:
        """获取图像宽度"""
        return self._width

    @property
    def height(self) -> int:
        """获取图像高度"""
        return self._height

    @property
    def widthXheight(self) -> tuple[int, int]:
        """获取图像的高度和宽度"""
        return self._width, self._height


class ImageFile:
    """
    图像文件的包装类，支持按需加载和保存
    通过 mode 参数控制读写模式：
    - "r": 只读模式，按需加载图像
    - "w": 写入模式，允许设置和保存图像
    """

    _image: Image | None = None
    _path: Path
    _mode: Literal["r"] | Literal["w"]

    def __init__(self, file_path: Path, mode: Literal["r"] | Literal["w"] = "r"):
        self._path = file_path
        self._mode = mode

    @staticmethod
    def load(dir: Path) -> "dict[Path, ImageFile]":
        """创建 ImageFile 实例"""
        image_files = []
        for ext in SUPPORTED_EXTS:
            image_files.extend(dir.glob(ext))
        return {fp: ImageFile(fp) for fp in sorted(image_files)}

    @property
    def image(self) -> Image | None:
        """获取图像对象，按需加载"""
        if self._mode == "r":
            __touch_file__(self)
            if self._image is None:
                self._image = Image.from_file(self._path)
        return self._image

    @image.setter
    def image(self, image: Image):
        """设置图像对象"""
        if self._mode == "w":
            self._image = image
        else:
            raise ValueError("只能在写模式下设置图像")

    def __del__(self):
        """析构时释放图像数据"""
        if self._image is not None:
            self._image = None
            gc.collect()

    @property
    def path(self) -> Path:
        return self._path

    def save(self) -> bool:
        """保存图像到文件"""
        if self._mode != "w":
            raise ValueError("只能在写模式下保存图像")
        if self._image is None:
            raise ValueError("没有图像数据可保存")
        return self._image.save(self._path)


# if __name__ == "__main__":
#     # 测试代码
#     image = Image.from_file(Path("images/basic/DSC_3551.tif"))
#     print(image.rgb.dtype)
#     print(image.rgb.shape)
#     image.save(Path("test_out.tif"))


__image_file_timestamps__: dict[ImageFile, float] = {}


def __drop_images__():
    while len(__image_file_timestamps__) > MIN_KEEP_FILES:
        # 找到最旧的文件并卸载
        oldest_file = min(
            __image_file_timestamps__, key=lambda k: __image_file_timestamps__[k]
        )
        oldest_file._image = None
        del __image_file_timestamps__[oldest_file]
    gc.collect()


def __touch_file__(file: ImageFile) -> None:
    touch_time = time()
    if file in __image_file_timestamps__:
        __image_file_timestamps__[file] = touch_time

    if len(__image_file_timestamps__) >= MAX_LOADED_FILES:
        __drop_images__()

    __image_file_timestamps__[file] = touch_time
