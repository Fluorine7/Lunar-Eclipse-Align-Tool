"""
图像IO功能单元测试
"""

import pytest
import numpy as np
from pathlib import Path
from lunar_eclipse_align.utils.image import Image, TiffIO, PngIO, JpegIO


class TestImageIO:
    """测试图像IO功能"""

    def test_tiff_read_16bit(self):
        """测试TIFF读取16位图像"""
        # 读取测试图像
        img = Image.from_file(Path("images/basic/DSC_3549.tif"))

        assert img is not None, "图像读取失败"
        assert img.rgb.dtype == np.uint16, f"期望uint16，实际 {img.rgb.dtype}"
        assert len(img.rgb.shape) == 3, "期望3维数组(H,W,C)"
        assert img.rgb.shape[2] == 3, "期望RGB三通道"

    def test_tiff_icc_profile(self):
        """测试TIFF读取ICC配置"""
        img = Image.from_file(Path("images/basic/DSC_3549.tif"))

        assert img is not None, "图像读取失败"
        assert img.icc is not None, "ICC配置应该存在"
        assert isinstance(img.icc, bytes), f"ICC应为bytes类型，实际 {type(img.icc)}"
        assert len(img.icc) > 0, "ICC数据不应为空"

    def test_tiff_roundtrip(self, tmp_path):
        """测试TIFF往返读写"""
        # 读取原始图像
        img1 = TiffIO.read(Path("images/basic/DSC_3549.tif"))
        assert img1 is not None

        # 保存
        output_path = tmp_path / "test_output.tif"
        success = TiffIO.write(img1, output_path)
        assert success, "保存失败"

        # 读回
        img2 = TiffIO.read(output_path)
        assert img2 is not None, "读回失败"

        # 验证
        assert img2.rgb.dtype == img1.rgb.dtype, "位深度不一致"
        assert img2.rgb.shape == img1.rgb.shape, "形状不一致"
        assert np.array_equal(img1.rgb, img2.rgb), "像素数据不一致"
        assert img2.icc == img1.icc, "ICC配置不一致"

    def test_png_16bit_write_read(self, tmp_path):
        """测试PNG 16位读写"""
        # 创建16位测试数据
        rgb16 = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
        icc_data = b"test_icc_profile_data" * 10
        img1 = Image(rgb=rgb16, icc=icc_data)

        # 保存为PNG
        output_path = tmp_path / "test_16bit.png"
        success = PngIO.write(img1, output_path)
        assert success, "PNG保存失败"

        # 读回
        img2 = PngIO.read(output_path)
        assert img2 is not None, "PNG读取失败"

        # 验证
        assert img2.rgb.dtype == np.uint16, f"期望uint16，实际 {img2.rgb.dtype}"
        assert np.array_equal(rgb16, img2.rgb), "像素数据不一致"
        assert img2.icc == icc_data, "ICC配置不一致"

    def test_jpeg_8bit_conversion(self, tmp_path):
        """测试JPEG自动转换16位到8位"""
        # 创建16位测试数据
        rgb16 = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
        icc_data = b"test_icc_profile"
        img1 = Image(rgb=rgb16, icc=icc_data)

        # 保存为JPEG（应自动转8位）
        output_path = tmp_path / "test.jpg"
        success = JpegIO.write(img1, output_path)
        assert success, "JPEG保存失败"

        # 读回
        img2 = JpegIO.read(output_path)
        assert img2 is not None, "JPEG读取失败"

        # 验证
        assert img2.rgb.dtype == np.uint8, f"JPEG应为uint8，实际 {img2.rgb.dtype}"
        assert img2.icc == icc_data, "ICC配置不一致"

    def test_image_save_dispatch(self, tmp_path):
        """测试Image.save的格式调度"""
        # 创建测试图像
        img = Image.from_file(Path("images/basic/DSC_3549.tif"))
        assert img is not None

        # 测试保存为不同格式
        formats = [
            (tmp_path / "test.tif", np.uint16),
            (tmp_path / "test.png", np.uint16),
            (tmp_path / "test.jpg", np.uint8),
        ]

        for path, expected_dtype in formats:
            success = img.save(path)
            assert success, f"保存 {path.suffix} 失败"

            # 验证可以读回
            img_read = Image.from_file(path)
            assert img_read is not None, f"读取 {path.suffix} 失败"
            assert img_read.rgb.dtype == expected_dtype, \
                f"{path.suffix} 期望 {expected_dtype}，实际 {img_read.rgb.dtype}"
