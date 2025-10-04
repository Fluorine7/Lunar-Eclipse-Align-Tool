"""
pytest 配置文件
为测试提供共享的夹具和配置
"""

import pytest
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def sample_image_path():
    """示例图像文件路径"""
    # 这里返回一个测试用的图像路径
    # 实际使用时需要放置测试图像文件
    return os.path.join(os.path.dirname(__file__), "test_data", "sample_lunar.jpg")


@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


def pytest_configure(config):
    """pytest配置钩子"""
    # 添加自定义标记
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "gui: mark test as requiring GUI components")
