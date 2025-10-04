"""
测试包初始化文件
"""
# 这个文件使tests目录成为一个Python包
# 可以在这里添加测试相关的共同配置或工具函数

import sys
import os

# 确保能导入src目录下的模块
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)