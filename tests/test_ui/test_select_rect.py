"""
UI组件单元测试 - select_rect模块
测试ResizeHandle和EditableRect的修复功能
"""

import pytest


@pytest.mark.unit
@pytest.mark.gui
def test_resize_handle_creation_with_parent(qtbot):
    """测试ResizeHandle使用Qt父子关系创建（验证修复）"""
    from lunar_eclipse_align.ui.select_rect import EditableRect
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QGraphicsScene

    # 创建场景（QGraphicsScene 不需要添加到 qtbot，它不是 QWidget）
    scene = QGraphicsScene()

    # 创建父矩形
    rect = QRectF(100, 100, 200, 150)
    editable_rect = EditableRect(rect)

    # 添加到场景
    scene.addItem(editable_rect)

    # 验证句柄已创建且有正确的父子关系
    assert len(editable_rect.handles) == 8

    # 验证每个句柄都有正确的父项
    for handle_type, handle in editable_rect.handles.items():
        assert handle.parentItem() == editable_rect
        assert handle.parent_rect == editable_rect
        assert handle.handle_type == handle_type

    # 验证句柄在场景中（通过父子关系自动添加）
    scene_items = scene.items()
    assert len(scene_items) == 9  # 1个矩形 + 8个句柄


@pytest.mark.unit
@pytest.mark.gui
def test_editable_rect_creation(qtbot):
    """测试EditableRect创建和句柄自动生成"""
    from lunar_eclipse_align.ui.select_rect import EditableRect
    from PySide6.QtCore import QRectF

    # 创建矩形（EditableRect 是 QGraphicsItem，不是 QWidget，不需要添加到 qtbot）
    rect = QRectF(50, 50, 100, 80)
    editable_rect = EditableRect(rect)

    # 验证基本属性
    assert editable_rect.rect() == rect

    # 验证句柄创建
    expected_handle_types = [
        "top-left",
        "top",
        "top-right",
        "right",
        "bottom-right",
        "bottom",
        "bottom-left",
        "left",
    ]
    assert len(editable_rect.handles) == 8
    assert set(editable_rect.handles.keys()) == set(expected_handle_types)

    # 验证每个句柄都正确配置
    for handle_type, handle in editable_rect.handles.items():
        assert handle.handle_type == handle_type
        assert handle.parent_rect == editable_rect
