#!/usr/bin/env python3
"""
WD Tagger GUI — 基于 PyQt6 的 WD 图片智能打标工具
==============================================
支持 CUDA GPU 加速 | 单图/批量打标 | Trigger Words | 黑名单

用法:
    python main.py

首次使用请先安装依赖:
    pip install -r requirements.txt
    # GPU 加速 (需要 CUDA 环境):
    pip install onnxruntime-gpu
"""

import os
import sys

# 设置环境变量：解决 PyQt6 在某些 Windows 环境下的缩放问题
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow, DARK_STYLE


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("WD Tagger GUI")
    app.setOrganizationName("OpenSquilla")
    app.setStyleSheet(DARK_STYLE)

    # 强制启用高 DPI
    if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
