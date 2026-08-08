"""
WD Tagger GUI - 主窗口
=====================
基于 PyQt6 的 Waifu Diffusion 图片打标工具。
支持单图打标和批量打标，标签页切换。
"""

import csv
import os
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QUrl,
)
from PyQt6.QtGui import (
    QPixmap, QFont, QIcon, QTextCursor, QPalette, QColor,
    QDragEnterEvent, QDropEvent,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QSlider, QDoubleSpinBox,
    QComboBox, QTextEdit, QGroupBox, QFileDialog, QProgressBar,
    QCheckBox, QSplitter, QFrame, QScrollArea, QLineEdit,
    QGridLayout, QMessageBox, QStatusBar, QSizePolicy, QListWidget,
    QListWidgetItem, QAbstractItemView, QSpinBox,
)

from tagger_backend import (
    WDTagger, AVAILABLE_MODELS, IMAGE_EXTENSIONS,
    get_available_providers, has_cuda, has_gpu, detect_device,
    CAT_GENERAL, CAT_ARTIST, CAT_COPYRIGHT, CAT_CHARACTER, CAT_RATING,
    find_images, DEFAULT_CACHE, preprocess_batch, extract_tags,
)


# ─── 样式表 ────────────────────────────────────────────────

DARK_STYLE = """
QMainWindow {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background-color: #313244;
    color: #a6adc8;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background-color: #45475a;
    color: #cdd6f4;
}
QTabBar::tab:hover:!selected {
    background-color: #3a3b4e;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #cdd6f4;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #89b4fa;
}
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 6px 16px;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #585b70;
}
QPushButton:pressed {
    background-color: #6c7086;
}
QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}
QPushButton#primaryBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}
QPushButton#primaryBtn:hover {
    background-color: #74c7ec;
}
QPushButton#dangerBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
}
QPushButton#dangerBtn:hover {
    background-color: #eba0ac;
}
QPushButton#successBtn {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}
QPushButton#successBtn:hover {
    background-color: #94e2d5;
}
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 28px;
}
QComboBox:hover {
    border-color: #89b4fa;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
    border: 1px solid #45475a;
}
QSlider::groove:horizontal {
    height: 6px;
    background-color: #313244;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background-color: #89b4fa;
    border-radius: 3px;
}
QDoubleSpinBox, QSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 28px;
}
QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #89b4fa;
}
QTextEdit {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px;
}
QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 28px;
}
QLineEdit:focus {
    border-color: #89b4fa;
}
QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    text-align: center;
    color: #cdd6f4;
    min-height: 22px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 5px;
}
QCheckBox {
    color: #cdd6f4;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #45475a;
}
QLabel#titleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #89b4fa;
}
QListWidget {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
}
QListWidget::item:selected {
    background-color: #45475a;
}
QListWidget::item:hover {
    background-color: #313244;
}
QSplitter::handle {
    background-color: #45475a;
    width: 3px;
}
QFrame#separator {
    background-color: #45475a;
    max-height: 1px;
}
QLabel#tagLabel {
    background-color: #313244;
    border-radius: 4px;
    padding: 2px 8px;
    color: #a6e3a1;
}
QLabel#tagLabelChar {
    background-color: #313244;
    border-radius: 4px;
    padding: 2px 8px;
    color: #f9e2af;
}
QLabel#tagLabelRating {
    background-color: #313244;
    border-radius: 4px;
    padding: 2px 8px;
    color: #cba6f7;
}
"""


# ─── 工作线程 ──────────────────────────────────────────────

class TaggerLoadThread(QThread):
    """后台加载模型线程。"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, error_msg

    def __init__(self, tagger: WDTagger, model_id: str, use_gpu: bool,
                 cache_dir: str, device: str = "cpu",
                 local_model_dir: str | None = None):
        super().__init__()
        self.tagger = tagger
        self.model_id = model_id
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.device = device
        self.local_model_dir = local_model_dir

    def run(self):
        try:
            self.tagger.device = self.device
            self.tagger.load(
                model_id=self.model_id,
                use_gpu=self.use_gpu,
                cache_dir=self.cache_dir,
                local_model_dir=self.local_model_dir,
                progress_callback=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))


class BatchTagThread(QThread):
    """批量打标线程。"""
    progress = pyqtSignal(int, int, str)  # current, total, current_file
    item_done = pyqtSignal(int, str, str)  # index, filename, tags
    finished = pyqtSignal(int, int)  # success, errors
    error = pyqtSignal(str)

    def __init__(self, tagger: WDTagger, image_paths: list[str],
                 threshold: float, char_threshold: float,
                 exclude_tags: set, cat_flags: dict,
                 trigger_words: str, batch_size: int):
        super().__init__()
        self.tagger = tagger
        self.image_paths = image_paths
        self.threshold = threshold
        self.char_threshold = char_threshold
        self.exclude_tags = exclude_tags
        self.cat_flags = cat_flags
        self.trigger_words = trigger_words
        self.batch_size = batch_size

    def run(self):
        success = 0
        errors = 0
        try:
            total = len(self.image_paths)
            for batch_start in range(0, total, self.batch_size):
                batch_paths = self.image_paths[batch_start:batch_start + self.batch_size]
                try:
                    batch_arr = preprocess_batch(batch_paths)
                    outputs = self.tagger.session.run(
                        None, {self.tagger.input_name: batch_arr}
                    )
                    probs_list = outputs[0]

                    for j, img_path in enumerate(batch_paths):
                        tag_results = extract_tags(
                            probs_list[j],
                            self.tagger.tag_names,
                            self.tagger.categories,
                            self.threshold,
                            self.char_threshold,
                            self.exclude_tags,
                            self.cat_flags,
                        )
                        tags = [t[0].replace("_", " ") for t in tag_results]
                        if self.trigger_words.strip():
                            tw_list = [w.strip() for w in self.trigger_words.split(",") if w.strip()]
                            tags = tw_list + tags

                        tag_text = ", ".join(tags)
                        idx = batch_start + j
                        self.item_done.emit(idx, os.path.basename(img_path), tag_text)
                        success += 1

                except Exception as e:
                    for img_path in batch_paths:
                        idx = batch_start + batch_paths.index(img_path)
                        self.item_done.emit(idx, os.path.basename(img_path), f"[ERROR] {e}")
                        errors += 1

                current = min(batch_start + self.batch_size, total)
                self.progress.emit(current, total,
                                   batch_paths[-1] if batch_paths else "")

        except Exception as e:
            self.error.emit(str(e))

        self.finished.emit(success, errors)


# ─── 主窗口 ────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WD Tagger GUI — 图片智能打标工具 🦐")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # 状态
        self.tagger = WDTagger()
        self.current_image_path: str | None = None
        self.batch_image_paths: list[str] = []
        self.batch_output_dir: str = ""
        self.batch_results: list[tuple[str, str]] = []  # (path, tags)
        self._auto_save_batch: bool = True   # 批量自动保存

        self._init_ui()
        self._detect_device()

        # 允许拖放
        self.setAcceptDrops(True)

    def _init_ui(self):
        """初始化界面。"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── 顶部：标题 + 模型设置 ──
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        # ── 标签页 ──
        self.tab_widget = QTabWidget()

        self.single_tab = SingleImageTab(self)
        self.batch_tab = BatchImageTab(self)

        self.tab_widget.addTab(self.single_tab, "🖼️  单图打标")
        self.tab_widget.addTab(self.batch_tab, "📁  批量打标")

        main_layout.addWidget(self.tab_widget, 1)

        # ── 状态栏 ──
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("就绪 — 请先加载模型")
        self.setStatusBar(self.status_bar)

    def _create_top_bar(self) -> QWidget:
        """创建顶部栏：模型选择、阈值、加载。"""
        widget = QWidget()
        widget.setFixedHeight(130)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 第一行：模型选择
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        title_lbl = QLabel("WD Tagger GUI")
        title_lbl.setObjectName("titleLabel")
        row1.addWidget(title_lbl)

        row1.addStretch()

        row1.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        for m in AVAILABLE_MODELS:
            self.model_combo.addItem(f"{m['name']}  [{m['id'].split('/')[-1]}]", m)
        self.model_combo.setCurrentIndex(0)
        row1.addWidget(self.model_combo)

        self.gpu_check = QCheckBox("GPU 加速")
        self.gpu_check.setChecked(False)
        row1.addWidget(self.gpu_check)

        self.load_model_btn = QPushButton("🔽 加载模型")
        self.load_model_btn.setObjectName("primaryBtn")
        self.load_model_btn.setFixedWidth(120)
        self.load_model_btn.clicked.connect(self._on_load_model)
        row1.addWidget(self.load_model_btn)

        self.model_status = QLabel("未加载")
        self.model_status.setStyleSheet("color: #f9e2af; font-style: italic;")
        row1.addWidget(self.model_status)

        layout.addLayout(row1)

        # 第二行：阈值滑块
        row2 = QHBoxLayout()
        row2.setSpacing(16)

        # 通用标签阈值
        row2.addWidget(QLabel("通用阈值:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(35)
        self.threshold_slider.setFixedWidth(150)
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider)
        row2.addWidget(self.threshold_slider)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.35)
        self.threshold_spin.setFixedWidth(80)
        self.threshold_spin.valueChanged.connect(self._on_threshold_spin)
        row2.addWidget(self.threshold_spin)

        row2.addSpacing(20)

        # 角色阈值
        row2.addWidget(QLabel("角色阈值:"))
        self.char_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.char_threshold_slider.setRange(0, 100)
        self.char_threshold_slider.setValue(85)
        self.char_threshold_slider.setFixedWidth(150)
        self.char_threshold_slider.valueChanged.connect(self._on_char_threshold_slider)
        row2.addWidget(self.char_threshold_slider)

        self.char_threshold_spin = QDoubleSpinBox()
        self.char_threshold_spin.setRange(0.0, 1.0)
        self.char_threshold_spin.setSingleStep(0.01)
        self.char_threshold_spin.setDecimals(2)
        self.char_threshold_spin.setValue(0.85)
        self.char_threshold_spin.setFixedWidth(80)
        self.char_threshold_spin.valueChanged.connect(self._on_char_threshold_spin)
        row2.addWidget(self.char_threshold_spin)

        row2.addStretch()

        # Trigger Words
        row2.addWidget(QLabel("Trigger Words:"))
        self.trigger_words_input = QLineEdit()
        self.trigger_words_input.setPlaceholderText("如: masterpiece, best quality")
        self.trigger_words_input.setFixedWidth(250)
        row2.addWidget(self.trigger_words_input)

        # Blacklist
        row2.addWidget(QLabel("黑名单:"))
        self.blacklist_input = QLineEdit()
        self.blacklist_input.setPlaceholderText("逗号分隔, 如: bad hands, blurry")
        self.blacklist_input.setFixedWidth(250)
        row2.addWidget(self.blacklist_input)

        row2.addStretch()

        layout.addLayout(row2)

        return widget

    def _detect_device(self):
        """检测可用设备。"""
        device = detect_device()
        providers = get_available_providers()

        if device == "cuda":
            self.gpu_check.setChecked(True)
            self.status_bar.showMessage(f"检测到 CUDA — GPU 加速已启用 | 可用提供器: {', '.join(providers[:3])}")
        elif device == "rocm":
            self.gpu_check.setChecked(True)
            self.status_bar.showMessage(f"检测到 ROCm — GPU 加速已启用")
        elif device == "directml":
            self.gpu_check.setChecked(True)
            self.status_bar.showMessage(f"检测到 DirectML — GPU 加速已启用")
        else:
            self.gpu_check.setChecked(False)
            self.gpu_check.setEnabled(False)
            self.status_bar.showMessage("未检测到 GPU — 仅 CPU 模式可用 (安装 onnxruntime-gpu 以启用 CUDA)")

    # ── 信号处理 ──

    def _on_threshold_slider(self, value):
        if not self._slider_spin_locked:
            self._slider_spin_locked = True
            self.threshold_spin.setValue(value / 100.0)
            self._slider_spin_locked = False

    def _on_threshold_spin(self, value):
        if not self._slider_spin_locked:
            self._slider_spin_locked = True
            self.threshold_slider.setValue(int(value * 100))
            self._slider_spin_locked = False

    def _on_char_threshold_slider(self, value):
        if not self._char_slider_spin_locked:
            self._char_slider_spin_locked = True
            self.char_threshold_spin.setValue(value / 100.0)
            self._char_slider_spin_locked = False

    def _on_char_threshold_spin(self, value):
        if not self._char_slider_spin_locked:
            self._char_slider_spin_locked = True
            self.char_threshold_slider.setValue(int(value * 100))
            self._char_slider_spin_locked = False

    _slider_spin_locked = False
    _char_slider_spin_locked = False

    def _on_load_model(self):
        """加载模型。"""
        model_data = self.model_combo.currentData()
        if not model_data:
            return

        model_id = model_data["id"]
        use_gpu = self.gpu_check.isChecked()
        device = detect_device() if use_gpu else "cpu"
        self.load_model_btn.setEnabled(False)
        self.model_status.setText("下载中...")
        self.model_status.setStyleSheet("color: #f9e2af; font-style: italic;")

        self.load_thread = TaggerLoadThread(
            self.tagger, model_id, use_gpu, DEFAULT_CACHE, device
        )
        self.load_thread.progress.connect(self._on_load_progress)
        self.load_thread.finished.connect(self._on_load_finished)
        self.load_thread.start()

    def _on_load_progress(self, msg: str):
        self.model_status.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_load_finished(self, success: bool, error_msg: str):
        self.load_model_btn.setEnabled(True)

        if success:
            active_providers = self.tagger.session.get_providers()
            if "CUDAExecutionProvider" in active_providers:
                device = "CUDA"
            elif "DmlExecutionProvider" in active_providers:
                device = "DirectML"
            elif "ROCMExecutionProvider" in active_providers:
                device = "ROCm"
            else:
                device = "CPU"
            self.model_status.setText(f"✅ 已加载 ({device}, {self.tagger.tag_count} 标签)")
            self.model_status.setStyleSheet("color: #a6e3a1;")
            self.status_bar.showMessage(
                f"模型已就绪 — {self.tagger.model_name} | {self.tagger.tag_count} 个标签 | {device}"
            )
            # 通知子标签页
            self.single_tab.on_model_loaded()
            self.batch_tab.on_model_loaded()
        else:
            self.model_status.setText("❌ 加载失败")
            self.model_status.setStyleSheet("color: #f38ba8;")
            self.status_bar.showMessage(f"加载失败: {error_msg}")
            QMessageBox.critical(self, "模型加载失败", error_msg)

    def get_params(self) -> dict:
        """获取当前参数。"""
        exclude_tags = set()
        if self.blacklist_input.text().strip():
            exclude_tags = {t.strip() for t in self.blacklist_input.text().split(",") if t.strip()}

        return {
            "threshold": self.threshold_spin.value(),
            "character_threshold": self.char_threshold_spin.value(),
            "exclude_tags": exclude_tags,
            "trigger_words": self.trigger_words_input.text().strip(),
            "cat_flags": {
                CAT_GENERAL: True,
                CAT_ARTIST: True,
                CAT_COPYRIGHT: True,
                CAT_CHARACTER: True,
                CAT_RATING: True,
            },
        }


# ─── 单图打标标签页 ─────────────────────────────────────────

class SingleImageTab(QWidget):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.mw = main_window
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── 左侧：图片区域 ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)

        # 缩略图预览
        self.image_label = QLabel("拖放图片到此处\n或点击下方按钮选择")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet(
            "border: 2px dashed #45475a; border-radius: 12px; "
            "background-color: #181825; color: #6c7086; font-size: 16px;"
        )
        self.image_label.setAcceptDrops(True)
        left_layout.addWidget(self.image_label, 1)

        # 按钮行
        btn_row = QHBoxLayout()
        self.open_btn = QPushButton("📂 打开图片")
        self.open_btn.setObjectName("primaryBtn")
        self.open_btn.clicked.connect(self._on_open_image)
        btn_row.addWidget(self.open_btn)

        self.tag_btn = QPushButton("🏷️ 开始打标")
        self.tag_btn.setObjectName("successBtn")
        self.tag_btn.setEnabled(False)
        self.tag_btn.clicked.connect(self._on_tag_image)
        btn_row.addWidget(self.tag_btn)

        left_layout.addLayout(btn_row)

        # 图片信息
        self.img_info = QLabel("")
        self.img_info.setStyleSheet("color: #a6adc8; font-size: 12px;")
        left_layout.addWidget(self.img_info)

        splitter.addWidget(left)

        # ── 右侧：标签结果 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)

        # 类别过滤
        cat_group = QGroupBox("标签类别")
        cat_layout = QHBoxLayout(cat_group)
        self.cat_general = QCheckBox("General"); self.cat_general.setChecked(True)
        self.cat_artist = QCheckBox("Artist"); self.cat_artist.setChecked(True)
        self.cat_copyright = QCheckBox("Copyright"); self.cat_copyright.setChecked(True)
        self.cat_character = QCheckBox("Character"); self.cat_character.setChecked(True)
        self.cat_rating = QCheckBox("Rating"); self.cat_rating.setChecked(True)
        for cb in [self.cat_general, self.cat_artist, self.cat_copyright,
                    self.cat_character, self.cat_rating]:
            cat_layout.addWidget(cb)
        right_layout.addWidget(cat_group)

        # 结果文本框
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("打标结果将显示在这里...")
        right_layout.addWidget(self.result_text, 1)

        # 操作按钮
        result_btns = QHBoxLayout()
        self.copy_btn = QPushButton("📋 复制标签")
        self.copy_btn.clicked.connect(lambda: self._copy_tags())
        self.copy_btn.setEnabled(False)
        result_btns.addWidget(self.copy_btn)

        self.save_btn = QPushButton("💾 保存 .txt")
        self.save_btn.clicked.connect(lambda: self._save_tags())
        self.save_btn.setEnabled(False)
        result_btns.addWidget(self.save_btn)

        self.save_json_btn = QPushButton("💾 保存 .json")
        self.save_json_btn.clicked.connect(lambda: self._save_tags_json())
        self.save_json_btn.setEnabled(False)
        result_btns.addWidget(self.save_json_btn)

        self.auto_save_single = QCheckBox("自动保存 .txt 到原图目录")
        self.auto_save_single.setChecked(False)
        result_btns.addWidget(self.auto_save_single)

        result_btns.addStretch()
        right_layout.addLayout(result_btns)

        splitter.addWidget(right)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter)

        # 支持拖放
        self.setAcceptDrops(True)

    def on_model_loaded(self):
        self.tag_btn.setEnabled(True)

    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif);;所有文件 (*.*)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        self.mw.current_image_path = path
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "错误", f"无法加载图片: {path}")
            return
        scaled = pixmap.scaled(
            self.image_label.size() - QSize(20, 20),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setStyleSheet("border: 2px solid #45475a; border-radius: 12px;")
        self.img_info.setText(f"📄 {os.path.basename(path)}  ({pixmap.width()}×{pixmap.height()})")

    def _on_tag_image(self):
        if not self.mw.tagger.is_loaded:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        if not self.mw.current_image_path:
            QMessageBox.warning(self, "提示", "请先选择图片")
            return

        self.tag_btn.setEnabled(False)
        self.tag_btn.setText("打标中...")

        params = self.mw.get_params()
        params["cat_flags"] = {
            CAT_GENERAL: self.cat_general.isChecked(),
            CAT_ARTIST: self.cat_artist.isChecked(),
            CAT_COPYRIGHT: self.cat_copyright.isChecked(),
            CAT_CHARACTER: self.cat_character.isChecked(),
            CAT_RATING: self.cat_rating.isChecked(),
        }

        try:
            tag_text, tag_results = self.mw.tagger.tag_image(
                self.mw.current_image_path,
                threshold=params["threshold"],
                character_threshold=params["character_threshold"],
                exclude_tags=params["exclude_tags"],
                cat_flags=params["cat_flags"],
                trigger_words=params["trigger_words"],
            )

            # 显示结果
            self._display_results(tag_text, tag_results,
                                  self.mw.tagger.categories,
                                  self.mw.tagger.tag_names)
            self._last_tag_text = tag_text
            self.copy_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.save_json_btn.setEnabled(True)

            # ── 单图自动保存 ──
            if self.auto_save_single.isChecked() and self.mw.current_image_path:
                try:
                    txt_path = os.path.join(
                        os.path.dirname(self.mw.current_image_path),
                        Path(self.mw.current_image_path).stem + ".txt"
                    )
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(tag_text)
                except Exception:
                    pass

        except Exception as e:
            QMessageBox.critical(self, "打标失败", str(e))
        finally:
            self.tag_btn.setEnabled(True)
            self.tag_btn.setText("🏷️ 开始打标")

    _last_tag_text = ""

    def _display_results(self, tag_text: str, tag_results: list, categories: list, tag_names: list):
        """格式化显示标签结果——分类展示，带置信度颜色标识。"""
        html = '<div style="font-family: sans-serif;">'

        # 根据实际标签类别分组
        char_tags = [(t, c) for t, c in tag_results
                     if categories[tag_names.index(t)] == CAT_CHARACTER]
        rating_tags = [(t, c) for t, c in tag_results
                       if categories[tag_names.index(t)] == CAT_RATING]
        artist_tags = [(t, c) for t, c in tag_results
                       if categories[tag_names.index(t)] == CAT_ARTIST]
        copyright_tags = [(t, c) for t, c in tag_results
                          if categories[tag_names.index(t)] == CAT_COPYRIGHT]
        general_tags = [(t, c) for t, c in tag_results
                        if categories[tag_names.index(t)] == CAT_GENERAL]

        sections = [
            ("🎯 Rating", rating_tags, "#cba6f7"),
            ("👤 Characters", char_tags, "#f9e2af"),
            ("🎨 Artists", artist_tags, "#94e2d5"),
            ("© Copyright", copyright_tags, "#89dceb"),
            ("🏷️ General", general_tags, "#a6e3a1"),
        ]

        for title, tags, color in sections:
            if not tags:
                continue
            html += f'<p><b style="color:{color};">{title}</b> ({len(tags)}):</p>'
            html += '<p style="line-height:2.0;">'
            for tag, conf in tags:
                # 置信度颜色渐变
                if conf >= 0.7:
                    c = "#a6e3a1"  # 绿色高置信
                elif conf >= 0.5:
                    c = "#f9e2af"  # 黄色中等
                else:
                    c = "#f38ba8"  # 红色低置信
                html += (
                    f'<span style="background-color:#313244;border-radius:4px;'
                    f'padding:2px 8px;margin:2px;display:inline-block;">'
                    f'{tag.replace("_", " ")} '
                    f'<span style="font-size:10px;color:{c};">{conf:.1%}</span>'
                    f'</span> '
                )
            html += '</p>'

        html += f'<hr style="border-color:#45475a;">'
        html += f'<p><b>📝 纯文本:</b></p>'
        html += f'<p style="color:#cdd6f4;background:#181825;padding:8px;border-radius:6px;">{tag_text}</p>'
        html += '</div>'

        self.result_text.setHtml(html)
        # 同时存储纯文本
        self._last_tag_text = tag_text

    def _copy_tags(self):
        if hasattr(self, '_last_tag_text'):
            QApplication.clipboard().setText(self._last_tag_text)
            self.mw.status_bar.showMessage("标签已复制到剪贴板", 3000)

    def _save_tags(self):
        if not hasattr(self, '_last_tag_text') or not self.mw.current_image_path:
            return
        default_name = Path(self.mw.current_image_path).stem + ".txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存标签文件", default_name,
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._last_tag_text)
            self.mw.status_bar.showMessage(f"已保存: {path}", 3000)

    def _save_tags_json(self):
        if not hasattr(self, '_last_tag_text') or not self.mw.current_image_path:
            return
        import json
        default_name = Path(self.mw.current_image_path).stem + ".json"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存标签 JSON", default_name,
            "JSON 文件 (*.json);;所有文件 (*.*)"
        )
        if path:
            data = {
                "image": os.path.basename(self.mw.current_image_path),
                "tags": self._last_tag_text,
                "params": {
                    "threshold": self.mw.threshold_spin.value(),
                    "character_threshold": self.mw.char_threshold_spin.value(),
                    "model": self.mw.tagger.model_name,
                }
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.mw.status_bar.showMessage(f"已保存: {path}", 3000)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
                self._load_image(path)


# ─── 批量打标标签页 ─────────────────────────────────────────

class BatchImageTab(QWidget):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.mw = main_window
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # ── 顶部控制栏 ──
        control = QGroupBox("批量打标设置")
        control_layout = QGridLayout(control)
        control_layout.setSpacing(8)

        # 输入文件夹
        control_layout.addWidget(QLabel("输入文件夹:"), 0, 0)
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("选择包含图片的文件夹...")
        control_layout.addWidget(self.folder_input, 0, 1, 1, 2)
        self.browse_btn = QPushButton("📂 浏览")
        self.browse_btn.clicked.connect(self._on_browse_folder)
        control_layout.addWidget(self.browse_btn, 0, 3)

        # 输出文件夹
        control_layout.addWidget(QLabel("输出文件夹:"), 1, 0)
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("留空则输出到图片所在文件夹")
        control_layout.addWidget(self.output_input, 1, 1, 1, 2)
        self.browse_out_btn = QPushButton("📂 浏览")
        self.browse_out_btn.clicked.connect(self._on_browse_output)
        control_layout.addWidget(self.browse_out_btn, 1, 3)

        # 选项行
        self.recursive_check = QCheckBox("递归扫描子文件夹")
        self.recursive_check.setChecked(True)
        control_layout.addWidget(self.recursive_check, 2, 0)

        control_layout.addWidget(QLabel("输出格式:"), 2, 1)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["每图一个 .txt", "合并 .txt", "合并 .csv"])
        control_layout.addWidget(self.format_combo, 2, 2)

        control_layout.addWidget(QLabel("批大小:"), 2, 3)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        control_layout.addWidget(self.batch_size_spin, 2, 4)

        # 自动保存复选框
        self.auto_save_check = QCheckBox("自动保存 .txt 到图片目录")
        self.auto_save_check.setChecked(True)
        control_layout.addWidget(self.auto_save_check, 3, 0, 1, 3)

        layout.addWidget(control)

        # ── 中间：文件列表 + 进度 ──
        mid_layout = QHBoxLayout()

        # 图片列表
        list_group = QGroupBox("图片列表")
        list_layout = QVBoxLayout(list_group)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setAlternatingRowColors(True)
        list_layout.addWidget(self.file_list)

        list_btns = QHBoxLayout()
        self.scan_btn = QPushButton("🔍 扫描图片")
        self.scan_btn.clicked.connect(self._on_scan_images)
        list_btns.addWidget(self.scan_btn)

        self.clear_list_btn = QPushButton("🗑️ 清空")
        self.clear_list_btn.clicked.connect(self._on_clear_list)
        list_btns.addWidget(self.clear_list_btn)

        self.remove_btn = QPushButton("❌ 移除选中")
        self.remove_btn.clicked.connect(self._on_remove_selected)
        list_btns.addWidget(self.remove_btn)
        list_layout.addLayout(list_btns)

        mid_layout.addWidget(list_group, 1)

        # 进度区域
        progress_group = QGroupBox("打标进度")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("等待开始...")
        self.progress_label.setStyleSheet("color: #a6adc8;")
        progress_layout.addWidget(self.progress_label)

        self.progress_detail = QTextEdit()
        self.progress_detail.setReadOnly(True)
        self.progress_detail.setMaximumHeight(300)
        self.progress_detail.setPlaceholderText("详细进度将在这里显示...")
        progress_layout.addWidget(self.progress_detail)

        mid_layout.addWidget(progress_group, 1)

        layout.addLayout(mid_layout, 1)

        # ── 底部操作按钮 ──
        bottom = QHBoxLayout()

        self.start_batch_btn = QPushButton("🚀 开始批量打标")
        self.start_batch_btn.setObjectName("successBtn")
        self.start_batch_btn.setMinimumHeight(36)
        self.start_batch_btn.setEnabled(False)
        self.start_batch_btn.clicked.connect(self._on_start_batch)
        bottom.addWidget(self.start_batch_btn)

        self.stop_btn = QPushButton("⏹️ 停止")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_batch)
        bottom.addWidget(self.stop_btn)

        bottom.addStretch()

        self.save_all_btn = QPushButton("💾 导出合并文件")
        self.save_all_btn.setEnabled(False)
        self.save_all_btn.clicked.connect(self._on_save_all)
        bottom.addWidget(self.save_all_btn)

        layout.addLayout(bottom)

        self.setAcceptDrops(True)

        # 线程引用
        self._batch_thread: BatchTagThread | None = None
        self._batch_results: list[tuple[str, str]] = []

    def on_model_loaded(self):
        self.start_batch_btn.setEnabled(True)

    def _on_browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.folder_input.setText(path)

    def _on_browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_input.setText(path)

    def _on_scan_images(self):
        folder = self.folder_input.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "提示", "请先选择有效的文件夹")
            return

        recursive = self.recursive_check.isChecked()
        self.mw.status_bar.showMessage("正在扫描图片...")
        images = find_images(folder, recursive=recursive)
        self.file_list.clear()
        for img in images:
            item = QListWidgetItem(os.path.relpath(img, folder))
            item.setData(Qt.ItemDataRole.UserRole, img)
            self.file_list.addItem(item)

        self.mw.status_bar.showMessage(
            f"扫描完成: 找到 {len(images)} 张图片 (递归={recursive})"
        )

    def _on_clear_list(self):
        self.file_list.clear()
        self._batch_results = []
        self.progress_bar.setValue(0)
        self.progress_label.setText("等待开始...")
        self.progress_detail.clear()

    def _on_remove_selected(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _on_start_batch(self):
        if not self.mw.tagger.is_loaded:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "提示", "请先扫描图片")
            return

        params = self.mw.get_params()

        # 收集图片路径
        image_paths = []
        for i in range(self.file_list.count()):
            image_paths.append(self.file_list.item(i).data(Qt.ItemDataRole.UserRole))

        self._batch_results = []
        self.progress_bar.setMaximum(len(image_paths))
        self.progress_bar.setValue(0)
        self.progress_detail.clear()
        self.start_batch_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_all_btn.setEnabled(False)

        self._batch_thread = BatchTagThread(
            self.mw.tagger, image_paths,
            params["threshold"], params["character_threshold"],
            params["exclude_tags"], params["cat_flags"],
            params["trigger_words"],
            self.batch_size_spin.value(),
        )
        self._batch_thread.progress.connect(self._on_batch_progress)
        self._batch_thread.item_done.connect(self._on_batch_item_done)
        self._batch_thread.finished.connect(self._on_batch_finished)
        self._batch_thread.error.connect(self._on_batch_error)
        self._batch_thread.start()

    def _on_stop_batch(self):
        if self._batch_thread and self._batch_thread.isRunning():
            self._batch_thread.terminate()
            self._batch_thread.wait(2000)
            self.start_batch_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.save_all_btn.setEnabled(True)
            self.mw.status_bar.showMessage("已停止批量打标")

    def _on_batch_progress(self, current: int, total: int, current_file: str):
        self.progress_bar.setValue(current)
        self.progress_label.setText(
            f"进度: {current}/{total} ({current/total*100:.1f}%) — {os.path.basename(current_file)}"
        )

    def _on_batch_item_done(self, idx: int, filename: str, tags: str):
        # 在结果列表中添加条目
        img_path = self.file_list.item(idx).data(Qt.ItemDataRole.UserRole)
        self._batch_results.append((img_path, tags))

        # ── 自动保存 .txt ──
        if self.auto_save_check.isChecked() and tags and not tags.startswith("[ERROR]"):
            try:
                output_dir = self.output_input.text().strip()
                if output_dir and os.path.isdir(output_dir):
                    txt_path = os.path.join(output_dir, Path(img_path).stem + ".txt")
                else:
                    txt_path = os.path.join(os.path.dirname(img_path), Path(img_path).stem + ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(tags)
            except Exception:
                pass  # 自动保存失败不中断流程

        # 更新进度详情
        tag_count = len(tags.split(", ")) if tags else 0
        self.progress_detail.append(
            f"[{idx+1}] {filename} → {tag_count} tags"
        )

        # 滚动到底部
        cursor = self.progress_detail.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.progress_detail.setTextCursor(cursor)

    def _on_batch_finished(self, success: int, errors: int):
        self.start_batch_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_all_btn.setEnabled(True)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_label.setText(f"完成! 成功: {success}, 失败: {errors}")
        self.mw.status_bar.showMessage(
            f"批量打标完成 — 成功 {success}/{self.file_list.count()} 张, 失败 {errors} 张"
        )

    def _on_batch_error(self, msg: str):
        QMessageBox.critical(self, "打标错误", msg)
        self.start_batch_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_save_all(self):
        if not self._batch_results:
            QMessageBox.warning(self, "提示", "没有可保存的结果")
            return

        output_dir = self.output_input.text().strip()
        fmt = self.format_combo.currentIndex()

        if fmt == 0:  # 每图一个 .txt
            if self.auto_save_check.isChecked():
                QMessageBox.information(
                    self, "提示",
                    "自动保存已开启，每张图片的 .txt 已经写入到图片所在目录。\n"
                    "如需重新写入，请取消勾选\"自动保存\"后再次点击导出。"
                )
                return
            saved = 0
            for img_path, tags in self._batch_results:
                if output_dir:
                    txt_path = os.path.join(output_dir, Path(img_path).stem + ".txt")
                else:
                    txt_path = os.path.join(os.path.dirname(img_path), Path(img_path).stem + ".txt")
                os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(tags)
                saved += 1
            self.mw.status_bar.showMessage(f"已保存 {saved} 个 .txt 文件", 5000)

        elif fmt == 1:  # 合并 .txt
            save_path = os.path.join(output_dir or self.folder_input.text(), "tags_all.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                for img_path, tags in self._batch_results:
                    f.write(f"{os.path.basename(img_path)}\t{tags}\n")
            self.mw.status_bar.showMessage(f"已保存合并 .txt: {save_path}", 5000)

        elif fmt == 2:  # 合并 .csv
            save_path = os.path.join(output_dir or self.folder_input.text(), "tags_all.csv")
            with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "tags"])
                for img_path, tags in self._batch_results:
                    writer.writerow([os.path.basename(img_path), tags])
            self.mw.status_bar.showMessage(f"已保存合并 .csv: {save_path}", 5000)
        QMessageBox.information(self, "保存完成",
                                f"已保存 {len(self._batch_results)} 个打标结果")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            first = urls[0].toLocalFile()
            if os.path.isdir(first):
                self.folder_input.setText(first)
                self._on_scan_images()
