# 🦐 WD Tagger GUI

> 基于 PyQt6 的 WD（Waifu Diffusion）图片智能打标工具，支持 CUDA GPU 加速、单图/批量打标、Trigger Words 与黑名单 Tags。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![GUI](https://img.shields.io/badge/GUI-PyQt6-41cd52)
![Acceleration](https://img.shields.io/badge/Acceleration-CUDA%20%2F%20CPU-76b900)
![License](https://img.shields.io/badge/License-MIT-yellow)

WD Tagger GUI 是一个本地运行的图片自动打标工具，使用 [SmilingWolf](https://huggingface.co/SmilingWolf) 系列 WD14 ONNX 模型（含最新 v3 系列），无需联网推理即可为图片生成 Danbooru 风格的标签。所有推理均在本地完成（CPU 或 NVIDIA CUDA GPU），图片数据不会上传。

---

## ✨ 功能特性

### 核心能力
- **单图打标**：实时预览图片与打标结果，标签按置信度着色（绿/黄/红）
- **批量打标**：选择文件夹一键批量处理，支持**递归扫描子文件夹**
- **完整支持非英文（中文等）文件夹名与文件名**，全程 UTF-8 处理
- **CUDA GPU 加速**：自动检测并优先使用 GPU（onnxruntime-gpu），无 GPU 时自动回退 CPU

### 标签控制
- **通用标签阈值**：滑块 + 数字输入框双控件联动（0~1，步长 0.01）
- **角色标签阈值**：独立滑块 + 数字输入框
- **标签类别过滤**：General / Character / Artist / Copyright / Rating 可自由勾选
- **黑名单 Tags**：输入逗号分隔的标签，批量打标时自动排除
- **Trigger Words**：批量打标时自动将触发词写入每个标签文件开头

### 模型管理
- **10 款模型可选**（下拉菜单），覆盖 v3 最新系列与 v2 经典系列
- 首次使用自动从 HuggingFace 下载模型，支持断点续传
- 支持自定义 `HF_ENDPOINT` 镜像加速下载

### 界面与输出
- 深色 Catppuccin 主题，标签页切换「单图打标 / 批量打标」
- 批量打标实时进度条、当前文件显示、可调批大小（1~64）
- 输出格式：每图独立 `.txt` / 合并 `.txt` / 合并 `.csv`
- 单图结果支持一键复制、保存 `.txt`、保存 `.json`
- 支持拖放图片/文件夹到窗口

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Windows / Linux / macOS
- （可选）NVIDIA GPU + CUDA 环境

### 安装

```bash
# 克隆项目
git clone https://github.com/tiengalaxy/wd-tagger-gui.git
cd wd-tagger-gui

# 安装依赖
pip install -r requirements.txt

# （可选）NVIDIA GPU 用户安装 CUDA 加速版
pip install onnxruntime-gpu
```

### 运行

```bash
# 命令行启动
python main.py

# Windows 用户也可直接双击
启动打标工具.bat
```

> 首次使用时会自动从 HuggingFace 下载所选模型（约 300~600MB）。
> 国内网络可设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`

---

## 🎯 使用指南

### 单图打标
1. 切换到「单图打标」标签页
2. 打开或拖入一张图片
3. 选择模型（首次使用需等待下载）
4. 调整通用/角色阈值，勾选需要的标签类别
5. 点击「开始打标」，结果按置信度显示

### 批量打标
1. 切换到「批量打标」标签页
2. 选择图片文件夹（可勾选「递归子文件夹」）
3. 填写 Trigger Words（可选，写入每个标签文件开头）
4. 填写黑名单 Tags（可选，逗号分隔，自动排除）
5. 选择输出格式与批大小，点击「开始批量打标」

---

## 🧠 支持的模型

| 模型 | 系列 | 特点 |
|------|------|------|
| `SmilingWolf/wd-eva02-large-tagger-v3` | v3 | 最新最精准，体积最大 |
| `SmilingWolf/wd-swinv2-tagger-v3` | v3 | 综合表现最佳（默认） |
| `SmilingWolf/wd-vit-large-tagger-v3` | v3 | 精度与速度均衡 |
| `SmilingWolf/wd-vit-tagger-v3` | v3 | 标准快速 |
| `SmilingWolf/wd-convnext-tagger-v3` | v3 | 极速，适合大批量 |
| `SmilingWolf/wd-v1-4-convnextv2-tagger-v2` | v2 | 经典高性能 |
| `SmilingWolf/wd-v1-4-moat-tagger-v2` | v2 | 经典 |
| `SmilingWolf/wd-v1-4-swinv2-tagger-v2` | v2 | 经典 |
| `SmilingWolf/wd-v1-4-convnext-tagger-v2` | v2 | 经典快速 |
| `SmilingWolf/wd-v1-4-vit-tagger-v2` | v2 | 经典均衡 |

---

## 📁 项目结构

```
wd_tagger_gui/
├── main.py                  # 程序入口
├── tagger_backend.py        # 打标引擎（ONNX Runtime + HuggingFace）
├── requirements.txt         # Python 依赖
├── 启动打标工具.bat          # Windows 一键启动脚本
└── gui/
    ├── __init__.py
    └── main_window.py       # GUI 主窗口（PyQt6 深色主题）
```

---

## 🔧 技术栈

- **GUI**：PyQt6（QTabWidget 标签页、QSlider + QDoubleSpinBox 联动、QThread 多线程）
- **推理**：ONNX Runtime（自动选择 CUDA / CPU ExecutionProvider）
- **模型**：HuggingFace Hub 自动下载 WD14 Tagger ONNX 模型
- **图像**：Pillow（支持 PNG / JPG / WEBP / BMP 等格式）

---

## ⚠️ 说明

- 本项目仅用于学习与合法用途，请遵守模型原作者的开源许可（Apache-2.0）
- 标签结果由 AI 生成，可能存在误差，请人工复核关键标签
- 模型下载需联网，推理全程离线本地执行

## 📄 许可证

[MIT](LICENSE)
