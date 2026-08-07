"""
WD Tagger 后端核心模块
====================
基于 SmilingWolf ONNX 模型的图片打标引擎。
支持 CUDA (onnxruntime-gpu) 和 CPU 推理。
"""

import csv
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ─── 依赖懒加载 ───────────────────────────────────────────
_ort = None
_hf_hub_download = None


def _get_ort():
    global _ort
    if _ort is None:
        import onnxruntime as ort
        _ort = ort
    return _ort


def _get_hf_hub():
    global _hf_hub_download
    if _hf_hub_download is None:
        from huggingface_hub import hf_hub_download
        _hf_hub_download = hf_hub_download
    return _hf_hub_download


# ─── 常量 ─────────────────────────────────────────────────
TARGET_SIZE = 448
MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "selected_tags.csv"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "wd14_tagger")

CAT_GENERAL = 0
CAT_ARTIST = 1
CAT_COPYRIGHT = 3
CAT_CHARACTER = 4
CAT_RATING = 5

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# ─── 预定义模型列表 ────────────────────────────────────────
AVAILABLE_MODELS = [
    {
        "id": "SmilingWolf/wd-eva02-large-tagger-v3",
        "name": "EVA02-Large v3 (最精准)",
        "description": "最新 V3 大模型，精度最高，速度较慢",
    },
    {
        "id": "SmilingWolf/wd-swinv2-tagger-v3",
        "name": "SwinV2 v3 (高精度)",
        "description": "V3 高精度模型，综合表现优秀",
    },
    {
        "id": "SmilingWolf/wd-vit-large-tagger-v3",
        "name": "ViT-Large v3 (均衡)",
        "description": "V3 大模型，精度与速度均衡",
    },
    {
        "id": "SmilingWolf/wd-vit-tagger-v3",
        "name": "ViT v3 (快速)",
        "description": "V3 标准模型，速度较快",
    },
    {
        "id": "SmilingWolf/wd-convnext-tagger-v3",
        "name": "ConvNext v3 (极速)",
        "description": "V3 最快模型，适合大批量处理",
    },
    {
        "id": "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "name": "MOAT v2 (经典)",
        "description": "V2 MOAT 架构，精度极高",
    },
    {
        "id": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "name": "SwinV2 v2 (经典)",
        "description": "V2 经典模型",
    },
    {
        "id": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        "name": "ConvNextV2 v2 (经典)",
        "description": "V2 经典快速模型",
    },
    {
        "id": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "name": "ConvNext v2 (经典)",
        "description": "V2 早期模型",
    },
    {
        "id": "SmilingWolf/wd-v1-4-vit-tagger-v2",
        "name": "ViT v2 (经典)",
        "description": "V2 早期模型",
    },
]


def get_available_providers() -> list[str]:
    """检测可用的 ONNX Runtime 执行提供器。"""
    try:
        ort = _get_ort()
        available = ort.get_available_providers()
        return available
    except Exception:
        return []


def has_cuda() -> bool:
    """检测 CUDA 是否可用。"""
    return "CUDAExecutionProvider" in get_available_providers()


def detect_device() -> str:
    """检测最佳可用设备。"""
    providers = get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    elif "ROCMExecutionProvider" in providers:
        return "rocm"
    elif "DmlExecutionProvider" in providers:
        return "directml"
    else:
        return "cpu"


# ─── 模型管理 ─────────────────────────────────────────────


def download_model(
    model_name: str,
    cache_dir: str = DEFAULT_CACHE,
    progress_callback: Optional[callable] = None,
) -> tuple[str, str]:
    """从 HuggingFace 下载模型文件，返回 (model_path, tags_path)。"""
    hf_hub_download = _get_hf_hub()

    local_dir = os.path.join(cache_dir, model_name.replace("/", "_"))
    safe_local_dir = local_dir  # 保留非 ASCII 路径
    model_path = os.path.join(safe_local_dir, MODEL_FILENAME)
    tags_path = os.path.join(safe_local_dir, TAGS_FILENAME)

    if progress_callback:
        progress_callback(f"检查模型: {model_name}")

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        if progress_callback:
            progress_callback(f"下载模型文件... (约 300-600MB)")
        os.makedirs(safe_local_dir, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=MODEL_FILENAME,
                local_dir=safe_local_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"下载模型失败: {e}\n"
                f"提示: 国内网络可设置环境变量 HF_ENDPOINT=https://hf-mirror.com\n"
                f"或手动下载模型放到: {safe_local_dir}"
            )

    if not os.path.exists(tags_path):
        if progress_callback:
            progress_callback("下载标签文件...")
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=TAGS_FILENAME,
                local_dir=safe_local_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise RuntimeError(f"下载标签文件失败: {e}")

    return model_path, tags_path


def load_model(
    model_path: str, use_gpu: bool = False
) -> "ort.InferenceSession":
    """加载 ONNX 模型。"""
    ort = _get_ort()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        if use_gpu:
            # 回退到 CPU
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        else:
            raise
    return session


def load_tags(tags_path: str) -> tuple[list[str], list[int]]:
    """从 CSV 加载标签列表。"""
    tag_names: list[str] = []
    categories: list[int] = []
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 3:
                tag_names.append(row[1].strip())
                try:
                    categories.append(int(row[2]))
                except ValueError:
                    categories.append(0)
    return tag_names, categories


# ─── 图片处理 ─────────────────────────────────────────────


def preprocess_image(image_path: str, target_size: int = TARGET_SIZE) -> np.ndarray:
    """预处理单张图片。"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_size, target_size), Image.BICUBIC)
    arr = np.array(image, dtype=np.float32)
    return arr


def preprocess_batch(image_paths: list[str], target_size: int = TARGET_SIZE) -> np.ndarray:
    """批量预处理。"""
    batch = [preprocess_image(p, target_size) for p in image_paths]
    return np.stack(batch, axis=0)


# ─── 标签提取 ─────────────────────────────────────────────


def extract_tags(
    probs: np.ndarray,
    tag_names: list[str],
    categories: list[int],
    threshold: float,
    character_threshold: float,
    exclude_tags: set[str] | None = None,
    cat_flags: dict[int, bool] | None = None,
) -> list[tuple[str, float]]:
    """从模型输出提取标签，返回 [(tag, confidence), ...] 按置信度降序。"""
    if exclude_tags is None:
        exclude_tags = set()
    if cat_flags is None:
        cat_flags = {}

    results: list[tuple[str, float]] = []
    for i, (name, cat, prob) in enumerate(zip(tag_names, categories, probs)):
        if name in exclude_tags:
            continue
        if cat_flags and not cat_flags.get(cat, True):
            continue
        thresh = character_threshold if cat == CAT_CHARACTER else threshold
        if prob >= thresh:
            results.append((name, float(prob)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def extract_tags_flat(
    probs: np.ndarray,
    tag_names: list[str],
    categories: list[int],
    threshold: float,
    character_threshold: float,
    exclude_tags: set[str] | None = None,
    cat_flags: dict[int, bool] | None = None,
) -> str:
    """提取标签并返回逗号分隔字符串。"""
    tags = extract_tags(probs, tag_names, categories, threshold, character_threshold,
                        exclude_tags, cat_flags)
    return ", ".join(t[0].replace("_", " ") for t in tags)


# ─── 文件查找 ─────────────────────────────────────────────


def find_images(input_dir: str, recursive: bool = False) -> list[str]:
    """查找目录中所有图片文件（支持非英文路径）。"""
    images: list[str] = []
    input_path = Path(input_dir)

    if recursive:
        for root, _, files in os.walk(str(input_path)):
            for f in sorted(files):
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(os.path.join(root, f))
    else:
        for f in sorted(input_path.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(str(f))
    return images


# ─── Tagger 类（面向 GUI） ─────────────────────────────────


class WDTagger:
    """WD Tagger 封装类，供 GUI 调用。"""

    def __init__(self):
        self.session = None
        self.tag_names: list[str] = []
        self.categories: list[int] = []
        self.input_name: str = ""
        self.model_name: str = ""
        self.use_gpu: bool = False
        self._model_path: str = ""
        self._tags_path: str = ""
        self._model_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def tag_count(self) -> int:
        return len(self.tag_names)

    def load(
        self,
        model_id: str,
        use_gpu: bool = False,
        cache_dir: str = DEFAULT_CACHE,
        local_model_dir: str | None = None,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """加载模型。"""
        if local_model_dir:
            mp = os.path.join(local_model_dir, MODEL_FILENAME)
            tp = os.path.join(local_model_dir, TAGS_FILENAME)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"模型文件不存在: {mp}")
            self._model_path = mp
            self._tags_path = tp
        else:
            self._model_path, self._tags_path = download_model(
                model_id, cache_dir, progress_callback
            )

        if progress_callback:
            progress_callback("加载 ONNX 模型...")
        self.session = load_model(self._model_path, use_gpu)
        self.input_name = self.session.get_inputs()[0].name
        self.tag_names, self.categories = load_tags(self._tags_path)
        self.model_name = model_id
        self.use_gpu = use_gpu
        self._model_loaded = True

        return True

    def unload(self):
        """卸载模型释放显存。"""
        self.session = None
        self.tag_names = []
        self.categories = []
        self._model_loaded = False

    def tag_image(
        self,
        image_path: str,
        threshold: float = 0.35,
        character_threshold: float = 0.85,
        exclude_tags: set[str] | None = None,
        cat_flags: dict[int, bool] | None = None,
        trigger_words: str = "",
    ) -> tuple[str, list[tuple[str, float]]]:
        """对单张图片打标，返回 (tag_text, [(tag, conf), ...])。"""
        if not self._model_loaded:
            raise RuntimeError("模型未加载")

        arr = preprocess_image(image_path)
        arr = np.expand_dims(arr, axis=0)  # (1, 448, 448, 3)
        outputs = self.session.run(None, {self.input_name: arr})
        probs = outputs[0][0]

        tag_results = extract_tags(
            probs, self.tag_names, self.categories,
            threshold, character_threshold, exclude_tags, cat_flags,
        )

        # 插入 trigger words
        tags = [t[0].replace("_", " ") for t in tag_results]
        if trigger_words.strip():
            tw_list = [w.strip() for w in trigger_words.split(",") if w.strip()]
            # trigger words 放最前面
            tags = tw_list + tags

        tag_text = ", ".join(tags)
        return tag_text, tag_results

    def tag_batch(
        self,
        image_paths: list[str],
        threshold: float = 0.35,
        character_threshold: float = 0.85,
        exclude_tags: set[str] | None = None,
        cat_flags: dict[int, bool] | None = None,
        trigger_words: str = "",
        batch_size: int = 8,
        progress_callback: Optional[callable] = None,
    ) -> list[tuple[str, str, list[tuple[str, float]]]]:
        """批量打标，返回 [(path, tag_text, [(tag, conf), ...]), ...]。"""
        if not self._model_loaded:
            raise RuntimeError("模型未加载")

        results: list[tuple[str, str, list[tuple[str, float]]]] = []
        total = len(image_paths)

        for batch_start in range(0, total, batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]

            try:
                batch_arr = preprocess_batch(batch_paths)
                outputs = self.session.run(None, {self.input_name: batch_arr})
                probs = outputs[0]

                for j, img_path in enumerate(batch_paths):
                    tag_results = extract_tags(
                        probs[j], self.tag_names, self.categories,
                        threshold, character_threshold, exclude_tags, cat_flags,
                    )
                    tags = [t[0].replace("_", " ") for t in tag_results]
                    if trigger_words.strip():
                        tw_list = [w.strip() for w in trigger_words.split(",") if w.strip()]
                        tags = tw_list + tags

                    tag_text = ", ".join(tags)
                    results.append((img_path, tag_text, tag_results))

            except Exception as e:
                for img_path in batch_paths:
                    results.append((img_path, f"[ERROR: {e}]", []))

            if progress_callback:
                progress = min(batch_start + batch_size, total)
                progress_callback(progress, total, batch_paths[-1] if batch_paths else "")

        return results
