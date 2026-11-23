"""TesseraMap - Model Weights Downloader
Downloads fine-tuned YOLO26x and SegFormer-B5 model weights.
Usage:
    python download_weights.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
YOLO_DIR = WEIGHTS_DIR / "yolo"
SEGFORMER_DIR = WEIGHTS_DIR / "segformer"

# Configured for GitHub Releases (can also be overridden via environment variables)
YOLO_URL = os.getenv(
    "YOLO_WEIGHTS_URL",
    "https://github.com/mrkerchick/TesseraMap/releases/download/v1.0.0/yolo26x_best.pt"
)
SEGFORMER_MODEL_URL = os.getenv(
    "SEGFORMER_MODEL_URL",
    "https://github.com/mrkerchick/TesseraMap/releases/download/v1.0.0/segformer_b5_model.safetensors"
)
SEGFORMER_CONFIG_URL = os.getenv(
    "SEGFORMER_CONFIG_URL",
    "https://github.com/mrkerchick/TesseraMap/releases/download/v1.0.0/segformer_config.json"
)


def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"✓ Уже загружен: {dest_path.name} ({dest_path.stat().st_size // (1024*1024)} MB)")
        return

    print(f"Загрузка {dest_path.name} из {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✓ Успешно: {dest_path.name}")
    except Exception as e:
        print(f"✗ Ошибка загрузки {dest_path.name}: {e}")
        print("  Вы можете скачать файл вручную и положить в:", dest_path.parent)


def main():
    print("=== TesseraMap: Загрузка весов моделей ===")
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    SEGFORMER_DIR.mkdir(parents=True, exist_ok=True)

    download_file(YOLO_URL, YOLO_DIR / "best.pt")
    download_file(SEGFORMER_MODEL_URL, SEGFORMER_DIR / "model.safetensors")
    download_file(SEGFORMER_CONFIG_URL, SEGFORMER_DIR / "config.json")
    print("=== Готово! Веса размещены в папке weights/ ===")


if __name__ == "__main__":
    main()
