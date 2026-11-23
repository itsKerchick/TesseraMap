"""TesseraMap - Model Weights Downloader
Automatically downloads fine-tuned YOLO26x and SegFormer-B5 model weights from Dropbox
and unpacks them into the proper local structure:
    weights/
      ├── yolo/
      │     └── best.pt
      └── segformer/
            ├── model.safetensors
            └── config.json

Usage:
    python download_weights.py
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
YOLO_DIR = WEIGHTS_DIR / "yolo"
SEGFORMER_DIR = WEIGHTS_DIR / "segformer"

DROPBOX_ZIP_URL = os.getenv(
    "MODELS_ZIP_URL",
    "https://www.dropbox.com/scl/fo/tlmopc7m8ztcufy1cb7of/ALnVYS1YMse7pVmTnhtjYcA?rlkey=carjt86vleudkh35sf0vdf648&st=havn7ppz&dl=1",
)


def format_bytes(bytes_num: int) -> str:
    if bytes_num < 1024:
        return f"{bytes_num} B"
    elif bytes_num < 1024 * 1024:
        return f"{bytes_num / 1024:.1f} KB"
    elif bytes_num < 1024 * 1024 * 1024:
        return f"{bytes_num / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_num / (1024 * 1024 * 1024):.2f} GB"


def download_with_progress(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TesseraMap/2.0"})

    print(f"Подключение к Dropbox...")
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        total_len = resp.headers.get("Content-Length")
        total_bytes = int(total_len) if total_len and total_len.isdigit() else 0
        downloaded = 0
        start_time = time.time()
        last_print = 0

        while True:
            chunk = resp.read(1024 * 1024)  # 1 MB chunk
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print > 0.4:
                elapsed = max(0.1, now - start_time)
                speed = downloaded / elapsed
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    print(
                        f"\r  Загрузка: {pct:5.1f}% | {format_bytes(downloaded)} / {format_bytes(total_bytes)} | {format_bytes(int(speed))}/s    ",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  Загружено: {format_bytes(downloaded)} | {format_bytes(int(speed))}/s    ",
                        end="",
                        flush=True,
                    )
                last_print = now

        print(f"\r  Загрузка завершена! Всего: {format_bytes(downloaded)}                           ")


def unpack_and_organize(zip_path: Path):
    print("Распаковка архива моделей...")
    extract_temp = WEIGHTS_DIR / "_temp_extract"
    extract_temp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_temp)

    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    SEGFORMER_DIR.mkdir(parents=True, exist_ok=True)

    found_yolo = False
    found_segformer = False

    # Search extracted files and place them in target folders
    for root_dir, _, files in os.walk(extract_temp):
        for f in files:
            src = Path(root_dir) / f
            fname_lower = f.lower()

            if "best.pt" in fname_lower or (fname_lower.endswith(".pt") and "yolo" in fname_lower):
                dest = YOLO_DIR / "best.pt"
                shutil.move(str(src), str(dest))
                print(f"  ✓ YOLO26x: {dest.relative_to(ROOT)} ({format_bytes(dest.stat().st_size)})")
                found_yolo = True

            elif fname_lower.endswith(".safetensors") or fname_lower == "pytorch_model.bin":
                dest = SEGFORMER_DIR / f
                shutil.move(str(src), str(dest))
                print(f"  ✓ SegFormer-B5 веса: {dest.relative_to(ROOT)} ({format_bytes(dest.stat().st_size)})")
                found_segformer = True

            elif fname_lower == "config.json":
                dest = SEGFORMER_DIR / "config.json"
                shutil.move(str(src), str(dest))
                print(f"  ✓ SegFormer config: {dest.relative_to(ROOT)}")

    # Clean up temp folder and zip
    shutil.rmtree(extract_temp, ignore_errors=True)
    try:
        zip_path.unlink()
    except Exception:
        pass

    if found_yolo and found_segformer:
        print("\n=== Все веса моделей успешно установлены и готовы к работе! ===")
    else:
        print("\nВнимание: проверьте папку weights/, не все файлы были автоматически сопоставлены.")


def main():
    print("==================================================")
    print("      TesseraMap - Загрузка весов моделей        ")
    print("==================================================")

    yolo_file = YOLO_DIR / "best.pt"
    segformer_file = SEGFORMER_DIR / "model.safetensors"

    if yolo_file.exists() and segformer_file.exists():
        print(f"✓ Веса уже установлены в папке {WEIGHTS_DIR.relative_to(ROOT)}:")
        print(f"  - YOLO26x: {yolo_file.relative_to(ROOT)} ({format_bytes(yolo_file.stat().st_size)})")
        print(f"  - SegFormer-B5: {segformer_file.relative_to(ROOT)} ({format_bytes(segformer_file.stat().st_size)})")
        return

    temp_zip = WEIGHTS_DIR / "Models_tessera.zip"
    try:
        download_with_progress(DROPBOX_ZIP_URL, temp_zip)
        unpack_and_organize(temp_zip)
    except Exception as e:
        print(f"\n✗ Ошибка при скачивании или распаковке: {e}")
        print(f"Вы можете скачать архив вручную по ссылке:")
        print(f"  {DROPBOX_ZIP_URL}")
        print(f"И распаковать файлы в: {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
