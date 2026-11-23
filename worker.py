"""TesseraMap - Home PC GPU ML Worker
Runs on the home PC with NVIDIA RTX 4060 Ti (CUDA + PyTorch).
Connects to the VPS in Germany via an OUTGOING WebSocket tunnel.
No white IP, no port forwarding, and no router configuration required!
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import websockets

# Ensure SegFormer-Benchmark directory is in sys.path
ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = Path(r"L:\Local project 2\SegFormer-Benchmark").resolve()
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

LOG = logging.getLogger("tesseramap.worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Hardware Monitoring (NVML + psutil)
HAS_NVML = False
GPU_NAME = "NVIDIA GeForce GPU"
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_NAME = pynvml.nvmlDeviceGetName(handle)
    if isinstance(GPU_NAME, bytes):
        GPU_NAME = GPU_NAME.decode("utf-8")
    LOG.info("NVML detected GPU: %s", GPU_NAME)
except Exception as e:
    LOG.warning("pynvml not initialized: %s", e)

WORKER_START_TIME = time.time()
EXECUTOR = ThreadPoolExecutor(max_workers=1)
MODELS = None


def get_models():
    global MODELS
    if MODELS is None:
        LOG.info("Загрузка нейросетевых моделей SegFormer и YOLO...")
        import viewer_ensemble_compare_slider_fixed_v2 as viewer
        MODELS = viewer
        LOG.info("Модели успешно загружены на устройство: %s", viewer.DEVICE)
    return MODELS


def get_gpu_telemetry():
    load = 0
    mem_str = "-- / -- MB"
    if HAS_NVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            load = int(util.gpu)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = int(mem_info.used // 1048576)
            total_mb = int(mem_info.total // 1048576)
            mem_str = f"{used_mb} / {total_mb} MB"
        except Exception:
            pass

    uptime_sec = int(time.time() - WORKER_START_TIME)
    uptime_str = str(timedelta(seconds=uptime_sec))

    return {
        "gpu_name": GPU_NAME,
        "gpu_load": load,
        "gpu_memory": mem_str,
        "uptime": uptime_str,
    }


def decode_browser_image(image_b64: str, expected_w: int, expected_h: int):
    raw = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    data = base64.b64decode(raw)
    with Image.open(BytesIO(data)) as im:
        img = im.convert("RGB")
        if img.size != (expected_w, expected_h):
            img = img.resize((expected_w, expected_h), Image.Resampling.BILINEAR)
    return img


def run_selected_model(im: Image.Image, p: dict, viewer):
    started = time.perf_counter()
    mode = p.get("mode", "Тайлы")
    model_mode = p.get("model_mode", "ensemble")
    single_size = p.get("single_size", 768)
    tile_size = p.get("tile_size", 512)
    overlap = p.get("overlap", 128)
    seg_weight = p.get("seg_weight", 60)
    weight = seg_weight / 100.0

    def predict_one(patch: Image.Image, size: int):
        if model_mode == "yolo":
            probs, _ = viewer.yolo_probs(patch, size)
            return probs.detach().float().cpu()
        if model_mode == "segformer":
            probs, _ = viewer.segformer_probs(patch, size)
            return probs.detach().float().cpu()
        if weight <= 0:
            probs, _ = viewer.yolo_probs(patch, size)
            return probs.detach().float().cpu()
        if weight >= 1:
            probs, _ = viewer.segformer_probs(patch, size)
            return probs.detach().float().cpu()
        yp, _ = viewer.yolo_probs(patch, size)
        sp, _ = viewer.segformer_probs(patch, size)
        return (1 - weight) * yp.detach().float().cpu() + weight * sp.detach().float().cpu()

    if mode == "Один проход":
        probs = predict_one(im, single_size)
        mask = probs.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
        return mask, round((time.perf_counter() - started) * 1000, 1)

    xs = viewer.make_positions(im.width, tile_size, overlap)
    ys = viewer.make_positions(im.height, tile_size, overlap)
    accumulated = np.zeros((viewer.N, im.height, im.width), dtype=np.float32)
    normalizer = np.zeros((im.height, im.width), dtype=np.float32)
    feather = max(1, overlap // 2) if overlap else 0

    for y in ys:
        for x in xs:
            crop = im.crop((x, y, min(im.width, x + tile_size), min(im.height, y + tile_size)))
            prediction = predict_one(crop, tile_size).squeeze(0).numpy()
            h, w = prediction.shape[-2:]
            window = viewer.blend_window(h, w, feather)
            accumulated[:, y : y + h, x : x + w] += prediction * window[None]
            normalizer[y : y + h, x : x + w] += window

    accumulated /= np.maximum(normalizer[None], 1e-6)
    mask = accumulated.argmax(axis=0).astype(np.uint8)
    return mask, round((time.perf_counter() - started) * 1000, 1)


def process_segmentation(payload: dict):
    t0 = time.perf_counter()
    viewer = get_models()

    img = decode_browser_image(
        payload["image_b64"],
        payload["width"],
        payload["height"]
    )

    mask, infer_ms = run_selected_model(img, payload, viewer)

    selected_class = payload.get("selected_class", -1)
    opacity = payload.get("opacity", 0.65)

    rgba = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    rgba[:, :, :3] = viewer.COLORS[mask]
    visible = (mask == selected_class) if selected_class >= 0 else (mask < viewer.N)
    rgba[:, :, 3] = np.where(visible, int(round(opacity * 255)), 0).astype(np.uint8)

    out = BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG")
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    mask_b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return mask_b64, infer_ms, total_ms


async def telemetry_broadcaster(ws: websockets.WebSocketClientProtocol):
    while True:
        try:
            telem = get_gpu_telemetry()
            msg = {
                "type": "telemetry",
                **telem,
            }
            await ws.send(json.dumps(msg))
        except Exception:
            break
        await asyncio.sleep(2.0)


async def handle_segment_request(ws: websockets.WebSocketClientProtocol, data: dict):
    task_id = data.get("task_id")
    payload = data.get("payload", {})
    loop = asyncio.get_running_loop()

    try:
        LOG.info("Начало обработки задачи сегментации [id=%s]...", task_id)
        mask_b64, infer_ms, total_ms = await loop.run_in_executor(
            EXECUTOR, process_segmentation, payload
        )
        resp = {
            "type": "segment_response",
            "task_id": task_id,
            "ok": True,
            "mask_b64": mask_b64,
            "infer_ms": infer_ms,
            "total_ms": total_ms,
        }
        await ws.send(json.dumps(resp))
        LOG.info("Задача [id=%s] завершена за %s ms (инференс: %s ms)", task_id, total_ms, infer_ms)
    except Exception as exc:
        LOG.error("Ошибка при выполнении задачи [id=%s]: %s", task_id, exc)
        resp = {
            "type": "segment_response",
            "task_id": task_id,
            "ok": False,
            "error": str(exc),
        }
        try:
            await ws.send(json.dumps(resp))
        except Exception:
            pass


async def worker_loop(vps_url: str, token: str):
    full_url = f"{vps_url}?token={token}"
    while True:
        try:
            LOG.info("Подключение к VPS Gateway (%s)...", vps_url)
            async with websockets.connect(
                full_url,
                ping_interval=20,
                ping_timeout=25,
                max_size=25_000_000,
            ) as ws:
                LOG.info("=== УСПЕШНО ПОДКЛЮЧЕНО К VPS ===")
                LOG.info("Домашний сервер готов принимать запросы на сегментацию от пользователей!")

                broadcaster_task = asyncio.create_task(telemetry_broadcaster(ws))
                try:
                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                        except Exception:
                            continue

                        if msg.get("type") == "segment_request":
                            asyncio.create_task(handle_segment_request(ws, msg))
                finally:
                    broadcaster_task.cancel()

        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as exc:
            LOG.warning("Связь с VPS потеряна (%s). Повтор подключения через 3 секунды...", exc)
        except Exception as exc:
            LOG.error("Непредвиденная ошибка воркера: %s. Повтор через 3 секунды...", exc)

        await asyncio.sleep(3.0)


def main():
    parser = argparse.ArgumentParser(description="TesseraMap Home PC ML Worker")
    parser.add_argument(
        "--vps",
        default=os.getenv("VPS_URL", "ws://127.0.0.1:7861/ws/worker"),
        help="URL WebSocket эндпоинта VPS (например, ws://your-vps-ip:7861/ws/worker или wss://domain.com/ws/worker)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("WORKER_SECRET", "tessera-secret-2026"),
        help="Секретный токен для авторизации воркера на VPS",
    )
    args = parser.parse_args()

    print("==================================================")
    print("      TesseraMap - Home PC GPU ML Worker         ")
    print(f"      GPU: {GPU_NAME}")
    print(f"      VPS URL: {args.vps}")
    print("==================================================")

    # Preload models
    get_models()

    try:
        asyncio.run(worker_loop(args.vps, args.token))
    except KeyboardInterrupt:
        print("\nВоркер остановлен пользователем.")


if __name__ == "__main__":
    main()
