"""TesseraMap Backend Server
Provides real-time telemetry, model segmentation, and MapTiler integration.
"""
from __future__ import annotations

import asyncio
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

import httpx
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Ensure SegFormer-Benchmark directory is in sys.path
ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = Path(r"L:\Local project 2\SegFormer-Benchmark").resolve()
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

HTML_FILE = ROOT / "index.html"
PDF_FILE = ROOT / "maptiler_guide.pdf"

CUSTOM_URL_TEMPLATE = os.getenv("SATELLITE_XYZ_URL", "").strip()
MAPTILER_KEY = os.getenv("MAPTILER_API_KEY", "").strip()
URL_TEMPLATE = CUSTOM_URL_TEMPLATE
TILE_SIZE = 256
MAX_VIEWPORT_W = 1024
MAX_VIEWPORT_H = 768
MAX_TILE_COUNT = 35
MAX_ZOOM = 21
MIN_SCAN_ZOOM = 16

MODEL_LOCK = threading.Lock()
MODELS = None
EXECUTOR = ThreadPoolExecutor(max_workers=1)

START_TIME = time.time()
ACTIVE_CLIENTS: dict[str, float] = {}
CLIENT_LOCK = threading.Lock()

app = FastAPI(title="TesseraMap")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG = logging.getLogger("tesseramap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EVENTS = deque(maxlen=400)
EVENT_COUNTER = 0
EVENT_LOCK = threading.Lock()

# Initialize NVML if available
HAS_NVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception as e:
    LOG.warning("pynvml not initialized: %s", e)

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def add_log(kind: str, msg: str, **fields):
    global EVENT_COUNTER
    with EVENT_LOCK:
        EVENT_COUNTER += 1
        obj = {
            "seq": EVENT_COUNTER,
            "time": datetime.now().strftime("%H:%M:%S"),
            "kind": kind,
            "msg": msg,
            **fields,
        }
        EVENTS.append(obj)
    safe_fields = {k: v for k, v in fields.items() if k not in ("url", "secret", "token", "image_b64")}
    LOG.info("%s %s %s", kind, msg, json.dumps(safe_fields, ensure_ascii=False))
    return obj


def get_system_telemetry(client_id: str | None = None, req_client_ip: str = "127.0.0.1"):
    now = time.time()
    with CLIENT_LOCK:
        if client_id:
            ACTIVE_CLIENTS[client_id] = now
        # Purge clients inactive for > 15 seconds
        dead = [k for k, last in ACTIVE_CLIENTS.items() if now - last > 15]
        for k in dead:
            del ACTIVE_CLIENTS[k]
        active_count = max(1, len(ACTIVE_CLIENTS))

    uptime_sec = int(now - START_TIME)
    hours = uptime_sec // 3600
    minutes = (uptime_sec % 3600) // 60
    seconds = uptime_sec % 60
    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    gpu_name = "N/A"
    gpu_load_str = "0%"
    gpu_mem_str = "0 / 0 MB"

    if HAS_NVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            gpu_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_load_str = f"{util.gpu}%"
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = int(mem.used / (1024 * 1024))
            total_mb = int(mem.total / (1024 * 1024))
            gpu_mem_str = f"{used_mb} / {total_mb} MB"
        except Exception:
            gpu_name = "NVIDIA GPU"
    else:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                used_mb = int(torch.cuda.memory_allocated() / (1024 * 1024))
                total_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                gpu_mem_str = f"{used_mb} / {total_mb} MB"
        except Exception:
            pass

    cpu_load_str = "0%"
    ram_str = "0 GB"
    if HAS_PSUTIL:
        try:
            cpu_load_str = f"{psutil.cpu_percent()}%"
            ram = psutil.virtual_memory()
            ram_str = f"{ram.used / (1024**3):.1f} / {ram.total / (1024**3):.1f} GB"
        except Exception:
            pass

    return {
        "gpu_name": gpu_name,
        "gpu_load": gpu_load_str,
        "gpu_mem": gpu_mem_str,
        "cpu_load": cpu_load_str,
        "ram_usage": ram_str,
        "active_users": active_count,
        "uptime": uptime_str,
        "uptime_sec": uptime_sec,
        "client_ip": req_client_ip,
    }


class ViewportRequest(BaseModel):
    zoom: int = Field(ge=0, le=MAX_ZOOM)
    px: int
    py: int
    width: int = Field(ge=128, le=MAX_VIEWPORT_W)
    height: int = Field(ge=128, le=MAX_VIEWPORT_H)
    mode: str = "Тайлы"
    model_mode: str = "ensemble"
    single_size: int = 768
    tile_size: int = 512
    overlap: int = 128
    seg_weight: int = Field(default=60, ge=0, le=100)
    opacity: float = Field(default=0.65, ge=0, le=1)
    selected_class: int = Field(default=-1, ge=-1, le=7)
    image_b64: str | None = None
    maptiler_key: str | None = None


class ClientLog(BaseModel):
    level: str = "info"
    text: str = Field(max_length=1000)
    request_id: str = Field(default="", max_length=80)


class CheckKeyRequest(BaseModel):
    key: str


def decode_browser_image(image_b64: str, expected_w: int, expected_h: int, request_id: str):
    t = time.perf_counter()
    raw = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    data = base64.b64decode(raw)
    if len(data) > 10_000_000:
        raise HTTPException(413, "Слишком большое изображение от браузера")
    with Image.open(BytesIO(data)) as im:
        img = im.convert("RGB")
        if img.size != (expected_w, expected_h):
            img = img.resize((expected_w, expected_h), Image.Resampling.BILINEAR)
    elapsed = round((time.perf_counter() - t) * 1000, 1)
    add_log("server", "Получен снимок из браузера", req=request_id, ms=elapsed, size=list(img.size))
    return img, 0, elapsed


def get_models():
    global MODELS
    if MODELS is None:
        import viewer_ensemble_compare_slider_fixed_v2 as viewer
        MODELS = viewer
    return MODELS


def run_selected_model(im: Image.Image, p: ViewportRequest, viewer):
    started = time.perf_counter()
    weight = p.seg_weight / 100.0

    def predict_one(patch: Image.Image, size: int):
        if p.model_mode == "yolo":
            probs, _ = viewer.yolo_probs(patch, size)
            return probs.detach().float().cpu()
        if p.model_mode == "segformer":
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

    if p.mode == "Один проход":
        probs = predict_one(im, p.single_size)
        mask = probs.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
        return mask, round((time.perf_counter() - started) * 1000, 1), 1

    xs = viewer.make_positions(im.width, p.tile_size, p.overlap)
    ys = viewer.make_positions(im.height, p.tile_size, p.overlap)
    accumulated = np.zeros((viewer.N, im.height, im.width), dtype=np.float32)
    normalizer = np.zeros((im.height, im.width), dtype=np.float32)
    feather = max(1, p.overlap // 2) if p.overlap else 0
    for y in ys:
        for x in xs:
            crop = im.crop((x, y, min(im.width, x + p.tile_size), min(im.height, y + p.tile_size)))
            prediction = predict_one(crop, p.tile_size).squeeze(0).numpy()
            h, w = prediction.shape[-2:]
            window = viewer.blend_window(h, w, feather)
            accumulated[:, y : y + h, x : x + w] += prediction * window[None]
            normalizer[y : y + h, x : x + w] += window
    accumulated /= np.maximum(normalizer[None], 1e-6)
    mask = accumulated.argmax(axis=0).astype(np.uint8)
    return mask, round((time.perf_counter() - started) * 1000, 1), len(xs) * len(ys)


def segment_image(im: Image.Image, p: ViewportRequest, req: str):
    t = time.perf_counter()
    with MODEL_LOCK:
        viewer = get_models()
        add_log(
            "server",
            "Начат инференс нейросетей",
            req=req,
            device=str(viewer.DEVICE),
            image=list(im.size),
            mode=p.mode,
            model_mode=p.model_mode,
            tile=p.tile_size,
            overlap=p.overlap,
            segformer=p.seg_weight,
        )
        mask, infer_ms, ml_tiles = run_selected_model(im, p, viewer)
    if mask.shape != (im.height, im.width):
        raise RuntimeError(f"Размер предсказания {mask.shape} не равен {(im.height, im.width)}")
    rgba = np.zeros((im.height, im.width, 4), dtype=np.uint8)
    rgba[:, :, :3] = viewer.COLORS[mask]
    visible = (mask == p.selected_class) if p.selected_class >= 0 else (mask < viewer.N)
    rgba[:, :, 3] = np.where(visible, int(round(p.opacity * 255)), 0).astype(np.uint8)
    out = BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG")
    total_ms = round((time.perf_counter() - t) * 1000, 1)
    counts = np.bincount(mask.reshape(-1), minlength=viewer.N)[: viewer.N]
    top_classes = sorted(zip(viewer.CLASSES, counts.tolist()), key=lambda item: item[1], reverse=True)[:4]
    gpu_mb = (
        round(__import__("torch").cuda.memory_allocated() / 1048576, 1)
        if viewer.DEVICE.type == "cuda"
        else 0
    )
    add_log(
        "server",
        "Маска готова",
        req=req,
        ms=total_ms,
        infer_ms=infer_ms,
        ml_tiles=ml_tiles,
        model_mode=p.model_mode,
        shape=list(mask.shape),
        top_classes=top_classes,
        gpu_allocated_mb=gpu_mb,
    )
    return out.getvalue(), infer_ms, total_ms


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "127.0.0.1"


@app.get("/", response_class=HTMLResponse)
async def index():
    if not HTML_FILE.exists():
        raise HTTPException(404, "index.html не найден")
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/maptiler_guide.pdf")
async def maptiler_guide():
    if not PDF_FILE.exists():
        raise HTTPException(404, "Инструкция maptiler_guide.pdf не найдена в корне проекта")
    return FileResponse(PDF_FILE, media_type="application/pdf", filename="maptiler_guide.pdf")


@app.get("/api/ping")
async def ping():
    return {"pong": True, "time": time.time()}


@app.post("/api/heartbeat")
async def heartbeat(request: Request, client_id: str | None = None):
    ip = get_client_ip(request)
    return get_system_telemetry(client_id, ip)


@app.get("/api/system-stats")
async def system_stats(request: Request, client_id: str | None = None):
    ip = get_client_ip(request)
    return get_system_telemetry(client_id, ip)


@app.get("/api/status")
async def status(request: Request, client_id: str | None = None):
    ip = get_client_ip(request)
    stats = get_system_telemetry(client_id, ip)
    return {
        "status": "ready",
        "maptiler_key": MAPTILER_KEY,
        "min_scan_zoom": MIN_SCAN_ZOOM,
        "max_width": MAX_VIEWPORT_W,
        "max_height": MAX_VIEWPORT_H,
        **stats,
    }


@app.post("/api/check-key")
async def check_key(payload: CheckKeyRequest):
    key = payload.key.strip()
    if not key:
        return {"valid": False, "error": "Ключ пуст"}
    test_url = f"https://api.maptiler.com/tiles/satellite-v2/0/0/0.jpg?key={key}"
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(test_url)
            if resp.status_code == 200 and len(resp.content) > 500:
                return {"valid": True}
            return {"valid": False, "status_code": resp.status_code, "error": "Неверный ключ"}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


@app.get("/api/logs")
async def logs(since: int = 0):
    with EVENT_LOCK:
        return [row for row in EVENTS if row["seq"] > since]


@app.post("/api/client-log")
async def client_log(msg: ClientLog):
    add_log("client", msg.text, level=msg.level, req=msg.request_id)
    return {"ok": True}


@app.post("/api/segment")
async def segment(payload: ViewportRequest):
    if payload.zoom < MIN_SCAN_ZOOM:
        raise HTTPException(
            422,
            f"Сканирование доступно только при zoom >= {MIN_SCAN_ZOOM}; получен zoom {payload.zoom}.",
        )
    if payload.mode not in ("Тайлы", "Один проход"):
        raise HTTPException(422, "Неверный режим")
    if payload.model_mode not in ("yolo", "segformer", "ensemble"):
        raise HTTPException(422, "Неверный режим модели")
    if payload.single_size not in (512, 768, 1024) or payload.tile_size not in (384, 512, 768, 1024):
        raise HTTPException(422, "Недопустимый размер инференса или тайла")
    if payload.overlap < 0 or payload.overlap >= payload.tile_size:
        raise HTTPException(422, "Перекрытие должно быть меньше размера тайла")

    req = uuid.uuid4().hex[:8]
    try:
        if not payload.image_b64:
            raise HTTPException(400, "Браузер не передал снимок карты (image_b64).")
        image, tiles, fetch_ms = decode_browser_image(payload.image_b64, payload.width, payload.height, req)
        loop = asyncio.get_running_loop()
        image_png, infer_ms, total_ms = await loop.run_in_executor(EXECUTOR, segment_image, image, payload, req)
        return Response(
            image_png,
            media_type="image/png",
            headers={
                "X-Request-ID": req,
                "X-Tile-Count": str(tiles),
                "X-Fetch-MS": str(fetch_ms),
                "X-Inference-MS": str(infer_ms),
                "X-Total-MS": str(total_ms),
                "Cache-Control": "no-store",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        add_log("server", "Ошибка сегментации", req=req, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        LOG.exception("Failed segmentation, request=%s", req)
        raise HTTPException(500, f"{type(exc).__name__}: {str(exc)[:200]}") from exc


if __name__ == "__main__":
    import uvicorn
    add_log("server", "Запуск сервера TesseraMap", min_zoom=MIN_SCAN_ZOOM)
    uvicorn.run(app, host="127.0.0.1", port=7861, log_level="info")
