"""TesseraMap - VPS Web Server & Reverse Gateway
Hosted on VPS (Germany / Cloud) 24/7.
Serves the web client and proxies ML inference tasks to the Home PC worker over an outbound WebSocket tunnel.
No PyTorch/CUDA required on VPS!
"""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timedelta
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
HTML_FILE = ROOT / "index.html"
PDF_FILE = ROOT / "maptiler_guide.pdf"
FAVICON_ICO = ROOT / "favicon.ico"
FAVICON_PNG = ROOT / "favicon.png"
FAVICON_SVG = ROOT / "favicon.svg"

# Shared secret token for worker authentication
WORKER_SECRET = os.getenv("WORKER_SECRET", "tessera-secret-2026")
MIN_SCAN_ZOOM = 16

VPS_START_TIME = time.time()
ACTIVE_CLIENTS: dict[str, float] = {}
GEO_CACHE: dict[tuple, str] = {}

# Active home worker connection and pending request registry
ACTIVE_WORKER: WebSocket | None = None
ACTIVE_WORKER_INFO: dict = {}
PENDING_TASKS: dict[str, asyncio.Future] = {}
WORKER_LOCK = asyncio.Lock()

app = FastAPI(title="TesseraMap VPS Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG = logging.getLogger("tesseramap.vps")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ViewportRequest(BaseModel):
    width: int = Field(ge=256, le=2048)
    height: int = Field(ge=256, le=2048)
    zoom: int = Field(ge=1, le=22)
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


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "127.0.0.1"


def clean_active_clients():
    now = time.time()
    stale = [cid for cid, last_seen in ACTIVE_CLIENTS.items() if now - last_seen > 15]
    for cid in stale:
        ACTIVE_CLIENTS.pop(cid, None)


# -----------------------------------------------------------------------------
# Static Web Pages (Always available 24/7 on VPS even if Home PC is off)
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    if not HTML_FILE.exists():
        raise HTTPException(404, "index.html не найден на VPS")
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/maptiler_guide.pdf")
async def maptiler_guide():
    if not PDF_FILE.exists():
        raise HTTPException(404, "maptiler_guide.pdf не найден на VPS")
    return FileResponse(PDF_FILE, media_type="application/pdf", filename="maptiler_guide.pdf")


@app.get("/favicon.ico")
async def favicon_ico():
    if FAVICON_ICO.exists():
        return FileResponse(FAVICON_ICO, media_type="image/x-icon")
    raise HTTPException(404)


@app.get("/favicon.png")
async def favicon_png():
    if FAVICON_PNG.exists():
        return FileResponse(FAVICON_PNG, media_type="image/png")
    raise HTTPException(404)


@app.get("/favicon.svg")
async def favicon_svg():
    if FAVICON_SVG.exists():
        return FileResponse(FAVICON_SVG, media_type="image/svg+xml")
    raise HTTPException(404)


# -----------------------------------------------------------------------------
# WebSocket Tunnel for Home PC Worker (Reverse Connection bypasses NAT/Gray IP)
# -----------------------------------------------------------------------------

@app.websocket("/ws/worker")
async def worker_websocket_endpoint(websocket: WebSocket, token: str = ""):
    global ACTIVE_WORKER, ACTIVE_WORKER_INFO
    if token != WORKER_SECRET:
        LOG.warning("Worker connection rejected: invalid secret token")
        await websocket.close(code=1008, reason="Unauthorized")
        return

    await websocket.accept()
    worker_ip = websocket.client.host if websocket.client else "unknown"
    LOG.info(">>> Домашний ML-сервер успешно подключился с IP: %s", worker_ip)

    async with WORKER_LOCK:
        if ACTIVE_WORKER is not None:
            try:
                await ACTIVE_WORKER.close(code=1000, reason="Replaced by new worker")
            except Exception:
                pass
        ACTIVE_WORKER = websocket
        ACTIVE_WORKER_INFO = {
            "connected_at": time.time(),
            "worker_ip": worker_ip,
            "gpu_name": "Инициализация...",
            "gpu_load": 0,
            "gpu_memory": "-- / -- MB",
            "uptime": "00:00:00",
        }

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                msg = json.loads(raw_text)
            except Exception:
                continue

            msg_type = msg.get("type")
            if msg_type == "telemetry":
                # Update cached telemetry received from home PC
                ACTIVE_WORKER_INFO.update({
                    "gpu_name": msg.get("gpu_name", "NVIDIA GPU"),
                    "gpu_load": msg.get("gpu_load", 0),
                    "gpu_memory": msg.get("gpu_memory", "-- / -- MB"),
                    "uptime": msg.get("uptime", "--:--:--"),
                    "last_ping": time.time(),
                })
            elif msg_type == "segment_response":
                task_id = msg.get("task_id")
                if task_id in PENDING_TASKS:
                    fut = PENDING_TASKS.pop(task_id)
                    if not fut.done():
                        fut.set_result(msg)

    except WebSocketDisconnect:
        LOG.warning("<<< Домашний ML-сервер отключился (WebSocket closed)")
    except Exception as exc:
        LOG.error("Worker connection exception: %s", exc)
    finally:
        async with WORKER_LOCK:
            if ACTIVE_WORKER == websocket:
                ACTIVE_WORKER = None
                ACTIVE_WORKER_INFO.clear()
        # Cancel any pending requests with error
        for tid, fut in list(PENDING_TASKS.items()):
            if not fut.done():
                fut.set_exception(HTTPException(503, "ML воркер отключился во время обработки"))
        PENDING_TASKS.clear()


# -----------------------------------------------------------------------------
# Client REST API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/ping")
async def ping():
    return {
        "pong": True,
        "worker_online": ACTIVE_WORKER is not None,
        "time": time.time()
    }


@app.get("/api/status")
async def status(request: Request, client_id: str | None = None):
    return await system_stats(request, client_id)


@app.get("/api/system-stats")
async def system_stats(request: Request, client_id: str | None = None):
    ip = get_client_ip(request)
    now = time.time()
    if client_id:
        ACTIVE_CLIENTS[client_id] = now
    clean_active_clients()

    is_online = ACTIVE_WORKER is not None
    vps_uptime_sec = int(now - VPS_START_TIME)
    vps_uptime = str(timedelta(seconds=vps_uptime_sec))

    if is_online:
        return {
            "worker_online": True,
            "status": "ready",
            "gpu_name": ACTIVE_WORKER_INFO.get("gpu_name", "NVIDIA GPU"),
            "gpu_load": ACTIVE_WORKER_INFO.get("gpu_load", 0),
            "gpu_memory": ACTIVE_WORKER_INFO.get("gpu_memory", "-- / -- MB"),
            "uptime": ACTIVE_WORKER_INFO.get("uptime", vps_uptime),
            "vps_uptime": vps_uptime,
            "users": max(1, len(ACTIVE_CLIENTS)),
            "ping": 0,
            "ip": ip,
            "min_scan_zoom": MIN_SCAN_ZOOM,
        }
    else:
        return {
            "worker_online": False,
            "status": "worker_offline",
            "gpu_name": "Сервер оффлайн",
            "gpu_load": 0,
            "gpu_memory": "0 / 0 MB",
            "uptime": "--:--:--",
            "vps_uptime": vps_uptime,
            "users": max(1, len(ACTIVE_CLIENTS)),
            "ping": 0,
            "ip": ip,
            "min_scan_zoom": MIN_SCAN_ZOOM,
        }


@app.post("/api/segment")
async def segment(payload: ViewportRequest):
    if ACTIVE_WORKER is None:
        raise HTTPException(
            503,
            "ML сервер недоступен: домашний ПК сейчас выключен или не подключен к VPS.",
        )
    if payload.zoom < MIN_SCAN_ZOOM:
        raise HTTPException(
            422,
            f"Сканирование доступно только при zoom >= {MIN_SCAN_ZOOM}; получен zoom {payload.zoom}.",
        )
    if not payload.image_b64:
        raise HTTPException(400, "Браузер не передал снимок карты (image_b64).")

    task_id = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    PENDING_TASKS[task_id] = fut

    outbound_msg = {
        "type": "segment_request",
        "task_id": task_id,
        "payload": payload.model_dump(),
    }

    try:
        await ACTIVE_WORKER.send_text(json.dumps(outbound_msg))
    except Exception as exc:
        PENDING_TASKS.pop(task_id, None)
        raise HTTPException(503, f"Не удалось передать задачу воркеру: {exc}")

    try:
        # Wait up to 60 seconds for worker to process tile/ensemble inference
        result = await asyncio.wait_for(fut, timeout=60.0)
    except asyncio.TimeoutError:
        PENDING_TASKS.pop(task_id, None)
        raise HTTPException(504, "Таймаут: домашний ML сервер не ответил за 60 секунд.")

    if not result.get("ok"):
        error_msg = result.get("error", "Ошибка инференса на ML воркере")
        raise HTTPException(500, error_msg)

    mask_b64 = result.get("mask_b64", "")
    png_bytes = base64.b64decode(mask_b64)
    infer_ms = result.get("infer_ms", 0)
    total_ms = result.get("total_ms", 0)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Infer-Time-Ms": str(infer_ms),
            "X-Total-Time-Ms": str(total_ms),
            "X-Worker-Host": ACTIVE_WORKER_INFO.get("worker_ip", "home-pc"),
        },
    )


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


@app.get("/api/geocode")
async def geocode(lat: float, lon: float, zoom: int):
    precision = 2 if zoom <= 7 else (3 if zoom <= 13 else 4)
    cache_key = (round(lat, precision), round(lon, precision), min(zoom, 18))
    if cache_key in GEO_CACHE:
        return {"place": GEO_CACHE[cache_key]}

    osm_zoom = 3 if zoom < 6 else (10 if zoom < 14 else 16)
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&zoom={osm_zoom}&format=json&accept-language=ru"
    headers = {"User-Agent": "TesseraMap-SatelliteApp/2.0"}

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                addr = data.get("address", {})
                place = (
                    addr.get("city")
                    or addr.get("town")
                    or addr.get("village")
                    or addr.get("municipality")
                    or addr.get("state")
                    or addr.get("country")
                    or ""
                )
                if place:
                    GEO_CACHE[cache_key] = place
                    return {"place": place}
    except Exception:
        pass

    return {"place": ""}


@app.post("/api/client-log")
async def client_log(msg: ClientLog):
    LOG.info("[ClientLog %s] %s", msg.level, msg.text)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7861"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"=== TesseraMap VPS Gateway running on http://{host}:{port} ===")
    print(f"=== Waiting for Home PC ML Worker to connect to ws://{host}:{port}/ws/worker ===")
    uvicorn.run("vps_server:app", host=host, port=port, reload=False)
