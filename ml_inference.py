"""TesseraMap - Neural Network Inference Engine
Encapsulates YOLO26x-seg, SegFormer-B5, and Weighted Ensemble blending.
Supports local 'weights/' directory with automatic fallback to dev workspace.
"""
from __future__ import annotations

import io
import math
import os
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

CLASSES = [
    'bareland',
    'rangeland',
    'developed_space',
    'road',
    'tree',
    'water',
    'agriculture_land',
    'building',
]
N = len(CLASSES)
IGNORE = 255
COLORS = np.array([
    [176, 124, 76],   # bareland
    [88, 190, 148],   # rangeland
    [25, 190, 200],   # developed_space
    [130, 130, 130],  # road
    [32, 65, 100],    # tree
    [200, 70, 150],   # water
    [240, 145, 60],   # agriculture_land
    [240, 235, 40],   # building
], dtype=np.uint8)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DEVICE_STR = 'cuda' if DEVICE.type == 'cuda' else 'cpu'

_YOLO_MODEL: YOLO | None = None
_SEGFORMER_MODEL: SegformerForSemanticSegmentation | None = None


def resolve_yolo_weights() -> Path:
    local = ROOT / "weights" / "yolo" / "best.pt"
    if local.exists():
        return local
    fallback = ROOT.parent / "YOLO26-Benchmark" / "results" / "openearthmap-x-v1" / "weights" / "best.pt"
    if fallback.exists():
        return fallback
    try:
        import download_weights
        download_weights.ensure_weights()
        if local.exists():
            return local
    except Exception as exc:
        print(f"Ошибка при автоматической загрузке весов: {exc}")
    raise FileNotFoundError(
        f"Не найдены веса YOLO: {local}\n"
        "Запустите 'python download_weights.py' для загрузки весов."
    )


def resolve_segformer_dir() -> Path:
    local = ROOT / "weights" / "segformer"
    if (local / "model.safetensors").exists() or (local / "pytorch_model.bin").exists():
        return local
    fallback = ROOT.parent / "SegFormer-Benchmark" / "results" / "segformer-b5-v1" / "best"
    if (fallback / "model.safetensors").exists() or (fallback / "pytorch_model.bin").exists():
        return fallback
    try:
        import download_weights
        download_weights.ensure_weights()
        if (local / "model.safetensors").exists() or (local / "pytorch_model.bin").exists():
            return local
    except Exception as exc:
        print(f"Ошибка при автоматической загрузке весов: {exc}")
    raise FileNotFoundError(
        f"Не найдены веса SegFormer: {local}\n"
        "Запустите 'python download_weights.py' для загрузки весов."
    )


def load_models():
    global _YOLO_MODEL, _SEGFORMER_MODEL

    # Ensure weights are present (downloads from Dropbox if missing)
    local_yolo = ROOT / "weights" / "yolo" / "best.pt"
    fallback_yolo = ROOT.parent / "YOLO26-Benchmark" / "results" / "openearthmap-x-v1" / "weights" / "best.pt"
    if not local_yolo.exists() and not fallback_yolo.exists():
        try:
            import download_weights
            download_weights.ensure_weights()
        except Exception as exc:
            print(f"Автоматическая загрузка моделей не удалась: {exc}")

    if _YOLO_MODEL is None:
        yolo_path = resolve_yolo_weights()
        print(f"Загрузка YOLO26x ({yolo_path.name})...")
        _YOLO_MODEL = YOLO(str(yolo_path))

    if _SEGFORMER_MODEL is None:
        seg_dir = resolve_segformer_dir()
        print(f"Загрузка SegFormer-B5 ({seg_dir.name})...")
        _SEGFORMER_MODEL = SegformerForSemanticSegmentation.from_pretrained(str(seg_dir)).to(DEVICE)
        _SEGFORMER_MODEL.eval()

    return _YOLO_MODEL, _SEGFORMER_MODEL


def square_canvas_from_image(image: Image.Image, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = image.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas.paste(resized, (0, 0))
    return canvas, (0, 0, nw, nh)


def crop_probs_to_original(probs_square: torch.Tensor, box: tuple[int, int, int, int], orig_size: tuple[int, int]) -> torch.Tensor:
    left, top, nw, nh = box
    cropped = probs_square[:, :, top:top + nh, left:left + nw]
    return F.interpolate(cropped, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)


def yolo_probs(image: Image.Image, size: int) -> tuple[torch.Tensor, np.ndarray]:
    load_models()
    canvas, box = square_canvas_from_image(image, size)
    source = np.asarray(canvas, dtype=np.uint8).copy()
    kwargs = dict(source=source, imgsz=int(size), device=0 if DEVICE.type == 'cuda' else 'cpu', verbose=False)

    if _YOLO_MODEL.predictor is None:
        with torch.inference_mode():
            _YOLO_MODEL.predict(**kwargs)

    backend = _YOLO_MODEL.predictor.model
    inference_net = getattr(backend, 'model', backend)
    heads = [m for m in inference_net.modules() if m.__class__.__name__ == 'SemanticSegment']
    if not heads:
        raise RuntimeError('Не найдена SemanticSegment в модели YOLO predictor.')

    captured = []

    def capture(_module, _inputs, output):
        candidates = []
        if isinstance(output, torch.Tensor):
            candidates = [output]
        elif isinstance(output, (tuple, list)):
            candidates = [v for v in output if isinstance(v, torch.Tensor)]
        candidates = [v for v in candidates if v.ndim == 4 and v.shape[1] == N]
        if candidates:
            captured.append(candidates[0].detach().float())

    handles = [head.register_forward_hook(capture) for head in heads]
    try:
        with torch.inference_mode():
            _YOLO_MODEL.predict(**kwargs)
    finally:
        for handle in handles:
            handle.remove()

    if not captured:
        raise RuntimeError('Не удалось получить тензор [1,8,H,W] из головы YOLO.')

    raw = captured[-1]
    logits = F.interpolate(raw, size=(size, size), mode='bilinear', align_corners=False)
    raw_min, raw_max = float(raw.min().item()), float(raw.max().item())
    sum_error = float((raw.sum(dim=1) - 1).abs().mean().item())
    if raw_min >= -1e-5 and raw_max <= 1.00001 and sum_error < 0.01:
        probs_square = logits.clamp_min(0)
        probs_square = probs_square / probs_square.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        probs_square = torch.softmax(logits, dim=1)

    probs = crop_probs_to_original(probs_square, box, image.size)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mask = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return probs.cpu(), mask


def segformer_probs(image: Image.Image, size: int) -> tuple[torch.Tensor, np.ndarray]:
    load_models()
    canvas, box = square_canvas_from_image(image, size)
    arr = np.asarray(canvas, dtype=np.uint8).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    tensor = ((tensor - MEAN) / STD).to(DEVICE)
    with torch.inference_mode(), torch.autocast(device_type=TORCH_DEVICE_STR, enabled=(DEVICE.type == 'cuda')):
        logits = _SEGFORMER_MODEL(pixel_values=tensor).logits.float()
    logits_square = F.interpolate(logits, size=(size, size), mode='bilinear', align_corners=False)
    probs_square = torch.softmax(logits_square, dim=1)
    probs = crop_probs_to_original(probs_square, box, image.size)
    mask = probs.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return probs.detach().float().cpu(), mask


def make_positions(length: int, tile: int, overlap: int) -> List[int]:
    if tile >= length:
        return [0]
    stride = max(1, tile - overlap)
    positions = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def blend_window(h: int, w: int, feather: int) -> np.ndarray:
    if feather <= 0:
        return np.ones((h, w), dtype=np.float32)
    y = np.minimum(np.arange(h) + 1, np.arange(h, 0, -1)).astype(np.float32)
    x = np.minimum(np.arange(w) + 1, np.arange(w, 0, -1)).astype(np.float32)
    wy = np.clip(y / feather, 0.0, 1.0)
    wx = np.clip(x / feather, 0.0, 1.0)
    return np.outer(wy, wx).astype(np.float32)


def color_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(COLORS):
        out[mask == i] = c
    out[mask == IGNORE] = (70, 70, 70)
    return out


def overlay(image: Image.Image, mask: np.ndarray | None, alpha: float, selected: str = 'Все классы') -> Image.Image | None:
    if mask is None:
        return None
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    if mask.shape != rgb.shape[:2]:
        mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
    if selected == 'Все классы':
        active = mask != IGNORE
    else:
        active = mask == CLASSES.index(selected)
    tint = color_mask(mask)
    out = rgb.copy()
    if np.any(active):
        out[active] = np.clip(rgb[active] * (1.0 - alpha) + tint[active] * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
