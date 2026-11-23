@echo off
title TesseraMap - Home PC GPU ML Worker
echo ========================================================
echo   TesseraMap ML Worker (NVIDIA RTX 4060 Ti)
echo   Connecting to VPS via WebSocket tunnel...
echo ========================================================

REM VPS WebSocket URL:
REM For local testing: ws://127.0.0.1:7861/ws/worker
REM For production VPS: wss://tessera.mrkerchick.ru/ws/worker
set "VPS_URL=wss://tessera.mrkerchick.ru/ws/worker"
set "WORKER_SECRET=tessera-supersecret2026"

"..\SegFormer-Benchmark\.venv\Scripts\python.exe" worker.py --vps "%VPS_URL%" --token "%WORKER_SECRET%"
pause
