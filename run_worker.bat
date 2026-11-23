@echo off
chcp 65001 >nul
title TesseraMap - Home PC GPU ML Worker
echo ========================================================
echo   Запуск домашнего ML-воркера TesseraMap (RTX 4060 Ti)
echo   Подключение к VPS через исходящий WebSocket туннель...
echo ========================================================
:: Чтобы подключиться к удаленному VPS, укажите его адрес:
:: set VPS_URL=wss://your-vps-domain.com/ws/worker
:: или
:: set VPS_URL=ws://YOUR_VPS_IP:7861/ws/worker
set VPS_URL=ws://127.0.0.1:7861/ws/worker
set WORKER_SECRET=tessera-secret-2026

"..\SegFormer-Benchmark\.venv\Scripts\python.exe" worker.py --vps "%VPS_URL%" --token "%WORKER_SECRET%"
pause
