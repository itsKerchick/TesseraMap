@echo off
chcp 65001 >nul
title TesseraMap - VPS Gateway
echo Starting TesseraMap VPS Gateway on http://127.0.0.1:7861 ...
"..\SegFormer-Benchmark\.venv\Scripts\python.exe" vps_server.py
pause
