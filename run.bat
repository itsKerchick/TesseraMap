@echo off
chcp 65001 >nul
title TesseraMap Server
echo Starting TesseraMap Server on http://127.0.0.1:7861 ...
"..\SegFormer-Benchmark\.venv\Scripts\python.exe" server.py
pause
