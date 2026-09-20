@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\stop_sandbox.py --data data\qwen4b-webui
