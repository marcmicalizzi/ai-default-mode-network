@echo off
cd /d "%~dp0"
if not defined DMN_OPENWEBUI_PYTHON (
  echo Set DMN_OPENWEBUI_PYTHON to the Python executable in your Open WebUI environment.
  pause
  exit /b 1
)
"%~dp0.venv-gpu\Scripts\python.exe" "%~dp0scripts\openwebui_sandbox.py" --webui-python "%DMN_OPENWEBUI_PYTHON%" --data data\qwen4b-webui --webui-port 3032 --dmn-port 8768 --runtime-config examples\qwen4b-trial.json --reuse
if errorlevel 1 pause
