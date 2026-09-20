@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: start_dmn.cmd INSTANCE_DIRECTORY [additional run arguments]
  echo Launch this from a separate Command Prompt or Windows Terminal window.
  exit /b 2
)
set "DMN_PYTHON=%~dp0.venv-gpu\Scripts\python.exe"
if not exist "%DMN_PYTHON%" set "DMN_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%DMN_PYTHON%" (
  echo Install the DMN Python environment first. See README.md.
  exit /b 2
)
rem No restart loop. Strict native restoration and durable holds are enforced.
rem Pass the instance via --instance in the argument list below.
"%DMN_PYTHON%" -m dmn run --instance %*
set "DMN_EXIT=%ERRORLEVEL%"
echo DMN exited with code %DMN_EXIT%.
exit /b %DMN_EXIT%
