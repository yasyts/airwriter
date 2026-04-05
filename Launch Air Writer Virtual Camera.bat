@echo off
cd /d "%~dp0"
python desktop_app.py --virtual-camera --windowed
if errorlevel 1 (
  echo.
  echo Virtual camera mode could not start.
  echo Install OBS Studio or Unity Capture, then try again.
  pause
)
