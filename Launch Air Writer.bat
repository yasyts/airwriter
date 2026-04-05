@echo off
cd /d "%~dp0"
python desktop_app.py
if errorlevel 1 (
  echo.
  echo Air Writer closed with an error.
  pause
)
