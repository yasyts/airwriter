@echo off
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
  echo Python 3 is required to install Air Writer.
  echo Install Python from https://www.python.org/downloads/ and run this file again.
  pause
  exit /b 1
)

echo Installing Air Writer dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
  echo.
  echo Installation failed. Check the messages above and try again.
  pause
  exit /b 1
)

echo.
echo Air Writer is ready.
echo Use "Launch Air Writer.bat" for the normal app.
echo Use "Launch Air Writer Virtual Camera.bat" for Zoom, Meet, Teams, and similar apps.
pause
