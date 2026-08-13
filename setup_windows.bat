@echo off
setlocal

echo ============================================================
echo  SKU Annotation Tool - Windows local setup
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on this machine.
    echo Install Python 3.11 from https://www.python.org/downloads/
    echo IMPORTANT: on the install screen, check "Add python.exe to PATH".
    echo Then re-run this script.
    pause
    exit /b 1
)

echo Found:
python --version
echo.

if not exist ".venv" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
) else (
    echo .venv already exists, reusing it.
)

call .venv\Scripts\activate.bat

echo.
echo Installing CPU-only PyTorch (recommended default for this hardware -
echo see README.md "Trying this on a low-spec Windows laptop" section)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo [ERROR] PyTorch install failed - check your internet connection and retry.
    pause
    exit /b 1
)

echo.
echo Installing the remaining dependencies (this can take a few minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency install failed. If the error mentions pycocotools
    echo needing a compiler, install "Visual Studio Build Tools" (C++ workload)
    echo from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and retry.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete.
echo ============================================================
echo  Next steps:
echo    1. Run benchmark_windows.bat first - gives a quick timing
echo       readout so you know what to expect before using the app.
echo    2. Run run_windows.bat to start the app at http://127.0.0.1:8010
echo ============================================================
pause
