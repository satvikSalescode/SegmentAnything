@echo off
setlocal

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found - run setup_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

rem Same safe default as benchmark_windows.bat - see README.md to try the GPU.
set FORCE_DEVICE=cpu

echo Starting the app - once you see "Application startup complete", open
echo http://127.0.0.1:8010 in your browser.
echo.
echo The FIRST time you click "Add missed object", it will download a
echo segmentation model (~180 MB) from the internet - this only happens once.
echo.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
pause
