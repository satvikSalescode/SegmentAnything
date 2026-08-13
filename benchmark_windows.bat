@echo off
setlocal

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found - run setup_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

rem Force CPU: this hardware's discrete GPU has very little VRAM and could
rem crash (out-of-memory) instead of gracefully falling back - CPU is the
rem safe default here. See README.md for how to try the GPU anyway.
set FORCE_DEVICE=cpu

echo Keep this laptop plugged in and awake until this finishes...
echo.
python backend\_bench_local.py
pause
