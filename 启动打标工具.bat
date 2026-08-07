@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   WD Tagger GUI - 图片智能打标工具
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 检查依赖
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

:: 启动 GUI
echo [启动] 正在启动 WD Tagger GUI...
python main.py

pause
