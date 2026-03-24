@echo off
rem URL Classifier — 环境安装脚本 (Windows)
rem 双击运行，或在 cmd/PowerShell 中执行: setup.bat

echo === URL Classifier 环境安装 ===

where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: conda 未安装。请先安装 Miniconda 或 Anaconda.
    pause
    exit /b 1
)

set ENV_NAME=url-classifier

rem 检查环境是否存在
conda env list | findstr /C:"%ENV_NAME% " >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo 环境 '%ENV_NAME%' 已存在，跳过创建。
) else (
    echo 创建 conda 环境: %ENV_NAME% ...
    conda env create -f environment.yml
)

echo.
echo === 安装完成 ===
echo.
echo 激活环境:
echo   conda activate %ENV_NAME%
echo.
echo 运行推理:
echo   python src\infer.py "https://example.com/product/123"
echo.
echo 运行训练:
echo   python src\train.py
echo.
pause
