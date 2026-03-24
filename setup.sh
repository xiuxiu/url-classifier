#!/bin/bash
# URL Classifier — 环境安装脚本
# 用法: bash setup.sh

set -e

echo "=== URL Classifier 环境安装 ==="

# 检测 conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda 未安装。请先安装 Miniconda 或 Anaconda."
    exit 1
fi

ENV_NAME="url-classifier"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 '${ENV_NAME}' 已存在，跳过创建。"
else
    echo "创建 conda 环境: ${ENV_NAME}..."
    conda env create -f environment.yml
fi

echo ""
echo "=== 安装完成 ==="
echo ""
echo "激活环境:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "运行推理:"
echo "  python src/infer.py \"https://example.com/product/123\""
echo ""
echo "运行训练:"
echo "  python src/train.py"
