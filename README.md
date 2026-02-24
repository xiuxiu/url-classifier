# URL Page Type Classifier

基于 Qwen2.5-1.5B + LoRA 的URL类型分类模型，用于判断URL是列表页还是详情页。

## 功能

- 判断URL是列表页 (List Page) 还是详情页 (Detail Page)
- 支持GPU和CPU推理
- 使用LoRA微调，模型轻量

## 环境要求

- Python 3.10+
- PyTorch 2.5+
- Transformers
- PEFT
- 建议: NVIDIA GPU (RTX 4060+)

## 安装

```bash
# 创建conda环境
conda create -n url-classifier python=3.11
conda activate url-classifier

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets accelerate
```

## 训练

### 1. 准备数据

从 HuggingFace 下载数据集:

```python
from datasets import load_dataset
import json
import random

random.seed(42)

ds = load_dataset('IowaCat/page_type_inference_dataset')
train_data = ds['train']

# 采样
list_pages = [url for url, label in zip(train_data['url'], train_data['label']) if label == 0]
detail_pages = [url for url, label in zip(train_data['url'], train_data['label']) if label == 1]

sampled_list = random.sample(list_pages, 5000)
sampled_detail = random.sample(detail_pages, 5000)

# 转换为训练格式
data = []
for url in sampled_list:
    data.append({
        "instruction": "请判断以下URL是列表页还是详情页。",
        "input": url,
        "output": "列表页 (List Page)"
    })
for url in sampled_detail:
    data.append({
        "instruction": "请判断以下URL是列表页还是详情页。",
        "input": url,
        "output": "详情页 (Detail Page)"
    })

random.shuffle(data)

with open('data/data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

### 2. 训练模型

```bash
python src/train.py
```

训练配置:
- 模型: Qwen2.5-1.5B
- LoRA rank: 16
- Epochs: 3
- Batch size: 2
- Gradient accumulation: 8
- Learning rate: 2e-4

## 推理

### GPU推理

```bash
python src/infer.py "https://example.com/product/12345"
```

### CPU推理

```bash
python src/infer.py "https://example.com/products/list" --cpu
```

## 项目结构

```
url-classifier/
├── data/
│   └── data.json          # 训练数据
├── src/
│   ├── __init__.py
│   ├── train.py          # 训练脚本
│   ├── infer.py          # 推理脚本
│   └── utils.py          # 工具函数
├── output/                # 模型输出目录 (不提交)
├── .gitignore
├── README.md
└── requirements.txt
```

## 测试结果

| 测试集 | 准确率 |
|--------|--------|
| 100条验证集 | 99% |

## 模型下载

训练完成后，模型保存在 `output/checkpoint-300/` 目录。

### 🤗 HuggingFace 模型

模型已上传至 HuggingFace，可直接下载使用：

```bash
# 方法1: 使用 git
git lfs install
git clone https://huggingface.co/windlx/url-classifier-model

# 方法2: 使用 Python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("windlx/url-classifier-model")
```

**HuggingFace 模型地址**: https://huggingface.co/windlx/url-classifier-model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B')
model = PeftModel.from_pretrained(base_model, 'output/checkpoint-300')
model.merge_and_unload()
model.save_pretrained('output/merged-model')
```

## License

[MIT License](LICENSE)
