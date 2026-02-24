"""
URL Page Type Classifier Inference
使用Qwen2.5-1.5B + LoRA模型进行URL分类
"""

import sys
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 默认配置
DEFAULT_MODEL_PATH = 'Qwen/Qwen2.5-1.5B'
DEFAULT_CHECKPOINT = 'output/checkpoint-300'

def load_model(model_path=DEFAULT_MODEL_PATH, checkpoint_path=DEFAULT_CHECKPOINT, use_cpu=False):
    """加载模型"""
    print("Loading model...")
    
    device = 'cpu' if use_cpu else 'auto'
    dtype = torch.float32 if use_cpu else torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    print("Model loaded!")
    return model, tokenizer

def classify_url(url, model, tokenizer):
    """分类URL"""
    prompt = f'''请判断以下URL是列表页还是详情页。

URL: {url}
类型: '''
    
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device if hasattr(model, 'device') else 'cpu')
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取答案
    if 'Detail Page' in response or '详情页' in response:
        result = 'Detail Page (详情页)'
    elif 'List Page' in response or '列表页' in response:
        result = 'List Page (列表页)'
    else:
        result = 'Unknown (未知)'
    
    return result

def main():
    parser = argparse.ArgumentParser(description='URL Page Type Classifier')
    parser.add_argument('url', help='URL to classify')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for inference')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Model checkpoint path')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Base model name or path')
    
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model, args.checkpoint, args.cpu)
    result = classify_url(args.url, model, tokenizer)
    
    print(f"\nURL: {args.url}")
    print(f"Type: {result}")

if __name__ == '__main__':
    main()
