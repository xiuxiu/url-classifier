"""
URL Page Type Classifier Inference
使用Qwen2.5-1.5B + LoRA模型进行URL分类
修复版：与训练时的 prompt 格式一致
"""

import sys
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 默认配置 - 从HuggingFace加载
DEFAULT_MODEL_PATH = 'windlx/url-classifier-model'

def load_model(model_path=DEFAULT_MODEL_PATH, use_cpu=False):
    """加载模型 - 直接从HuggingFace获取"""
    print(f"Loading model from HuggingFace: {model_path}...")
    
    device = 'cpu' if use_cpu else 'auto'
    dtype = torch.float32 if use_cpu else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    print("Model loaded!")
    return model, tokenizer

def classify_url(url, model, tokenizer):
    """分类URL - 使用与训练时一致的 prompt 格式"""
    # 关键：使用与训练时一致的 prompt 格式
    prompt = f'''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
请判断以下URL是列表页还是详情页。

URL: {url}<|im_end|>
<|im_start|>assistant
'''
    
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device if hasattr(model, 'device') else 'cpu')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 只取新生成的 token
    input_len = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # 提取答案
    if 'Detail Page' in response or '详情页' in response:
        result = 'Detail Page (详情页)'
    elif 'List Page' in response or '列表页' in response:
        result = 'List Page (列表页)'
    else:
        result = f'Unknown (未知) - {response[:50]}'
    
    return result

def main():
    parser = argparse.ArgumentParser(description='URL Page Type Classifier')
    parser.add_argument('url', help='URL to classify')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for inference')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='HuggingFace model path')
    
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model, args.cpu)
    result = classify_url(args.url, model, tokenizer)
    
    print(f"\nURL: {args.url}")
    print(f"Type: {result}")

if __name__ == '__main__':
    main()
