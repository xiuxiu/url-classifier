"""
URL Page Type Classifier Inference - 修复版v2
使用Qwen2.5-1.5B + LoRA模型进行URL分类
从HuggingFace加载模型
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEFAULT_MODEL_PATH = 'windlx/url-classifier-model'

def load_model(model_path=DEFAULT_MODEL_PATH, use_cpu=False):
    """加载模型"""
    print(f"Loading model from HuggingFace: {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # 关键修复: 不使用 device_map，直接指定 device
    device = 'cpu' if use_cpu else 'cuda:0'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if use_cpu else torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    
    model.eval()
    print("Model loaded!")
    return model, tokenizer

def classify_url(url, model, tokenizer):
    """分类URL"""
    prompt = f'''请判断以下URL是列表页还是详情页。

URL: {url}
类型: '''
    
    # 关键修复: 禁用所有可能的 cache 机制
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    with torch.no_grad():
        # 禁用 cache，使用 static 模式
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
            use_cache=False,  # 关键修复
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 只取新生成的 token
    input_len = input_ids.shape[1]
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # 提取答案
    if 'Detail Page' in response or '详情页' in response:
        result = 'Detail Page (详情页)'
    elif 'List Page' in response or '列表页' in response:
        result = 'List Page (列表页)'
    else:
        result = f'Unknown - {response.strip()[:50]}'
    
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
