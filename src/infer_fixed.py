"""
URL Page Type Classifier Inference - 修复版
使用Qwen2.5-1.5B + LoRA模型进行URL分类
从HuggingFace加载模型
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEFAULT_MODEL_PATH = 'windlx/url-classifier-model'

def load_model(model_path=DEFAULT_MODEL_PATH, use_cpu=False):
    """加载模型 - 修复版"""
    print(f"Loading model from HuggingFace: {model_path}...")
    
    # 使用更兼容的方式加载
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=False  # 使用慢速 tokenizer 更稳定
    )
    
    # 设置 device_map
    device = 'cpu' if use_cpu else 'auto'
    
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
    """分类URL - 修复版"""
    prompt = f'''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
请判断以下URL是列表页还是详情页。

URL: {url}<|im_end|>
<|im_start|>assistant
'''
    
    # 正确 tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # 确保在正确的设备上
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # 修复: 使用 embed_tokens 的方式避免 cache 问题
    with torch.no_grad():
        # 直接获取 embedding
        inputs_embeds = model.model.embed_tokens(input_ids)
        
        # 手动运行 forward 获取隐藏状态
        hidden_states = inputs_embeds
        
        # 通过所有 transformer 层
        for layer in model.model.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                use_cache=False
            )
            hidden_states = layer_output[0]
        
        # 最终的 layernorm
        hidden_states = model.model.norm(hidden_states)
        
        # 投影到 vocab
        logits = model.lm_head(hidden_states)
        
        # 获取最后一个 token 的 logits
        next_token_logits = logits[:, -1, :]
        
        # 贪心采样
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # 判断
        token_str = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        
    # 提取答案
    if 'Detail Page' in token_str or '详情页' in token_str:
        result = 'Detail Page (详情页)'
    elif 'List Page' in token_str or '列表页' in token_str:
        result = 'List Page (列表页)'
    else:
        # 尝试直接生成
        result = generate_with_fallback(url, model, tokenizer)
    
    return result

def generate_with_fallback(url, model, tokenizer):
    """备用生成方法"""
    prompt = f'''请判断以下URL是列表页还是详情页。

URL: {url}
类型: '''
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids.to(next(model.parameters()).device)
        
        # 简单生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=None,
            )
        
        # 只取新生成的 token
        new_tokens = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if 'Detail' in response or '详情' in response:
            return 'Detail Page (详情页)'
        elif 'List' in response or '列表' in response:
            return 'List Page (列表页)'
        else:
            return f'Unknown - {response[:30]}'
            
    except Exception as e:
        return f'Error: {str(e)[:50]}'

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
