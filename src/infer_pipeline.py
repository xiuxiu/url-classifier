"""
URL Page Type Classifier Inference - 使用Pipeline
"""

import argparse
from transformers import pipeline

DEFAULT_MODEL_PATH = 'windlx/url-classifier-model'

def classify_url(url, classifier):
    """分类URL"""
    prompt = f'''请判断以下URL是列表页还是详情页。

URL: {url}
类型: '''
    
    result = classifier(prompt, max_new_tokens=20)
    
    # 提取答案
    response = result[0]['generated_text']
    
    if 'Detail Page' in response or '详情页' in response:
        return 'Detail Page (详情页)'
    elif 'List Page' in response or '列表页' in response:
        return 'List Page (列表页)'
    else:
        return f'Unknown - {response[-100:]}'

def main():
    parser = argparse.ArgumentParser(description='URL Page Type Classifier')
    parser.add_argument('url', help='URL to classify')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for inference')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='HuggingFace model path')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}...")
    device = -1 if args.cpu else 0
    classifier = pipeline(
        "text-generation",
        model=args.model,
        device=device,
        trust_remote_code=True,
    )
    print("Model loaded!")
    
    result = classify_url(args.url, classifier)
    
    print(f"\nURL: {args.url}")
    print(f"Type: {result}")

if __name__ == '__main__':
    main()
