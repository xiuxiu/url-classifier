"""
批量测试 URL 分类器 - 使用本地 LoRA 模型
"""

import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def load_local_model():
    """加载本地模型"""
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B',
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', trust_remote_code=True)
    
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, 'C:/Users/windlx/.openclaw/workspace/qwen_output/checkpoint-300')
    model.eval()
    
    print("Model loaded!")
    return model, tokenizer

def classify_url(url, model, tokenizer):
    """分类URL"""
    prompt = f'''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
请判断以下URL是列表页还是详情页。

URL: {url}<|im_end|>
<|im_start|>assistant
'''
    
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取答案
    if 'Detail Page' in response or '详情页' in response:
        result = 'Detail Page'
    elif 'List Page' in response or '列表页' in response:
        result = 'List Page'
    else:
        result = 'Unknown'
    
    return result

# 随机生成100个测试URL
def generate_test_urls():
    """生成随机测试URL"""
    list_patterns = [
        "https://{domain}/products",
        "https://{domain}/category/{cat}",
        "https://{domain}/search?q={query}",
        "https://{domain}/list/all",
        "https://{domain}/items",
        "https://{domain}/gallery",
        "https://{domain}/posts",
        "https://{domain}/tags/{tag}",
        "https://{domain}/blog",
        "https://{domain}/news",
    ]
    detail_patterns = [
        "https://{domain}/product/{id}",
        "https://{domain}/item/{id}",
        "https://{domain}/detail/{id}",
        "https://{domain}/article/{id}",
        "https://{domain}/post/{id}",
        "https://{domain}/news/{id}",
        "https://{domain}/page/{id}",
        "https://{domain}/info/{id}",
    ]
    
    domains = ['example.com', 'shop.com', 'store.io', 'blog.net', 'news.org', 'test.cn', 'demo.io']
    categories = ['electronics', 'books', 'clothing', 'food', 'toys', 'sports', 'music']
    tags = ['popular', 'new', 'sale', 'trending', 'hot', 'featured']
    queries = ['test', 'hello', 'search', 'find', 'buy', 'shop']
    ids = ['12345', 'abc123', 'xyz789', '67890', 'item001', 'prod-001', 'news-001']
    
    urls = []
    
    # 50个列表页
    for _ in range(50):
        pattern = random.choice(list_patterns)
        url = pattern.format(
            domain=random.choice(domains),
            cat=random.choice(categories),
            tag=random.choice(tags),
            query=random.choice(queries)
        )
        urls.append(('List Page', url))
    
    # 50个详情页
    for _ in range(50):
        pattern = random.choice(detail_patterns)
        url = pattern.format(
            domain=random.choice(domains),
            id=random.choice(ids)
        )
        urls.append(('Detail Page', url))
    
    # 随机打乱
    random.shuffle(urls)
    return urls

def main():
    print("Loading model...")
    model, tokenizer = load_local_model()
    
    print("Generating 100 test URLs...")
    test_cases = generate_test_urls()
    
    print("Testing...")
    results = []
    correct = 0
    
    for i, (expected, url) in enumerate(test_cases):
        result = classify_url(url, model, tokenizer)
        is_correct = (expected == result)
        if is_correct:
            correct += 1
        results.append({
            'url': url,
            'expected': expected,
            'result': result,
            'correct': is_correct
        })
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/100, Accuracy: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")
    
    print(f"\n=== 测试完成 ===")
    print(f"准确率: {correct}/100 = {correct}%")
    print()
    print("前20个结果:")
    for r in results[:20]:
        status = "✓" if r['correct'] else "✗"
        print(f"{status} | 预期: {r['expected']:10} | 预测: {r['result']:10} | {r['url']}")

if __name__ == "__main__":
    main()
