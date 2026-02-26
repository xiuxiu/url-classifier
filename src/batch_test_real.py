"""
批量测试 URL 分类器 - 使用真实训练数据
"""

from transformers import pipeline
import random

DEFAULT_MODEL_PATH = 'windlx/url-classifier-model'

def load_model():
    """加载模型"""
    print(f"Loading model from HuggingFace: {DEFAULT_MODEL_PATH}...")
    classifier = pipeline(
        "text-generation",
        model=DEFAULT_MODEL_PATH,
        device=-1,  # CPU
        trust_remote_code=True,
    )
    print("Model loaded!")
    return classifier

def classify_url(url, classifier):
    """分类URL - 使用与训练时一致的 prompt 格式"""
    prompt = f'''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
请判断以下URL是列表页还是详情页。

URL: {url}<|im_end|>
<|im_start|>assistant
'''
    
    result = classifier(prompt, max_new_tokens=20)
    response = result[0]['generated_text']
    
    # 提取答案
    if 'Detail Page' in response or '详情页' in response:
        return 'Detail Page'
    elif 'List Page' in response or '列表页' in response:
        return 'List Page'
    else:
        return 'Unknown'

# 加载真实训练数据
def load_real_data():
    """从url_train.txt加载真实数据"""
    list_pages = []
    detail_pages = []
    
    with open('C:/Users/windlx/.openclaw/workspace/url_train.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('__label__0 '):
                url = line.replace('__label__0 ', '')
                list_pages.append(url)
            elif line.startswith('__label__1 '):
                url = line.replace('__label__1 ', '')
                detail_pages.append(url)
    
    return list_pages, detail_pages

def main():
    print("Loading model...")
    classifier = load_model()
    
    print("Loading real training data...")
    list_pages, detail_pages = load_real_data()
    print(f"Total: {len(list_pages)} list pages, {len(detail_pages)} detail pages")
    
    # 随机采样100条（50列表页+50详情页）
    random.seed(42)
    test_list = random.sample(list_pages, 50)
    test_detail = random.sample(detail_pages, 50)
    
    test_cases = [(url, 'List Page') for url in test_list] + [(url, 'Detail Page') for url in test_detail]
    random.shuffle(test_cases)
    
    print("Testing...")
    results = []
    correct = 0
    
    for i, (url, expected) in enumerate(test_cases):
        result = classify_url(url, classifier)
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
        status = "OK" if r['correct'] else "FAIL"
        print(f"{status} | Expected: {r['expected']:10} | Predicted: {r['result']:10} | {r['url'][:60]}")

if __name__ == "__main__":
    main()
