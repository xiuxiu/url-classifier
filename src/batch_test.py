"""
批量测试 URL 分类器
"""

import json
from infer import load_model, classify_url

# 生成测试 URLs
test_urls = [
    # 列表页
    "https://example.com/products",
    "https://example.com/category/electronics",
    "https://example.com/search?q=test",
    "https://example.com/list/all",
    "https://shop.com/items",
    "https://store.com/gallery",
    "https://blog.io/posts",
    "https://news.net/latest",
    # 详情页
    "https://example.com/product/12345",
    "https://example.com/item/abc123",
    "https://shop.com/detail/67890",
    "https://store.com/item/xyz789",
    "https://blog.io/post/123",
    "https://news.net/article/456",
]

def main():
    print("Loading model...")
    model, tokenizer = load_model(use_cpu=True)
    
    results = []
    correct = 0
    total = len(test_urls)
    
    for url in test_urls:
        result = classify_url(url, model, tokenizer)
        results.append({"url": url, "result": result})
        print(f"{url} -> {result}")
    
    print(f"\n=== 测试完成 ===")
    print(f"总计: {total} 条")

if __name__ == "__main__":
    main()
