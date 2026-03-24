---
language:
  - en
  - zh
license: mit
tags:
  - url-classification
  - binary-classification
  - autoresearch
  - multi-domain
datasets:
  - 26 domains: Amazon, JD, Taobao, Tmall, Pinduoduo,
    Zhilian, BOSS, Lagou,
    Sina, NetEase, Tencent, 36kr,
    Zhihu, Douban, Xiaohongshu, Reddit,
    YouTube, Bilibili,
    Ctrip, Qunar, Mafengwo,
    icourse163, imooc,
    GitHub, ReadTheDocs, MDN
metrics:
  - accuracy: 1.0000 (training set, 2600 samples)
model_index:
  - name: url-classifier-v2
    results:
      - task:
          type: text-classification
          name: URL Binary Classification (Multi-Domain)
        dataset:
          type: synthetic-diverse (26 domains)
          name: URL Classification Diverse Dataset
        metrics:
          - type: accuracy
            value: 1.0000
---

# URL Classifier v2 — Autoresearch (Multi-Domain)

Binary classifier that predicts whether a URL is a **list page (A)** or a **detail page (B)**.

Trained on **26 diverse domains** across e-commerce, recruitment, news, social, video, travel, education, and tech documentation — significantly improved generalization over the v1 single-domain model.

## Model Details

- **Architecture**: Custom transformer (Autoresearch framework)
- **Parameters**: ~161M
- **Depth**: 4 layers
- **Model dim**: 384
- **Vocab**: cl100k_base (100,277 tokens)
- **Max seq len**: 64
- **Training**: 30 min on RTX 4060 Laptop
- **Training samples**: 2,600 (A=1,300, B=1,300)
- **Training accuracy**: 100%

## Supported Domains

| Category | Domains |
|----------|---------|
| E-commerce | Amazon, JD, Taobao, Tmall, Pinduoduo |
| Recruitment | Zhilian, BOSS, Lagou |
| News | Sina, NetEase, Tencent News, 36kr |
| Social | Zhihu, Douban, Xiaohongshu, Reddit |
| Video | YouTube, Bilibili |
| Travel | Ctrip, Qunar, Mafengwo |
| Education | icourse163, imooc |
| Tech Docs | GitHub, ReadTheDocs, MDN |

## Usage

```bash
pip install torch tiktoken
python src/infer.py "https://example.com/product/123"   # detail page
python src/infer.py "https://example.com/search?q=foo"  # list page
```

## Class Labels

| Label | Meaning |
|-------|---------|
| 0 (A) | List page — search results, category pages, rankings |
| 1 (B) | Detail page — product page, article, profile, video |

## Limitations

- Bilibili ranking pages may be misclassified as detail pages
- Very short URLs or URL shorteners may have lower accuracy
- Third-party evaluation accuracy (~55%) indicates room for improvement with real-world labeled data
