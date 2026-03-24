---
language:
  - en
  - zh
license: mit
tags:
  - url-classification
  - binary-classification
  - autoresearch
datasets:
  - iowacat (from url-classifier project)
metrics:
  - accuracy: 0.9962
model_index:
  - name: url-classifier
    results:
      - task:
          type: text-classification
          name: URL Binary Classification
        dataset:
          type: iowacat
          name: URL Classification Dataset
        metrics:
          - type: accuracy
            value: 0.9962
---

# URL Classifier — Autoresearch

Binary classifier that predicts whether a URL is a **list page (A)** or a **detail page (B)**.

## Model Details

- **Architecture**: Custom transformer (Autoresearch framework)
- **Parameters**: ~161M
- **Depth**: 4 layers
- **Model dim**: 384
- **Vocab**: cl100k_base (100,277 tokens)
- **Max seq len**: 64
- **Training time**: 5 minutes on RTX 4060 Laptop

## Training

Trained with the Autoresearch framework, which combines:
- **Muon** optimizer for attention/MLP layers
- **AdamW** for embeddings
- **Sliding window attention** (SSSL pattern)
- **Value embeddings** for alternating layers

Final loss: ~0.002 | Accuracy: **99.62%**

## Usage

```python
from src.prepare import Tokenizer

tokenizer = Tokenizer.from_directory()
# Encode a URL
ids = tokenizer.encode("https://example.com/product/123")
# Run through model + class_head for classification
```

## Class Labels

| Label | Meaning |
|-------|---------|
| 0 | A — List page |
| 1 | B — Detail page |
