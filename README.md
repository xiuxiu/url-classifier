# URL Classifier — CPU-Optimized

Binary classifier: predicts whether a URL is a **list page** or a **detail page**.

Built with **TF-IDF (char 3-6 gram) + LogisticRegression** — CPU-first, no GPU required, ~3,200 URLs/second on a single core.

## Results

| Metric | Value |
|--------|-------|
| **Test accuracy** | 97.2% |
| **CV accuracy** | 97.6% ± 0.5% |
| **Manual test** | 22/22 = 100% |
| **Inference speed** | 3,200 URLs/sec (1 core) |
| **Model size** | ~50 MB |

## Architecture

```
URL → TfidfVectorizer (char 3-6 gram, 100k features)
    → LogisticRegression (multi-class)
    → list | detail

Rule Engine (backup):
  - 17 high-precision patterns (Amazon dp, arxiv abs, YouTube watch, etc.)
  - Segment-level matching (avoids false positives like "hotel" → "hot")
  - Numeric ID detection
```

## Quick Start

```python
from inference import UrlClassifier

clf = UrlClassifier("data/models/url_classifier.pkl")
label, conf = clf.classify("https://github.com/facebook/react")
# → ('detail', 0.80)
```

Or from command line (batch):

```python
# Single URL
python inference.py

# Batch
urls = ["https://github.com/facebook/react", "https://arxiv.org/abs/2301.00001"]
results = clf.classify_batch(urls)
```

## Training

```bash
python train_fasttext.py
```

Requires: `scikit-learn`, `numpy`

## Project Structure

```
url-classifier/
  train_fasttext.py      # Training pipeline
  inference.py           # Inference (ML + rule hybrid)
  data/
    models/
      url_classifier.pkl # Trained model (vectorizer + classifier)
    urls_enhanced.json   # Enhanced training data
    real_urls/           # Manually labeled real URLs
  src/                   # Old transformer approach (deprecated)
```

## Rule Coverage

These patterns are intercepted by the rule engine (zero model uncertainty):

| Pattern | Site | Page type |
|---------|------|-----------|
| `/dp/[ASIN]` | amazon.com | detail |
| `/Hotel_Review...-d{id}` | tripadvisor.com | detail |
| `/itm/{digit}` | ebay.com | detail |
| `/abs/{digit}`, `/pdf/{digit}` | arxiv.org | detail |
| `/watch?v=` | youtube.com | detail |
| `/comments/` | reddit.com | detail |
| `/project/{name}` | pypi.org | detail |
| `/{6+ digit ID}$` | * | detail |
| `/search`, `/browse`, `/category` | * | list |

## Background

Originally trained with a 161M-parameter transformer (Autoresearch framework, 30 min on RTX 4060).

Replaced with TF-IDF + LR for production CPU deployment: **300× faster**, **< 1% accuracy loss**, runs on any machine without GPU.

## Citation

```bibtex
@software{url_classifier_cpu,
  title = {URL Classifier — CPU-Optimized},
  author = {xiuxiu},
  url = {https://github.com/xiuxiu/url-classifier}
}
```
