# URL Classifier — Autoresearch

Binary classifier (A=列表页 / B=详情页) built with the **Autoresearch** framework on top of a custom transformer.

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.62% |
| Dataset | iowacat (from url-classifier) |
| Training time | 5 minutes (RTX 4060 Laptop) |
| Final loss | ~0.002 |
| Model params | ~161M |

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Depth | 4 |
| Model dim | 384 |
| Head dim | 128 |
| Num heads | 3 |
| Vocab | 100,277 (cl100k_base) |
| Max seq len | 64 |
| Window pattern | SSSL |

## Quick Start

### Training

```bash
conda activate qwenfinetune
cd url-classifier

# Run training (5-30 min)
python -m src.train
```

### Inference

```python
import torch
from src.prepare import Tokenizer

tokenizer = Tokenizer.from_directory()
# Load checkpoint
# (see configs/model_config.json for architecture details)
```

## Project Structure

```
url-classifier/
├── configs/
│   └── model_config.json    # Model hyperparameters
├── checkpoints/
│   └── checkpoint_pre_eval.pt  # Trained weights (not in git)
├── src/
│   ├── prepare.py           # Data loading + tiktoken tokenizer
│   └── train.py             # Training script (autoresearch framework)
├── model_card.md            # HuggingFace model card
└── README.md
```

## Dataset

Trained on `iowacat` from the [url-classifier](https://github.com/xiuxiu/url-classifier) project, which contains labeled URL pairs (A=列表页, B=详情页).

## Citation

```bibtex
@software{url_classifier_autoresearch,
  title = {URL Classifier — Autoresearch},
  author = {xiuxiu},
  url = {https://huggingface.co/xiuxiu/url-classifier}
}
```
