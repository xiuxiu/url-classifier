# URL Classifier — Autoresearch

Binary classifier that predicts whether a URL is a **list page (A)** or a **detail page (B)**.

Built with the **Autoresearch** framework — a custom transformer trained from scratch, not a LoRA fine-tune.

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.88% |
| Dataset | iowacat (from url-classifier project) |
| Training time | 30 min (RTX 4060 Laptop) |
| Final loss | ~0.002 |
| Model params | ~161M |

## Quick Start

### 1. Install environment

**macOS / Linux:**
```bash
bash setup.sh
```

**Windows:**
```
双击运行 setup.bat
```

Or manually:
```bash
conda env create -f environment.yml
conda activate url-classifier
```

### 2. Run inference

```bash
python src/infer.py "https://example.com/product/12345"   # detail page
python src/infer.py "https://example.com/search?q=foo"   # list page
```

### 3. Train (optional)

```bash
python src/train.py
```

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
| Checkpoint | ~413 MB |

## Project Structure

```
url-classifier/
├── src/
│   ├── prepare.py       # Data loading + tiktoken tokenizer
│   ├── train.py         # Training script (autoresearch framework)
│   └── infer.py         # Inference script
├── configs/
│   └── model_config.json   # Model hyperparameters
├── environment.yml       # Conda environment spec
├── setup.sh             # Linux/macOS install script
├── setup.bat            # Windows install script
├── model_card.md        # HuggingFace model card
└── README.md
```

## Dataset

Trained on `iowacat` from the [url-classifier](https://github.com/xiuxiu/url-classifier) project — labeled URL pairs (A=列表页, B=详情页).

## Citation

```bibtex
@software{url_classifier_autoresearch,
  title = {URL Classifier — Autoresearch},
  author = {xiuxiu},
  url = {https://huggingface.co/xiuxiu/url-classifier}
}
```
