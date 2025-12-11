# CIFAR-10 Benchmark with timm Models

This directory contains a PyTorch implementation for training CIFAR-10 using models from the `timm` library.

## Prerequisites

Ensure you have the required packages installed:

```bash
pip install -r requirements.txt
```

## Supported Models

- `efficientnet` (EfficientNet-B0)
- `mobilenet` (MobileNetV3 Small)
- `wideresnet` (WideResNet-28-10)
- `vit` (ViT-Base Patch16 224)

## Usage

### Training

To train a model, run `train.py` with the `--model` argument:

```bash
# Train EfficientNet
python train.py --model efficientnet

# Train ViT (Vision Transformer)
python train.py --model vit --batch-size 64
```

### Arguments

- `--model`: Model architecture (required).
- `--epochs`: Number of epochs (default: 100).
- `--batch-size`: Batch size (default: 128).
- `--lr`: Learning rate (default: 0.1).
- `--no-cuda`: Disable CUDA usage.
- `--dry-run`: Run a single batch for debugging.

### Example

Run a quick test to verify everything works:

```bash
python train.py --model efficientnet --dry-run --no-cuda
```
