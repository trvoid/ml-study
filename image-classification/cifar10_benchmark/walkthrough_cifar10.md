# CIFAR-10 Benchmark Walkthrough

This guide explains how to use the CIFAR-10 training scripts located in `image-classification/cifar10_benchmark`.

## Files

- `models.py`: Contains model definitions (EfficientNet, MobileNetV3, ViT, WideResNet-28-10).
- `train.py`: Main training script with data loading and training loop.
- `evaluate.py`: Script to evaluate models or checkpoints.
- `requirements.txt`: List of dependencies.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Running the Training

You can train any of the supported models using the `train.py` script.

### 1. EfficientNet
```bash
python train.py --model efficientnet --epochs 100
```

### 2. MobileNetV3
```bash
python train.py --model mobilenet --epochs 100
```

### 3. WideResNet-28-10
```bash
python train.py --model wideresnet --epochs 200
```

### 4. Vision Transformer (ViT)
Note: ViT uses 224x224 images, so training might be slower.
```bash
python train.py --model vit --epochs 100
```

### Options

- `--batch-size`: Set batch size (default: 128).
- `--lr`: Set initial learning rate (default: 0.1).
- `--no-cuda`: Force CPU training.
- `--data-dir`: Directory to store dataset (default: `./data`).
- `--log-file`: Path to save logs (default: `train.log`).
- **Plots**: Training history (loss/accuracy) will be saved as `[model]_history.png` in the checkpoints directory.

## Evaluation

You can evaluate a trained model using `evaluate.py`.

```bash
python evaluate.py --model efficientnet --checkpoint ./checkpoints/efficientnet_best.pth
```

If no checkpoint is provided, it will evaluate the untrained model (random weights).

### Options
- `--log-file`: Path to save logs (default: `evaluate.log`).
- **Plots**: A confusion matrix will be saved as `[model]_confusion_matrix.png` (next to checkpoint or in current dir).

## Results
Checkpoints will be saved in the `./checkpoints` directory.
