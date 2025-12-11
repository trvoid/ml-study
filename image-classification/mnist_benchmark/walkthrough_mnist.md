# MNIST Benchmark Walkthrough

This guide explains how to use the MNIST training scripts located in `image-classification/mnist_benchmark`.

## Files

- `models.py`: Contains model definitions (SimpleCNN, MLP, ResNet18).
- `train.py`: Main training script.
- `evaluate.py`: Evaluation script.
- `requirements.txt`: List of dependencies.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Running the Training

Navigate to `image-classification/mnist_benchmark` and run:

### 1. SimpleCNN
```bash
python train.py --model simplecnn --epochs 10
```

### 2. MLP
```bash
python train.py --model mlp --epochs 10
```

### 3. ResNet18
```bash
python train.py --model resnet18 --epochs 10
```

### Options

- `--batch-size`: Set batch size (default: 64).
- `--lr`: Set initial learning rate (default: 0.01).
- `--no-cuda`: Force CPU training.
- `--data-dir`: Directory to store dataset (default: `./data`).
- `--log-file`: Path to save logs (default: `train.log`).
- **Plots**: Training history (loss/accuracy) will be saved as `[model]_history.png` in the checkpoints directory.

## Evaluation

Evaluation works similarly to training:

```bash
python evaluate.py --model simplecnn --checkpoint ./checkpoints/simplecnn_best.pth
```

### Options
- `--log-file`: Path to save logs (default: `evaluate.log`).
- **Plots**: A confusion matrix will be saved as `[model]_confusion_matrix.png` (next to checkpoint or in current dir).

## Results
Checkpoints will be saved in the `./checkpoints` directory.
