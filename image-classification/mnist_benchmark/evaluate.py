import argparse
import os
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import get_model

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate MNIST models')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['simplecnn', 'mlp', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint file')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size')
    parser.add_argument('--log-file', type=str, default='evaluate.log', help='Path to save logs')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def plot_confusion_matrix(targets, preds, classes, save_path):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def main():
    args = get_args()
    setup_logging(args.log_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info(f"Using device: {device}")

    # Data Preparation
    logging.info('==> Preparing data..')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    testset = torchvision.datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    classes = [str(i) for i in range(10)]

    # Model
    logging.info(f'==> Building model: {args.model}..')
    net = get_model(args.model, num_classes=10)
    net = net.to(device)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            logging.info(f"==> Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'net' in checkpoint:
                net.load_state_dict(checkpoint['net'])
            else:
                net.load_state_dict(checkpoint)
        else:
            logging.error(f"Error: No checkpoint found at {args.checkpoint}")
            return

    criterion = nn.CrossEntropyLoss()
    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        with tqdm(total=len(testloader), desc='Evaluating', unit='batch') as pbar:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                pbar.set_postfix({'Loss': test_loss/(batch_idx+1), 'Acc': 100.*correct/total})
                pbar.update(1)

    acc = 100.*correct/total
    logging.info(f'Test Set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {acc:.2f}%')
    
    # Plot confusion matrix
    cm_path = f'{args.model}_confusion_matrix.png'
    if args.checkpoint:
        ckpt_dir = os.path.dirname(args.checkpoint)
        if ckpt_dir:
            cm_path = os.path.join(ckpt_dir, f'{args.model}_confusion_matrix.png')
            
    plot_confusion_matrix(all_targets, all_preds, classes, cm_path)
    logging.info(f'Confusion matrix saved to {cm_path}')

if __name__ == '__main__':
    main()
