import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import get_model

def get_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 models using timm')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['efficientnet', 'mobilenet', 'wideresnet', 'vit'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--weights-path', type=str, default='./checkpoints', help='Path to save weights')
    parser.add_argument('--log-file', type=str, default='train.log', help='Path to save logs')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--dry-run', action='store_true', help='Run a single batch for debugging')
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

def plot_history(train_losses, test_losses, train_accs, test_accs, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy History')
    
    plt.savefig(save_path)
    plt.close()

def main():
    args = get_args()
    if not os.path.exists(os.path.dirname(args.log_file)) and os.path.dirname(args.log_file) != '':
        os.makedirs(os.path.dirname(args.log_file))
        
    setup_logging(args.log_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info(f"Using device: {device}")

    # Hyperparameters & Settings
    best_acc = 0
    start_epoch = 0
    
    if not os.path.exists(args.weights_path):
        os.makedirs(args.weights_path)

    # Data Preparation
    logging.info('==> Preparing data..')
    
    # Image size determination
    if args.model == 'vit':
        img_size = 224
        logging.info(f"Model is ViT, resizing images to {img_size}x{img_size}")
    else:
        img_size = 32
        
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=4 if img_size==32 else 0), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    logging.info(f'==> Building model: {args.model}..')
    net = get_model(args.model, pretrained=args.pretrained, num_classes=10, img_size=img_size)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.model == 'vit':
        optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # History tracking
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Dry run logic
        if args.dry_run:
            logging.info("Dry run: running only 1 batch")
            limit = 1
        else:
            limit = len(trainloader)
            
        with tqdm(total=limit, desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch') as pbar:
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx >= limit:
                    break
                    
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'Loss': train_loss/(batch_idx+1), 'Acc': 100.*correct/total})
                pbar.update(1)
        
        epoch_loss = train_loss/limit
        epoch_acc = 100.*correct/total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        logging.info(f'Epoch {epoch+1} Train: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        if args.dry_run:
            limit = 1
        else:
            limit = len(testloader)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if batch_idx >= limit:
                    break
                    
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = test_loss/limit
        epoch_acc = 100.*correct/total
        test_losses.append(epoch_loss)
        test_accs.append(epoch_acc)
        logging.info(f'Epoch {epoch+1} Test: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

        if epoch_acc > best_acc and not args.dry_run:
            logging.info(f'Saving.. (New best: {epoch_acc:.2f}%)')
            state = {
                'net': net.state_dict(),
                'acc': epoch_acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(args.weights_path, f'{args.model}_best.pth'))
            best_acc = epoch_acc

    epochs_to_run = 1 if args.dry_run else args.epochs
    
    for epoch in range(start_epoch, epochs_to_run):
        train(epoch)
        test(epoch)
        scheduler.step()
        
    # Plot history
    if not args.dry_run:
        plot_path = os.path.join(args.weights_path, f'{args.model}_history.png')
        plot_history(train_losses, test_losses, train_accs, test_accs, plot_path)
        logging.info(f'Training history saved to {plot_path}')
    
if __name__ == '__main__':
    main()
