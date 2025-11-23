#!/usr/bin/env python3
"""
Distributed PyTorch training example for LANTA cluster
Uses torch.distributed.launch for multi-node, multi-GPU training
"""

import os
import time
import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Configure logging
def setup_logging(rank, world_size):
    """Setup logging for distributed training"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f'[Node {rank}/{world_size}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'distributed_training_rank_{rank}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """Residual block for deeper network"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetCNN(nn.Module):
    """ResNet-style CNN for CIFAR-10"""
    
    def __init__(self, num_classes=10):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def create_data_loaders(data_path, batch_size, world_size, rank):
    """Create data loaders with distributed sampler"""
    
    # Enhanced data augmentation for better generalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # Create data loaders with optimized settings for distributed training
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=train_sampler,
        persistent_workers=True, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=test_sampler,
        persistent_workers=True, prefetch_factor=2
    )
    
    return train_loader, test_loader, train_sampler

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, scaler=None):
    """Train for one epoch with mixed precision support"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 50 == 0 and dist.get_rank() == 0:
            logger.info(
                f'Epoch: {epoch}, Batch: {batch_idx}, '
                f'Loss: {loss.item():.4f}, '
                f'Acc: {100.*correct/total:.2f}%'
            )
    
    # Aggregate metrics across all processes
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Convert to tensors for all-reduce
    loss_tensor = torch.tensor(avg_loss).to(device)
    acc_tensor = torch.tensor(accuracy).to(device)
    
    # All-reduce to get average across all processes
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
    
    return loss_tensor.item(), acc_tensor.item()

def validate(model, test_loader, criterion, device, logger):
    """Validate the model with distributed evaluation"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    # Aggregate metrics across all processes
    loss_tensor = torch.tensor(avg_loss).to(device)
    acc_tensor = torch.tensor(accuracy).to(device)
    
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
    
    return loss_tensor.item(), acc_tensor.item()

def setup_distributed():
    """Setup distributed training environment"""
    
    # Get environment variables from SLURM or torchrun
    if 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Get master node information
        master_addr = os.environ.get('SLURM_SRUN_COMM_HOST', '127.0.0.1')
        master_port = '29500'
    else:
        # torchrun environment
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    return rank, world_size, local_rank, device

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Distributed PyTorch Training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument('--output-dir', type=str, default='./output', help='output directory')
    parser.add_argument('--world-size', type=int, default=1, help='world size')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    logger = setup_logging(rank, world_size)
    
    logger.info(f'Distributed setup complete: rank={rank}, world_size={world_size}, local_rank={local_rank}')
    
    # Create output directory (only on rank 0)
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Wait for rank 0 to create directory
    dist.barrier()
    
    # Device information
    logger.info(f'Using device: {device}')
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(local_rank)}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB')
    
    # Create model and wrap with DDP
    model = ResNetCNN().to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    # Create data loaders
    train_loader, test_loader, train_sampler = create_data_loaders(
        args.data_path, args.batch_size, world_size, rank
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and torch.cuda.is_available() else None
    
    # Training loop
    best_accuracy = 0.0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logger.info('Starting distributed training...')
    logger.info(f'Mixed precision training: {args.use_amp and torch.cuda.is_available()}')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for proper shuffling
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device, logger)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log results (only on rank 0)
        if rank == 0:
            logger.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_acc:.2f}%'
            )
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Save best model (only on rank 0)
        if rank == 0 and val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'training_history': training_history,
                'world_size': world_size
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f'New best model saved with accuracy: {best_accuracy:.2f}%')
    
    # Save final model and training history (only on rank 0)
    if rank == 0:
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'training_history': training_history,
            'final_accuracy': val_acc,
            'world_size': world_size
        }, os.path.join(args.output_dir, 'final_model.pth'))
        
        # Save training history as JSON
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Log final results (only on rank 0)
    if rank == 0:
        logger.info(f'Training completed in {total_time:.2f} seconds')
        logger.info(f'Best validation accuracy: {best_accuracy:.2f}%')
        
        logger.info('='*60)
        logger.info('DISTRIBUTED TRAINING COMPLETED')
        logger.info('='*60)
        logger.info(f'Number of Nodes: {os.environ.get("SLURM_JOB_NUM_NODES", "N/A")}')
        logger.info(f'World Size (Total Processes): {world_size}')
        logger.info(f'GPUs per Node: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
        logger.info(f'Total GPUs: {world_size}')
        logger.info(f'Batch Size per GPU: {args.batch_size}')
        logger.info(f'Total Effective Batch Size: {args.batch_size * world_size}')
        logger.info(f'Mixed Precision: {args.use_amp and torch.cuda.is_available()}')
        logger.info(f'Final Validation Accuracy: {val_acc:.2f}%')
        logger.info(f'Best Validation Accuracy: {best_accuracy:.2f}%')
        logger.info(f'Total Training Time: {total_time:.2f} seconds')
    
    # Cleanup distributed training
    cleanup_distributed()

if __name__ == '__main__':
    main()