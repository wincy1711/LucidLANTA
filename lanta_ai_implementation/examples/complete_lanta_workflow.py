#!/usr/bin/env python3
"""
Complete LANTA cluster AI workflow example
Demonstrates integration of all components: job submission, monitoring, checkpointing, and cleanup
"""

import os
import time
import json
import logging
import argparse
from pathlib import Path

# Configure logging for distributed training
def setup_logging(rank=0):
    """Setup logging with rank-specific files for distributed training"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'workflow_rank_{rank}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class LANTAWorkflow:
    """Complete workflow manager for LANTA cluster AI jobs"""
    
    def __init__(self, config_path="workflow_config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.logger = None
        
        # Initialize components
        self.checkpoint_manager = None
        self.resource_monitor = None
        
    def load_config(self):
        """Load workflow configuration"""
        default_config = {
            "framework": "pytorch",
            "model_type": "simple_cnn",
            "dataset": "cifar10",
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "save_interval": 1800,  # 30 minutes
            "monitor_interval": 60,  # 1 minute
            "max_checkpoints": 5,
            "output_dir": "./workflow_output",
            "data_dir": "./data",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
            "distributed": False,
            "mixed_precision": True,
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.001
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def setup_environment(self):
        """Setup environment for LANTA cluster"""
        # Create directories
        for dir_path in [
            self.config["output_dir"],
            self.config["data_dir"],
            self.config["checkpoint_dir"],
            self.config["log_dir"]
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Log environment info
        self.logger.info(f"Environment setup complete")
        self.logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        self.logger.info(f"Output directory: {self.config['output_dir']}")
    
    def create_model(self):
        """Create AI model based on configuration"""
        if self.config["framework"] == "pytorch":
            return self.create_pytorch_model()
        elif self.config["framework"] == "tensorflow":
            return self.create_tensorflow_model()
        else:
            raise ValueError(f"Unsupported framework: {self.config['framework']}")
    
    def create_pytorch_model(self):
        """Create PyTorch model"""
        import torch
        import torch.nn as nn
        
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                
                self.fc1 = nn.Linear(128 * 4 * 4, 512)
                self.fc2 = nn.Linear(512, num_classes)
                
                self.bn1 = nn.BatchNorm2d(32)
                self.bn2 = nn.BatchNorm2d(64)
                self.bn3 = nn.BatchNorm2d(128)
            
            def forward(self, x):
                x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
                x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
                x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
                
                x = self.dropout1(x)
                x = x.view(-1, 128 * 4 * 4)
                x = nn.functional.relu(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)
                
                return x
        
        return SimpleCNN()
    
    def create_tensorflow_model(self):
        """Create TensorFlow model"""
        import tensorflow as tf
        from tensorflow import keras
        
        model = keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def create_data_loaders(self):
        """Create data loaders"""
        if self.config["framework"] == "pytorch":
            return self.create_pytorch_data_loaders()
        elif self.config["framework"] == "tensorflow":
            return self.create_tensorflow_data_loaders()
    
    def create_pytorch_data_loaders(self):
        """Create PyTorch data loaders"""
        import torch
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        # Data transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load dataset
        if self.config["dataset"] == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.config["data_dir"], train=True, 
                download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.config["data_dir"], train=False, 
                download=True, transform=transform_test
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.config['dataset']}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], 
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.config["batch_size"], 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, test_loader
    
    def create_tensorflow_data_loaders(self):
        """Create TensorFlow data loaders"""
        import tensorflow as tf
        
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        
        def preprocess_data(images, labels, training=True):
            images = tf.cast(images, tf.float32) / 255.0
            if training:
                images = data_augmentation(images)
            return images, labels
        
        # Load dataset
        if self.config["dataset"] == "cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.config['dataset']}")
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(10000)
        train_dataset = train_dataset.map(lambda x, y: preprocess_data(x, y, True))
        train_dataset = train_dataset.batch(self.config["batch_size"])
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.map(lambda x, y: preprocess_data(x, y, False))
        test_dataset = test_dataset.batch(self.config["batch_size"])
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, test_dataset
    
    def train_pytorch(self):
        """Train PyTorch model"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        
        # Create model and data
        model = self.create_model().to(device)
        train_loader, test_loader = self.create_data_loaders()
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Mixed precision
        scaler = None
        if self.config["mixed_precision"] and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        
        # Training loop
        best_accuracy = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        self.logger.info("Starting PyTorch training...")
        start_time = time.time()
        
        for epoch in range(self.config["epochs"]):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
                    self.logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_loss /= len(test_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log results
            self.logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                           f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save checkpoint
            if self.checkpoint_manager.should_save_checkpoint() or val_acc > best_accuracy:
                metrics = {
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }
                
                self.checkpoint_manager.save_pytorch_checkpoint(
                    model, optimizer, epoch, batch_idx, metrics
                )
                
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
        
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time:.2f} seconds')
        
        # Save final model
        final_path = Path(self.config["output_dir"]) / "final_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'final_accuracy': val_acc,
            'best_accuracy': best_accuracy
        }, final_path)
        
        self.logger.info(f'Final model saved: {final_path}')
        
        return history, best_accuracy
    
    def run_workflow(self):
        """Run the complete workflow"""
        # Setup logging
        rank = int(os.environ.get('SLURM_PROCID', 0))
        self.logger = setup_logging(rank)
        
        self.logger.info("Starting LANTA cluster AI workflow")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Setup environment
        self.setup_environment()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config["checkpoint_dir"],
            max_checkpoints=self.config["max_checkpoints"],
            save_interval=self.config["save_interval"]
        )
        
        # Train model
        if self.config["framework"] == "pytorch":
            history, best_accuracy = self.train_pytorch()
        else:
            raise ValueError(f"Unsupported framework: {self.config['framework']}")
        
        # Save final configuration
        self.config["final_accuracy"] = best_accuracy
        self.config["training_completed"] = datetime.now().isoformat()
        self.save_config()
        
        self.logger.info("Workflow completed successfully")
        self.logger.info(f"Best accuracy: {best_accuracy:.2f}%")
        
        return {
            "best_accuracy": best_accuracy,
            "history": history,
            "config": self.config
        }

def main():
    parser = argparse.ArgumentParser(description='LANTA Cluster AI Workflow')
    parser.add_argument('--config', type=str, default='workflow_config.json', help='Configuration file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create and run workflow
    workflow = LANTAWorkflow(config_path=args.config)
    
    if args.resume:
        print("Resuming from checkpoint...")
        # Add resume logic here
    
    # Run workflow
    results = workflow.run_workflow()
    
    # Print summary
    print("\n" + "="*60)
    print("LANTA WORKFLOW COMPLETED")
    print("="*60)
    print(f"Framework: {results['config']['framework']}")
    print(f"Model type: {results['config']['model_type']}")
    print(f"Dataset: {results['config']['dataset']}")
    print(f"Best accuracy: {results['best_accuracy']:.2f}%")
    print(f"Output directory: {results['config']['output_dir']}")
    print(f"Total checkpoints: {results['config'].get('final_accuracy', 'N/A')}")
    print("="*60)

if __name__ == '__main__':
    main()