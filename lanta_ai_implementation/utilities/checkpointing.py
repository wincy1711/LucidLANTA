#!/usr/bin/env python3
"""
Checkpoint management utilities for LANTA cluster
Provides automatic checkpointing and recovery for long-running AI jobs
"""

import os
import time
import json
import pickle
import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and recovery"""
    
    def __init__(self, checkpoint_dir="checkpoints", max_checkpoints=5, save_interval=3600):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval  # seconds
        self.last_save_time = 0
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        
        # Load or create metadata
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """Load checkpoint metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {
            "created_at": datetime.now().isoformat(),
            "checkpoints": [],
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "total_checkpoints": 0
        }
    
    def save_metadata(self):
        """Save checkpoint metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit"""
        checkpoints = self.metadata["checkpoints"]
        
        if len(checkpoints) > self.max_checkpoints:
            # Sort by timestamp and remove oldest
            checkpoints.sort(key=lambda x: x["timestamp"])
            to_remove = checkpoints[:-self.max_checkpoints]
            
            for checkpoint_info in to_remove:
                checkpoint_path = Path(checkpoint_info["path"])
                if checkpoint_path.exists():
                    if checkpoint_path.is_dir():
                        shutil.rmtree(checkpoint_path)
                    else:
                        checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                
                self.metadata["checkpoints"].remove(checkpoint_info)
            
            self.save_metadata()
    
    def create_checkpoint_name(self, epoch=None, step=None, metric_name=None, metric_value=None):
        """Create checkpoint name based on parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = ["checkpoint", timestamp]
        
        if epoch is not None:
            name_parts.append(f"epoch_{epoch}")
        
        if step is not None:
            name_parts.append(f"step_{step}")
        
        if metric_name and metric_value is not None:
            name_parts.append(f"{metric_name}_{metric_value:.4f}")
        
        return "_".join(name_parts)
    
    def save_pytorch_checkpoint(self, model, optimizer, epoch, step, metrics=None, **kwargs):
        """Save PyTorch model checkpoint"""
        checkpoint_name = self.create_checkpoint_name(epoch, step)
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict() if hasattr(model, 'state_dict') else model,
            "optimizer_state_dict": optimizer.state_dict() if hasattr(optimizer, 'state_dict') else optimizer,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "framework": "pytorch"
        }
        
        # Add additional kwargs
        checkpoint_data.update(kwargs)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update metadata
        checkpoint_info = {
            "name": checkpoint_name,
            "path": str(checkpoint_path),
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "framework": "pytorch",
            "metrics": metrics or {},
            "size_mb": checkpoint_path.stat().st_size / (1024 * 1024)
        }
        
        self.metadata["checkpoints"].append(checkpoint_info)
        self.metadata["latest_checkpoint"] = checkpoint_name
        self.metadata["total_checkpoints"] += 1
        
        # Update best checkpoint if metric provided
        if metrics:
            for metric_name, metric_value in metrics.items():
                if "loss" in metric_name.lower():
                    # Lower is better for loss
                    if (self.metadata["best_checkpoint"] is None or 
                        metric_value < self.metadata.get("best_metric_value", float('inf'))):
                        self.metadata["best_checkpoint"] = checkpoint_name
                        self.metadata["best_metric_name"] = metric_name
                        self.metadata["best_metric_value"] = metric_value
                else:
                    # Higher is better for accuracy and other metrics
                    if (self.metadata["best_checkpoint"] is None or 
                        metric_value > self.metadata.get("best_metric_value", 0)):
                        self.metadata["best_checkpoint"] = checkpoint_name
                        self.metadata["best_metric_name"] = metric_name
                        self.metadata["best_metric_value"] = metric_value
        
        self.save_metadata()
        self.cleanup_old_checkpoints()
        
        logger.info(f"Saved PyTorch checkpoint: {checkpoint_path}")
        self.last_save_time = time.time()
        
        return checkpoint_path
    
    def save_tensorflow_checkpoint(self, model, optimizer, epoch, step, metrics=None, **kwargs):
        """Save TensorFlow model checkpoint"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available")
            return None
        
        checkpoint_name = self.create_checkpoint_name(epoch, step)
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}"
        
        # Create checkpoint
        checkpoint = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=tf.Variable(epoch),
            step=tf.Variable(step)
        )
        
        # Save checkpoint
        save_path = checkpoint.save(str(checkpoint_path))
        
        # Create metadata file
        metadata = {
            "epoch": epoch,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "framework": "tensorflow",
            "checkpoint_path": save_path
        }
        
        metadata.update(kwargs)
        
        with open(f"{checkpoint_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update metadata
        checkpoint_info = {
            "name": checkpoint_name,
            "path": str(checkpoint_path),
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "framework": "tensorflow",
            "metrics": metrics or {},
            "size_mb": sum(f.stat().st_size for f in Path(save_path).parent.glob(f"{checkpoint_name}*")) / (1024 * 1024)
        }
        
        self.metadata["checkpoints"].append(checkpoint_info)
        self.metadata["latest_checkpoint"] = checkpoint_name
        self.metadata["total_checkpoints"] += 1
        
        # Update best checkpoint
        if metrics:
            for metric_name, metric_value in metrics.items():
                if "loss" in metric_name.lower():
                    if (self.metadata["best_checkpoint"] is None or 
                        metric_value < self.metadata.get("best_metric_value", float('inf'))):
                        self.metadata["best_checkpoint"] = checkpoint_name
                        self.metadata["best_metric_name"] = metric_name
                        self.metadata["best_metric_value"] = metric_value
                else:
                    if (self.metadata["best_checkpoint"] is None or 
                        metric_value > self.metadata.get("best_metric_value", 0)):
                        self.metadata["best_checkpoint"] = checkpoint_name
                        self.metadata["best_metric_name"] = metric_name
                        self.metadata["best_metric_value"] = metric_value
        
        self.save_metadata()
        self.cleanup_old_checkpoints()
        
        logger.info(f"Saved TensorFlow checkpoint: {save_path}")
        self.last_save_time = time.time()
        
        return Path(save_path)
    
    def load_pytorch_checkpoint(self, checkpoint_name=None, model=None, optimizer=None):
        """Load PyTorch checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = self.metadata.get("latest_checkpoint")
        
        if checkpoint_name is None:
            logger.warning("No checkpoint available to load")
            return None
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            if model and "model_state_dict" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model_state_dict"])
            
            # Load optimizer state
            if optimizer and "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            
            logger.info(f"Loaded PyTorch checkpoint: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def load_tensorflow_checkpoint(self, checkpoint_name=None, model=None, optimizer=None):
        """Load TensorFlow checkpoint"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available")
            return None
        
        if checkpoint_name is None:
            checkpoint_name = self.metadata.get("latest_checkpoint")
        
        if checkpoint_name is None:
            logger.warning("No checkpoint available to load")
            return None
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}"
        metadata_path = f"{checkpoint_path}_metadata.json"
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create checkpoint
            checkpoint = tf.train.Checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=tf.Variable(metadata.get("epoch", 0)),
                step=tf.Variable(metadata.get("step", 0))
            )
            
            # Restore checkpoint
            checkpoint.restore(str(checkpoint_path))
            
            logger.info(f"Loaded TensorFlow checkpoint: {checkpoint_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def should_save_checkpoint(self, force=False):
        """Check if it's time to save a checkpoint"""
        if force:
            return True
        
        current_time = time.time()
        return (current_time - self.last_save_time) >= self.save_interval
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        return self.metadata["checkpoints"]
    
    def get_checkpoint_info(self, checkpoint_name):
        """Get information about a specific checkpoint"""
        for checkpoint in self.metadata["checkpoints"]:
            if checkpoint["name"] == checkpoint_name:
                return checkpoint
        return None
    
    def remove_checkpoint(self, checkpoint_name):
        """Remove a specific checkpoint"""
        checkpoint_info = self.get_checkpoint_info(checkpoint_name)
        if not checkpoint_info:
            logger.warning(f"Checkpoint not found: {checkpoint_name}")
            return False
        
        checkpoint_path = Path(checkpoint_info["path"])
        
        # Remove checkpoint files
        if checkpoint_path.exists():
            if checkpoint_path.is_dir():
                shutil.rmtree(checkpoint_path)
            else:
                checkpoint_path.unlink()
        
        # Remove metadata file for TensorFlow checkpoints
        metadata_path = f"{checkpoint_path}_metadata.json"
        if Path(metadata_path).exists():
            Path(metadata_path).unlink()
        
        # Update metadata
        self.metadata["checkpoints"] = [c for c in self.metadata["checkpoints"] if c["name"] != checkpoint_name]
        
        if self.metadata["latest_checkpoint"] == checkpoint_name:
            self.metadata["latest_checkpoint"] = None
        
        if self.metadata["best_checkpoint"] == checkpoint_name:
            self.metadata["best_checkpoint"] = None
        
        self.metadata["total_checkpoints"] = len(self.metadata["checkpoints"])
        self.save_metadata()
        
        logger.info(f"Removed checkpoint: {checkpoint_name}")
        return True
    
    def print_checkpoint_summary(self):
        """Print summary of checkpoints"""
        print("\n" + "="*60)
        print("CHECKPOINT SUMMARY")
        print("="*60)
        
        print(f"Total checkpoints: {self.metadata['total_checkpoints']}")
        print(f"Latest checkpoint: {self.metadata.get('latest_checkpoint', 'None')}")
        print(f"Best checkpoint: {self.metadata.get('best_checkpoint', 'None')}")
        
        if self.metadata['best_checkpoint']:
            print(f"Best metric: {self.metadata.get('best_metric_name', 'Unknown')} = {self.metadata.get('best_metric_value', 'Unknown')}")
        
        print(f"\nAll checkpoints:")
        for checkpoint in self.metadata["checkpoints"]:
            print(f"  - {checkpoint['name']}")
            print(f"    Framework: {checkpoint['framework']}")
            print(f"    Created: {checkpoint['timestamp']}")
            print(f"    Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"    Step: {checkpoint.get('step', 'N/A')}")
            print(f"    Size: {checkpoint.get('size_mb', 0):.2f} MB")
            if checkpoint['metrics']:
                print(f"    Metrics: {checkpoint['metrics']}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Checkpoint Management Tool')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--max-checkpoints', type=int, default=5, help='Maximum number of checkpoints to keep')
    parser.add_argument('--save-interval', type=int, default=3600, help='Auto-save interval in seconds')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all checkpoints')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a checkpoint')
    remove_parser.add_argument('checkpoint_name', help='Name of checkpoint to remove')
    
    # Clean command
    subparsers.add_parser('clean', help='Clean up old checkpoints')
    
    args = parser.parse_args()
    
    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=args.max_checkpoints,
        save_interval=args.save_interval
    )
    
    # Execute command
    if args.command == 'list':
        manager.print_checkpoint_summary()
    elif args.command == 'remove':
        if manager.remove_checkpoint(args.checkpoint_name):
            print(f"Successfully removed checkpoint: {args.checkpoint_name}")
        else:
            print(f"Failed to remove checkpoint: {args.checkpoint_name}")
    elif args.command == 'clean':
        manager.cleanup_old_checkpoints()
        print("Checkpoint cleanup completed")
    else:
        manager.print_checkpoint_summary()

if __name__ == '__main__':
    main()