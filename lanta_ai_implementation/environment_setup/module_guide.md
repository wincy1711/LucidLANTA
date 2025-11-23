# LANTA Cluster Module Guide

## Overview

This guide provides detailed information about available AI modules on the LANTA cluster, based on official documentation.

## Available AI Frameworks

### PyTorch
```bash
# Load PyTorch with CUDA support
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load PyTorch/1.12.0

# Verify installation
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### TensorFlow
```bash
# Load TensorFlow with CUDA support
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load TensorFlow/2.9.1

# Verify installation
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python3 -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

### JAX
```bash
# Load JAX with CUDA support
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load JAX/0.3.15

# Verify installation
python3 -c "import jax; print('JAX version:', jax.__version__)"
python3 -c "import jax; print('Devices:', jax.devices())"
```

## Module Dependencies

### Required Foundation Modules
All AI frameworks require these base modules:
- `GCC/11.3.0` - Compiler
- `OpenMPI/4.1.4` - MPI implementation
- `CUDA/11.7.0` - CUDA toolkit

### Optional Supporting Modules
```bash
# Scientific computing
module load SciPy-bundle/2022.05
module load matplotlib/3.5.2

# Machine learning
module load scikit-learn/1.1.2
module load pandas/1.4.3

# Development
module load git/2.36.0
```

## Environment Variables

### CUDA Optimizations
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3    # Available GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # PyTorch memory management
export NCCL_DEBUG=INFO                   # NCCL debugging
export NCCL_SOCKET_IFNAME=ib0           # InfiniBand network interface
```

### TensorFlow Optimizations
```bash
export TF_CPP_MIN_LOG_LEVEL=2           # Reduce TensorFlow logging
export TF_GPU_MEMORY_GROWTH=true        # Dynamic GPU memory allocation
```

### General Optimizations
```bash
export PYTHONUNBUFFERED=1               # Unbuffered Python output
export MALLOC_TRIM_THRESHOLD_=100000000 # Memory optimization
```

## Best Practices

### 1. Module Loading Order
Always load modules in this order:
1. Compiler (GCC)
2. MPI (OpenMPI)
3. CUDA toolkit
4. AI framework
5. Additional tools

### 2. Environment Setup
Create a consistent environment:
```bash
# Save environment
cat > ~/.lanta_ai_env << 'EOF'
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load PyTorch/1.12.0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF

# Use in jobs
source ~/.lanta_ai_env
```

### 3. GPU Memory Management
```bash
# For PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For TensorFlow
export TF_GPU_MEMORY_GROWTH=true

# Monitor GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 4. Distributed Training
```bash
# Set NCCL environment variables for multi-node training
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_TREE_THRESHOLD=0
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   # Enable gradient checkpointing
   # Use mixed precision training
   ```

2. **NCCL errors in distributed training**
   ```bash
   # Check network interface
   export NCCL_SOCKET_IFNAME=ib0
   
   # Reduce NCCL buffer size
   export NCCL_BUFFSIZE=8388608
   ```

3. **Module conflicts**
   ```bash
   # Always start with module purge
   module purge
   
   # Check for conflicts
   module list
   ```

### Verification Commands
```bash
# Check GPU availability
nvidia-smi

# Test PyTorch
cat > test_pytorch.py << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
EOF
python3 test_pytorch.py

# Test TensorFlow
cat > test_tensorflow.py << 'EOF'
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
EOF
python3 test_tensorflow.py
```

## Performance Optimization

### 1. CPU Affinity
```bash
# Bind processes to specific CPU cores
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### 2. Memory Optimization
```bash
# Preallocate memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use pinned memory
export PYTORCH_PINNED_MEMORY=1
```

### 3. Network Optimization
```bash
# For multi-node training
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
```

## Module Availability

To check available modules:
```bash
module avail AI
module avail CUDA
module avail PyTorch
module avail TensorFlow
```

## Support

For module-related issues:
1. Check official LANTA documentation
2. Contact LANTA support team
3. Use `module help <module_name>` for specific module information