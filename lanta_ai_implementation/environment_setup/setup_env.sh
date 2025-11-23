#!/bin/bash
# LANTA Cluster AI Environment Setup Script
# Based on official LANTA documentation

echo "Setting up AI environment on LANTA cluster..."

# Function to check if module exists
check_module() {
    module avail $1 2>&1 | grep -q "No module(s) or extension(s) found" && return 1 || return 0
}

# Function to safely load module
safe_load_module() {
    local module=$1
    if check_module $module; then
        echo "Loading module: $module"
        module load $module
    else
        echo "Warning: Module $module not found, skipping..."
    fi
}

# Clear any existing modules
echo "Clearing existing modules..."
module purge

# Load basic compiler and MPI modules (required foundation)
echo "Loading basic compiler and MPI modules..."
safe_load_module "GCC/11.3.0"
safe_load_module "OpenMPI/4.1.4"

# Load CUDA toolkit (essential for GPU computing)
echo "Loading CUDA toolkit..."
safe_load_module "CUDA/11.7.0"

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    echo "CUDA compiler available: $(nvcc --version | grep release)"
else
    echo "Warning: CUDA compiler not found"
fi

# Load AI framework modules
echo "Loading AI framework modules..."

# PyTorch options
if [ "$1" = "pytorch" ] || [ -z "$1" ]; then
    echo "Setting up PyTorch environment..."
    safe_load_module "PyTorch/1.12.0"
    
    # Verify PyTorch installation
    if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
        echo "PyTorch successfully loaded"
        python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
        python3 -c "import torch; print('GPU count:', torch.cuda.device_count())"
    else
        echo "Warning: PyTorch not available"
    fi
fi

# TensorFlow options
if [ "$1" = "tensorflow" ]; then
    echo "Setting up TensorFlow environment..."
    safe_load_module "TensorFlow/2.9.1"
    
    # Verify TensorFlow installation
    if python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>/dev/null; then
        echo "TensorFlow successfully loaded"
        python3 -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
    else
        echo "Warning: TensorFlow not available"
    fi
fi

# JAX options
if [ "$1" = "jax" ]; then
    echo "Setting up JAX environment..."
    safe_load_module "JAX/0.3.15"
    
    # Verify JAX installation
    if python3 -c "import jax; print('JAX version:', jax.__version__)" 2>/dev/null; then
        echo "JAX successfully loaded"
        python3 -c "import jax; print('Devices:', jax.devices())"
    else
        echo "Warning: JAX not available"
    fi
fi

# Load additional useful modules
echo "Loading additional modules..."

# Scientific computing
safe_load_module "SciPy-bundle/2022.05"
safe_load_module "matplotlib/3.5.2"

# Machine learning utilities
safe_load_module "scikit-learn/1.1.2"
safe_load_module "pandas/1.4.3"

# Development tools
safe_load_module "git/2.36.0"

# Set environment variables for optimal performance
echo "Setting environment variables..."

# CUDA optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0

# TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=2
export TF_GPU_MEMORY_GROWTH=true

# Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:$HOME/.local/lib/python3.9/site-packages"

# Memory optimizations
export MALLOC_TRIM_THRESHOLD_=100000000

# Display loaded modules
echo "Currently loaded modules:"
module list

# Display environment information
echo "Environment setup complete!"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "NCCL_DEBUG: $NCCL_DEBUG"

# Test GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "Warning: nvidia-smi not available"
fi

# Create useful aliases
echo "Creating useful aliases..."
alias gpu-info='nvidia-smi'
alias gpu-watch='watch -n 1 nvidia-smi'
alias job-status='squeue -u $USER'
alias cluster-info='sinfo -N -l'

# Save environment to file for reuse
cat > ~/.lanta_ai_env << EOF
# LANTA AI Environment Variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
export NCCL_DEBUG=$NCCL_DEBUG
export TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL
export TF_GPU_MEMORY_GROWTH=$TF_GPU_MEMORY_GROWTH
export PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export PYTHONPATH=$PYTHONPATH
export MALLOC_TRIM_THRESHOLD_=$MALLOC_TRIM_THRESHOLD_

# Aliases
alias gpu-info='nvidia-smi'
alias gpu-watch='watch -n 1 nvidia-smi'
alias job-status='squeue -u $USER'
alias cluster-info='sinfo -N -l'
EOF

echo "Environment saved to ~/.lanta_ai_env"
echo "To reload this environment in future sessions, run: source ~/.lanta_ai_env"

# Display final status
echo ""
echo "=== LANTA AI Environment Setup Complete ==="
echo "Framework: ${1:-pytorch}"
echo "Available GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')"
echo "Python version: $(python3 --version)"
echo ""
echo "Quick test commands:"
echo "  PyTorch: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "  TensorFlow: python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo "  JAX: python3 -c 'import jax; print(jax.devices())'"