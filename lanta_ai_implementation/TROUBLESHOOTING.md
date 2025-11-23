# LANTA Cluster AI Implementation Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when running AI workloads on the LANTA cluster.

## Quick Diagnostic Commands

```bash
# Check job status
squeue -u $USER

# Check GPU availability
nvidia-smi

# Check loaded modules
module list

# Check disk usage
df -h

# Check memory usage
free -h

# Monitor real-time resource usage
htop
```

## Common Issues and Solutions

### 1. Job Submission Issues

#### Problem: Job stuck in pending state
```bash
# Check job status
squeue -j <JOB_ID>

# Check reason for pending
squeue -j <JOB_ID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Common reasons and solutions:
# - Resources: Wait for resources to become available
# - Priority: Job has low priority, will run when resources available
# - Dependency: Job waiting for dependency, check with scontrol show job <JOB_ID>
# - Partition: Job submitted to wrong partition, check with sinfo
```

#### Problem: Job fails immediately after submission
```bash
# Check job output
cat slurm_<JOB_ID>.out
cat slurm_<JOB_ID>.err

# Common causes:
# - Incorrect shebang in script
# - Missing executable permissions
# - Module loading errors
# - Insufficient memory request
```

### 2. GPU Issues

#### Problem: CUDA out of memory
```python
# PyTorch - Reduce batch size
dataloader = DataLoader(dataset, batch_size=32)  # Reduce from 64 to 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

```python
# TensorFlow - Memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Limit memory
tf.config.experimental.set_memory_limit(gpu, 4096)  # 4GB
```

#### Problem: GPU not detected
```bash
# Check GPU availability
nvidia-smi

# Check CUDA driver
nvidia-smi --query-gpu=name,driver_version --format=csv

# Verify modules are loaded
module list

# Reload modules if needed
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load PyTorch/1.12.0
```

#### Problem: Multi-GPU training fails
```bash
# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_TREE_THRESHOLD=0

# Check network interface
ip addr show

# Test NCCL
python -c "import torch; print(torch.cuda.nccl.version())"
```

### 3. Memory Issues

#### Problem: System out of memory
```bash
# Check memory usage
free -h
top -o %MEM

# Kill memory-intensive processes
kill -9 <PID>

# Request more memory in job script
#SBATCH --mem=64G
```

#### Problem: Python memory leaks
```python
# Force garbage collection
import gc
gc.collect()

# Use memory profiling
import tracemalloc
tracemalloc.start()
# ... code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

### 4. Storage Issues

#### Problem: Disk quota exceeded
```bash
# Check disk usage
df -h /home /scratch
quota -s

# Find large files
find /home/$USER -type f -size +100M -exec ls -lh {} \;

# Clean up temporary files
./utilities/cleanup.sh temp
./utilities/cleanup.sh cache
```

#### Problem: I/O bottlenecks
```bash
# Monitor I/O usage
iostat -x 1

# Use local scratch for temporary files
export TMPDIR=/scratch/$USER/tmp
mkdir -p $TMPDIR

# Optimize data loading
# - Use multiple workers
# - Pin memory
# - Use SSD if available
```

### 5. Network Issues

#### Problem: Distributed training communication errors
```bash
# Check network connectivity
ping $(hostname)

# Test MPI
mpirun -np 2 hostname

# Set network interface for NCCL
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand
export NCCL_SOCKET_IFNAME=eth0 # Ethernet

# Check InfiniBand status
ibstatus
ibv_devinfo
```

#### Problem: Slow data transfer
```bash
# Use efficient transfer methods
rsync -avh --info=progress2 source/ dest/

# Compress before transfer
tar -czf archive.tar.gz data/

# Use multiple streams
rsync -avh --info=progress2 --whole-file source/ dest/ &
rsync -avh --info=progress2 --whole-file source2/ dest2/ &
wait
```

### 6. Module and Environment Issues

#### Problem: Module conflicts
```bash
# Clear all modules
module purge

# Load in correct order
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load CUDA/11.7.0
module load PyTorch/1.12.0

# Check for conflicts
module list
```

#### Problem: Python package conflicts
```bash
# Create isolated environment
python -m venv myenv
source myenv/bin/activate
pip install --upgrade pip

# Or use conda
conda create -n myenv python=3.9
conda activate myenv
```

#### Problem: Version mismatches
```bash
# Check versions
python -c "import torch; print(torch.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"
nvcc --version

# Load specific versions
module load PyTorch/1.12.0
module load TensorFlow/2.9.1
```

### 7. Performance Issues

#### Problem: Slow training
```bash
# Profile the code
python -m cProfile -o profile.out training_script.py
python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative').print_stats(20)"

# GPU profiling
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1

# Use performance monitoring
./monitoring/profile_gpu.py --duration 60 --save
```

#### Problem: CPU bottleneck
```python
# Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,  # Increase workers
    pin_memory=True,  # Use pinned memory
    persistent_workers=True
)

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
```

#### Problem: Poor scaling with multiple GPUs
```python
# Optimize distributed training
# - Increase batch size
# - Use gradient accumulation
# - Tune learning rate
# - Use efficient data loading

# PyTorch DDP optimization
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
```

### 8. Software-Specific Issues

#### PyTorch Issues
```python
# CUDA error: device-side assert triggered
try:
    # Your code
except RuntimeError as e:
    if "device-side assert" in str(e):
        torch.cuda.empty_cache()
        # Check for NaN/Inf in data
        torch.autograd.set_detect_anomaly(True)

# DataLoader worker issues
dataloader = DataLoader(
    dataset,
    num_workers=0,  # Reduce if issues
    batch_size=32
)
```

#### TensorFlow Issues
```python
# ResourceExhaustedError
try:
    # Your code
except tf.errors.ResourceExhaustedError:
    # Reduce batch size
    # Use gradient checkpointing
    # Enable XLA
    tf.config.optimizer.set_jit(True)

# Mixed precision issues
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### 9. Job Recovery Issues

#### Problem: Job interrupted
```bash
# Check if checkpoint exists
ls -la checkpoints/

# Resume from checkpoint
python training_script.py --resume-from checkpoints/latest.pth

# Use job arrays for automatic resubmission
#SBATCH --array=1-10%1
```

#### Problem: Checkpoint corruption
```bash
# Verify checkpoint integrity
python -c "import torch; torch.load('checkpoint.pth')"

# Keep multiple checkpoint versions
# Save best and latest checkpoints separately
```

## Advanced Debugging

### 1. System-Level Debugging

```bash
# Check system logs
journalctl -u slurmctld
journalctl -u slurmd

# Monitor system resources
sar -u 1 10    # CPU usage
sar -r 1 10    # Memory usage
sar -d 1 10    # Disk I/O

# Check network statistics
netstat -i
ss -tuln
```

### 2. Application-Level Debugging

```python
# Add debugging hooks
import torch
import traceback
import sys

def exception_hook(exc_type, exc_value, exc_traceback):
    print(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    sys.exit(1)

sys.excepthook = exception_hook

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Memory debugging
torch.cuda.memory_summary()
```

### 3. Performance Profiling

```bash
# Use profiling tools
nsys profile -o profile_report python training_script.py
ncu --set full python training_script.py

# Python profiling
python -m cProfile -o profile.stats training_script.py
py-spy top --pid $PID
```

## Best Practices

### 1. Job Design
- Always request appropriate resources
- Use checkpointing for long jobs
- Monitor resource usage
- Handle interruptions gracefully

### 2. Resource Management
- Don't request more resources than needed
- Use shared storage efficiently
- Clean up temporary files
- Monitor quota usage

### 3. Error Handling
- Implement robust error handling
- Use try-catch blocks
- Log important information
- Save intermediate results

### 4. Performance Optimization
- Profile before optimizing
- Use appropriate batch sizes
- Optimize data loading
- Use mixed precision when possible

## Getting Help

### 1. Documentation
- Check LANTA cluster documentation
- Review AI framework documentation
- Consult this troubleshooting guide

### 2. Support Channels
- LANTA support team
- Community forums
- Framework-specific support

### 3. Reporting Issues
When reporting issues, include:
- Job ID and submission script
- Error messages and logs
- System information (modules loaded)
- Steps to reproduce
- Expected vs actual behavior

## Emergency Procedures

### 1. Job Cancellation
```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all user jobs
scancel -u $USER

# Cancel job array
scancel <JOB_ID>_<ARRAY_INDEX>
```

### 2. System Recovery
```bash
# Clear stuck processes
ps aux | grep python | grep $USER
kill -9 <PID>

# Clear GPU memory
nvidia-smi --gpu-reset -i <GPU_ID>

# Clear file locks
lsof | grep deleted
```

### 3. Data Recovery
```bash
# Check for automatic backups
ls -la /backup/$USER/

# Recover from checkpoints
python training_script.py --resume-from checkpoints/latest.pth

# Check temporary files
find /scratch/$USER -name "*tmp*" -type f
```

This troubleshooting guide should help resolve most common issues. For specific problems not covered here, consult the LANTA cluster documentation or contact support.