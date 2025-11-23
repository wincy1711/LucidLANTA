#!/bin/bash
#SBATCH --job-name=multi_gpu_ai_job      # Job name
#SBATCH --partition=gpu                  # Partition (queue)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=16               # CPU cores per task
#SBATCH --gres=gpu:4                     # Number of GPUs (4 GPUs)
#SBATCH --time=06:00:00                  # Time limit
#SBATCH --output=slurm_multi_gpu_%j.out  # Standard output log
#SBATCH --error=slurm_multi_gpu_%j.err   # Standard error log
#SBATCH --mem=128G                       # Memory per node

# Load required modules
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load PyTorch/1.12.0

# Verify all GPUs are available
nvidia-smi

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO

# Create working directory
WORKDIR=/scratch/$USER/multi_gpu_job_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

# Copy multi-GPU training script
cp /path/to/your/multi_gpu_training_script.py .

# Run distributed data parallel training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    multi_gpu_training_script.py \
    --world-size 4 \
    --rank $SLURM_PROCID \
    --epochs 100 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --data-path /path/to/dataset \
    --output-dir $WORKDIR/output

# Copy results back
mkdir -p $HOME/multi_gpu_results/$SLURM_JOB_ID
cp -r $WORKDIR/output/* $HOME/multi_gpu_results/$SLURM_JOB_ID/

# Cleanup
cd $HOME
rm -rf $WORKDIR

echo "Multi-GPU job completed successfully"