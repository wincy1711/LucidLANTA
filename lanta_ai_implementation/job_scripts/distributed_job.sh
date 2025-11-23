#!/bin/bash
#SBATCH --job-name=distributed_ai_job     # Job name
#SBATCH --partition=gpu                   # Partition (queue)
#SBATCH --nodes=2                         # Number of nodes
#SBATCH --ntasks-per-node=4               # Number of tasks per node
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --gres=gpu:4                      # Number of GPUs per node
#SBATCH --time=12:00:00                   # Time limit
#SBATCH --output=slurm_distributed_%j.out # Standard output log
#SBATCH --error=slurm_distributed_%j.err  # Standard error log
#SBATCH --mem=256G                        # Memory per node

# Load required modules
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load PyTorch/1.12.0

# Get node list
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES=($NODELIST)
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname --ip-address)

echo "Head node: $HEAD_NODE"
echo "Head node IP: $HEAD_NODE_IP"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0

# Create working directory
WORKDIR=/scratch/$USER/distributed_job_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

# Copy distributed training script
cp /path/to/your/distributed_training_script.py .

# Launch distributed training across nodes
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HEAD_NODE_IP:29500 \
    distributed_training_script.py \
    --epochs 200 \
    --batch-size 512 \
    --learning-rate 0.001 \
    --data-path /path/to/dataset \
    --output-dir $WORKDIR/output \
    --world-size $SLURM_NTASKS

# Copy results back
mkdir -p $HOME/distributed_results/$SLURM_JOB_ID
cp -r $WORKDIR/output/* $HOME/distributed_results/$SLURM_JOB_ID/

# Cleanup
cd $HOME
rm -rf $WORKDIR

echo "Distributed training job completed successfully"