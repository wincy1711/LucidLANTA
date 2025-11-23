#!/bin/bash
#SBATCH --job-name=pytorch_ai_job      # Job name
#SBATCH --partition=gpu                # Partition (queue)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --cpus-per-task=8              # CPU cores per task
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --time=02:00:00                # Time limit
#SBATCH --output=slurm_%j.out          # Standard output log
#SBATCH --error=slurm_%j.err           # Standard error log
#SBATCH --mem=32G                      # Memory per node

# Load required modules based on LANTA documentation
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load PyTorch/1.12.0

# Verify GPU availability
nvidia-smi

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create working directory
WORKDIR=/scratch/$USER/pytorch_job_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

# Copy training script (modify path as needed)
cp /path/to/your/training_script.py .

# Run the training script
python training_script.py \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --data-path /path/to/dataset \
    --output-dir $WORKDIR/output

# Copy results back to home
mkdir -p $HOME/pytorch_results/$SLURM_JOB_ID
cp -r $WORKDIR/output/* $HOME/pytorch_results/$SLURM_JOB_ID/

# Cleanup
cd $HOME
rm -rf $WORKDIR

echo "Job completed successfully"