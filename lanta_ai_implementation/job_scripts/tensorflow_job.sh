#!/bin/bash
#SBATCH --job-name=tensorflow_ai_job     # Job name
#SBATCH --partition=gpu                 # Partition (queue)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --time=04:00:00                 # Time limit
#SBATCH --output=slurm_tf_%j.out        # Standard output log
#SBATCH --error=slurm_tf_%j.err         # Standard error log
#SBATCH --mem=32G                       # Memory per node

# Load required modules based on LANTA documentation
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0
module load TensorFlow/2.9.1

# Verify GPU availability and TensorFlow GPU support
nvidia-smi
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

# Set environment variables for TensorFlow
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
export TF_GPU_MEMORY_GROWTH=true

# Create working directory
WORKDIR=/scratch/$USER/tensorflow_job_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

# Copy training script (modify path as needed)
cp /path/to/your/tensorflow_training_script.py .

# Run the TensorFlow training script
python tensorflow_training_script.py \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --data-path /path/to/dataset \
    --model-dir $WORKDIR/models \
    --log-dir $WORKDIR/logs

# Copy results back to home
mkdir -p $HOME/tensorflow_results/$SLURM_JOB_ID
cp -r $WORKDIR/* $HOME/tensorflow_results/$SLURM_JOB_ID/

# Cleanup
cd $HOME
rm -rf $WORKDIR

echo "TensorFlow job completed successfully"