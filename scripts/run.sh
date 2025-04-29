#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=/network/scratch/t/theo.saulus/coinche/slurm/slurm-%A_%a.out

# Example usage:
# sbatch --partition=long scripts/run.sh

# module purge
module load miniconda/3
module load cuda/11.8
conda activate coinche

# export WANDB_MODE=offline
# export WANDB_DIR=$SCRATCH/coinche-params/$GROUP_NAME/jobs/$SLURM_ARRAY_JOB_ID
# mkdir -p $WANDB_DIR

# Run the training script
# SEED=$(($SLURM_ARRAY_TASK_ID + $i))
# echo "Running for seed: $SEED"

python train.py --config train/config.yaml