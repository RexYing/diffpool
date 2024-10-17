#!/bin/bash

#SBATCH --job-name="diffpool"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=14-00:00:00
#SBATCH --output="stdout.%j.%N"
#SBATCH --ntasks-per-node=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=""
#SBATCH --gres=gpu:1

#module load python/3.11
#python -V

conda activate
conda activate diffpool
which python
#module load cuda/11.8.0
export CUDA_CMLOCAL_ROOT="/mmfs1/cm/local/apps/cuda/libs/current"

nvidia-smi
nvcc --version

date

# ENZYMES - Diffpool
#python testgpu.py
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=1 --num-classes=6 --method=soft-assign

date

