#!/bin/bash
#SBATCH -A gpu-wizard #Replace with whatever your GPU project is, if not gpu-wizard
#SBATCH -t 4:00:00 #Change if more time is needed
#SBATCH -J UNet_training
#SBATCH -o JOB_LOG_%x_%J.out
#SBATCH -p ursa
#SBATCH -q gpu #change to gpuwf if using windfall allocation
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 24
#SBATCH --mem=0
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:2

export OMP_NUM_THREADS=24 #must match --cpus-per-task
nvidia-smi
echo "start at $(date)"

source /scratch3/BMC/wrfruc/gge/AI/ai4da/load_ai4da.sh 
srun python3 -u model_training.py

echo "finish at $(date)"
