#!/bin/bash
#SBATCH --job-name=FC_RNN_GRU_GPU_0_HDBA_yyyymmdd_example
#SBATCH --account=project_xxxxxxx
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,FAIL,END #uncomment to enable mail

module load pytorch/2.1
pip install --user torcheval

srun python3 artisTrain.py -p /scratch/project_xxxxxxx/Models/PAR/FC_RNN_GRU_GPU_0_HDBA_yyyymmdd_example.txt