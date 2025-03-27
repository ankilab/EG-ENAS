#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:59:00
#SBATCH --partition=a100
#SBATCH --job-name=/home/woody/iwb3/iwb3021h/EGENAS_RESULTS/hpcruns/evonas
#SBATCH --export=NONE
#SBATCH --mail-user=mateo.avila@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV
module load python/pytorch-1.13py3.10
#module load cuda/11.6.1
source .venv/bin/activate
#cd /home/hpc/iwb3/iwb3021h/NAS_CHALLENGE/NAS_Challenge_AutoML_2024
make -f Makefile save_folder=/home/woody/iwb3/iwb3021h/EGENAS_RESULTS \
        submission=egenas \
        mode=T0+ \
        augment=Proxy \
        seed=4 \
        all

deactivate