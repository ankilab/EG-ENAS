#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:59:00
#SBATCH --partition=a100
#SBATCH --job-name=/hpcruns/egenas
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python/pytorch-1.13py3.10
#module load cuda/11.6.1
source .venv/bin/activate
make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T0+ \
        augment=Proxy \
        seed=1 \
        all

deactivate