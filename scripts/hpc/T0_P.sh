#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:59:00
#SBATCH --partition=a100
#SBATCH --job-name=/hpcruns/egenas
#SBATCH --export=NONE

#Low cost EG-ENAS
# fisher + jacob_cov zero-cost proxies based augmentation selection + Random Forest regressor for best model estimation
unset SLURM_EXPORT_ENV
module load python/pytorch-1.13py3.10
source .venv/bin/activate
make -f Makefile save_folder=EGENAS \
        submission=egenas \
        mode=T0 \
        augment=Proxy \
        seed=1 \
        all

deactivate