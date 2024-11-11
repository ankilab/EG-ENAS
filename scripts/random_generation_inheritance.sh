#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --job-name=hpcruns/Testjob_evonas
#SBATCH --export=NONE
#SBATCH --mail-user=mateo.avila@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV
module load python/pytorch-1.13py3.10
#module load cuda/11.6.1
source /home/hpc/iwb3/iwb3021h/NAS_CHALLENGE/NAS_Challenge_AutoML_2024/.testvenv/bin/activate
cd /home/hpc/iwb3/iwb3021h/NAS_CHALLENGE/NAS_Challenge_AutoML_2024
srun python3 -W "ignore" egenas/tests/train_random_gen_inheritance.py 

deactivate