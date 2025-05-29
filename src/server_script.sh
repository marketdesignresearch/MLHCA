#!/bin/bash
#SBATCH --job-name=LSVM_national
#SBATCH --nodes 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2
#SBATCH -t 0-12:00:00
#SBATCH --partition=minion_slow,minion_superslow,minion_fast
#SBATCH --exclude=minion[01-09,12-17,36]




source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlca_dq_clean

cpu_counter=1
echo $workingDir

PYTHONPATH=pwd wandb agent anonymized/SERVER-SRVM-HPO-Biddertype_national_mixed_v1.5/vhvtq057

