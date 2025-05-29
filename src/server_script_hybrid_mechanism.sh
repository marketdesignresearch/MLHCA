#!/bin/bash
#SBATCH --job-name=LSVM_national
#SBATCH --nodes 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2
#SBATCH -t 10-00:00:00
## SBATCH -t 00-12:00:00
#SBATCH --partition=minion_superslow,minion_slow
## SBATCH --exclude=minion[01-10,12-28]
## SBATCH --exclude=minion[01-09,12,13,15-20]
#SBATCH --exclude=minion[01-09,12-17,36]




source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlca_dq_clean

cpu_counter=1
file_name=./sim_mlca_dq_hybrid.py
echo $workingDir


PYTHONPATH=pwd python3 $file_name --domain $1 --init_method $2 --cca_dq $3 --seed $4 --new_query_option $5 --forbid_individual_bundles $6
