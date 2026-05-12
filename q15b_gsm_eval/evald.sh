#!/bin/bash
#SBATCH -A naiss2026-4-815
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

#SBATCH --job-name=0509val_q15b_gsm_expd_ep100_prompt1_1024
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ha5083li-s@student.lu.se

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1

source /mimer/NOBACKUP/groups/naiss2026-4-815/HaoranLi/env/vllm_eval/bin/activate

VLLM_WORKER_MULTIPROC_METHOD=spawn python run.py 0506_gsm_exp_newprompt/d_eval.py -a vllm -r latest --debug
