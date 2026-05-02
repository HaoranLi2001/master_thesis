#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p gpua40
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=qwen25_gkd_0426_expd2_b7
#SBATCH --output=/home/ha5083li/Downloads/llm_related/knowledge_distillation_llm/logs/%x_%j.out
#SBATCH --error=/home/ha5083li/Downloads/llm_related/knowledge_distillation_llm/logs/%x_%j.err
#SBATCH -D /home/ha5083li/Downloads/llm_related/knowledge_distillation_llm

set -euo pipefail

PROJECT_DIR=/home/ha5083li/Downloads/llm_related/knowledge_distillation_llm
PY_SCRIPT=${PROJECT_DIR}/ms_gkd.py

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

source /home/ha5083li/miniconda3/etc/profile.d/conda.sh
#conda activate train_env
conda activate swift_env

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_HF=1

# 每个节点只有1张A40，所以 nproc_per_node=1；多节点总进程数由 --nnodes 决定
srun --chdir="${PROJECT_DIR}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=1 \
  --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "'"${PY_SCRIPT}"'" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher_model Qwen/Qwen2.5-Math-7B-Instruct \
  --lmbda 0.5 \
  --train_dataset "'"${PROJECT_DIR}"'/data/Distilled_Data_Qwen14B/0422_OPR_1k_123.json" \
  --val_dataset "'"${PROJECT_DIR}"'/data/MATH_val_1k.json" \
  --output_dir "'"${PROJECT_DIR}"'/week13-output/qwen25_15b_gkd_expd2" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 7 \
  --per_device_eval_batch_size 7 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --max_length 4096 \
  --deepspeed zero3 \
  --attn_impl sdpa   
'
