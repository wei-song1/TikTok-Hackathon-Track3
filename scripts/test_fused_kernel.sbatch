#!/bin/bash

#SBATCH --job-name=test-fused-ln
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=xgpf
#SBATCH --gpus=1
#SBATCH --output=test-fused-%j.out
#SBATCH --error=test-fused-%j.err

set -euo pipefail

source /home/w/weisong/venvs/torch-benchmark/bin/activate

PROJECT_DIR=/home/w/weisong/transformer-kernel

cd "$PROJECT_DIR"

export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv

echo "Testing standalone kernel"

srun --ntasks=1 --kill-on-bad-exit=1 \
    python -u test_fused_residual_layernorm.py

echo "Testing Transformer case 1"

srun --ntasks=1 --kill-on-bad-exit=1 \
    python -u torch_transformer_benchmark.py \
    --device cuda \
    --dtype float16 \
    --causal \
    --batch-size 64 \
    --seq-len 128 \
    --d-model 128 \
    --heads 4 \
    --ffn-dim 128 \
    --layers 4 \
    --accuracy-trials 20 \
    --warmup 20 \
    --repeats 100 \
    --benchmark-rounds 3