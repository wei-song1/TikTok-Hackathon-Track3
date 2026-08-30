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

# Submit from the repository root:
#
#   sbatch scripts/test_fused_kernel.sbatch
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/torch-benchmark}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Virtual environment not found: $VENV_PATH" >&2
    echo "Set VENV_PATH to the desired virtual environment." >&2
    exit 1
fi

source "$VENV_PATH/bin/activate"
cd "$PROJECT_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export MAX_JOBS="${MAX_JOBS:-$SLURM_CPUS_PER_TASK}"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv

# Fail clearly if the extension has not been built.
python -c '
import torch
import fused_residual_layernorm_cuda
print("Custom CUDA extension: available")
'

echo "Testing standalone kernel"

srun --ntasks=1 --kill-on-bad-exit=1 \
    python -u "$PROJECT_DIR/test_fused_residual_layernorm.py"

echo "Testing Transformer case 1"

srun --ntasks=1 --kill-on-bad-exit=1 \
    python -u "$PROJECT_DIR/torch_transformer_benchmark.py" \
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