#!/bin/bash

#SBATCH --job-name=fused-shapes
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=xgpf
#SBATCH --gpus=1
#SBATCH --array=0-12%2
#SBATCH --output=fused-shape-%A-%a.out
#SBATCH --error=fused-shape-%A-%a.err

set -euo pipefail

# Submit from the repository root:
#
#   sbatch scripts/run_fused_shapes.sbatch
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

# Fail clearly if the extension has not been built.
python -c '
import torch
import fused_residual_layernorm_cuda
print("Custom CUDA extension: available")
'

BATCH_SIZE=(64 1 4 16 128 10000 64 64 64 64 64 64 64)
D_MODEL=(128 128 128 128 128 128 32 1024 128 128 128 128 128)
HEADS=(4 4 4 4 4 4 4 4 1 2 16 4 4)
SEQ_LEN=(128 128 128 128 128 128 128 128 128 128 128 32 1024)
LAYERS=(4 4 4 4 4 4 4 4 4 4 4 4 4)
FFN_DIM=(128 128 128 128 128 128 32 1024 128 128 128 128 128)

I="$SLURM_ARRAY_TASK_ID"
CASE_NUMBER=$((I + 1))

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $I"
echo "Published case: $CASE_NUMBER"
echo "Node: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "batch=${BATCH_SIZE[$I]}"
echo "d_model=${D_MODEL[$I]}"
echo "heads=${HEADS[$I]}"
echo "seq_len=${SEQ_LEN[$I]}"
echo "layers=${LAYERS[$I]}"
echo "ffn_dim=${FFN_DIM[$I]}"

nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv

srun --ntasks=1 --kill-on-bad-exit=1 \
    python -u "$PROJECT_DIR/torch_transformer_benchmark.py" \
    --device cuda \
    --dtype float16 \
    --causal \
    --batch-size "${BATCH_SIZE[$I]}" \
    --d-model "${D_MODEL[$I]}" \
    --heads "${HEADS[$I]}" \
    --seq-len "${SEQ_LEN[$I]}" \
    --layers "${LAYERS[$I]}" \
    --ffn-dim "${FFN_DIM[$I]}" \
    --accuracy-trials 5 \
    --warmup 20 \
    --repeats 100 \
    --benchmark-rounds 3