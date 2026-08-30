#!/bin/bash

#SBATCH --job-name=build-fused-ln
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=xgpf
#SBATCH --gpus=1
#SBATCH --output=build-fused-%j.out
#SBATCH --error=build-fused-%j.err

set -euo pipefail

# Submit from the repository root:
#
#   sbatch scripts/build_fused_kernel.sbatch
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
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv

nvcc --version
g++ --version | head -n 1

python -c '
import ninja
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Extension CUDA_HOME:", CUDA_HOME)
print("Ninja:", ninja.__version__)
'

python "$PROJECT_DIR/setup.py" build_ext --inplace

echo "Built extension:"
ls -lh "$PROJECT_DIR"/fused_residual_layernorm_cuda*.so

# Import torch first so libc10.so and other PyTorch libraries are loaded.
python -c '
import torch
import fused_residual_layernorm_cuda
print("Extension import: PASS")
'