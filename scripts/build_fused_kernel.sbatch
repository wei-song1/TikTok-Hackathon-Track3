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

source /home/w/weisong/venvs/torch-benchmark/bin/activate

PROJECT_DIR=/home/w/weisong/transformer-kernel

cd "$PROJECT_DIR"

export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

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

python setup.py build_ext --inplace

echo "Built extension:"
ls -lh fused_residual_layernorm_cuda*.so

python -c '
import fused_residual_layernorm_cuda
print("Extension import: PASS")
'