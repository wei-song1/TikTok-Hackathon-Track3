#!/bin/bash

#SBATCH --job-name=torch-shapes
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=xgpf
#SBATCH --gpus=1
#SBATCH --array=0-12%2
#SBATCH --output=torch-shape-%A-%a.out
#SBATCH --error=torch-shape-%A-%a.err

set -euo pipefail

source /home/w/weisong/venvs/torch-benchmark/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Published test cases 1-13.
BATCH_SIZE=(64 1 4 16 128 10000 64 64   64 64 64 64 64)
D_MODEL=(    128 128 128 128 128 128 32 1024 128 128 128 128 128)
HEADS=(      4   4   4   4   4   4   4  4    1   2   16  4   4)
SEQ_LEN=(    128 128 128 128 128 128 128 128  128 128 128 32  1024)
LAYERS=(     4   4   4   4   4   4   4  4    4   4   4   4   4)
FFN_DIM=(    128 128 128 128 128 128 32 1024 128 128 128 128 128)

I="$SLURM_ARRAY_TASK_ID"
CASE_NUMBER=$((I + 1))

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Published case: $CASE_NUMBER"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv

echo "Configuration:"
echo "  batch=${BATCH_SIZE[$I]}"
echo "  d_model=${D_MODEL[$I]}"
echo "  heads=${HEADS[$I]}"
echo "  seq_len=${SEQ_LEN[$I]}"
echo "  layers=${LAYERS[$I]}"
echo "  ffn_dim=${FFN_DIM[$I]}"

srun --ntasks=1 --kill-on-bad-exit=1 \
    python -u /home/w/weisong/torch_transformer_benchmark.py \
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