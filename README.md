# Shape-Aware Transformer Kernels for NVIDIA T4

An optimized PyTorch implementation of a causal Transformer benchmark for the **Implement a GPU Kernel for a Transformer Layer** hackathon track. The project combines safe graph-level optimizations with a custom CUDA kernel for fused residual addition and LayerNorm, targeting an NVIDIA Tesla T4 (compute capability 7.5).

Across the 13 runnable published test cases, the final implementation passes the required element-wise accuracy rule and achieves:

- **1.305x arithmetic-mean speedup** across cases
- **1.297x geometric-mean speedup** across cases
- **1.230x aggregate speedup** by combined median latency
- **18.7% lower combined median latency**
- A best observed speedup of **1.537x**

## Project overview

The supplied baseline implements a pre-normalization Transformer with explicit multi-head self-attention, causal masking, FP32 softmax, feed-forward layers, residual connections, and LayerNorm. The optimized implementation preserves its public interface and weights while reducing redundant GPU work.

The main optimizations are:

1. **Packed QKV projection** — combines the query, key, and value projections into one `F.linear` call, reducing kernel-launch overhead. A shape guard falls back to separate projections where packed QKV changes FP16 rounding enough to threaten the accuracy requirement.
2. **Mask fast paths and caching** — caches causal masks and bypasses redundant key/output masking when every token is valid.
3. **Fused residual-add + LayerNorm CUDA kernel** — performs residual addition, Welford statistics, normalization, and affine transformation in one CUDA launch.
4. **Numerically faithful mixed-precision behavior** — the fused kernel rounds the residual to the output dtype before computing LayerNorm statistics, matching the baseline's FP16 operation boundary, while accumulating statistics in FP32.
5. **Shape-aware dispatch** — uses the custom kernel only on tested shapes where it is both accurate and beneficial; all other shapes use the safe native PyTorch path.

The custom residual/LayerNorm kernel is selected for published cases 2, 3, 4, 9, and 12. Other cases still benefit from packed QKV and mask optimizations.

## Repository layout

```text
├── README.md
├── setup.py
├── torch_transformer_benchmark.py
├── fused_residual_layernorm.cpp
├── fused_residual_layernorm_cuda.cu
├── test_fused_residual_layernorm.py
├── scripts/
│   ├── build_fused_kernel.sbatch
│   ├── test_fused_kernel.sbatch
│   └── run_fused_shapes.sbatch
└── results/
```

## Tested environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA Tesla T4, 15 GB |
| Compute capability | 7.5 (`sm_75`) |
| CPU | 2 x Intel Xeon Silver 4116, 2.1 GHz |
| System memory | 256 GB DDR4 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| CUDA toolkit / NVCC | 13.3 |
| Python | 3.12 |
| Compiler | GCC/G++ 13.3 |
| Scheduler | Slurm |

The CUDA 13.3 toolkit produces a minor-version mismatch warning with the CUDA 13.0 version used to build PyTorch. It built and ran successfully in the environment above, but using a toolkit that exactly matches the installed PyTorch build is preferable when available.

## Setup and installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

All commands and Slurm submissions below assume the current directory is the repository root.

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch ninja
```

Install the PyTorch build appropriate for the cluster's driver and CUDA environment if the generic command does not select a compatible build.

### 3. Configure CUDA

For the tested T4 system:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=7.5
```

Confirm the toolchain before building:

```bash
nvidia-smi
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
```

### 4. Build the CUDA extension

Interactively on an allocated GPU node:

```bash
python setup.py build_ext --inplace
python -c 'import torch; import fused_residual_layernorm_cuda; print("Extension import: PASS")'
```

Or submit the provided build job from the repository root:

```bash
sbatch --export=ALL,VENV_PATH="$PWD/.venv" scripts/build_fused_kernel.sbatch
```

The scripts default to `$HOME/venvs/torch-benchmark`. Set `VENV_PATH` as shown above when using another environment. `PROJECT_DIR` defaults to `SLURM_SUBMIT_DIR`, so submitting from the repository root keeps the scripts path-portable.

The included Slurm directives (`--partition=gpu` and `--constraint=xgpf`) are specific to the test cluster. Adjust them for another cluster.

## Reproducing the results

### Recommended Slurm workflow

Submit the build, standalone correctness tests, and 13-case benchmark with job dependencies:

```bash
BUILD_JOB=$(sbatch --parsable \
    --export=ALL,VENV_PATH="$PWD/.venv" \
    scripts/build_fused_kernel.sbatch)

TEST_JOB=$(sbatch --parsable \
    --dependency="afterok:$BUILD_JOB" \
    --export=ALL,VENV_PATH="$PWD/.venv" \
    scripts/test_fused_kernel.sbatch)

SHAPES_JOB=$(sbatch --parsable \
    --dependency="afterok:$TEST_JOB" \
    --export=ALL,VENV_PATH="$PWD/.venv" \
    scripts/run_fused_shapes.sbatch)

echo "build=$BUILD_JOB test=$TEST_JOB shapes=$SHAPES_JOB"
```

Monitor the jobs:

```bash
squeue -j "$BUILD_JOB,$TEST_JOB,$SHAPES_JOB"
```

The array script runs published cases 1–13 in FP16 with five accuracy trials, 20 warm-up iterations, 100 timed repetitions, and three benchmark rounds. When it completes, summarize the logs with:

```bash
grep -H -E 'summary:|baseline :|optimized:|speedup' fused-shape-*.out
```

Every output must satisfy the benchmark's element-wise correctness condition:

```text
absolute_error <= 0.002 OR relative_error <= 2%
```

### Run one case directly

After obtaining an interactive GPU allocation and activating the environment, case 1 can be run with:

```bash
python torch_transformer_benchmark.py \
    --device cuda \
    --dtype float16 \
    --causal \
    --batch-size 64 \
    --seq-len 128 \
    --d-model 128 \
    --heads 4 \
    --ffn-dim 128 \
    --layers 4 \
    --accuracy-trials 5 \
    --warmup 20 \
    --repeats 100 \
    --benchmark-rounds 3
```

## Results

The following measurements were collected on a Tesla T4 using FP16. Latencies are medians measured with CUDA events; random input generation is excluded.

| Case | B | S | D | Heads | Layers | FFN | Accuracy | Baseline (ms) | Optimized (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 64 | 128 | 128 | 4 | 4 | 128 | PASS | 5.5624 | 4.7004 | 1.183x |
| 2 | 1 | 128 | 128 | 4 | 4 | 128 | PASS | 3.9075 | 2.5415 | 1.537x |
| 3 | 4 | 128 | 128 | 4 | 4 | 128 | PASS | 3.9796 | 2.5888 | 1.537x |
| 4 | 16 | 128 | 128 | 4 | 4 | 128 | PASS | 3.9301 | 2.7487 | 1.430x |
| 5 | 128 | 128 | 128 | 4 | 4 | 128 | PASS | 11.5466 | 9.3483 | 1.235x |
| 6 | 10,000 | 128 | 128 | 4 | 4 | 128 | PASS | 897.0486 | 729.7964 | 1.229x |
| 7 | 64 | 128 | 32 | 4 | 4 | 32 | PASS | 4.5089 | 3.7584 | 1.200x |
| 8 | 64 | 128 | 1,024 | 4 | 4 | 1,024 | PASS | 30.0458 | 26.9645 | 1.114x |
| 9 | 64 | 128 | 128 | 1 | 4 | 128 | PASS | 3.5468 | 2.4442 | 1.451x |
| 10 | 64 | 128 | 128 | 2 | 4 | 128 | PASS | 3.9482 | 3.3577 | 1.176x |
| 11 | 64 | 128 | 128 | 16 | 4 | 128 | PASS | 15.6761 | 12.9182 | 1.213x |
| 12 | 64 | 32 | 128 | 4 | 4 | 128 | PASS | 3.9060 | 2.7391 | 1.426x |
| 13 | 64 | 1,024 | 128 | 4 | 4 | 128 | PASS | 232.2404 | 188.0709 | 1.235x |

Small run-to-run changes are expected because of GPU clock state, thermals, node contention, and software versions. Compare implementations on the same node and in the same job whenever possible.

## Correctness testing

`test_fused_residual_layernorm.py` exercises:

- Hidden dimensions 32, 128, and 1,024
- 128, 2,048, and 8,192 rows
- FP16 and FP32
- Multiple random seeds
- Exact residual-output comparison
- LayerNorm comparison using the benchmark's tolerance rule

In addition to the standalone suite, the selected shape-dispatch paths were stress-tested with additional accuracy trials and random seeds. Shapes that produced even rare failures were removed from custom-kernel dispatch and use the native fallback.

## Limitations and future improvements

### Current limitations

- **Hardware specialization:** the supplied build scripts compile for `sm_75`. Other GPU architectures require a different `TORCH_CUDA_ARCH_LIST` and should be retuned.
- **Exact-shape dispatch:** the fastest path is deliberately restricted to validated published shapes. Unseen shapes remain correct through native PyTorch but may not receive the same speedup.
- **Dense attention scaling:** both the supplied baseline and this implementation retain an explicit attention matrix with quadratic memory and compute complexity in sequence length.
- **Published case 14 is not runnable with the supplied dense reference on a 15 GB T4:** its attention scores alone contain `32 × 16 × 100000 × 100000 = 5.12 trillion` FP16 elements, requiring approximately 10.24 TB before temporary buffers. Because the reference also allocates dense scores, candidate accuracy and latency cannot be compared without changing the benchmark methodology.
- **Inference-only custom operation:** the CUDA extension implements the forward path and does not provide a custom backward kernel for training.
- **Partial fusion:** GEMMs, attention softmax, and the FFN still rely on PyTorch/CUDA library kernels. The custom kernel targets only residual addition and LayerNorm.
- **Toolchain sensitivity:** extension build and timing may vary with PyTorch, CUDA, compiler, and GPU-driver versions.

### Given more time

I would:

1. Profile with Nsight Systems, Nsight Compute and Valgrind.
2. Autotune block sizes and dispatch thresholds per GPU instead of using a fixed T4 system.
3. Explore deeper fusion around QKV projection, head reshaping, softmax, output projection, GELU, and FFN residuals.
4. Add backward kernels, automated tests across different GPU architectures, reproducible containers, and CI build checks.
