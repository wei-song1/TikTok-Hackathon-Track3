#!/usr/bin/env python3

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

import fused_residual_layernorm_cuda as fused_layernorm


RTOL = 0.02
ATOL = 0.002


def benchmark_rule(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[bool, int, float]:
    reference_fp32 = reference.float()
    candidate_fp32 = candidate.float()

    absolute_error = (
        candidate_fp32 - reference_fp32
    ).abs()

    passed_elements = (
        (absolute_error <= ATOL)
        | (absolute_error <= RTOL * reference_fp32.abs())
    )

    failed = int((~passed_elements).sum().item())
    maximum_absolute_error = float(
        absolute_error.max().item()
    )

    return (
        failed == 0,
        failed,
        maximum_absolute_error,
    )


def run_case(
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")

    input_tensor = torch.randn(
        rows,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    update = torch.randn(
        rows,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    weight = (
        torch.ones(
            hidden_size,
            device=device,
            dtype=dtype,
        )
        + 0.1
        * torch.randn(
            hidden_size,
            device=device,
            dtype=dtype,
        )
    )

    bias = (
        0.1
        * torch.randn(
            hidden_size,
            device=device,
            dtype=dtype,
        )
    )

    eps = 1e-5

    with torch.inference_mode():
        reference_residual = input_tensor + update

        reference_normalized = F.layer_norm(
            reference_residual,
            (hidden_size,),
            weight,
            bias,
            eps,
        )

        candidate_residual, candidate_normalized = (
            fused_layernorm.forward(
                input_tensor.contiguous(),
                update.contiguous(),
                weight.contiguous(),
                bias.contiguous(),
                eps,
            )
        )

    residual_exact = torch.equal(
        reference_residual,
        candidate_residual,
    )

    normalized_passed, failed, max_abs = benchmark_rule(
        reference_normalized,
        candidate_normalized,
    )

    status = (
        "PASS"
        if residual_exact and normalized_passed
        else "FAIL"
    )

    print(
        f"{status} | "
        f"dtype={dtype} | "
        f"rows={rows} | "
        f"hidden={hidden_size} | "
        f"seed={seed} | "
        f"residual_exact={residual_exact} | "
        f"norm_failed={failed} | "
        f"norm_max_abs={max_abs:.8g}"
    )

    if not residual_exact:
        raise AssertionError(
            "Residual output is not bit-exact"
        )

    if not normalized_passed:
        raise AssertionError(
            f"LayerNorm failed for {failed} elements"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)

    hidden_sizes = (32, 128, 1024)
    row_counts = (128, 2048, 8192)
    dtypes = (torch.float16, torch.float32)
    seeds = (1234, 5678, 9876)

    for hidden_size, rows, dtype, seed in itertools.product(
        hidden_sizes,
        row_counts,
        dtypes,
        seeds,
    ):
        run_case(
            rows=rows,
            hidden_size=hidden_size,
            dtype=dtype,
            seed=seed,
        )

    torch.cuda.synchronize()
    print("All standalone fused LayerNorm tests passed.")


if __name__ == "__main__":
    main()