#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


struct WelfordState {
    float mean;
    float m2;
    int count;
};


__device__ __forceinline__ WelfordState combine_welford(
    const WelfordState left,
    const WelfordState right
) {
    if (left.count == 0) {
        return right;
    }

    if (right.count == 0) {
        return left;
    }

    const int combined_count = left.count + right.count;
    const float delta = right.mean - left.mean;

    WelfordState result;
    result.count = combined_count;

    result.mean = left.mean + delta *
        (static_cast<float>(right.count) /
         static_cast<float>(combined_count));

    result.m2 = left.m2 + right.m2 +
        delta * delta *
        (static_cast<float>(left.count) *
         static_cast<float>(right.count) /
         static_cast<float>(combined_count));

    return result;
}


__device__ __forceinline__ WelfordState warp_reduce_welford(
    WelfordState value
) {
    constexpr unsigned int full_mask = 0xffffffffu;

    for (int offset = 16; offset > 0; offset /= 2) {
        WelfordState other;

        other.mean = __shfl_down_sync(
            full_mask,
            value.mean,
            offset
        );
        other.m2 = __shfl_down_sync(
            full_mask,
            value.m2,
            offset
        );
        other.count = __shfl_down_sync(
            full_mask,
            value.count,
            offset
        );

        value = combine_welford(value, other);
    }

    return value;
}


template <typename scalar_t>
__global__ void fused_residual_layernorm_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ residual_output,
    scalar_t* __restrict__ normalized_output,
    int64_t row_count,
    int hidden_size,
    float eps
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);

    if (row >= row_count) {
        return;
    }

    const int thread_index = threadIdx.x;
    const int lane = thread_index & 31;
    const int warp_index = thread_index >> 5;
    const int warp_count = blockDim.x >> 5;

    const int64_t row_offset = row * hidden_size;

    WelfordState local;
    local.mean = 0.0f;
    local.m2 = 0.0f;
    local.count = 0;

    /*
     * Compute the residual in FP32, round it to the output dtype, and then
     * convert it back to FP32 for LayerNorm statistics.
     *
     * This preserves the baseline operation boundary:
     *
     *     residual = input + update
     *     normalized = LayerNorm(residual)
     */
    for (
        int feature = thread_index;
        feature < hidden_size;
        feature += blockDim.x
    ) {
        const int64_t index = row_offset + feature;

        const float input_value =
            static_cast<float>(input[index]);
        const float update_value =
            static_cast<float>(update[index]);

        const float unrounded_residual =
            input_value + update_value;

        const scalar_t rounded_residual =
            static_cast<scalar_t>(unrounded_residual);

        residual_output[index] = rounded_residual;

        const float value =
            static_cast<float>(rounded_residual);

        local.count += 1;

        const float delta = value - local.mean;
        local.mean += delta / static_cast<float>(local.count);

        const float delta2 = value - local.mean;
        local.m2 += delta * delta2;
    }

    local = warp_reduce_welford(local);

    __shared__ float shared_mean[32];
    __shared__ float shared_m2[32];
    __shared__ int shared_count[32];

    if (lane == 0) {
        shared_mean[warp_index] = local.mean;
        shared_m2[warp_index] = local.m2;
        shared_count[warp_index] = local.count;
    }

    __syncthreads();

    WelfordState block_state;
    block_state.mean = 0.0f;
    block_state.m2 = 0.0f;
    block_state.count = 0;

    if (warp_index == 0) {
        if (lane < warp_count) {
            block_state.mean = shared_mean[lane];
            block_state.m2 = shared_m2[lane];
            block_state.count = shared_count[lane];
        }

        block_state = warp_reduce_welford(block_state);

        if (lane == 0) {
            shared_mean[0] = block_state.mean;

            const float variance =
                block_state.m2 /
                static_cast<float>(block_state.count);

            shared_m2[0] = rsqrtf(variance + eps);
        }
    }

    __syncthreads();

    const float mean = shared_mean[0];
    const float inverse_std = shared_m2[0];

    for (
        int feature = thread_index;
        feature < hidden_size;
        feature += blockDim.x
    ) {
        const int64_t index = row_offset + feature;

        const float value =
            static_cast<float>(residual_output[index]);

        const float gamma =
            static_cast<float>(weight[feature]);

        const float beta =
            static_cast<float>(bias[feature]);

        const float normalized =
            (value - mean) * inverse_std;

        const float affine =
            normalized * gamma + beta;

        normalized_output[index] =
            static_cast<scalar_t>(affine);
    }
}


template <typename scalar_t>
void launch_fused_residual_layernorm(
    const torch::Tensor& input,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& residual_output,
    torch::Tensor& normalized_output,
    int64_t row_count,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads;

    if (hidden_size <= 32) {
        threads = 32;
    } else if (hidden_size <= 128) {
        threads = 128;
    } else {
        threads = 256;
    }

    const dim3 blocks(
        static_cast<unsigned int>(row_count)
    );

    fused_residual_layernorm_kernel<scalar_t>
        <<<blocks, threads, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            update.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            residual_output.data_ptr<scalar_t>(),
            normalized_output.data_ptr<scalar_t>(),
            row_count,
            hidden_size,
            eps
        );
}


std::vector<torch::Tensor> fused_residual_layernorm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
    const int hidden_size =
        static_cast<int>(input.size(-1));

    const int64_t row_count =
        input.numel() / hidden_size;

    auto residual_output = torch::empty_like(input);
    auto normalized_output = torch::empty_like(input);

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "fused_residual_layernorm_cuda",
        [&] {
            launch_fused_residual_layernorm<scalar_t>(
                input,
                update,
                weight,
                bias,
                residual_output,
                normalized_output,
                row_count,
                hidden_size,
                static_cast<float>(eps),
                stream
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        residual_output,
        normalized_output,
    };
}