#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> fused_residual_layernorm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
);


std::vector<torch::Tensor> fused_residual_layernorm_forward(
    const torch::Tensor& input,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(update.is_cuda(), "update must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(update.is_contiguous(), "update must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(
        input.scalar_type() == torch::kFloat16 ||
        input.scalar_type() == torch::kFloat32,
        "only float16 and float32 are supported"
    );

    TORCH_CHECK(
        input.scalar_type() == update.scalar_type(),
        "input and update must have the same dtype"
    );
    TORCH_CHECK(
        input.scalar_type() == weight.scalar_type(),
        "input and weight must have the same dtype"
    );
    TORCH_CHECK(
        input.scalar_type() == bias.scalar_type(),
        "input and bias must have the same dtype"
    );

    TORCH_CHECK(
        input.sizes() == update.sizes(),
        "input and update must have the same shape"
    );
    TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
    TORCH_CHECK(bias.dim() == 1, "bias must be one-dimensional");

    const auto hidden_size = input.size(-1);

    TORCH_CHECK(
        weight.numel() == hidden_size,
        "weight size must equal the final input dimension"
    );
    TORCH_CHECK(
        bias.numel() == hidden_size,
        "bias size must equal the final input dimension"
    );

    TORCH_CHECK(
        input.device() == update.device() &&
        input.device() == weight.device() &&
        input.device() == bias.device(),
        "all tensors must be on the same CUDA device"
    );

    TORCH_CHECK(eps > 0.0, "eps must be positive");

    return fused_residual_layernorm_cuda(
        input,
        update,
        weight,
        bias,
        eps
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "forward",
        &fused_residual_layernorm_forward,
        "Fused residual addition and LayerNorm"
    );
}