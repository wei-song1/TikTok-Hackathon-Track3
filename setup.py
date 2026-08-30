from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="fused_residual_layernorm_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_residual_layernorm_cuda",
            sources=[
                "fused_residual_layernorm.cpp",
                "fused_residual_layernorm_cuda.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,
        )
    },
)