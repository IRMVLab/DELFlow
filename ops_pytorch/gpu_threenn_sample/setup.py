from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="no_sort_knn",
    ext_modules=[
        CUDAExtension(
            "no_sort_knn_cuda",
            ["no_sort_knn_g.cpp", "no_sort_knn_go.cu"],
            extra_compile_args={
                "cxx": ['-g'],
                "nvcc": ['-O2']
            })
        
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)