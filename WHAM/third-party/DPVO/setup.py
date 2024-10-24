import os.path as osp
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))


if 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

setup(
    name='dpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': ['-O3'],
            }),
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'dpvo/lietorch/include'),
                osp.join(ROOT, 'third-party/eigen-3.4.0')],
            sources=[
                'dpvo/lietorch/src/lietorch.cpp',
                'dpvo/lietorch/src/lietorch_gpu.cu',
                'dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

