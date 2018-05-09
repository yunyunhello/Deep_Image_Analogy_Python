
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

math_functions='''
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void mul_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}
'''

def caffe_gpu_mul(N,a,b,y):
	mod=SourceModule(math_functions)
	mul_kernel=mod.get_function("mul_kernel")
	mul_kernel(N,a,b,y,block=(512,1,1),grid=((10000+512-1)/512,1,1))



