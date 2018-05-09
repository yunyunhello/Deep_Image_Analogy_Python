
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

__global__ void set_kernel(const int n, const float alpha, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}
'''

def caffe_gpu_mul(N,a,b,y):
	mod=SourceModule(math_functions)
	mul_kernel=mod.get_function("mul_kernel")
	mul_kernel(N,a,b,y,block=(512,1,1),grid=((N+512-1)/512,1,1))

def caffe_gpu_set(N, alpha, Y):
	if(alpha==0):
		cuda.memset_d32(Y,0,int(N))
		return
	mod=SourceModule(math_functions)
	set_kernel=mod.get_function('set_kernel')
	set_kernel(N,alpha,Y,block=(512,1,1),grid=((N+512-1)/512,1,1))

	
