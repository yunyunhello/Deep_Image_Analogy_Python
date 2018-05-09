
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

math_functions='''
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
	   
template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}
'''

a=gpuarray.to_gpu(numpy.ones(shape=(100,100),dtype=ny.float32))
result=cuda.mem_alloc(100*100*(np.dtype(np.float32).itemsize))

mod=SourceModule(math_functions)
mul_kernel=mod.get_function(math_functions)
mul_kernel=(result,block=512,grid=(N+512-1)/512)


"""
def caffe_gpu_sub(N,a,b,y):
	mod=SourceModule(math_functions)
	mul_kernel=mod.get_function(math_functions)
"""