
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import skcuda.cublas as cublas

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

__global__ void powx_kernel(const int n, const float* a,
    const float alpha, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
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
	
def caffe_gpu_gemv(trans, M, N, alpha, A, x, beta, y):
	cuTransA='t' if TransA=='n' else 'n'
	h=cublas.cublasCreate()
	cublas.cublasSgemv(h,cuTransA,N,M,alpha,A,N,x,1,beta,y,1)

def caffe_gpu_powx(n, a, b, y):
	mod=SourceModule(math_functions)
	powx_kernel=mod.get_function('powx_kernel')
	powx_kernel(N,a,alpha,y,block=(512,1,1),grid=((N+512-1)/512,1,1))
	
	

	
