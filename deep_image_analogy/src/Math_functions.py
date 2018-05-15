
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

__global__ void sub_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
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

__global__ void add_scalar_kernel(const int n, const float alpha, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

__global__ void div_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
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
	
def caffe_gpu_gemv(TransA, M, N, alpha, A, x, beta, y):
	h=cublas.cublasCreate()
	cuTransA='t' if TransA=='n' else 'n'
	cublas.cublasSgemv(h,cuTransA,N,M,alpha,A,N,x,1,beta,y,1)

def caffe_gpu_powx(N, a, alpha, y):
	mod=SourceModule(math_functions)
	powx_kernel=mod.get_function('powx_kernel')
	powx_kernel(N,a,alpha,y,block=(512,1,1),grid=((N+512-1)/512,1,1))
	
def caffe_gpu_add_scalar(N, alpha, Y):
	mod=SourceModule(math_functions)
	add_scalar_kernel=mod.get_function('add_scalar_kernel')
	add_scalar_kernel(N,alpha,Y,block=(512,1,1),grid=((N+512-1)/512,1,1))
	
def caffe_gpu_scal(N, alpha, X):
	cublas.cublasSscal(cublas.cublasCreate(),N, alpha, X, 1)
	
def caffe_gpu_gemm(TransA, TransB, M, N, K, alpha, A, B, beta, C):
	lda=K if TransA =='n' else M
	ldb=N if TransB =='n' else K
	cuTransA='n' if TransA=='n' else 't'
	cuTransB='n' if TransB=='n' else 't'
	cublas.cublasSgemm(cublas.cublasCreate(),cuTransB, cuTransA, N, M, K, alpha, B, ldb, A, lda, beta, C, N)

def caffe_gpu_div(N, a, b, y):
	mod=SourceModule(math_functions)
	div_kernel=mod.get_function('div_kernel')
	div_kernel(N, a, b, y, block=(512,1,1),grid=((N+512-1)/512,1,1))
	
def caffe_gpu_sub(N, a, b, y): 	
	mod=SourceModule(math_functions)
	sub_kernel=mod.get_function('sub_kernel')
	sub_kernel(N, a, b, y, block=(512,1,1),grid=((N+512-1)/512,1,1))
	
def caffe_gpu_asum(n, x):
	return cublas.cublasSasum(cublas.cublasCreate(), n, x, 1)





