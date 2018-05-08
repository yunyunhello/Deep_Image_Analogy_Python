
from pycuda.compiler import SourceModule

math_functions='''
template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}
'''

def caffe_gpu_sub(N,a,b,y):
	mod=SourceModule(math_functions)
	mul_kernel=mod.get_function(math_functions)