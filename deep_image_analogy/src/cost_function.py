import pycuda.driver as cuda
import Math_functions as math_func
import numpy as np

class cost_function:
	def __init__(self, numDimensions):
		self._m_numDimensions=numDimensions
		
	@abstractmethod 
	def f_gradf(d_x, d_f, d_gradf):
		pass
	
	def getNumberOfUnknowns(self):
		return m_numDimensions
		
class my_cost_function(cost_function):
	def __init__(self, classifier, layer1, d_y, num1, layer2, num2, id1, id2):
		super(my_cost_function,self).__init__(num2)
		self.__m_classifier = classifier
		self.__m_dy = d_y
		self.__m_num1 = num1
		self.__m_num2 = num2
		self.__m_id1 = id1
		self.__m_id2 = id2
		
		self.__m_layer1 = layer1
		self.__m_layer2 = layer2
	
	def f_gradf(self, d_x, d_f, d_gradf):
		self.__m_classifier.net_._forward(self.__m_id2 + 1, m_id1)
		
		src=cuda.to_device(self.__m_classifier.net_.blobs[self.__m_layer1].data)
		diff=src
		math_func.caffe_gpu_sub(self.__m_num1, src, m_dy, diff)
		
		diff2=cuda.mem_alloc(self.__m_num1*(np.dtype(np.float).itemsize))
		math_func.caffe_gpu_mul(m_num1, diff, diff, diff2)
		
		total=math_func.caffe_gpu_asum(m_num1, diff2)
		
		
		self.__m_classifier.net_._backward(self.__m_id1, self.__m_id2 + 1)
		
		diff_gpu=cuda.to_device(self.__m_classifier.net_.blobs[self.__m_layer2].diff)
		cuda.memcpy_dtod(d_gradf,diff_gpu,self.__m_num2*(np.dtype(np.float32).itemsize))
		cuda.memcpy_htod(d_f, total)
		
		diff2.free()
	
		
		