
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context
import cv2
import math
import Classifier
import time
import GeneralizedPatchMatch
import caffe
import Math_functions as math_func

class parameters:
	def __init__(self):
		self.layers=[] #which layers  used as content
	
		self.patch_size0=0
		self.iter=0
		
#??? Ignore __host__ ???
def norm(dst, src, smooth, dim):
	count=dim.channel*dim.height*dim.width #type  np.int32
	x=src
	x2=cuda.mem_alloc(count*(np.dtype(np.float).itemsize))
	
	math_func.caffe_gpu_mul(count, x, x, x2)
	
	#caculate dis
	sum=cuda.mem_alloc(dim.height*dim.width*(np.dtype(np.float).itemsize))
	ones=cuda.mem_alloc(dim.channel*(np.dtype(np.float).itemsize))
#	caffe_gpu_set(dim.channel, 1.0, ones)
#	caffe_gpu_gemv(CblasTrans, dim.channel, dim.height*dim.width, 1.0, x2, ones, 0.0, sum)
	
	
	
	
	
	
	
	
	


class DeepAnalogy:
	#Construction Method
	def __init__(self):
		self.__resizeRatio=1
		self.__weightLevel = 3
		self.__photoTransfer = False
		self.__file_A = ""
		self.__file_BP = ""
		self.__path_output = ""
		self.__path_model = ""
		
	#Destructor Method???
	
	def SetRatio(self, ratio):
		self.__resizeRatio=ratio
	
	def SetBlendWeight(self, level):
		self.__weightLevel = level;
		
	def UsePhotoTransfer(self, flag):
		self.__photoTransfer = flag
		
	def SetModel(self, path):
		self.__path_model =path
	
	def SetA(self, f_a):
		self.__file_A=f_a	
	def SetBPrime(self, f_bp):
		self.__file_BP = f_bp
	
	def SetOutputDir(self, f_o):
		self.__path_output=f_o
	
	#???
	def SetGPU(self, no):
		if(no!=0):
			cuda.Device(no).make_context()
		
	def LoadInputs(self):
		ori_AL=cv2.imread(self.__file_A) #type numpy.ndarry
		ori_BPL=cv2.imread(self.__file_BP)
		if ori_AL is None or ori_BPL is None:
			print "image cannot read!"
			cv2.waitKey(0);
			sys.exit()
		else:
			self.__ori_A_cols=ori_AL.shape[1]
			self.__ori_A_rows=ori_AL.shape[0]
			self.__ori_BP_cols=ori_BPL.shape[1]
			self.__ori_BP_rows=ori_BPL.shape[0]
#			print ori_AL.shape

		# Geometric Transformations of Images: Transforming the orginal loading image(ori_AL and ori_BPL) to 200<=size<=700
		if ori_AL.shape[0]>700:
			ratio=700.0/ori_AL.shape[0]
			self.__img_AL=cv2.resize(ori_AL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_AL=self.__img_AL.copy() #refer to the different object(np.ndarray)
			
		if ori_AL.shape[1]>700:
			ratio=700.0/ori_AL.shape[1]
			self.__img_AL=cv2.resize(ori_AL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_AL=self.__img_AL.copy()
			
		if ori_AL.shape[0]<200:
			ratio=200.0/ori_AL.shape[0]
			self.__img_AL=cv2.resize(ori_AL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_AL=self.__img_AL.copy()
		
		if ori_AL.shape[1]<200:
			ratio=200.0/ori_AL.shape[1]
			self.__img_AL=cv2.resize(ori_AL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_AL=self.__img_AL.copy()
			
		if ori_BPL.shape[0]>700:
			ratio=700.0/ori_BPL.shape[0]
			self.__img_BPL=cv2.resize(ori_BPL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_BPL=self.__img_BPL.copy()
		
		if ori_BPL.shape[1]>700:
			ratio=700.0/ori_BPL.shape[1]
			self.__img_BPL=cv2.resize(ori_BPL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_BPL=self.__img_BPL.copy()
			
		if ori_BPL.shape[0]<200:
			ratio=200.0/ori_BPL.shape[0]
			self.__img_BPL=cv2.resize(ori_BPL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_BPL=self.__img_BPL.copy()
		
		if ori_BPL.shape[1]<200:
			ratio=200.0/ori_BPL.shape[1]
			self.__img_BPL=cv2.resize(ori_BPL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_BPL=self.__img_BPL.copy()
			
		# Geometric Transformations of Images: Transforming the transformed loading image(ori_AL and ori_BPL) to the total area which are less than 350000
		if (ori_AL.shape[0]*ori_AL.shape[1])>350000:
			ratio=math.sqrt(350000.0/(ori_AL.shape[0]*ori_AL.shape[1]))
			self.__img_AL=cv2.resize(ori_AL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_AL=self.__img_AL.copy()
			
		if (ori_BPL.shape[0]*ori_AL.shape[1])>350000:
			ratio=math.sqrt(350000.0/(ori_BPL.shape[0]*ori_BPL.shape[1]))
			self.__img_BPL=cv2.resize(ori_BPL, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
			ori_BPL=self.__img_BPL.copy()
			
		# Check if the images are transformed to the required range
		maxLateral=max(max(ori_AL.shape[0],ori_AL.shape[1]),max(ori_BPL.shape[0],ori_BPL.shape[1]))
		minLateral=min(min(ori_AL.shape[0],ori_AL.shape[1]),min(ori_BPL.shape[0],ori_BPL.shape[1]))
		
		if maxLateral>700 or minLateral<200:
			print "The sizes of images are not permitted. (One side cannot be larger than 700 or smaller than 200 and the area should not be larger than 350000)"
			cv2.waitKey(0)
			sys.exit()
			
		#Recording the number of rows and columns of transformed images	
		cur_A_cols=ori_AL.shape[1]
		cur_A_rows=ori_AL.shape[0]
		cur_BP_cols=ori_BPL.shape[1]
		cur_BP_rows=ori_BPL.shape[0]
		
		#If original images are transformed, notify this change
		if self.__ori_A_cols!=ori_AL.shape[1]:
			print "The input image A has been resized to %d x %d.\n" % (cur_A_cols,cur_A_rows)
		if self.__ori_BP_cols!=ori_BPL.shape[1]:
			print "The input image B prime has been resized to %d x %d.\n" % (cur_BP_cols,cur_BP_rows)
		
		#???
		self.__img_AL=cv2.resize(ori_AL, None, fx=float(cur_A_cols)/ori_AL.shape[1],fy=float(cur_A_rows)/ori_AL.shape[0],interpolation=cv2.INTER_CUBIC)
		self.__img_BPL=cv2.resize(ori_AL, None, fx=float(cur_BP_cols)/ori_BPL.shape[1],fy=float(cur_BP_rows)/ori_BPL.shape[0],interpolation=cv2.INTER_CUBIC)
		
	def ComputeAnn(self):
		if self.__img_BPL is None or self.__img_AL is None:
			cv2.waitKey(0)
			sys.exit()
			
		#???
		param_size=8
		
		params_host=np.empty(param_size,dtype=np.int)

		params=parameters()
		params.layers.append("conv5_1")
		params.layers.append("conv4_1")
		params.layers.append("conv3_1")
		params.layers.append("conv2_1")
		params.layers.append("conv1_1")
		params.layers.append("data")
		
		weight=[]
		weight.append(1.0)
		if self.__weightLevel==1:
			weight.append(0.7)
			weight.append(0.6)
			weight.append(0.5)
			weight.append(0.0)
		elif self.__weightLevel==2:
			weight.append(0.8)
			weight.append(0.7)
			weight.append(0.6)
			weight.append(0.1)
		elif self.__weightLevel==3:
			weight.append(0.9)
			weight.append(0.8)
			weight.append(0.7)
			weight.append(0.2)
		else:
			weight.append(0.9)
			weight.append(0.8)
			weight.append(0.7)
			weight.append(0.2)
		
		weight.append(0.0)
		
		sizes=[]
		sizes.append(3)
		sizes.append(3)
		sizes.append(3)
		sizes.append(5)
		sizes.append(5)
		sizes.append(3)
		
		params.iter=10
		
		#scale and enhance
		ratio=self.__resizeRatio
		img_A=cv2.resize(self.__img_AL, None, fx=ratio, fy=ratio,interpolation=cv2.INTER_CUBIC)
		img_BP=cv2.resize(self.__img_BPL, None, fx=ratio, fy=ratio,interpolation=cv2.INTER_CUBIC)
		
		#???
		ranges=[]
		if img_A.shape[1]>img_A.shape[0]:
			ranges.append(img_A.shape[1]/16)
		else:
			ranges.append(img_A.shape[0]/16)
			
		ranges.append(6)
		ranges.append(6)
		ranges.append(4)
		ranges.append(4)
		ranges.append(2)
		
		#load caffe
		#???
		#::google::InitGoogleLogging("deepanalogy")
		model_file = "vgg19/VGG_ILSVRC_19_layers_deploy.prototxt"
		trained_file = "vgg19/VGG_ILSVRC_19_layers.caffemodel"
		
		classifier_A=Classifier.Classifier(self.__path_model + model_file, self.__path_model + trained_file)
		classifier_B=Classifier.Classifier(self.__path_model + model_file, self.__path_model + trained_file)
		
		data_A=[]
		data_AP=[]
		data_A_size=[]		
		
		print "The shape of img_A: "
		print img_A.shape #(256, 342, 3)
		classifier_A.Predict(img_A, params.layers, data_AP, data_A, data_A_size)	# type(img_A) numpy.ndarray
			
		data_B=[]
		data_BP=[]
		data_B_size=[]
		classifier_B.Predict(img_BP, params.layers, data_B, data_BP, data_B_size)

		start=time.clock()
		
		ann_size_AB=self.__img_AL.shape[0]*self.__img_AL.shape[1]
		ann_size_BA=self.__img_BPL.shape[0]*self.__img_BPL.shape[1]
		
		ann_host_AB=np.empty(ann_size_AB,dtype=np.uint)
		annd_host_AB=np.empty(ann_size_AB,dtype=np.float)
		ann_host_BA=np.empty(ann_size_BA,dtype=np.uint)
		annd_host_BA=np.empty(ann_size_BA,dtype=np.float)
		
		
		params_device_AB=cuda.mem_alloc(param_size*(np.dtype(np.int).itemsize))
		params_device_BA=cuda.mem_alloc(param_size*(np.dtype(np.int).itemsize))
		ann_device_AB=cuda.mem_alloc(ann_size_AB*(np.dtype(np.uint).itemsize))
		annd_device_AB=cuda.mem_alloc(ann_size_AB*(np.dtype(np.float).itemsize))
		ann_device_BA=cuda.mem_alloc(ann_size_BA*(np.dtype(np.uint).itemsize))
		annd_device_BA=cuda.mem_alloc(ann_size_BA*(np.dtype(np.float).itemsize))
		
		numlayer=len(params.layers)
		
		#feature match
		for curr_layer in range(numlayer-1):
			#set parameters
			params_host[0]=data_A_size[curr_layer].channel
			params_host[1]=data_A_size[curr_layer].height
			params_host[2]=data_A_size[curr_layer].width
			params_host[3]=data_B_size[curr_layer].height
			params_host[4]=data_B_size[curr_layer].width
			params_host[5]=sizes[curr_layer]
			params_host[6]=params.iter
			params_host[7]=ranges[curr_layer]
			
			#copy to device
			cuda.memcpy_htod(params_device_AB, params_host)
			
			#set parameters
			params_host[0]=data_B_size[curr_layer].channel
			params_host[1]=data_B_size[curr_layer].height
			params_host[2]=data_B_size[curr_layer].width
			params_host[3]=data_A_size[curr_layer].height
			params_host[4]=data_A_size[curr_layer].width
			
			#copy to device
			cuda.memcpy_htod(params_device_BA, params_host)			
			#set device pa, device pb, device ann and device annd
			blocksPerGridAB=(data_A_size[curr_layer].width / 20 + 1, data_A_size[curr_layer].height / 20 + 1, 1)
			threadsPerBlockAB=(20, 20, 1)
			ann_size_AB = data_A_size[curr_layer].width* data_A_size[curr_layer].height
			blocksPerGridBA=(data_B_size[curr_layer].width / 20 + 1, data_B_size[curr_layer].height / 20 + 1, 1)
			threadsPerBlockBA=(20, 20, 1)
			ann_size_BA = data_B_size[curr_layer].width* data_B_size[curr_layer].height
			
			mod=SourceModule(GeneralizedPatchMatch.GeneralizedPatchMatch_cu)
			#initialize ann if needed
			if curr_layer==0:
				initialAnn_kernel=mod.get_function('initialAnn_kernel')
				initialAnn_kernel(ann_device_AB, params_device_AB, block=threadsPerBlockAB,grid=threadsPerBlockAB)
				initialAnn_kernel(ann_device_BA, params_device_BA, block=threadsPerBlockBA,grid=threadsPerBlockBA)
			else:
			#upsampling, notice this block's dimension is twice the ann at this point
				ann_tmp=cuda.mem_alloc(ann_size_AB*(np.dtype(np.uint).itemsize))
					
				upSample_kernel=mod.get_function('upSample_kernel')
				print data_A_size[curr_layer - 1].width
				print type(data_A_size[curr_layer - 1].width)
				print data_A_size[curr_layer - 1].height
				print type(data_A_size[curr_layer - 1].height)
				#get new ann_device
				upSample_kernel(ann_device_AB, ann_tmp, params_device_AB, data_A_size[curr_layer - 1].width, data_A_size[curr_layer - 1].height, block=threadsPerBlockAB,grid=blocksPerGridAB)
				cuda.memcpy_dtod(ann_device_AB, ann_tmp, ann_size_AB * np.dtype(np.uint).itemsize)
				ann_tmp.free()
				
				ann_tmp=cuda.mem_alloc(ann_size_BA*(np.dtype(np.uint).itemsize))
				upSample_kernel(ann_device_BA, ann_tmp, params_device_BA, data_B_size[curr_layer - 1].width, data_B_size[curr_layer - 1].height, block=threadsPerBlockBA,grid=blocksPerGridBA)
				cuda.memcpy_dtod(ann_device_BA, ann_tmp, ann_size_BA * np.dtype(np.uint).itemsize)
				ann_tmp.free()
				
			#normarlize two data
			Ndata_A=cuda.mem_alloc(data_A_size[curr_layer].channel*data_A_size[curr_layer].width*data_A_size[curr_layer].height*(np.dtype(np.float).itemsize))
			Ndata_AP=cuda.mem_alloc(data_A_size[curr_layer].channel*data_A_size[curr_layer].width*data_A_size[curr_layer].height*(np.dtype(np.float).itemsize))
			response_A=cuda.mem_alloc(data_A_size[curr_layer].width*data_A_size[curr_layer].height*(np.dtype(np.float32).itemsize))
			Ndata_B=cuda.mem_alloc(data_B_size[curr_layer].channel*data_B_size[curr_layer].width*data_B_size[curr_layer].height*(np.dtype(np.float).itemsize))	
			Ndata_BP=cuda.mem_alloc(data_B_size[curr_layer].channel*data_B_size[curr_layer].width*data_B_size[curr_layer].height*(np.dtype(np.float).itemsize))	
			response_BP=cuda.mem_alloc(data_B_size[curr_layer].width*data_B_size[curr_layer].height*(np.dtype(np.float).itemsize))
				
			norm(Ndata_A, data_A[curr_layer], response_A, data_A_size[curr_layer])		
				
				
				
			
			
			
			
