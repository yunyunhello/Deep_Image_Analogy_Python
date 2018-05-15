
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
import skcuda.cublas as cublas
import Cv_func
import Deconv

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
	sum=cuda.mem_alloc(dim.height*dim.width*(np.dtype(np.float32).itemsize))
	ones=cuda.mem_alloc(dim.channel*(np.dtype(np.float).itemsize))
	math_func.caffe_gpu_set(dim.channel, np.float32(1.0), ones)
	math_func.caffe_gpu_gemv('t', dim.channel, dim.height*dim.width, 1.0, x2, ones, 0.0, sum)
	
	dis=cuda.mem_alloc(dim.height*dim.width*(np.dtype(np.float).itemsize))
	math_func.caffe_gpu_powx(dim.height*dim.width, sum, np.float32(0.5), dis)
	
	if(smooth!=None):
		minv=np.empty(1,dtype=np.float32)
		maxv=np.empty(1,dtype=np.float32)
		cuda.memcpy_dtod(smooth, sum, dim.height*dim.width*(np.dtype(np.float32).itemsize))
		index=cublas.cublasIsamin(cublas.cublasCreate(),dim.height*dim.width, sum, 1)
		cuda.memcpy_dtoh(minv,int(sum)+index-1) #???
		index=cublas.cublasIsamax(cublas.cublasCreate(),dim.height*dim.width, sum, 1)
		cuda.memcpy_dtoh(maxv,int(sum)+index-1)
		
		math_func.caffe_gpu_add_scalar(dim.height*dim.width, -minv[0], smooth)
		math_func.caffe_gpu_scal(dim.height*dim.width, np.float32(1.0) / (maxv[0] - minv[0]), smooth)
	
	math_func.caffe_gpu_gemm('n', 'n', dim.channel, dim.width*dim.height, 1, np.float32(1.0), ones, dis, np.float32(0.0), x2)
	math_func.caffe_gpu_div(count, src, x2, dst)
	
	x2.free()
	ones.free()
	dis.free()
	sum.free()


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
		
		#print "The shape of img_A: "
		#print img_A.shape #(256, 342, 3)
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
			
			mod=SourceModule(GeneralizedPatchMatch.GeneralizedPatchMatch_cu,no_extern_c=1)
			#initialize ann if needed
			if curr_layer==0:
				initialAnn_kernel=mod.get_function('initialAnn_kernel')
				initialAnn_kernel(ann_device_AB, params_device_AB, block=threadsPerBlockAB,grid=threadsPerBlockAB)
				initialAnn_kernel(ann_device_BA, params_device_BA, block=threadsPerBlockBA,grid=threadsPerBlockBA)
			else:
			#upsampling, notice this block's dimension is twice the ann at this point
				ann_tmp=cuda.mem_alloc(ann_size_AB*(np.dtype(np.uint).itemsize))
					
				upSample_kernel=mod.get_function('upSample_kernel')
				#print data_A_size[curr_layer - 1].width
				#print type(data_A_size[curr_layer - 1].width)
				#print data_A_size[curr_layer - 1].height
				#print type(data_A_size[curr_layer - 1].height)
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
			norm(Ndata_BP, data_BP[curr_layer], response_BP, data_B_size[curr_layer])
	
			temp1=cv2.resize(self.__img_AL,(data_A_size[curr_layer].width, data_A_size[curr_layer].height))	
			temp2=cv2.resize(self.__img_BPL,(data_B_size[curr_layer].width, data_B_size[curr_layer].height))	
			
			response1=np.ndarray(shape=(temp1.shape[0],temp1.shape[1]),dtype=np.float32)
			response2=np.ndarray(shape=(temp2.shape[0],temp2.shape[1]),dtype=np.float32)
			
			cuda.memcpy_dtoh(response1, response_A)
			cuda.memcpy_dtoh(response2,response_BP)
			
			response_byte1=Cv_func.convertTo(response1,np.uint8,255)
			response_byte2=Cv_func.convertTo(response2,np.uint8,255)
			
			blend=mod.get_function('blend')
			blend(response_A, data_A[curr_layer], data_AP[curr_layer], np.float32(weight[curr_layer]), params_device_AB, block=threadsPerBlockAB, grid=blocksPerGridAB)
			blend(response_BP, data_BP[curr_layer], data_B[curr_layer], np.float32(weight[curr_layer]), params_device_BA, block=threadsPerBlockBA, grid=blocksPerGridBA)
			
			norm(Ndata_AP, data_AP[curr_layer], None, data_A_size[curr_layer])
			norm(Ndata_B, data_B[curr_layer], None, data_B_size[curr_layer])
			
			#patchmatch
			print "Finding nearest neighbor field using PatchMatch Algorithm at layer: %s" % params.layers[curr_layer]
			patchmatch=mod.get_function('patchmatch')
			patchmatch(Ndata_AP, Ndata_BP, Ndata_A, Ndata_B, ann_device_AB, annd_device_AB, params_device_AB, block=threadsPerBlockAB, grid=blocksPerGridAB)
			patchmatch(Ndata_B, Ndata_A, Ndata_BP, Ndata_AP, ann_device_BA, annd_device_BA, params_device_BA, block=threadsPerBlockBA, grid=blocksPerGridBA)
			
			Ndata_A.free()
			Ndata_AP.free()
			Ndata_B.free()
			Ndata_BP.free()
			response_A.free()
			response_BP.free()
			
			#deconv
			if curr_layer < numlayer - 2:
				next_layer = curr_layer + 2
				
				#upsample
				#for better deconvolution
				params_host[0] = data_A_size[next_layer].channel
				params_host[1] = data_A_size[next_layer].height
				params_host[2] = data_A_size[next_layer].width
				params_host[3] = data_B_size[next_layer].height
				params_host[4] = data_B_size[next_layer].width
				params_host[5] = sizes[next_layer]
				params_host[6] = params.iter
				params_host[7] = ranges[next_layer]
				
				#copy to device
				cuda.memcpy_htod(params_device_AB, params_host)
				
				#set parameters
				params_host[0] = data_B_size[next_layer].channel
				params_host[1] = data_B_size[next_layer].height
				params_host[2] = data_B_size[next_layer].width
				params_host[3] = data_A_size[next_layer].height
				params_host[4] = data_A_size[next_layer].width
				
				#copy to device
				cuda.memcpy_htod(params_device_BA, params_host)
				
				#set device pa, pb, device ann and device annd
				blocksPerGridAB=(data_A_size[next_layer].width / 20 + 1, data_A_size[next_layer].height / 20 + 1, 1)
				threadsPerBlockAB=(20,20,1)
				ann_size_AB = data_A_size[next_layer].width* data_A_size[next_layer].height
				blocksPerGridSC=(data_B_size[next_layer].width / 20 + 1, data_B_size[next_layer].height / 20 + 1, 1)
				threadsPerBlockBA=(20,20,1)
				ann_size_BA = data_B_size[next_layer].width* data_B_size[next_layer].height
				
				ann_tmp=cuda.mem_alloc(ann_size_AB*(np.dtype(np.uint).itemsize))
				upSample_kernel=mod.get_function('upSample_kernel')
				upSample_kernel(ann_device_AB, ann_tmp, params_device_AB, data_A_size[curr_layer].width, data_A_size[curr_layer].height, block=threadsPerBlockAB, grid=blocksPerGridAB) #get new ann_devices
				avg_vote=mod.get_function('avg_vote')
				avg_vote(ann_tmp, data_BP[next_layer], data_AP[next_layer], params_device_AB, block=threadsPerBlockAB, grid=blocksPerGridAB)
				ann_tmp.free()
				
				ann_tmp=cuda.mem_alloc(ann_size_BA*(np.dtype(np.uint).itemsize))
				upSample_kernel(ann_device_BA, ann_tmp, params_device_BA, data_B_size[curr_layer].width, data_B_size[curr_layer].height, block=threadsPerBlockBA, grid=blocksPerGridBA) #get new ann_devices
				ann_tmp.free()
				
				#set parameters
				params_host[0] = data_A_size[curr_layer].channel;#channels
				params_host[1] = data_A_size[curr_layer].height;
				params_host[2] = data_A_size[curr_layer].width;
				params_host[3] = data_B_size[curr_layer].height;
				params_host[4] = data_B_size[curr_layer].width;
				params_host[5] = sizes[curr_layer];
				params_host[6] = params.iter;
				params_host[7] = ranges[curr_layer];
				
				#copy to device
				cuda.memcpy_htod(params_device_AB, params_host)
				
				#set parameters
				params_host[0] = data_B_size[curr_layer].channel; #channels
				params_host[1] = data_B_size[curr_layer].height;
				params_host[2] = data_B_size[curr_layer].width;
				params_host[3] = data_A_size[curr_layer].height;
				params_host[4] = data_A_size[curr_layer].width;		

				#copy to device
				cuda.memcpy_htod(params_device_BA, params_host)
				
				#set device pa, device pb, device ann and device annd
				blocksPerGridAB=(data_A_size[curr_layer].width / 20 + 1, data_A_size[curr_layer].height / 20 + 1, 1)
				threadsPerBlockAB =(20, 20, 1)
				ann_size_AB = data_A_size[curr_layer].width* data_A_size[curr_layer].height
				blocksPerGridBA = (data_B_size[curr_layer].width / 20 + 1, data_B_size[curr_layer].height / 20 + 1, 1)
				threadsPerBlockBA = (20, 20, 1)
				ann_size_BA = data_B_size[curr_layer].width* data_B_size[curr_layer].height
				
				num1 = data_A_size[curr_layer].channel*data_A_size[curr_layer].width*data_A_size[curr_layer].height
				num2 = data_A_size[next_layer].channel*data_A_size[next_layer].width*data_A_size[next_layer].height
				
				target=cuda.mem_alloc(num1*(np.dtype(np.float32).itemsize))
				avg_vote(ann_device_AB, data_BP[curr_layer], target, params_device_AB,block=threadsPerBlockAB,grid=blocksPerGridAB)
				Deconv.deconv(classifier_A, params.layers[curr_layer], target, data_A_size[curr_layer], params.layers[next_layer], data_AP[next_layer], data_A_size[next_layer])		
				target.free()
				
				num1 = data_B_size[curr_layer].channel*data_B_size[curr_layer].width*data_B_size[curr_layer].height
				num2 = data_B_size[next_layer].channel*data_B_size[next_layer].width*data_B_size[next_layer].height
				target=cuda.mem_alloc(num1*(np.dtype(np.float32).itemsize))
				avg_vote(ann_device_BA, data_A[curr_layer], target, params_device_BA,block=threadsPerBlockBA,grid=blocksPerGridBA)
				Deconv.deconv(classifier_B, params.layers[curr_layer], target, data_B_size[curr_layer], params.layers[next_layer], data_B[next_layer], data_B_size[next_layer])
				target.free()
		
		# upsample
		curr_layer = numlayer - 1
		
		print "HERE"
		# set parameters
		params_host[0] = np.int32(3) # channels
		params_host[1] = self.__img_AL.shape[0]
		params_host[2] = self.__img_AL.shape[1]
		params_host[3] = self.__img_BPL.shape[0]
		params_host[4] = self.__img_BPL.shape[1]
		params_host[5] = sizes[curr_layer]
		params_host[6] = params.iter
		params_host[7] = ranges[curr_layer]
		
		# copy to device
		cuda.memcpy_htod(params_device_AB, params_host)
			
		# set parameters
		params_host[0] = 3 # channels
		params_host[1] = self.__img_BPL.shape[0]
		params_host[2] = self.__img_BPL.shape[1]
		params_host[3] = self.__img_AL.shape[0]
		params_host[4] = self.__img_AL.shape[1]
			
		# copy to device
		cuda.memcpy_htod(params_device_BA, params_host)
			
		#set device pa, device pb, device ann and device annd
		blocksPerGridAB=(self.__img_AL.shape[1] / 20 + 1, self.__img_AL.shape[0] / 20 + 1, 1)
		threadsPerBlockAB=(20, 20, 1)
		ann_size_AB = self.__img_AL.shape[1]* self.__img_AL.shape[0]
		blocksPerGridBA=(self.__img_BPL.shape[1] / 20 + 1, self.__img_BPL.shape[0] / 20 + 1, 1)
		threadsPerBlockBA=(20, 20, 1)
		ann_size_BA = self.__img_BPL.shape[0]* self.__img_BPL.shape[1]
			
		mod=SourceModule(GeneralizedPatchMatch.GeneralizedPatchMatch_cu,no_extern_c=1)
			
		# updample
		ann_tmp=cuda.mem_alloc(ann_size_AB * (np.dtype(np.uint).itemsize))
		upSample_kernel=mod.get_function('upSample_kernel')
		upSample_kernel(ann_device_AB, ann_tmp, params_device_AB, data_A_size[curr_layer - 1].width, data_A_size[curr_layer - 1].height, block=threadsPerBlockAB, grid=blocksPerGridAB) #get new ann_device
		cuda.memcpy_dtod(ann_device_AB, ann_tmp, ann_size_AB * (np.dtype(np.uint).itemsize))
		ann_tmp.free()
			
		ann_tmp=cuda.mem_alloc(ann_size_BA * (np.dtype(np.uint).itemsize))
		upSample_kernel(ann_device_BA, ann_tmp, params_device_BA, data_B_size[curr_layer - 1].width, data_B_size[curr_layer - 1].height, block=threadsPerBlockAB, grid=blocksPerGridAB) #get new ann_device
		cuda.memcpy_dtod(ann_device_BA, ann_tmp, ann_size_BA * (np.dtype(np.uint).itemsize))
		ann_tmp.free()
			
		cuda.memcpy_dtoh(ann_host_AB, ann_device_AB)
		cuda.memcpy_dtoh(ann_host_BA, ann_device_BA)
			
		# free space in device, only need to free pa and pb which are created temporarily image downBAale
		flow = GeneralizedPatchMatch.reconstruct_dflow(self.__img_AL, self.__img_BPL, ann_host_AB, sizes[curr_layer])
		result_AB = GeneralizedPatchMatch.reconstruct_avg(img_AL, img_BPL, ann_host_AB, sizes[curr_layer])
			
		out=cv2.resize(result_AB, None, fx=float(self.__ori_A_cols) / cur_A_cols, fy=float(ori_A_rows) / cur_A_rows, interpolation=cv2.INTER_CUBIC)
		fname="resultAB.png"
		cv2.imwrite(self.__path_output+fname, out)
			
		flow = GeneralizedPatchMatch.reconstruct_dflow(self.__img_BPL, self.__img_AL, ann_host_BA, sizes[curr_layer]);
		result_BA = GeneralizedPatchMatch.reconstruct_avg(self.__img_BPL, self.__img_AL, ann_host_BA, sizes[curr_layer]);

		out=cv2.resize(result_BA, None, fx=float(self.__ori_BP_cols) / cur_BP_cols, fy=float(ori_BP_rows) / cur_BP_rows, interpolation=cv2.INTER_CUBIC)
		fname = "resultBA.png"
		cv2.imwrite(path_output + fname, out)
			
		if (self.__photoTransfer):
			print "Refining photo transfer."
				
			origin_A=Cv_func.convertTo(self.__img_AL, np.float32, 1/255.0)
			origin_B=Cv_func.convertTo(self.__img_BPL, np.float32, 1/255.0)
			res_AB=Cv_func.convertTo(result_AB, np.float32, 1/255.0)
			res_BA=Cv_func.convertTo(result_BA, np.float32, 1/255.0)
				
			print "Unfinished code"			
		
		
		print "Saving flow result."
		
		# save ann
	
		fname = "flowAB.txt"
		print "Unfinished code for saving ann"
		
		params_device_AB.free()
		ann_device_AB.free()
		annd_device_AB.free()
		params_device_BA.free()
		ann_device_BA.free()
		annd_device_BA.free()
		
		for i in range(numlayer):
			data_A[i].free()
			data_BP[i].free()
		
		finish=time.clock()
		duration = (double)(finish - start)
		print "Finished finding ann. Time : %s" % str(duration)
		
		classifier_A.DeleteNet()
		classifier_B.DeleteNet()
					
			
			
			
