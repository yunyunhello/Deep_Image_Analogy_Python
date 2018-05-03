
import numpy as np
import caffe
import glog as log
import Structure  

class Classifier:
	def __init__(self,model_file,trained_file):
		
		self.input_geometry_=Structure.Size()
		self.num_channels_=0
		#???
		#caffe.set_device(0)
		caffe.set_mode_gpu()
	
		#Load the network
		self.net_=caffe.Net(model_file, trained_file, caffe.TEST)
		#???
		print "Enter Clssifier.py\n"
		print self.net_.blobs['data'].data.shape #(1,3,224,224)
		log.check_eq(len(self.net_.inputs),1,"Network should have exactly one input.")
		log.check_eq(len(self.net_.outputs),1,"Network should have exactly one output.")
		self.num_channels_=self.net_.blobs['data'].shape[1]
		print "self.num_channels:"
		print self.num_channels_
		log.check(self.num_channels_==3 or self.num_channels_==1, "Input layer should have 1 or 3 channels.")
		#???
		self.mean_=[103.939, 116.779, 123.68]
		#print type(self.mean_)

	# Load the mean file in binaryproto format. 
		
	def Predict(self, img, layers, data_s, data_d, size):
		self.input_geometry_.width=img.shape[1]
		self.input_geometry_.height=img.shape[0]
		
		input_layer=self.net_.blobs['data']
		print "net_.blobs['data'].shape:"
		print input_layer.shape #<caffe._caffe.IntVec object at 0x7f4c3fa45f50>
		input_layer.reshape(1,self.num_channels_,self.input_geometry_.height,self.input_geometry_.width)
	        print "net_.blobs['data'].data.shape:" 
		print input_layer.data.shape #(1, 3, 256, 342)
		
		#Forward dimension change to all layers.
		self.net_.reshape()
		
		input_channels=[]
		self.WrapInputLayer(input_channels)
		#Preprocess(img,input_channels)
	

	"""Wrap the input layer of the network in separate np.ndarray(one per channel). This way we save 
	one memcpy operation and we don't need to rely on cudaMemcpy2D. The last preprocessing operation 
	will write the separate channels directly to the input layer."""	
	def WrapInputLayer(self,input_channels):
		
		input_layer=self.net_.blobs['data'].data #blob???
		print "The type of  net_.blobs['data']:"
		print type(self.net_.blobs['data']) #<class 'caffe._caffe.Blob'>
		print type(self.net_.blobs['data'].data) #<type 'numpy.ndarray'>
		width = input_layer.shape[3]
		height= input_layer.shape[2]
		input_data= input_layer.data #???mutable_cpu_type???<type 'numpy.ndarray'>
		print "The type of input_data: "
		print type(input_data) #??? <type 'numpy.ndarray'>
		#for(i=0;i<input_layer.shape[1],i++)
			#channel=np.ndaaray(shape=(height,width),dtype="float32")
			
