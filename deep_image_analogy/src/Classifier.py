
import numpy as np
import caffe
import glog as log
import Structure  
import cv2

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
		print "self.num_channels_:"
		print self.num_channels_
		log.check(self.num_channels_==3 or self.num_channels_==1, "Input layer should have 1 or 3 channels.")
		#???
		self.mean_=(103.939, 116.779, 123.68) #type tuple
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
		
		self.Preprocess(img)
		
		self.net_.Forward()
		
		print self.net_._blob_names
		print self.net_._layer_names
		#for i in range(len(layers)):
			
		
		
	

	"""Wrap the input layer of the network in separate np.ndarray(one per channel). This way we save 
	one memcpy operation and we don't need to rely on cudaMemcpy2D. The last preprocessing operation 
	will write the separate channels directly to the input layer."""	

	# Convert the input image to the input image format of the network. 
	def Preprocess(self,img):
		if img.shape[2]==3 and self.num_channels_==1:
			sample=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		elif img.shape[2]==4 and self.num_channels_==1:
			sample=cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
		elif img.shape[2]==4 and self.num_channels_==3:
			sample=cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
		elif img.shape[2]==1 and self.num_channels_==3:
			sample=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		else:
			sample=img
		print "The shape of img:"
		print  img.shape #(256, 342, 3)
		sample_float=sample.astype(np.float32)
		
		#??? Only dealing with 3 channels
		sample_normalized=np.ndarray(shape=sample_float.shape,dtype="float32")
		print type(self.mean_[0])#type float
		for i in range(self.num_channels_):
			sample_normalized[:,:,i]=cv2.subtract(sample_float[:,:,i],self.mean_[i])
		
		#This operation will write the separate BGR planes directly to the input layer 
		#of the network because it is wrapped by the numpy.ndarray in input_channels.
		input_layer=self.net_.blobs['data']
		input_data= input_layer.data
		for i in range(self.num_channels_):
			input_data[0,i,:,:]=sample_normalized[:,:,i]
	
		
