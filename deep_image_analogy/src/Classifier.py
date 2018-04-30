
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
		net_=caffe.Net(model_file, trained_file, caffe.TEST)
		#???
		print "Enter Clssifier.py\n"
		print net_.blobs['data'].data.shape #(1,3,224,224)
		log.check_eq(len(net_.inputs),1,"Network should have exactly one input.")
		log.check_eq(len(net_.outputs),1,"Network should have exactly one output.")
		self.num_channels_=net_.blobs['data'].shape[1]
		log.check(self.num_channels_==3 or self.num_channels_==1, "Input layer should have 1 or 3 channels.")
		#???
		self.mean_=[103.939, 116.779, 123.68]
		#print type(self.mean_)

	# Load the mean file in binaryproto format. 
		
	def Predict(self, img, layers, data_s, data_d, size):
		input_geometry_.width=img.shape[1]
		input_geometry_.height=img.shape[0]
		
		input_layer=net_.blobs['data']
		input_layer.reshape(1,self.num_channels_,input_geometry_.height,input_geometry_.width)
                print input_layer.data.shape
			
