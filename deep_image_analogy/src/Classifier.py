
import numpy as np
import caffe
import glog as log

class Classifier:
	def __init__(self,model_file,trained_file):
		
		#???
		#caffe.set_device(0)
		caffe.set_mode_gpu()
	
		#Load the network
		net_=caffe.Net(model_file, trained_file, caffe.TEST)
		#???
		print "Enter Clssifier.py\n"
		print net_.blobs['data'].data.shape 
		log.check_eq(len(net_.inputs),1,"Network should have exactly one input.")
		log.check_eq(len(net_.outputs),1,"Network should have exactly one output.")
		log.check(net_.blobs['data'].channels==3 or net_.blobs['data'].channels==1, "Input layer should have 1 or 3 channels.")
		#???
		self.mean_=[103.939, 116.779, 123.68]
		#print type(self.mean_)
