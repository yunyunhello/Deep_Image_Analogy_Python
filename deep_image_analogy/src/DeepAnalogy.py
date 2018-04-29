
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context
import cv2

class parameters:
	def __init__(self):
		self.layers=[] #which layers  used as content
	
		self.patch_size0=0
		self.iter=0


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
		ori_AL=cv2.imread(self.__file_A)
		ori_BPL=cv2.imread(self.__file_BP)
		if ori_AL is None or ori_BPL is None:
			print "image cannot read!"
		else:
			self.__ori_A_cols=ori_AL.shape[1]
			self.__ori_A_rows=ori_AL.shape[0]
			self.__ori_BP_cols=ori_BPL.shape[1]
			self.__ori_BP_rows=ori_BPL.shape[0]
#			print ori_AL.shape

