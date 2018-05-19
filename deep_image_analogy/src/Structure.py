import numpy as np

class Size:
	def __init__(self):
		self.width=np.int32(0)
		self.height=np.int32(0)
		
class Dim:
	def __init__(self, c, h, w):
		self.channel=np.int32(c)
		self.height=np.int32(h)
		self.width=np.int32(w)
