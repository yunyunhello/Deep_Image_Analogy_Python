import numpy as np

def convertTo(src, rtype, alpha=1, beta=0):
	result=src.astype(rtype)
	result=result*alpha+beta
	result.clip(min=0,max=255)
	return result