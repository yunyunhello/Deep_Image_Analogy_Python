import cost_function as cost_func
import lbfgs 
import skcuda.cublas as cublas

def string_replace(s1, s2, s3):
	pos=0
	
	while((pos=s1.find(s2,pos))!=-1):
		s1.replace(s2,s3)
		pos+=len(s3)

def(classifier, layer1, d_y, dim1, layer2, d_x, dim2):
	num1=dim1.channel*dim1.height*dim1.width
	num2=dim2.channel*dim2.height*dim2.width
	
	m_layer1=layer1
	m_layer2=layer2
	
	string_replace(layer1, "conv", "relu")
	string_replace(layer2, "conv", "relu")
	
	if(layer2 == string("data")):
		layer2 = string("input")
	layer_names = classifier.net_._layer_names
	for i in range(len(layer_names)):
		if(layer_names[i]==layer1):
			id1=i
		if(layer_names[i]==layer2):
			id2=i
	
	func=cost_func.my_cost_function(classifier, m_layer1, d_y, num1, m_layer2, num2, id1, id2)
	solver=lbfgs.lbfgs(func, cublas.cublasCreate())
	s=solver.minimize(d_x)
	