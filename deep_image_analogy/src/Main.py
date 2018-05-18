import DeepAnalogy
import sys


dp=DeepAnalogy.DeepAnalogy()

if(len(sys.argv)!=9):
	model="models/"
	
	#A="demo/content.png"
	#BP="demo/style.png"
	A="example/1photo2style/A/A44.png"
	BP="example/1photo2style/BP/BP44.png"
	#A="example/2style2style/A/A15.png"
	#BP="example/2style2style/BP/BP15.png"
	output="demo/output/"

	dp.SetModel(model)
	dp.SetA(A)
	dp.SetBPrime(BP)
	dp.SetOutputDir(output)
	dp.SetGPU(0)
	dp.SetRatio(0.5)
	dp.SetBlendWeight(2)
	dp.UsePhotoTransfer(False)
	dp.LoadInputs()
	dp.ComputeAnn()

else:
	dp.SetModel(sys.argv[1])
	dp.SetA(argv[2])
	dp.SetBPrime(argv[3])
	dp.SetOutputDir(argv[4])
	dp.SetGPU(atoi(argv[5]))
	dp.SetRatio(atof(argv[6]))
	dp.SetBlendWeight(atoi(argv[7]))
	if (atoi(argv[8]) == 1):
		dp.UsePhotoTransfer(True)
	else:
		dp.UsePhotoTransfer(False)
	#	dp.LoadInputs()
		#dp.ComputeAnn()

