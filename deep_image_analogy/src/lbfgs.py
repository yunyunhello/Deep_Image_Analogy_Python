from enum import Enum
import pycuda.driver as cuda
import numpy as np
import skcuda.cublas as cublas



class lbfgs:
	def __init__(self, cf, cublasHandle):
		self.__m_costFunction=cf
		self.__m_maxIter=100
		self.__m_maxEvals=max()
		self.__m_gradientEps=np.float(1e-4)
		self.__m_cublasHandle=cublasHandle
		self.status=Enum('status',(LBFGS_BELOW_GRADIENT_EPS, LBFGS_REACHED_MAX_ITER, LBFGS_REACHED_MAX_EVALS, LBFGS_LINE_SEARCH_FAILED))
	
	
	
	def minimize(self, d_x):
		return gpu_lbfgs(d_x)
	
	def gpu_lbfgs(self, d_x):
		update='''
		__global__ void update1(float *alpha_out, const float *sDotZ, const float *rho, float *minusAlpha_out)
		{
			*alpha_out      = *sDotZ * *rho;
			*minusAlpha_out = -*alpha_out;
		}
		
		__global__ void update2(float *alphaMinusBeta_out, const float *rho, const float *yDotZ, const float *alpha)
		{
			const float beta = *rho * *yDotZ;
			*alphaMinusBeta_out = *alpha - beta;
		}
		
		__global__ void update3(float *rho_out, float *H0_out, const float *yDotS, const float *yDotY)
		{
			*rho_out = 1.0f / *yDotS;

			if (*yDotY > 1e-5)
				*H0_out = *yDotS / *yDotY;
		}
		'''
		
		HISTORY_SIZE=4
		
		NX = self.__m_costFunction.getNumberOfUnknowns()
	
		d_gk=cuda.mem_alloc(NX *(np.dtype(np.float32).itemsize))
		d_gkm1=cuda.mem_alloc(NX *(np.dtype(np.float32).itemsize))
		d_z=cuda.mem_alloc(NX *(np.dtype(np.float32).itemsize))
		d_s=cuda.mem_alloc(HISTORY_SIZE*NX*(np.dtype(np.float32).itemsize))
		d_y=cuda.mem_alloc(HISTORY_SIZE*NX*(np.dtype(np.float32).itemsize))
		
		d_fk=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_fkm1=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_H0=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_temp=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_temp2=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_step=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_status=cuda.mem_alloc(np.dtype(np.int32).itemsize)
		
		d_rho=cuda.mem_alloc(HISTORY_SIZE*(np.dtype(np.float32).itemsize))
		d_alpha=cuda.mem_alloc(HISTORY_SIZE*(np.dtype(np.float32).itemsize))
		
		self.__m_costFunction.f_gradf(d_x, d_fk, d_gk)
		
		
		#??? ???
		#CudaCheckError();
		#cudaDeviceSynchronize();
		cuda.Context.synchronize()
		
		evals = 1
		#status stat = LBFGS_REACHED_MAX_ITER;
		
		one = np.float32(1.0)
		cuda.memcpy_htod(d_H0,one)
		
		for it in range(self.__m_maxIter):	
			#Check for convergence
			gkNormSquared=np.empty(1,dtype=np.float32)
			xkNormSquared=np.empty(1,dtype=np.float32)
			
			self.__dispatch_dot(NX, &xkNormSquared, d_x,  d_x)
			self.__dispatch_dot(NX, &gkNormSquared, d_gk, d_gk)
			
			if (gkNormSquared[0] < (self.__m_gradientEps * self.__m_gradientEps) * max(xkNormSquared, 1.0)):
				stat = self.__status.LBFGS_BELOW_GRADIENT_EPS
				break
			
			#Find search direction
			minusOne = np.array([-1.0],dtype=np.float32) 
			self.__dispatch_scale(NX, d_z, d_gk, minusOne) #z = -gk
			
			MAX_IDX = min(it, HISTORY_SIZE)
			 
			mod=SourceModule(update)
			update1=mod.get_function('update1')
			update2=mod.get_function('update2')
			update3=mod.get_function('update3')
			for i in range(1,MAX_IDX+1):
				idx = (it - i) % HISTORY_SIZE
				self.__dispatch_dot(NX, d_tmp, d_s + idx * NX, d_z) #tmp = sDotZ
				
				# alpha = tmp * rho
				# tmp = -alpha		
				update1(d_alpha + idx, d_tmp, d_rho + idx, d_tmp, block=(1,1,1), grid=(1,1,1))
				
				#CudaCheckError();
				#cudaDeviceSynchronize();
				cuda.Context.synchronize()
				
				#z += tmp * y
				self.__dispatch_axpy(NX, d_z, d_z, d_y + idx * NX, d_tmp)
				
			self.__dispatch_scale(NX, d_z, d_z, d_H0) #z = H0 * z
			
			i=MAX_IDX
			while i>0:
				idx = (it - i) % HISTORY_SIZE
				self.__dispatch_dot(NX, d_tmp, d_y + idx * NX, d_z) #tmp = yDotZ
							
				# beta = rho * tmp
				# tmp = alpha - beta
				update2(d_tmp, d_rho + idx, d_tmp, d_alpha + idx, block=(1,1,1), grid=(1,1,1))
				
				#CudaCheckError();
				#cudaDeviceSynchronize();
				cuda.Context.synchronize()
			
				# z += tmp * s
				self.__dispatch_axpy(NX, d_z, d_z, d_s + idx * NX, d_tmp)
				
				i=i-1
			
			cuda.memcpy_dtod(d_fkm1, d_fk, 1  * (np.dtype(np.float32).itemsize)) #fkm1 = fk
			cuda.memcpy_dtod(d_gkm1, d_gk, NX * (np.dtype(np.float32).itemsize)) #gkm1 = gk
			
			#timer *t_evals = NULL, *t_linesearch = NULL
			
			#line search defined in linesearch_gpu.h
			t_evals=None
			t_linesearch=None
			gpu_linesearch(d_x, d_z, d_fk, d_gk, evals, d_gkm1, d_fkm1, stat, d_step, self.__m_maxEvals, t_evals, t_linesearch, d_tmp, d_status))
			
			
	def __dispatch_dot(self, n, dst, d_x, d_y):
		cublas.cublasSdot(self.__m_cublasHandle, int(n), d_x, 1, d_y, 1)
		
	def __dispatch_scale(n, d_dst, d_x, a):
		if (d_dst != d_x):
			cuda.memcpy_dtod(d_dst, d_x, n * (np.dtype(np.float32).itemsize))
		
		cuda.cublasSscal(self.__m_cublasHandle, int(n), a, d_dst, 1)
		
	def __dispatch_axpy(self, n, d_dst, d_y, d_x, a):
		if d_dst != d_y:
			cuda.memcpy_dtod(d_dst, d_y, n * (np.dtype(np.float32).itemsize))
		
		cublas.cublasSaxpy(self.__m_cublasHandle, int(n), a, d_x, 1, d_dst, 1)
		
	def gpu_linesearch(self, d_x, d_z, d_fk, d_gk, evals, d_gkm1, d_fkm1, stat, step, maxEvals, timer_evals, timer_linesearch, d_tmp, d_status):	
						   
		#Step, function value and directional derivative at the starting point of the line search
		d_phi_prime_0=cuda.mem_alloc(np.dtype(np.float32).itemsize)		

		#Current, previous and correction step Correction step is (alpha_cur - alpha_old)		
		d_alpha_cur=cuda.mem_alloc(np.dtype(np.float32).itemsize)	
		d_alpha_old=cuda.mem_alloc(np.dtype(np.float32).itemsize)	
		d_alpha_correction=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		
		#Directional derivative at alpha
		d_phi_prime_alpha=cuda.mem_alloc(np.dtype(np.float32).itemsize
		
		# Low and high search interval boundaries
		alpha_low=cuda.mem_alloc(np.dtype(np.float32).itemsize
		d_alpha_high=cuda.mem_alloc(np.dtype(np.float32).itemsize
		d_phi_low=cuda.mem_alloc(np.dtype(np.float32).itemsize
		d_phi_high=cuda.mem_alloc(np.dtype(np.float32).itemsize
		d_phi_prime_low=cuda.mem_alloc(np.dtype(np.float32).itemsize
		d_phi_prime_high=cuda.mem_alloc(np.dtype(np.float32).itemsize	
						   
						   
		NX = self.__m_costFunction.getNumberOfUnknowns()
		
		phi_prime_0=np.empyt(1,dtype=np.float32)
		self.__dispatch_dot(NX, phi_prime_0, d_z, d_gk) #phi_prime_0 = z' * gk
		
		if phi_prime_0[0]>=0:
			stat=lbfgs::LBFGS_LINE_SEARCH_FAILED
			return False
			
		cuda.memcpy_htod(d_phi_prime_0, phi_prime_0, np.dtype(np.float32).itemsize)
	
		zero=np.float32(0.0)
		one=np.float32(1.0)
		cuda.memcpy_htod(d_alpha_cur, one)
		cuda.memcpy_htod(d_alpha_old, zero)
		cuda.memcpy_htod(d_alpha_correction, one)
		
		second_iter = False
		
		while 1:
			# go from (x + alpha_old * z)
			# to      (x + alpha     * z)

			# xk += (alpha - alpha_old) * z;
			
			self.__dispatch_axpy(NX, d_x, d_x, d_z, d_alpha_correction)
			break
		
		

		
		
		
		
		