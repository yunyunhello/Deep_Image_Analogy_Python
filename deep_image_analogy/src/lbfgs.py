from enum import Enum
import pycuda.driver as cuda
import numpy as np
import skcuda.cublas as cublas

"""
class status(Enum):
	LBFGS_BELOW_GRADIENT_EPS=0
	LBFGS_REACHED_MAX_ITER=1
	LBFGS_REACHED_MAX_EVALS=2
	LBFGS_LINE_SEARCH_FAILED=3
"""

class lbfgs:
	def __init__(self, cf, cublasHandle):
		self.__m_costFunction=cf
		self.__m_maxIter=100
		#self.__m_maxEvals=max()
		self.__m_gradientEps=np.float32(1e-4)
		self.__m_cublasHandle=cublasHandle
		self.status=Enum('status',('LBFGS_BELOW_GRADIENT_EPS', 'LBFGS_REACHED_MAX_ITER', 'LBFGS_REACHED_MAX_EVALS', 'LBFGS_LINE_SEARCH_FAILED'))
	
	
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
		
		one=np.array([1.0],dtype=np.float32)
		cuda.memcpy_htod(d_H0,one)
	
		for it in range(self.__m_maxIter):	
			#Check for convergence
			gkNormSquared=np.empty(1,dtype=np.float32)
			xkNormSquared=np.empty(1,dtype=np.float32)
			
			self.__dispatch_dot(NX, xkNormSquared, d_x,  d_x)
			self.__dispatch_dot(NX, gkNormSquared, d_gk, d_gk)
			
			if (gkNormSquared[0] < (self.__m_gradientEps * self.__m_gradientEps) * max(xkNormSquared, 1.0)):
				stat = self.status.LBFGS_BELOW_GRADIENT_EPS
				print "break" #Debug Info: break Here
				break
			
			#Find search direction
			minusOne = np.array([-1.0],dtype=np.float32) 
			self.__dispatch_scale(NX, d_z, d_gk, minusOne) #z = -gk
			
			MAX_IDX = min(it, HISTORY_SIZE)
			 
			mod=SourceModule(update)
			update1=mod.get_function('update1')
			update2=mod.get_function('update2')
			update3=mod.get_function('update3')
			print "Here"
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
			
			print "HAHA"	
			cuda.memcpy_dtod(d_fkm1, d_fk, 1  * (np.dtype(np.float32).itemsize)) #fkm1 = fk
			cuda.memcpy_dtod(d_gkm1, d_gk, NX * (np.dtype(np.float32).itemsize)) #gkm1 = gk
			
			#timer *t_evals = NULL, *t_linesearch = NULL
			
			#line search defined in linesearch_gpu.h
			t_evals=None
			t_linesearch=None
			if not self.__gpu_linesearch(d_x, d_z, d_fk, d_gk, evals, d_gkm1, d_fkm1, stat, d_step, t_evals, t_linesearch, d_tmp, d_status):
				break
				
			# Update s, y, rho and H_0
			# ------------------------

			# s   = x_k - x_{k-1} = step * z
			# y   = g_k - g_{k-1}
			# rho = 1 / (y^T s)
			# H_0 = (y^T s) / (y^T y)
			d_curS = d_s + it % HISTORY_SIZE * NX
			d_curY = d_y + it % HISTORY_SIZE * NX
			
			self.__dispatch_scale(NX, d_curS, d_z,  d_step) # s = step * z
			self.__dispatch_axpy (NX, d_curY, d_gk, d_gkm1, minusOne) # y = gk - gkm1
			
			self.__dispatch_dot(NX, d_tmp,  d_curY, d_curS) # tmp  = yDotS
			self.__dispatch_dot(NX, d_tmp2, d_curY, d_curY) # tmp2 = yDotY
		
			# rho = 1 / tmp
			# if (tmp2 > 1e-5)
			# H0 = tmp / tmp2
			update3(d_rho + it % HISTORY_SIZE, d_H0, d_tmp, d_tmp2, grid=(1,1,1), block=(1,1,1))
			
			#CudaCheckError();
			#cudaDeviceSynchronize();
			cuda.Context.synchronize()			
			
		d_gk.free()
		d_gkm1.free()
		d_z.free()
		d_s.free()
		d_y.free()
		
		return stat
			
			
			
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
		
	def __gpu_linesearch(self, d_x, d_z, d_fk, d_gk, evals, d_gkm1, d_fkm1, stat, step, timer_evals, timer_linesearch, d_tmp, d_status):	
		print "Enter to gpu_linesearch"	

		strongWolfe='''
		__global__ void strongWolfePhase1(bool second_iter)
		{
			const float c1 = 1e-4f;
			const float c2 = 0.9f;

			const float phi_alpha = fk;

			const bool armijo_violated = (phi_alpha > fkm1 + c1 * alpha_cur * phi_prime_0 || (second_iter && phi_alpha >= fkm1));
			const bool strong_wolfe    = (fabsf(phi_prime_alpha) <= -c2 * phi_prime_0);

			// If both Armijo and Strong Wolfe hold, we're done
			if (!armijo_violated && strong_wolfe)
			{
				step   = alpha_cur;
				status = 1;
				return;
			}

			// If Armijio condition is violated, we've bracketed a viable minimum
			// Interval is [alpha_0, alpha]
			if (armijo_violated)
			{
				alpha_low      = 0.0f;
				alpha_high     = alpha_cur;
				phi_low        = fkm1;
				phi_high       = phi_alpha;
				phi_prime_low  = phi_prime_0;
				phi_prime_high = phi_prime_alpha;

				status = 2;
			}
			// If the directional derivative at alpha is positive, we've bracketed a viable minimum
			// Interval is [alpha, alpha_0]
			else if (phi_prime_alpha >= 0)
			{
				alpha_low      = alpha_cur;
				alpha_high     = 0.0f;
				phi_low        = phi_alpha;
				phi_high       = fkm1;
				phi_prime_low  = phi_prime_alpha;
				phi_prime_high = phi_prime_0;

				status = 2;
			}

			if (status == 2)
			{
				// For details check the comment for the same code in phase 2
				alpha_old = alpha_cur;

				alpha_cur  = 0.5f * (alpha_low + alpha_high);
				alpha_cur += (phi_high - phi_low) / (phi_prime_low - phi_prime_high);

				if (alpha_cur < fminf(alpha_low, alpha_high) || alpha_cur > fmaxf(alpha_low, alpha_high))
					alpha_cur = 0.5f * (alpha_low + alpha_high);

				alpha_correction = alpha_cur - alpha_old;

				return;
			}

			// Else look to the "right" of alpha for a viable minimum
			float alpha_new  = alpha_cur + 4 * (alpha_cur - alpha_old);
			alpha_old        = alpha_cur;
			alpha_cur        = alpha_new;
			alpha_correction = alpha_cur - alpha_old;

			// No viable minimum found in the interval [0, 1e8]
			if (alpha_cur > 1e8f)
			{
				status = 3;
				return;
			}

			status = 0;
		}
		
		__global__ void strongWolfePhase2(size_t tries)
		{
			const float c1 = 1e-4f;
			const float c2 = 0.9f;

			const size_t minTries = 10;

			const float phi_0       = fkm1;
			const float phi_j       = fk;
			const float phi_prime_j = tmp;

			const bool armijo_violated = (phi_j > phi_0 + c1 * alpha_cur * phi_prime_0 || phi_j >= phi_low);
			const bool strong_wolfe    = (fabsf(phi_prime_j) <= -c2 * phi_prime_0);

			if (!armijo_violated && strong_wolfe)
			{
				// The Armijo and Strong Wolfe conditions hold
				step   = alpha_cur;
				status = 1;
				return;
			}
			else if (fabsf(alpha_high - alpha_low) < 1e-5f && tries > minTries)
			{
				// The search interval has become too small
				status = 2;
				return;
			}
			else if (armijo_violated)
			{
				alpha_high     = alpha_cur;
				phi_high       = phi_j;
				phi_prime_high = phi_prime_j;
			}
			else
			{
				if (tmp * (alpha_high - alpha_low) >= 0)
				{
					alpha_high     = alpha_low;
					phi_high       = phi_low;
					phi_prime_high = phi_prime_low;
				}

				alpha_low     = alpha_cur;
				phi_low       = phi_j;
				phi_prime_low = phi_prime_j;
			}

			// Quadratic interpolation:
			// Least-squares fit a parabola to (alpha_low, phi_low),
			// (alpha_high, phi_high) with gradients phi_prime_low and
			// phi_prime_high and select the minimum of that parabola as
			// the new alpha

			alpha_old = alpha_cur;

			alpha_cur  = 0.5f * (alpha_low + alpha_high);
			alpha_cur += (phi_high - phi_low) / (phi_prime_low - phi_prime_high);

			if (alpha_cur < fminf(alpha_low, alpha_high) || alpha_cur > fmaxf(alpha_low, alpha_high))
				alpha_cur = 0.5f * (alpha_low + alpha_high);

			alpha_correction = alpha_cur - alpha_old;

			status = 0;
		}
		'''
		
		#Step, function value and directional derivative at the starting point of the line search
		d_phi_prime_0=cuda.mem_alloc(np.dtype(np.float32).itemsize)		

		#Current, previous and correction step Correction step is (alpha_cur - alpha_old)		
		d_alpha_cur=cuda.mem_alloc(np.dtype(np.float32).itemsize)	
		d_alpha_old=cuda.mem_alloc(np.dtype(np.float32).itemsize)	
		d_alpha_correction=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		
		#Directional derivative at alpha
		d_phi_prime_alpha=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		
		# Low and high search interval boundaries
		alpha_low=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_alpha_high=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_phi_low=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_phi_high=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_phi_prime_low=cuda.mem_alloc(np.dtype(np.float32).itemsize)
		d_phi_prime_high=cuda.mem_alloc(np.dtype(np.float32).itemsize)	
						   
						   
		NX = self.__m_costFunction.getNumberOfUnknowns()
		
		phi_prime_0=np.empyt(1,dtype=np.float32)
		self.__dispatch_dot(NX, phi_prime_0, d_z, d_gk) #phi_prime_0 = z' * gk
		
		if phi_prime_0[0]>=0:
			stat=status.LBFGS_LINE_SEARCH_FAILED
			return False
			
		cuda.memcpy_htod(d_phi_prime_0, phi_prime_0, np.dtype(np.float32).itemsize)
	
		zero=np.float32(0.0)
		one=np.float32(1.0)
		cuda.memcpy_htod(d_alpha_cur, one)
		cuda.memcpy_htod(d_alpha_old, zero)
		cuda.memcpy_htod(d_alpha_correction, one)
		
		second_iter = False
		
		mod=SourceModule('strongWolfe')
		
		while 1:
			# go from (x + alpha_old * z)
			# to      (x + alpha     * z)

			# xk += (alpha - alpha_old) * z;
			
			#dispatch_axpy(NX, d_x, d_x, d_z, d_alpha_correction, true);			
			self.__dispatch_axpy(NX, d_x, d_x, d_z, d_alpha_correction)
			self.__m_costFunction.f_gradf(d_x, d_fk, d_gk)	
			
			#CudaCheckError();
			#cudaDeviceSynchronize();
			cuda.Context.synchronize()
			
			++evals
			
			# dispatch_dot(NX, d_phi_prime_alpha, d_z, d_gk, true); // phi_prime_alpha = z' * gk;
			self.dispatch_dot(NX, d_phi_prime_alpha, d_z, d_gk)
			strongWolfePhase1=mod.get_function('strongWolfePhase1')
			strongWolfePhase1(second_iter,block=(1,1,1),grid=(1,1,1))
			
			#CudaCheckError();
			#cudaDeviceSynchronize();
			cuda.Context.synchronize()

			ret=np.empty(1,dtype=np.int32)
			cuda.memcpy_dtoh(ret, d_status)
			
			#If both Armijo and Strong Wolfe hold, we're done
			if ret==1:
				return True
			
			if evals >= maxEvals:
				stat = self.status.LBFGS_REACHED_MAX_EVALS
				return False
				
			# We've bracketed a viable minimum, go find it in phase 2
			if ret==2:
				break
				
			# Coudln't find a viable minimum in the range [0, alpha_max=1e8]
			if ret==3:
				stat = self.status.LBFGS_LINE_SEARCH_FAILED
				return False
			
			second_iter = True
			
			print "Unfinished gpu_linesearch code!"

		# The minimum is now bracketed in [alpha_low, alpha_high]
		# Find it...
		tries = 0
		
		while 1:
			tries=tries+1
			
			# go from (x + alpha_old * z)
			# to      (x + alpha     * z)

			# xk += (alpha - alpha_old) * z;
			
			# dispatch_axpy(NX, d_x, d_x, d_z, d_alpha_correction, true);
			self.dispatch_axpy(NX, d_x, d_x, d_z, d_alpha_correction)
			
			self.__m_costFunction.f_gradf(d_x, d_fk, d_gk)
			
			#CudaCheckError();
			#cudaDeviceSynchronize();
			cuda.Context.synchronize()			
			
			++evals
			
			self.__dispatch_dot(NX, d_tmp, d_z, d_gk, true); # tmp = phi_prime_j = z' * gk;
			strongWolfePhase2=mod.get_function('strongWolfePhase2')
			strongWolfePhase2(tries, block=(1,1,1), grid=(1,1,1))
			
			#CudaCheckError();
			#cudaDeviceSynchronize();
			cuda.Context.synchronize()
			
			ret=np.empty(1, dtype=np.int32)
			
			if ret==1:
				# The Armijo and Strong Wolfe conditions hold
				return True
			
			if ret==2:
				# The search interval has become too small
				stat = self.status.LBFGS_LINE_SEARCH_FAILED
				return False
			
			if evals>=maxEvals:
				stat = self.status.LBFGS_REACHED_MAX_EVALS
				return False
			
			
	def minimize(self, d_x):
		return self.gpu_lbfgs(d_x)
		
		
		
