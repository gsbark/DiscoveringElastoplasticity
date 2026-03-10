import warp as wp 
from warp.types import matrix,vector

class matrix66(matrix(shape=(6,6),dtype=wp.float32)):
   pass

class vector6f(vector(length=6,dtype=wp.float32)):
   pass

@wp.func
def Identity3():
   v = wp.vec3f(wp.float32(1.0),wp.float32(1.0),wp.float32(1.0))
   return wp.diag(v)

@wp.func
def Identity6():
   v = vector6f(wp.float32(1.0),wp.float32(1.0),wp.float32(1.0),wp.float32(1.0),wp.float32(1.0),wp.float32(1.0))
   return wp.diag(v)

@wp.func
def mat_2_void(A:wp.mat33f):
   return vector6f(A[0,0],A[1,1],A[2,2],A[1,2],A[0,2],A[0,1])

@wp.func
def void_2_mat(A:vector6f):
   return wp.mat33f(A[0],A[5],A[4],
                    A[5],A[1],A[3],
                    A[4],A[3],A[2])

@wp.func
def Linear(x:wp.float32):
   return x

@wp.func
def ReLu(x:wp.float32):
   return wp.max(0.0,x)

@wp.func
def softplus(x:wp.float32):
   return wp.log(1.0+ wp.exp(x))

@wp.func
def ELU(x:wp.float32):
   if x>=0:
      return x
   else:
      return wp.exp(x)-1.0
   
@wp.func
def apply_grad_relu(x:wp.float32):
   if x>0:
      val = 1.0
   else:
      val = 0.0
   return val

@wp.func
def apply_grad_elu(x:wp.float32):
   if x>=0:
      val =1.0
   else:
      val = x+1.0
   return val

@wp.func
def safe_sqrt(x: wp.float32):   
   if x>0.0:
      return wp.sqrt(x)
   else:
      return 0.0

@wp.func
def eig33(sigma:wp.mat33f):
   U = wp.mat33f()
   V = wp.mat33f()
   s = wp.vec3f()
   s_out= wp.vec3f()
   wp.svd3(sigma,U,s,V)
   for i in range(3):
      d = wp.vec3f(U[0,i],U[1,i],U[2,i])
      ff = wp.dot(d,sigma@d)
      s_out[i] = wp.abs(s[i])*wp.sign(ff)
   return s_out

@wp.func
def eig33_v(sigma:wp.mat33f):
   U = wp.mat33f()
   V = wp.mat33f()
   s = wp.vec3f()
   wp.svd3(sigma,U,s,V)
   s_out= wp.vec3f()
   for i in range(3):
      d = wp.vec3f(U[0,i],U[1,i],U[2,i])
      ff = wp.dot(d,sigma@d)
      s_out[i] = wp.abs(s[i])*wp.sign(ff)
   return s_out,U

@wp.func
def matrix_exp(A:wp.mat33f):
   U = wp.mat33f()
   V = wp.mat33f()
   sigma_p = wp.vec3f()
   wp.svd3(A,U,sigma_p,V)
   s_out =  wp.vec3f()
   for i in range(3):
      d = wp.vec3f(U[0,i],U[1,i],U[2,i])
      ff = wp.dot(d,A@d)
      s_out[i] = wp.abs(sigma_p[i])*wp.sign(ff)
   v = wp.vec3f(wp.exp(s_out[0]),wp.exp(s_out[1]),wp.exp(s_out[2]))
   s_exp = wp.diag(v)
   return U@s_exp@wp.transpose(U)

@wp.func
def matrix_log(A:wp.mat33f):
   U = wp.mat33f()
   s = wp.vec3f()
   s_out= wp.vec3f()
   V = wp.mat33f()
   wp.svd3(A,U,s,V)
   for i in range(3):
      d = wp.vec3f(U[0,i],U[1,i],U[2,i])
      ff = wp.dot(d,A@d)
      s_out[i] = wp.abs(s[i])*wp.sign(ff)
   v = wp.vec3f(wp.log(s_out[0]),wp.log(s_out[1]),wp.log(s_out[2]))
   s_exp = wp.diag(v)
   return U@s_exp@wp.transpose(U)

@wp.func
def sig_2_rt(sigma:wp.mat33f):
   
   sigma_p = eig33(sigma)
   sigma_1 = sigma_p[0]
   sigma_2 = sigma_p[1]
   sigma_3 = sigma_p[2]

   if sigma_3 > sigma_2:sigma_3, sigma_2 = sigma_2, sigma_3
   if sigma_2 > sigma_1:sigma_2, sigma_1 = sigma_1, sigma_2
   if sigma_3 > sigma_2:sigma_3, sigma_2 = sigma_2, sigma_3

   sigma1_pp = wp.float32(wp.sqrt(2.0)/2.0)*sigma_1 - wp.float32(wp.sqrt(2.0)/2.0)*sigma_3
   sigma2_pp = -wp.float32(wp.sqrt(6.0)/6.0)*sigma_1 + wp.float32(wp.sqrt(6.0)/3.0)*sigma_2 - wp.float32(wp.sqrt(6.0)/6.0)*sigma_3
   
   rho   = safe_sqrt(sigma1_pp**wp.float32(2.0) + sigma2_pp**wp.float32(2.0))
   theta = wp.atan2(sigma2_pp,sigma1_pp)
   if theta<wp.float32(0.0):
      theta = theta + wp.float32(wp.pi)*wp.float32(2.0)
   return rho,theta

@wp.func
def prt_to_123(p:wp.float32, rho:wp.float32, theta:wp.float32):

   sigma1pp = rho*wp.cos(theta)
   sigma2pp = rho*wp.sin(theta)
   sigma3pp = wp.sqrt(3.0)*p
   sigma_pp = wp.vec3f(sigma1pp, sigma2pp, sigma3pp)

   R = wp.mat33f(
      wp.sqrt(2.0)/2.0, -wp.sqrt(6.0)/6.0, wp.sqrt(3.0)/3.0,
      0.0,  wp.sqrt(6.0)/3.0, wp.sqrt(3.0)/3.0,
      -wp.sqrt(2.0)/2.0, -wp.sqrt(6.0)/6.0, wp.sqrt(3.0)/3.0)
   
   sigma = R@sigma_pp
   sigma_1 = sigma[0]
   sigma_2 = sigma[1]
   sigma_3 = sigma[2]

   if sigma_3 > sigma_2: sigma_3, sigma_2 = sigma_2, sigma_3
   if sigma_2 > sigma_1: sigma_2, sigma_1 = sigma_1, sigma_2
   if sigma_3 > sigma_2: sigma_3, sigma_2 = sigma_2, sigma_3

   return sigma_1, sigma_2, sigma_3

@wp.kernel
def add(
   A:wp.array(dtype=wp.float32),
   B:wp.array(dtype=wp.float32)
   ): 
   
   i = wp.tid()
   if wp.isnan(B[i]) or wp.isinf(B[i]):
      a = 0.0
   else:
      a = B[i]   
   wp.atomic_add(A,i,a)

@wp.kernel
def mult_scalar(
   A:wp.array2d(dtype=wp.float32),
   scalar:wp.float32,
   B:wp.array(dtype=wp.float32)
   ):                    
   
   tid = wp.tid()
   B[tid] = A[0,tid]*scalar

@wp.kernel
def scale_df(
   A: wp.array3d(dtype=wp.float32),
   scale_inp:wp.array(dtype=wp.float32),
   scale_out:wp.float32,
   B:wp.array2d(dtype=wp.float32)
   ):  
   
   b,i,j = wp.tid()
   B[b,j] = A[b,i,j]*scale_inp[j]*scale_out


