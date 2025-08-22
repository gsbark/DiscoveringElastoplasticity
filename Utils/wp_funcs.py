import warp as wp 

@wp.func
def Linear(x:float):
   return x

@wp.func
def ReLu(x:float):
   return wp.max(0.0,x)

@wp.func
def softplus(x:float):
   return wp.log(1.0+ wp.exp(x))

@wp.func
def ELU(x:float):
   if x>=0:
      return x
   else:
      return wp.exp(x)-1.0
   
@wp.func
def safe_sqrt(x: float):   
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
def apply_grad_relu(x:float):
   if x>0:
      val = 1.0
   else:
      val = 0.0
   return val

@wp.func
def apply_grad_elu(x:float):
   if x>=0:
      val =1.0
   else:
      val = x+1.0
   return val

@wp.kernel
def Get_NN_input(sigma_n:wp.array(dtype=wp.mat33f),  # type:ignore
                 thetas:wp.array(dtype=wp.float32),  # type:ignore
                 scale:wp.array(dtype=wp.float32),   # type: ignore
                 inp1:wp.array2d(dtype=wp.float32),  # type:ignore
                 inp2:wp.array2d(dtype=wp.float32)): # type:ignore 
   
   tid = wp.tid()
   rho,theta = sig_2_rt(sigma_n[tid])
   thetas[tid] = theta

   inp1[0,tid] = rho * scale[0]
   inp1[1,tid] = wp.sin(theta*3.0) * scale[1]

   inp2[0,tid] = 0.0 * scale[0]
   inp2[1,tid] = wp.sin(theta*3.0) * scale[1]

@wp.kernel
def run_mlp(w1:wp.array2d(dtype=wp.float32),b1:wp.array(dtype=wp.float32),          # type:ignore
            w2:wp.array2d(dtype=wp.float32),b2:wp.array(dtype=wp.float32),          # type:ignore
            w3:wp.array2d(dtype=wp.float32),b3:wp.array(dtype=wp.float32),          # type:ignore
            inp:wp.array2d(dtype=wp.float32),out1:wp.array2d(dtype=wp.float32),     # type:ignore
            out2:wp.array2d(dtype=wp.float32),out3:wp.array2d(dtype=wp.float32)):   # type:ignore

   tid = wp.tid()
   wp.mlp(w1,b1,ELU,tid,inp,out1)
   wp.mlp(w2,b2,ELU,tid,out1,out2)
   wp.mlp(w3,b3,Linear,tid,out2,out3)

@wp.kernel
def scale_f(A: wp.array2d(dtype=wp.float32),  # type:ignore
            scale:wp.array(dtype=wp.float32), # type:ignore
            B: wp.array(dtype=wp.float32)):   # type:ignore

   i = wp.tid()
   B[i] = A[0,i]*scale[0]

@wp.kernel
def loss_fun(x_pred:wp.array(dtype=wp.vec3f),   # type:ignore
             x0:wp.array(dtype=wp.vec3f),       # type:ignore
             x_true:wp.array(dtype=wp.vec3f),   # type:ignore         
             loss:wp.array(dtype=wp.float32),   # type:ignore
             flag:wp.array(dtype=wp.float32),   # type:ignore
             dev:wp.float32):  

   tid = wp.tid()

   loss_i = ((x_pred[tid][0]-x0[tid][0]) - x_true[tid][0])**wp.float32(2.0) + \
            ((x_pred[tid][1]-x0[tid][1]) - x_true[tid][1])**wp.float32(2.0) + \
            ((x_pred[tid][2]-x0[tid][2]) - x_true[tid][2])**wp.float32(2.0)
   
   loss_i = flag[tid] * loss_i
   wp.atomic_add(loss,0,loss_i/dev)

@wp.kernel
def add(A:wp.array(dtype=wp.float32),B:wp.array(dtype=wp.float32)):               # type: ignore
   
   i = wp.tid()
   wp.atomic_add(A,i,B[i])

@wp.kernel
def mult_scalar(A:wp.array(dtype=wp.float32),norm:wp.float32):                    # type: ignore
   
   tid = wp.tid()
   A[tid] = A[tid]*norm

@wp.kernel
def Get_z(out1:wp.array2d(dtype=float), # type: ignore
          w:wp.array2d(dtype=float),    # type: ignore
          z:wp.array3d(dtype=float)):   # type: ignore
   
   j,i,k = wp.tid()
   x = out1[i,j]
   val = apply_grad_elu(x)
   z[j,i,k] = val*w[i,k]

@wp.kernel
def scale_df(A: wp.array3d(dtype=float),   # type: ignore
             scale1:wp.array(dtype=float), # type: ignore
             scale2:wp.array(dtype=float), # type: ignore
             B: wp.array2d(dtype=float)):  # type: ignore
   
   b,i,j = wp.tid()
   B[b,j] = A[b,i,j]*scale1[j]*scale2[0]

@wp.kernel
def copy_kernel(arr_3d: wp.array3d(dtype=wp.float32),   # type: ignore
                arr_2d: wp.array2d(dtype=wp.float32)):  # type: ignore
   
   batch, i, j = wp.tid()  
   arr_3d[batch, i, j] = arr_2d[i, j]  
