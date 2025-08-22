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
def safe_sqrt(x: float):   
   if x>0.0:
      return wp.sqrt(x)
   else:
      return 0.0

@wp.func
def YF_ellipse(rho:wp.float32,theta:wp.float32):
   K = 1.0
   sy = (2.0*K)/(1.0+K+(K-1.0)*wp.sin(3.0*theta))*100.0
   f = rho - sy
   return f

@wp.func
def Hosford_yf(rho:wp.float32,theta:wp.float32):
   s1,s2,s3 = convert_prt_to_123(100.0,rho,theta)
   n = 1.5
   sy = (0.5*wp.abs(s2-s3)**n + 0.5*wp.abs(s3-s1)**n+0.5*wp.abs(s1-s2)**n)**(1.0/n)
   return sy - 120.0

@wp.func
def Elasticity_tens(mu:wp.float32,lamda:wp.float32):
   Ce = matrix66(lamda+2.0*mu,lamda,lamda,0.0,0.0,0.0,
                 lamda,lamda+2.0*mu,lamda,0.0,0.0,0.0,
                 lamda,lamda,lamda+2.0*mu,0.0,0.0,0.0,
                 0.0,0.0,0.0,2.0*mu,0.0,0.0,
                 0.0,0.0,0.0,0.0,2.0*mu,0.0,
                 0.0,0.0,0.0,0.0,0.0,2.0*mu)
   return Ce

@wp.func
def Get_R_mat(dF:wp.mat33f,flow_vec:wp.mat33f,Ce:matrix66):
   
   mult = wp.diag(vector6f(1.0,1.0,1.0,2.0,2.0,2.0))
   vec = mat_2_void(flow_vec)
   F = mat_2_void(dF)
   out = wp.ddot(flow_vec,void_2_mat(F@Ce))
   out1 = wp.outer(vec,F@Ce)
   if out!=0.0:R_mat = 1.0/out*out1
   else: R_mat = matrix66()
   R_mat_out = R_mat@mult
   return R_mat_out

@wp.func
def get_pl(Ce:matrix66,dF:wp.mat33,flow_vector:wp.mat33,d_eps:wp.mat33,H1:float,H2:float):
   denom = wp.ddot(void_2_mat((mat_2_void(dF)@Ce)),flow_vector)
   d_gamma = H1*H2*wp.ddot(void_2_mat((mat_2_void(dF)@Ce)),d_eps)
   if denom>0:out = d_gamma/denom
   else:out = 0.0
   return out

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
def convert_prt_to_123(p:wp.float32, rho:wp.float32, theta:wp.float32):

   sigma1pp = rho*wp.cos(theta)
   sigma2pp = rho*wp.sin(theta)
   sigma3pp = wp.sqrt(3.0)*p
   sigma_pp = wp.vec3f(sigma1pp, sigma2pp, sigma3pp)

   R = wp.mat33f(wp.sqrt(2.0)/2.0, -wp.sqrt(6.0)/6.0, wp.sqrt(3.0)/3.0,
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

@wp.func
def sig_2_prt(sigma:wp.mat33f):
   
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
def Get_df_dsigma(df_drho:wp.float32,df_x2:wp.float32,sigma:wp.mat33):
   
   s_dev = sigma - (1.0/3.0)*wp.trace(sigma)*Identity3()
   denom = safe_sqrt(wp.ddot(s_dev,s_dev))
   if denom>0:
      drho_dsigma = s_dev/denom
   else:
      drho_dsigma = wp.mat33()
   sigma_p,sigma_v = eig33_v(sigma)
   
   sigma_1 = sigma_p[0]
   sigma_1_v = wp.vec3(sigma_v[0,0],sigma_v[1,0],sigma_v[2,0])
   sigma_2 = sigma_p[1]
   sigma_2_v = wp.vec3(sigma_v[0,1],sigma_v[1,1],sigma_v[2,1])
   sigma_3 = sigma_p[2]
   sigma_3_v = wp.vec3(sigma_v[0,2],sigma_v[1,2],sigma_v[2,2])

   if sigma_1 < sigma_2: 
      sigma_1, sigma_2 = sigma_2, sigma_1
      sigma_1_v, sigma_2_v = sigma_2_v, sigma_1_v
   if sigma_2 < sigma_3: 
      sigma_2, sigma_3 = sigma_3, sigma_2
      sigma_2_v, sigma_3_v = sigma_3_v, sigma_2_v
   if sigma_1 < sigma_2: 
      sigma_1, sigma_2 = sigma_2, sigma_1
      sigma_1_v, sigma_2_v = sigma_2_v, sigma_1_v

   y = (wp.sqrt(2.0)/2.0)*sigma_1 - (wp.sqrt(2.0)/2.0)*sigma_3
   x = (-wp.sqrt(6.0)/6.0)*sigma_1 + (wp.sqrt(6.0)/3.0)*sigma_2 - (wp.sqrt(6.0)/6.0)*sigma_3  
   theta = wp.atan2(x,y)
   
   if theta<wp.float32(0.0):
      theta = theta + wp.float32(wp.pi)*wp.float32(2.0)
   if (x**2.0+y**2.0)>0:
      df_ds1 = y/(x**2.0+y**2.0)*(-wp.sqrt(6.0)/6.0) - x/(x**2.0+y**2.0)*(wp.sqrt(2.0)/2.0)
      df_ds2 = y/(x**2.0+y**2.0)*(wp.sqrt(6.0)/3.0) 
      df_ds3 = y/(x**2.0+y**2.0)*(-wp.sqrt(6.0)/6.0) - x/(x**2.0+y**2.0)*(-wp.sqrt(2.0)/2.0)
     
   else:
      df_ds1 = 0.0
      df_ds2 = 0.0
      df_ds3 = 0.0
   dtheta_dsigma = df_ds1* wp.outer(sigma_1_v,sigma_1_v) + \
                  df_ds2* wp.outer(sigma_2_v,sigma_2_v)  + \
                  df_ds3* wp.outer(sigma_3_v,sigma_3_v) 

   df_dtheta = df_x2*wp.cos(3.0*theta)*3.0
   df_dsigma = df_drho*drho_dsigma + df_dtheta*dtheta_dsigma   
   return df_dsigma

#NN_update law
@wp.func
def stress_update33_NN(L_n:wp.mat33f,
                       E_n:wp.mat33f,
                       tau_n:wp.mat33f,
                       F_val:wp.float32,
                       F_max:wp.float32,
                       dF_x1:wp.float32,
                       dF_x2:wp.float32,
                       mu:wp.float32,
                       lamda:wp.float32,
                       dt:wp.float32,
                       m:wp.float32):
   
   dF = Identity3() + dt*L_n
   Bn = matrix_exp(2.0*E_n)
   Bn1_trial = dF@Bn@wp.transpose(dF)
   en1_trial = wp.float32(0.5)*matrix_log(Bn1_trial)
   e_e_tr = en1_trial - wp.float32(1.0/3.0)*wp.trace(en1_trial)* Identity3()
   s_dev = wp.float32(2.0) * mu * e_e_tr
   norm_dev = safe_sqrt(wp.ddot(s_dev,s_dev))

   if norm_dev>wp.float32(0.0):
      flow_vector = s_dev/norm_dev
   else:
      flow_vector = s_dev

   d_eps = en1_trial - E_n
   out = -F_max
   f = F_val

   H1 = (f/(out)+1.0)**m
   H2 = wp.float32(0.5)*wp.sign(wp.ddot(tau_n,d_eps)) + wp.float32(0.5)
   #H1 = wp.clamp(H1,0.0,1.0)
   
   df_dsigma = Get_df_dsigma(dF_x1,dF_x2,tau_n)
   Ce = Elasticity_tens(mu,lamda)

   R_mat = Get_R_mat(df_dsigma,flow_vector,Ce)
   d_tau = void_2_mat((Ce @(Identity6()-H1*H2*R_mat))@ mat_2_void(d_eps))
   tau_n1 =  tau_n + d_tau
   eps_n1 = E_n + void_2_mat((Identity6()-H1*H2*R_mat)@ mat_2_void(d_eps))

   #d_gamma = get_pl(Ce,dF,flow_vector,d_eps,H1,H2)
   #eps_n1 = E_n + (d_eps-d_gamma*flow_vector)
   #d_tau = Ce@mat_2_void(d_eps-d_gamma*flow_vector)
   #tau_n1 =  tau_n + void_2_mat(d_tau)
   return tau_n1,eps_n1

@wp.func
def stress_update33(L_n:wp.mat33f,
                    E_n:wp.mat33f,
                    mu:wp.float32,
                    K:wp.float32,
                    dt:wp.float32):

   dF = Identity3() + dt*L_n
   #Trial state
   Bn = matrix_exp(wp.float32(2.0)*E_n)
   Bn1_trial = dF@Bn@wp.transpose(dF)
   en1_trial = wp.float32(0.5)*matrix_log(Bn1_trial) 

   e_e_tr = en1_trial - wp.float32(1.0/3.0)*wp.trace(en1_trial)* Identity3()
   tau_dev = wp.float32(2.0) * mu * e_e_tr
   tau_vol = K * wp.trace(en1_trial) * Identity3()
   norm_dev = wp.sqrt(wp.ddot(tau_dev,tau_dev))
   tau_trial = tau_dev + tau_vol
   rho,theta = sig_2_prt(tau_trial)
   f_trial = YF_ellipse(rho,theta)
   if f_trial<=wp.float32(1e-4):
      tau_dev_n1 = tau_dev
      e_n1 = en1_trial
   else:
      flow_vec = tau_dev/norm_dev
      dgama = wp.float32(0.0)
      N_iter = int(0)
      while wp.abs(f_trial)>wp.float32(1e-4) and N_iter<20:
         dfdgama = wp.float32(2.0) * mu
         dgama = dgama + f_trial/dfdgama
         e_n1 = en1_trial - dgama * flow_vec
         e_dev_n1 = e_n1 - wp.float32(1.0/3.0)*wp.trace(en1_trial)*Identity3()
         tau_dev_n1 = wp.float32(2.0) * mu*e_dev_n1
         tau = tau_dev_n1 + K* wp.trace(en1_trial) * Identity3()
         rho,theta = sig_2_prt(tau)
         f_trial = YF_ellipse(rho,theta)
         N_iter +=1
   tau_n1 = tau_dev_n1 + tau_vol
   return tau_n1,e_n1

#Smooth return mapping 
@wp.func
def stress_update33_smooth(L_n:wp.mat33f,
                           E_n:wp.mat33f,
                           tau_n:wp.mat33f,
                           mu:wp.float32,
                           lamda:wp.float32,
                           dt:wp.float32):

   dF = Identity3() + dt*L_n
   #Trial state
   Bn = matrix_exp(wp.float32(2.0)*E_n)
   Bn1_trial = dF@Bn@wp.transpose(dF)
   en1_trial = wp.float32(0.5)*matrix_log(Bn1_trial) 
   
   e_e_tr  = en1_trial - wp.float32(1.0/3.0)*wp.trace(en1_trial)* Identity3()
   tau_dev = wp.float32(2.0) * mu * e_e_tr
   norm_dev = wp.sqrt(wp.ddot(tau_dev,tau_dev))
   if norm_dev>0:
      flow_vector = tau_dev/norm_dev
   else:
      flow_vector = tau_dev
      
   d_eps = en1_trial - E_n
   rho,theta = sig_2_prt(tau_n)
   out = -YF_ellipse(0.0,theta)
   f = YF_ellipse(rho,theta)

   H1 = (f/(out)+1.0)**5.0
   H2 = 0.5*wp.sign(wp.ddot(tau_dev,d_eps)) + 0.5
   H1 = wp.min(H1,1.0)
   
   Ce = Elasticity_tens(mu,lamda)
   dF = 100.0*flow_vector
   denom = wp.ddot(void_2_mat((mat_2_void(dF)@Ce)),flow_vector)
   d_gamma = H1*H2*wp.ddot(void_2_mat((mat_2_void(dF)@Ce)),d_eps)
   
   if denom>0:d_gamma = d_gamma/denom

   eps_n1 = E_n + (d_eps-d_gamma*flow_vector)
   d_tau = Ce@mat_2_void(d_eps-d_gamma*flow_vector)
   tau_n1 =  tau_n + void_2_mat(d_tau)

   # d_tau = void_2_mat((Ce @(Identity6()-H1*H2*R_mat))@ mat_2_void(d_eps))
   # tau_n1 =  tau_n + d_tau
   # eps_n1 = E_n + void_2_mat((Identity6()-H1*H2*R_mat)@ mat_2_void(d_eps))
   
   # d_e_pl = void_2_mat((H1*H2*R_mat)@ mat_2_void(d_eps))
   # e_pl_n1 = lamda_n +  safe_sqrt(wp.ddot(d_e_pl,d_e_pl))
   return tau_n1,eps_n1


