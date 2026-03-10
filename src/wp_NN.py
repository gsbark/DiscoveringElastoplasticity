import warp as wp 
import numpy as np 
from torch import Tensor
from src.wp_helpers import *

class wp_MLP:
   def __init__(self,params:Tensor,batch:int,steps:int,batch_size:int,device):

      self.w1 = wp.array(params['layers.0.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.b1 = wp.array(params['layers.0.bias'].numpy()[:,None],dtype=wp.float32,requires_grad=True,device=device)
      
      self.out1 = wp.zeros((steps,self.w1.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.out11 = wp.zeros((steps,self.w1.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.z1 = wp.zeros((steps,batch,self.w1.shape[0],self.w1.shape[1]),dtype=wp.float32,requires_grad=True,device=device)
      
      self.w2 = wp.array(params['layers.2.weight'].numpy(),requires_grad=True,device=device)
      self.b2 = wp.array(params['layers.2.bias'].numpy()[:,None],requires_grad=True,device=device)

      self.out2 = wp.zeros((steps,self.w2.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.out22 = wp.zeros((steps,self.w2.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.z2 = wp.zeros((steps,batch,self.w2.shape[0],self.w2.shape[1]),dtype=wp.float32,requires_grad=True,device=device)

      self.z2z1 = wp.zeros((steps,batch,self.w2.shape[1],self.w1.shape[1]),dtype=wp.float32,requires_grad=True,device=device)

      self.w3 = wp.array(params['layers.4.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.b3 = wp.array(params['layers.4.bias'].numpy()[:,None],dtype=wp.float32,requires_grad=True,device=device)
      self.out3 = wp.zeros((steps,self.w3.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.out33 = wp.zeros((steps,self.w3.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      
      self.f_val = wp.zeros((steps,batch),dtype=wp.float32,requires_grad=True,device=device)  
      self.f_max = wp.zeros((steps,batch),dtype=wp.float32,requires_grad=True,device=device)
      
      self.df_dx = wp.zeros((steps,batch,1,self.w1.shape[1]),dtype=wp.float32,requires_grad=True,device=device)
      self.df_dx_true = wp.zeros((steps,batch,self.w1.shape[1]),dtype=wp.float32,requires_grad=True,device=device)
      
      self.w1_copy = wp.array(params['layers.0.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.w2_copy =  wp.array(params['layers.2.weight'].numpy(),requires_grad=True,device=device)
      self.w3_copy = wp.array(params['layers.4.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)

   def reset(self):

      self.out1.zero_()
      self.out11.zero_()
      self.z1.zero_()
      
      self.out2.zero_()
      self.out22.zero_()
      self.z2.zero_()

      self.out3.zero_()
      self.out33.zero_()
      self.f_val.zero_()
      self.f_max.zero_()

      self.z2z1.zero_()
      self.df_dx.zero_()
      self.df_dx_true.zero_()

@wp.kernel
def Get_NN_input_2d(
   sigma:wp.array(dtype=wp.mat33f),    # Cauchy Stress
   thetas:wp.array(dtype=wp.float32),  # Lode angle theta
   scale:wp.array(dtype=wp.float32),   # Scale
   inp1:wp.array2d(dtype=wp.float32),  # NN input 1
   inp2:wp.array2d(dtype=wp.float32)   # NN input 2
   ): 
   
   tid = wp.tid()
   rho,theta = sig_2_rt(sigma[tid])
   thetas[tid] = theta

   inp1[0,tid] = rho * scale[0]
   inp1[1,tid] = wp.sin(theta*3.0) * scale[1]

   inp2[0,tid] = 0.0 * scale[0]
   inp2[1,tid] = wp.sin(theta*3.0) * scale[1]

@wp.kernel
def Get_NN_input_3d(
   sigma:wp.array(dtype=wp.mat33f),    # Cauchy Stress
   pl_mult:wp.array(dtype=wp.float32), # Plastic multiplier
   thetas:wp.array(dtype=wp.float32),  # Lode angle theta
   scale:wp.array(dtype=wp.float32),   # Scale
   inp1:wp.array2d(dtype=wp.float32),  # NN input 1
   inp2:wp.array2d(dtype=wp.float32)   # NN input 2
   ): 
   
   tid = wp.tid()
   rho,theta = sig_2_rt(sigma[tid])
   thetas[tid] = theta

   inp1[0,tid] = rho * scale[0]
   inp1[1,tid] = wp.sin(theta*3.0) * scale[1]
   inp1[2,tid] = pl_mult[tid] * scale[2]

   inp2[0,tid] = 0.0 * scale[0]
   inp2[1,tid] = wp.sin(theta*3.0) * scale[1]
   inp2[2,tid] = pl_mult[tid] * scale[2]

@wp.kernel
def Get_z(
   out1:wp.array2d(dtype=float), 
   w:wp.array2d(dtype=float),   
   z:wp.array3d(dtype=float)
   ):   
   
   j,i,k = wp.tid()
   x = out1[i,j]
   val = apply_grad_elu(x)
   z[j,i,k] = val*w[i,k]

@wp.kernel
def loss_fun(
   x_pred:wp.array(dtype=wp.vec3f),   
   x0:wp.array(dtype=wp.vec3f),       
   x_true:wp.array(dtype=wp.vec3f),
   x_prev:wp.array(dtype=wp.vec3f),
   loss:wp.array(dtype=wp.float32),
   flag:wp.array(dtype=wp.float32),   
   dev:wp.float32
   ):  

   tid = wp.tid()

   loss_i = ((x_pred[tid][0]) - x_true[tid][0])**wp.float32(2.0) + \
            ((x_pred[tid][1]) - x_true[tid][1])**wp.float32(2.0) + \
            ((x_pred[tid][2]) - x_true[tid][2])**wp.float32(2.0)
   
   loss_prev = ((x0[tid][0]) - x_prev[tid][0])**wp.float32(2.0) + \
               ((x0[tid][1]) - x_prev[tid][1])**wp.float32(2.0) + \
               ((x0[tid][2]) - x_prev[tid][2])**wp.float32(2.0)
   #loss_i = flag[tid] * loss_i
   wp.atomic_add(loss,0,loss_i-loss_prev)


