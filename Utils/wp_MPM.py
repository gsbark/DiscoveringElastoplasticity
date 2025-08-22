from torch import Tensor
import warp as wp 

@wp.struct
class MPM_state:
   
   x: wp.array2d(dtype=wp.vec3f)          #type:ignore
   v: wp.array2d(dtype=wp.vec3f)          #type:ignore
   F: wp.array2d(dtype=wp.mat33f)         #type:ignore
   L: wp.array2d(dtype=wp.mat33f)         #type:ignore
   stress:wp.array2d(dtype=wp.mat33f)     #type:ignore
   log_e:wp.array2d(dtype=wp.mat33f)      #type:ignore

   theta:wp.array2d(dtype=wp.float32)     #type:ignore
   flag:wp.array(dtype=wp.float32)        #type:ignore

   grid_v: wp.array4d(dtype=wp.vec3f)     #type:ignore
   grid_mv: wp.array4d(dtype=wp.vec3f)    #type:ignore
   grid_m: wp.array4d(dtype=wp.float32)   #type:ignore

   def initialize(self,steps:int,n_grid:int,num_particles:int,requires_grad:bool,device):

      self.x = wp.zeros(shape=(steps,num_particles),dtype=wp.vec3f,requires_grad=requires_grad,device=device)
      self.v = wp.zeros(shape=(steps,num_particles),dtype=wp.vec3f,requires_grad=requires_grad,device=device)
      self.F = wp.zeros(shape=(steps,num_particles),dtype=wp.mat33f,requires_grad=requires_grad,device=device)
      self.L = wp.zeros(shape=(steps,num_particles),dtype=wp.mat33f,requires_grad=requires_grad,device=device)
      self.stress = wp.zeros(shape=(steps,num_particles),dtype=wp.mat33f,requires_grad=requires_grad,device=device)
      self.log_e = wp.zeros(shape=(steps,num_particles),dtype=wp.mat33f,requires_grad=requires_grad,device=device)
      
      self.theta =  wp.zeros(shape=(steps,num_particles),dtype=wp.float32,requires_grad=requires_grad,device=device)
      self.flag = wp.zeros(num_particles,dtype=wp.float32,requires_grad=requires_grad,device=device)

      self.grid_v = wp.zeros(shape=(steps,n_grid,n_grid,n_grid),dtype=wp.vec3f,requires_grad=requires_grad,device=device)
      self.grid_mv = wp.zeros(shape=(steps,n_grid,n_grid,n_grid),dtype=wp.vec3f,requires_grad=requires_grad,device=device)
      self.grid_m = wp.zeros(shape=(steps,n_grid,n_grid,n_grid),dtype=wp.float32,requires_grad=requires_grad,device=device)
   
   def reset_particles(self):
      
      self.x.zero_()
      self.v.zero_()
      self.F.zero_()
      self.L.zero_()
      self.stress.zero_()
      self.log_e.zero_()
      self.flag.zero_()   

   def reset_grid(self):
      self.grid_v.zero_()
      self.grid_m.zero_()
      self.grid_mv.zero_()

   def reset_substeps(self,steps):
      self.x[0].assign(self.x[-1])
      self.v[0].assign(self.v[-1])
      self.F[0].assign(self.F[-1])
      self.L[0].assign(self.L[-1])
      self.stress[0].assign(self.stress[-1])
      self.log_e[0].assign(self.log_e[-1])
      for i in range(1,steps):
         self.x[i].zero_()
         self.v[i].zero_()
         self.F[i].zero_()
         self.L[i].zero_()
         self.stress[i].zero_()
         self.log_e[i].zero_()
      self.reset_grid()

@wp.struct
class MPM_vars:

   inv_dx:wp.float32
   dx:wp.float32
   dt:wp.float32
   p_rho:wp.float32
   p_vol:wp.float32
   p_mass:wp.float32
   l_edge:wp.float32
   n_grid:int

   length_x:wp.float32
   length_y:wp.float32
   length_z:wp.float32
   r0:wp.float32

   E:wp.float32
   n:wp.float32
   K:wp.float32
   mu:wp.float32
   lamda:wp.float32

   def initialize(self,Geometry,MPM_config,Analysis_config,Material):
      
      self.n_grid = MPM_config['n_grid']
      self.length_x = Geometry['length_x']
      self.length_y = Geometry['length_y']
      self.length_z = Geometry['length_z']
      self.r0 = Geometry['radius']
      self.p_rho = MPM_config['p_rho']
      self.l_edge =  MPM_config['l_edge']
      self.dt = Analysis_config['dt']
      self.dx = self.l_edge / float(self.n_grid)
      self.inv_dx = 1.0/ self.dx

      self.p_vol = self.dx * self.dx * self.dx / 8.0
      self.p_mass = self.p_vol * self.p_rho
      
      self.E = Material['E']
      self.n = Material['n']

      self.mu = float(self.E/(2*(1+self.n)))
      self.lamda = float((self.E*self.n)/((1.0+self.n)*(1.0-2.0*self.n)))
      self.K  = float(self.E/(3.0*(1.0-2.0*self.n)))
      
class wp_MLP:
   def __init__(self,params:Tensor,batch:int,steps:int,device,w_norm:bool):
      
      self.w1 = wp.array(params['layers.0.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.b1 = wp.array(params['layers.0.bias'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      
      self.w2 = wp.array(params['layers.2.weight'].numpy(),requires_grad=True,device=device)
      self.b2 = wp.array(params['layers.2.bias'].numpy(),requires_grad=True,device=device)
      
      self.w3 = wp.array(params['layers.4.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.b3 = wp.array(params['layers.4.bias'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      
      self.out1 = wp.zeros((steps,self.w1.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.out11 = wp.zeros((steps,self.w1.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.z1 = wp.zeros((steps,batch,self.w1.shape[0],self.w1.shape[1]),dtype=wp.float32,requires_grad=True,device=device)
      
      self.out2 = wp.zeros((steps,self.w2.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.out22 = wp.zeros((steps,self.w2.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.z2 = wp.zeros((steps,batch,self.w2.shape[0],self.w2.shape[1]),dtype=wp.float32,requires_grad=True,device=device)

      self.out3 = wp.zeros((steps,self.w3.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      self.out33 = wp.zeros((steps,self.w3.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)
      
      self.z2z1 = wp.zeros((steps,batch,self.w2.shape[1],self.w1.shape[1]),dtype=wp.float32,requires_grad=True,device=device)
      self.f_val = wp.zeros((steps,batch),dtype=wp.float32,requires_grad=True,device=device)  
      self.f_max = wp.zeros((steps,batch),dtype=wp.float32,requires_grad=True,device=device)
      
      self.w3_exp = wp.zeros((steps,batch,self.w3.shape[0],self.w3.shape[1]),dtype=wp.float32,requires_grad=True,device=device)
      self.df_dx = wp.zeros((steps,batch,1,self.w1.shape[1]),dtype=wp.float32,requires_grad=True,device=device)
      self.df_dx_true = wp.zeros((steps,batch,self.w1.shape[1]),dtype=wp.float32,requires_grad=True,device=device)
      
      self.w1_copy = wp.array(params['layers.0.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.w2_copy =  wp.array(params['layers.2.weight'].numpy(),requires_grad=True,device=device)
      self.w3_copy = wp.array(params['layers.4.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)

      self.dummy1 = wp.zeros_like(self.z2z1)
      self.dummy2 = wp.zeros_like(self.df_dx)

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
      self.w3_exp.zero_()
      self.df_dx.zero_()
      self.df_dx_true.zero_()
