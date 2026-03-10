import warp as wp 

@wp.struct
class MPM_state:
   
   x: wp.array2d(dtype=wp.vec3f)          
   v: wp.array2d(dtype=wp.vec3f)          
   F: wp.array2d(dtype=wp.mat33f)         
   L: wp.array2d(dtype=wp.mat33f)         
   
   stress:wp.array2d(dtype=wp.mat33f)     
   log_e:wp.array2d(dtype=wp.mat33f)   
   pl_mult:wp.array2d(dtype=wp.float32) 

   theta:wp.array2d(dtype=wp.float32)     
   flag:wp.array(dtype=wp.float32)        

   grid_v: wp.array4d(dtype=wp.vec3f)     
   grid_mv: wp.array4d(dtype=wp.vec3f)    
   grid_m: wp.array4d(dtype=wp.float32)   

   def initialize(self,steps:int,n_grid:int,num_particles:int,requires_grad:bool,device):

      self.x = wp.zeros(shape=(steps,num_particles),dtype=wp.vec3f,requires_grad=requires_grad,device=device)
      self.v = wp.zeros(shape=(steps,num_particles),dtype=wp.vec3f,requires_grad=requires_grad,device=device)
      self.F = wp.zeros(shape=(steps,num_particles),dtype=wp.mat33f,requires_grad=requires_grad,device=device)
      self.L = wp.zeros(shape=(steps,num_particles),dtype=wp.mat33f,requires_grad=requires_grad,device=device)
   
      self.stress = wp.zeros(shape=(steps,num_particles),dtype=wp.mat33f,requires_grad=requires_grad,device=device)
      self.log_e = wp.zeros(shape=(steps,num_particles),dtype=wp.mat33f,requires_grad=requires_grad,device=device)
      self.pl_mult =  wp.zeros(shape=(steps,num_particles),dtype=wp.float32,requires_grad=requires_grad,device=device)

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
      self.pl_mult.zero_()
      
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
      self.pl_mult[0].assign(self.pl_mult[-1])
      for i in range(1,steps):
         self.x[i].zero_()
         self.v[i].zero_()
         self.F[i].zero_()
         self.L[i].zero_()
         self.stress[i].zero_()
         self.log_e[i].zero_()
         self.pl_mult[i].zero_()
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
      self.inv_dx = 1.0 / self.dx

      self.p_vol = self.dx * self.dx * self.dx / 8.0
      self.p_mass = self.p_vol * self.p_rho
      
      self.E = Material['E']
      self.n = Material['n']

      self.mu = float(self.E/(2*(1+self.n)))
      self.lamda = float((self.E*self.n)/((1.0+self.n)*(1.0-2.0*self.n)))
      self.K  = float(self.E/(3.0*(1.0-2.0*self.n)))
      


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

@wp.kernel
def p2g(
    MPM:MPM_vars,
    x:wp.array(dtype=wp.vec3f),
    v:wp.array(dtype=wp.vec3f),
    F:wp.array(dtype=wp.mat33f),
    L:wp.array(dtype=wp.mat33f),
    stress:wp.array(dtype=wp.mat33f),
    grid_m:wp.array3d(dtype=wp.float32),
    grid_mv:wp.array3d(dtype=wp.vec3f)
    ):

   tid = wp.tid()

   base_x = int(x[tid][0]*MPM.inv_dx-wp.float32(0.5))
   base_y = int(x[tid][1]*MPM.inv_dx-wp.float32(0.5))
   base_z = int(x[tid][2]*MPM.inv_dx-wp.float32(0.5))

   f_x = x[tid][0]*MPM.inv_dx - wp.float32(base_x)
   f_y = x[tid][1]*MPM.inv_dx - wp.float32(base_y)
   f_z = x[tid][2]*MPM.inv_dx - wp.float32(base_z)

   w = wp.mat33f(
       wp.float32(0.5) *(wp.float32(1.5)-f_x)**wp.float32(2.0), 
       wp.float32(0.5) *(wp.float32(1.5)-f_y)**wp.float32(2.0),
       wp.float32(0.5) *(wp.float32(1.5)-f_z)**wp.float32(2.0),
       wp.float32(0.75)-(f_x-wp.float32(1.0))**wp.float32(2.0),
       wp.float32(0.75)-(f_y-wp.float32(1.0))**wp.float32(2.0),
       wp.float32(0.75)-(f_z-wp.float32(1.0))**wp.float32(2.0),
       wp.float32(0.5) *(f_x-wp.float32(0.5))**wp.float32(2.0),
       wp.float32(0.5) *(f_y-wp.float32(0.5))**wp.float32(2.0),
       wp.float32(0.5) *(f_z-wp.float32(0.5))**wp.float32(2.0)
       )
   
   Jdet = wp.determinant(F[tid])
   sigma = stress[tid]/Jdet
   MLS_stress = (-MPM.dt*MPM.p_vol*Jdet*4.0*MPM.inv_dx*MPM.inv_dx)*sigma
   MLS_affine = MLS_stress + MPM.p_mass*L[tid]

   for i in range(3):
      for j in range(3):
            for k in range(3):
               offset = wp.vec3f(wp.float32(i), wp.float32(j),wp.float32(k))
               dpos =  wp.vec3f((offset[0] - f_x) * MPM.dx, (offset[1] - f_y) * MPM.dx, (offset[2] - f_z) * MPM.dx)
               weight = w[i,0] * w[j,1]* w[k,2]
               momentum =  weight * (MPM.p_mass* v[tid] + MLS_affine * dpos)
               mass = weight * MPM.p_mass
               wp.atomic_add(grid_mv,i= base_x+i ,j=base_y+j,k=base_z+k,value=momentum)
               wp.atomic_add(grid_m, i=base_x+i ,j=base_y+j,k=base_z+k,value=mass)

@wp.kernel
def grid_update(
    MPM:MPM_vars,
    grid_m:wp.array3d(dtype=wp.float32),
    grid_mv:wp.array3d(dtype=wp.vec3f),
    grid_v:wp.array3d(dtype=wp.vec3f)
    ):
   
   i,j,k = wp.tid()
   if grid_m[i,j,k]>0.0:
      grid_v[i,j,k] = grid_mv[i,j,k]/grid_m[i,j,k]
   #Boundary conditions
   dist_z = MPM.dx * wp.float32(k)
   if dist_z < wp.float32(0.5) * (MPM.l_edge - MPM.length_z): 
      grid_v[i,j,k][0] = wp.float32(0.0)
      grid_v[i,j,k][1] = wp.float32(0.0)
      grid_v[i,j,k][2] = wp.float32(0.0)

   if dist_z > wp.float32(0.8) * MPM.l_edge: 
      grid_v[i,j,k][0] = wp.float32(-50.0)
      grid_v[i,j,k][1] = wp.float32(0.0)
      grid_v[i,j,k][2] = wp.float32(0.0)

@wp.kernel
def g2p(
    MPM:MPM_vars,
    x:wp.array(dtype=wp.vec3f),
    F:wp.array(dtype=wp.mat33f),
    grid_v:wp.array3d(dtype=wp.vec3f),
    x_new:wp.array(dtype=wp.vec3f),
    v_new:wp.array(dtype=wp.vec3f),
    F_new:wp.array(dtype=wp.mat33f),
    L_new:wp.array(dtype=wp.mat33f)
    ):            
   
   tid = wp.tid()
   base_x = int(x[tid][0]*MPM.inv_dx-wp.float32(0.5))
   base_y = int(x[tid][1]*MPM.inv_dx-wp.float32(0.5))
   base_z = int(x[tid][2]*MPM.inv_dx-wp.float32(0.5))

   f_x = x[tid][0]*MPM.inv_dx - wp.float32(base_x)
   f_y = x[tid][1]*MPM.inv_dx - wp.float32(base_y)
   f_z = x[tid][2]*MPM.inv_dx - wp.float32(base_z)

   w = wp.mat33f(
       wp.float32(0.5) *(wp.float32(1.5)-f_x)**wp.float32(2.0), 
       wp.float32(0.5) *(wp.float32(1.5)-f_y)**wp.float32(2.0),
       wp.float32(0.5) *(wp.float32(1.5)-f_z)**wp.float32(2.0),
       wp.float32(0.75)-(f_x-wp.float32(1.0))**wp.float32(2.0),
       wp.float32(0.75)-(f_y-wp.float32(1.0))**wp.float32(2.0),
       wp.float32(0.75)-(f_z-wp.float32(1.0))**wp.float32(2.0),
       wp.float32(0.5) *(f_x-wp.float32(0.5))**wp.float32(2.0),
       wp.float32(0.5) *(f_y-wp.float32(0.5))**wp.float32(2.0),
       wp.float32(0.5) *(f_z-wp.float32(0.5))**wp.float32(2.0)
       )
   
   new_v = wp.vec3f()
   new_L = wp.mat33f()
   for i in range(3):
      for j in range(3):
            for k in range(3):
               weight = w[i][0] * w[j][1] * w[k][2]
               g_v = grid_v[base_x + i,base_y + j, base_z + k]
               new_v = new_v + weight * g_v
               dpos =  wp.vec3((wp.float32(i)- f_x) * MPM.dx, (wp.float32(j) - f_y) * MPM.dx, (wp.float32(k) - f_z) * MPM.dx)
               new_L =  new_L + (4.0*weight*MPM.inv_dx*MPM.inv_dx)*wp.outer(g_v,dpos)
               
   v_new[tid] = new_v
   L_new[tid] = new_L
   x_new[tid] = x[tid] + MPM.dt * new_v
   F_new[tid] = F[tid] + MPM.dt * (new_L @ F[tid])