import warp as wp 
from Utils.wp_MPM import MPM_vars,MPM_state

'''
Geometry functions 

'''

@wp.func
def cube(x:wp.vec3f,MPM:MPM_vars):
    
   ls = wp.float32(0.0)
   x1 = wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_x)
   x2 = MPM.l_edge - wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_x)
   
   y1 = wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_y)
   y2 = MPM.l_edge - wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_y)

   z1 = wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_z)
   z2 = MPM.l_edge - wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_z)
   
   if x[0] > x1 and x[0] < x2 and x[1] > y1 and x[1] < y2 and x[2] > z1 and x[2] < z2:
      ls = wp.float32(1.0) # Material
   return ls

@wp.func
def Assymetric_holes(x:wp.vec3f,MPM:MPM_vars):
    
   ls = wp.float32(0.0)
   flag = wp.float32(1.0)

   x1 = wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_x)
   x2 = MPM.l_edge - wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_x)
   
   y1 = wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_y)
   y2 = MPM.l_edge - wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_y)

   z1 = wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_z)
   z2 = MPM.l_edge - wp.float32(1.0/2.0) * (MPM.l_edge - MPM.length_z)
   
   cz1 = MPM.l_edge*0.6
   cz2 = MPM.l_edge*0.4

   rr1 = wp.sqrt((x[0] - x1) **wp.float32(2.0) + (x[2] - cz1) **wp.float32(2.0))
   rr2 = wp.sqrt((x[0] - x2) **wp.float32(2.0) + (x[2] - cz2) **wp.float32(2.0))

   if x[0] > x1 and x[0] < x2 and x[1] > y1 and x[1] < y2 and x[2] > z1 and x[2] < z2:
      ls = wp.float32(1.0) # Material
   if rr1 < MPM.r0 : 
      ls = 0.0             # Hole
   if rr2 < MPM.r0 : 
      ls = 0.0             # Hole
   # if ls==1.0 and (y1+y2)/2.0-MPM.dx/2.0< x[1] and x[1] < (y1+y2)/2.0:
   #    flag=1.0
   return ls,flag

@wp.func
def Identity3():
   v = wp.vec3f(wp.float32(1.0),wp.float32(1.0),wp.float32(1.0))
   return wp.diag(v)

@wp.kernel
def geometry(state:MPM_state,
             MPM_vars:MPM_vars,
             particles_id:wp.array(dtype=int),   # type:ignore
             gp:wp.array(dtype=wp.vec3f), # type:ignore
             n_grid:int):

   for i in range(n_grid):
      for j in range(n_grid):
            for k in range(n_grid):
               xx = wp.vec3f(MPM_vars.dx*wp.float32(i),MPM_vars.dx*wp.float32(j),MPM_vars.dx*wp.float32(k))
               for p in range(8):
                  ls,f = Assymetric_holes(xx+gp[p],MPM_vars)
                  if ls >0.5:
                     v = wp.vec3f(wp.float32(0.0),wp.float32(0.0),wp.float32(0.0))
                     
                     state.x[0,particles_id[0]] = xx + gp[p]
                     state.v[0,particles_id[0]] = v
                     state.F[0,particles_id[0]] = Identity3()
                     state.L[0,particles_id[0]] = wp.mat33f()
                     state.flag[particles_id[0]] = f
                     wp.atomic_add(particles_id,0,1)