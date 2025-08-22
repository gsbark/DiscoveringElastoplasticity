import warp as wp 
from wp_MPM import MPM_vars


@wp.kernel
def p2g(MPM:MPM_vars,
        x:wp.array(dtype=wp.vec3f),
        v:wp.array(dtype=wp.vec3f),
        F:wp.array(dtype=wp.mat33f),
        L:wp.array(dtype=wp.mat33f),
        stress:wp.array(dtype=wp.mat33f),
        grid_m:wp.array3d(dtype=float),
        grid_mv:wp.array3d(dtype=wp.vec3f)):

   tid = wp.tid()

   base_x = int(x[tid][0]*MPM.inv_dx-wp.float32(0.5))
   base_y = int(x[tid][1]*MPM.inv_dx-wp.float32(0.5))
   base_z = int(x[tid][2]*MPM.inv_dx-wp.float32(0.5))

   f_x = x[tid][0]*MPM.inv_dx - wp.float32(base_x)
   f_y = x[tid][1]*MPM.inv_dx - wp.float32(base_y)
   f_z = x[tid][2]*MPM.inv_dx - wp.float32(base_z)

   w = wp.mat33f(wp.float32(0.5) *(wp.float32(1.5)-f_x)**wp.float32(2.0), 
                  wp.float32(0.5) *(wp.float32(1.5)-f_y)**wp.float32(2.0),
                  wp.float32(0.5) *(wp.float32(1.5)-f_z)**wp.float32(2.0),
                  wp.float32(0.75)-(f_x-wp.float32(1.0))**wp.float32(2.0),
                  wp.float32(0.75)-(f_y-wp.float32(1.0))**wp.float32(2.0),
                  wp.float32(0.75)-(f_z-wp.float32(1.0))**wp.float32(2.0),
                  wp.float32(0.5) *(f_x-wp.float32(0.5))**wp.float32(2.0),
                  wp.float32(0.5) *(f_y-wp.float32(0.5))**wp.float32(2.0),
                  wp.float32(0.5) *(f_z-wp.float32(0.5))**wp.float32(2.0))
   
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
def grid_update(MPM:MPM_vars,
                  grid_m:wp.array3d(dtype=float),
                  grid_mv:wp.array3d(dtype=wp.vec3f),
                  grid_v:wp.array3d(dtype=wp.vec3f)):
   
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
      grid_v[i,j,k][0] = wp.float32(-20.0)
      grid_v[i,j,k][1] = wp.float32(0.0)
      grid_v[i,j,k][2] = wp.float32(0.0)

@wp.kernel
def g2p(MPM:MPM_vars,
      x:wp.array(dtype=wp.vec3f),
      F:wp.array(dtype=wp.mat33f),
      grid_v:wp.array3d(dtype=wp.vec3f),
      x_new:wp.array(dtype=wp.vec3f),
      v_new:wp.array(dtype=wp.vec3f),
      F_new:wp.array(dtype=wp.mat33f),
      L_new:wp.array(dtype=wp.mat33f)):            
   
   tid = wp.tid()
   base_x = int(x[tid][0]*MPM.inv_dx-wp.float32(0.5))
   base_y = int(x[tid][1]*MPM.inv_dx-wp.float32(0.5))
   base_z = int(x[tid][2]*MPM.inv_dx-wp.float32(0.5))

   f_x = x[tid][0]*MPM.inv_dx - wp.float32(base_x)
   f_y = x[tid][1]*MPM.inv_dx - wp.float32(base_y)
   f_z = x[tid][2]*MPM.inv_dx - wp.float32(base_z)

   w = wp.mat33f(wp.float32(0.5) *(wp.float32(1.5)-f_x)**wp.float32(2.0), 
                  wp.float32(0.5) *(wp.float32(1.5)-f_y)**wp.float32(2.0),
                  wp.float32(0.5) *(wp.float32(1.5)-f_z)**wp.float32(2.0),
                  wp.float32(0.75)-(f_x-wp.float32(1.0))**wp.float32(2.0),
                  wp.float32(0.75)-(f_y-wp.float32(1.0))**wp.float32(2.0),
                  wp.float32(0.75)-(f_z-wp.float32(1.0))**wp.float32(2.0),
                  wp.float32(0.5) *(f_x-wp.float32(0.5))**wp.float32(2.0),
                  wp.float32(0.5) *(f_y-wp.float32(0.5))**wp.float32(2.0),
                  wp.float32(0.5) *(f_z-wp.float32(0.5))**wp.float32(2.0))
   
   new_v = wp.vec3f()
   new_L = wp.mat33f()
   for i in range(3):
      for j in range(3):
            for k in range(3):
               weight = w[i][0] * w[j][1] * w[k][2]
               g_v = grid_v[base_x + i,base_y + j, base_z + k]
               new_v = new_v + weight * g_v
               dpos =  wp.vec3((float(i)- f_x) * MPM.dx, (float(j) - f_y) * MPM.dx, (float(k) - f_z) * MPM.dx)
               new_L =  new_L + (4.0*weight*MPM.inv_dx*MPM.inv_dx)*wp.outer(g_v,dpos)
               
   v_new[tid] = new_v
   L_new[tid] = new_L
   x_new[tid] = x[tid] + MPM.dt * new_v
   F_new[tid] = F[tid] + MPM.dt * (new_L @ F[tid])
