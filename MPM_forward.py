import numpy as np 
import warp as wp 
import os 
import shutil
from src.wp_MPM import MPM_state,MPM_vars,p2g,grid_update,g2p
from src.material_laws import ST_update,ST_update_hardening
from src.utils import make_mat_points

class MPM_program:
   def __init__(self,MPM_config,Material,Analysis_config,Geometry,device,**kwargs):
   
      self.device = device
      self.num_particles = MPM_config['particles']
      self.n_grid = MPM_config['n_grid']
      self.steps = Analysis_config['steps']
      self.isHardening = Analysis_config['Hardening']
      
      if self.isHardening:
         self.path = './Input/3D_input'
      else:
         self.path = './Input/2D_input'
      
      self.MPM_var = MPM_vars()
      self.curr_state = MPM_state()
      self.curr_state.initialize(1,self.n_grid,self.num_particles,requires_grad=False,device=device)
      self.MPM_var.initialize(Geometry,MPM_config,Analysis_config,Material)

      self.mat_points_pos,_ = make_mat_points(self.MPM_var)
      self.x_dataset = np.zeros(shape=(self.steps+1,self.num_particles,3))
   
   def reset_geo(self):
      
      self.curr_state.reset_particles()
      self.curr_state.x[0].assign(self.mat_points_pos)
      self.curr_state.F[0].fill_(wp.diag(wp.vec3f(wp.float32(1.0),wp.float32(1.0),wp.float32(1.0))))
      self.x0 = np.copy(self.curr_state.x[0].numpy())
   
   def step(self):

      if self.isHardening:
         wp.launch(
         kernel=self.Stress_update_hardening,
         dim=[self.num_particles],
         inputs=[
            self.MPM_var,
            self.curr_state.log_e[0],
            self.curr_state.L[0],
            self.curr_state.stress[0],
            self.curr_state.pl_mult[0]
            ]
            )
      else:
         wp.launch(
            kernel=self.Stress_update,
            dim=[self.num_particles],
            inputs=[
               self.MPM_var,
               self.curr_state.log_e[0],
               self.curr_state.L[0],
               self.curr_state.stress[0]
               ]
               )
      
      wp.launch(
         kernel=p2g,
         dim=[self.num_particles],
         inputs=[
            self.MPM_var,
            self.curr_state.x[0],
            self.curr_state.v[0],
            self.curr_state.F[0],
            self.curr_state.L[0],
            self.curr_state.stress[0],
            self.curr_state.grid_m[0],
            self.curr_state.grid_mv[0]
            ]
            )
      
      wp.launch(
         kernel=grid_update,
         dim=[self.n_grid,self.n_grid,self.n_grid],
         inputs=[
            self.MPM_var,
            self.curr_state.grid_m[0],
            self.curr_state.grid_mv[0],
            self.curr_state.grid_v[0]
            ]
            )
      
      wp.launch(
         kernel=g2p,
         dim=[self.num_particles],
         inputs=[
            self.MPM_var,
            self.curr_state.x[0],
            self.curr_state.F[0],
            self.curr_state.grid_v[0],
            self.curr_state.x[0],
            self.curr_state.v[0],
            self.curr_state.F[0],
            self.curr_state.L[0]
            ]
            )
      
   def run_forward(self):
      self.reset_geo()
      self.x_dataset[0] = self.x0
      with wp.ScopedTimer('Sim'):
         for i in range(self.steps):
               self.curr_state.reset_grid()
               self.step() 
               self.x_dataset[i+1] = self.curr_state.x[0].numpy()
      
      np.save(self.path + '/positions.npy',self.x_dataset)
      print('Finishing..')
       
   @staticmethod
   @wp.kernel
   def Stress_update(
      MPM:MPM_vars,
      log_e:wp.array(dtype=wp.mat33f),
      L:wp.array(dtype=wp.mat33f),
      new_stress:wp.array(dtype=wp.mat33f)
      ):

      tid = wp.tid()
      tau,e = ST_update(
         L[tid],
         log_e[tid],
         MPM.mu,
         MPM.K,
         MPM.dt
         )
      new_stress[tid] = tau
      log_e[tid] = e

   @staticmethod
   @wp.kernel
   def Stress_update_hardening(
      MPM:MPM_vars,
      log_e:wp.array(dtype=wp.mat33f),
      L:wp.array(dtype=wp.mat33f),
      new_stress:wp.array(dtype=wp.mat33f),
      pl_mult:wp.array(dtype=wp.float32)
      ):

      tid = wp.tid()
      tau,e,e_pl = ST_update_hardening(
         L[tid],
         log_e[tid],
         pl_mult[tid],
         MPM.mu,
         MPM.K,
         MPM.dt
         )
      new_stress[tid] = tau
      log_e[tid] = e
      pl_mult[tid] = e_pl
                        
               
   