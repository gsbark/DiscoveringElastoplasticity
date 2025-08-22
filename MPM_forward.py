import numpy as np 
import warp as wp 
import os 
import shutil
from Utils.wp_MPM import MPM_state,MPM_vars
from Utils.material_laws import stress_update33
from Utils.geometry import geometry
from Utils.wp_MPM_funcs import p2g,grid_update,g2p

class MPM_program:
   def __init__(self,MPM_config,Material,Analysis_config,Geometry,device):
   
      self.device = device
      self.num_particles = MPM_config['particles']
      self.n_grid = MPM_config['n_grid']
      self.steps = Analysis_config['steps']

      self.MPM_var = MPM_vars()
      self.curr_state = MPM_state()

      self.curr_state.initialize(1,self.n_grid,self.num_particles,requires_grad=False,device=device)
      self.MPM_var.initialize(Geometry,MPM_config,Analysis_config,Material)
      
      self.particles_id = wp.zeros(1,dtype=int,device=device)
      a = np.sqrt(3.0)/6

      gauss_points = self.MPM_var.dx * np.array([[0.5 - a, 0.5 - a, 0.5 - a],
                                                 [0.5 - a, 0.5 + a, 0.5 - a],
                                                 [0.5 + a, 0.5 - a, 0.5 - a],
                                                 [0.5 + a, 0.5 + a, 0.5 - a],
                                                 [0.5 - a, 0.5 - a, 0.5 + a],
                                                 [0.5 - a, 0.5 + a, 0.5 + a],
                                                 [0.5 + a, 0.5 - a, 0.5 + a],
                                                 [0.5 + a, 0.5 + a, 0.5 + a]],dtype=np.float64)
      
      self.gp = wp.from_numpy(gauss_points,device=device,dtype=wp.vec3f)
      self.x_dataset = np.zeros(shape=(self.steps+1,self.num_particles,3))
   
      #Output
      shutil.rmtree('./Output/dataset',ignore_errors=True)
      os.makedirs('./Output/dataset',exist_ok=True)

      shutil.rmtree('./Output/Paraview',ignore_errors=True)
      os.makedirs('./Output/Paraview',exist_ok=True)

   def reset_geo(self):
      self.particles_id.zero_()
      self.curr_state.reset_particles()
      wp.launch(
         kernel=geometry,
         dim=[1],
         inputs=[self.curr_state,self.MPM_var,self.particles_id,self.gp,self.n_grid]
         )
      self.x0 = np.copy(self.curr_state.x[0].numpy())
      assert self.num_particles == self.particles_id.numpy()[0], self.particles_id.numpy()[0]

   def step(self):

      wp.launch(
         kernel=self.Stress_update,
         dim=[self.num_particles],
         inputs=[self.MPM_var,self.curr_state.log_e[0],
                 self.curr_state.L[0],self.curr_state.stress[0]]
                 )
      wp.launch(
         kernel=p2g,
         dim=[self.num_particles],
         inputs=[self.MPM_var,self.curr_state.x[0],self.curr_state.v[0],self.curr_state.F[0],self.curr_state.L[0],
                 self.curr_state.stress[0],self.curr_state.grid_m[0],self.curr_state.grid_mv[0]]
                 )
      wp.launch(
         kernel=grid_update,
         dim=[self.n_grid,self.n_grid,self.n_grid],
         inputs=[self.MPM_var,self.curr_state.grid_m[0],self.curr_state.grid_mv[0],
                 self.curr_state.grid_v[0]]
                 )
      wp.launch(
         kernel=g2p,
         dim=[self.num_particles],
         inputs=[self.MPM_var,self.curr_state.x[0],self.curr_state.F[0],self.curr_state.grid_v[0],
                 self.curr_state.x[0],self.curr_state.v[0],self.curr_state.F[0],self.curr_state.L[0]]
                 )
      
   def run_forward(self):
      self.reset_geo()
      self.x_dataset[0] = self.x0
      with wp.ScopedTimer('Sim'):
         for i in range(self.steps):
               self.curr_state.reset_grid()
               self.step() 
               self.x_dataset[i+1] = self.curr_state.x[0].numpy()
      np.save(f'./Output/dataset/positions.npy',self.x_dataset)
      print('Finishing..')
       
   @staticmethod
   @wp.kernel
   def Stress_update(MPM:MPM_vars,
                     log_e:wp.array(dtype=wp.mat33f),
                     L:wp.array(dtype=wp.mat33f),
                     new_stress:wp.array(dtype=wp.mat33f)):
      tid = wp.tid()
      tau,e = stress_update33(L[tid],log_e[tid],MPM.mu,MPM.K,MPM.dt)
      new_stress[tid] = tau
      log_e[tid] = e
                        
               
   