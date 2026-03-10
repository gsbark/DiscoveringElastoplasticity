import numpy as np 
from src.wp_MPM import MPM_vars,MPM_state
import torch
import vtk

'''
Geometry functions 

'''

def Assymetric_holes(x,MPM:MPM_vars):
    
   ls = 0.0
   flag = 1.0

   x1 = 0.5 * (MPM.l_edge - MPM.length_x)
   x2 = MPM.l_edge - 0.5 * (MPM.l_edge - MPM.length_x)
   
   y1 = 0.5 * (MPM.l_edge - MPM.length_y)
   y2 = MPM.l_edge - 0.5 * (MPM.l_edge - MPM.length_y)

   z1 = 0.5 * (MPM.l_edge - MPM.length_z)
   z2 = MPM.l_edge - 0.5 * (MPM.l_edge - MPM.length_z)
   
   cz1 = MPM.l_edge*0.6
   cz2 = MPM.l_edge*0.4

   rr1 = np.sqrt((x[0] - x1) **2.0 + (x[2] - cz1) **2.0)
   rr2 = np.sqrt((x[0] - x2) **2.0 + (x[2] - cz2) **2.0)

   if x[0] > x1 and x[0] < x2 and x[1] > y1 and x[1] < y2 and x[2] > z1 and x[2] < z2:
      ls = 1.0             # Material
   if rr1 < MPM.r0 : 
      ls = 0.0             # Hole
   if rr2 < MPM.r0 : 
      ls = 0.0             # Hole
   # if ls==1.0 and (y1+y2)/2.0-MPM.dx/2.0< x[1] and x[1] < (y1+y2)/2.0:
   #    flag=1.0
   return ls,flag

def make_mat_points(Input):
   n_grid = Input.n_grid
   dx = Input.dx

   points_coords = np.zeros(shape=(8*n_grid**3,3)) 
   flag = np.zeros(shape=(8*n_grid**3)) 
   def gauss_points_unit_cell(k):
      xi, w = np.polynomial.legendre.leggauss(k)      
      x = 0.5 * (xi + 1.0)     # map to [0,1]
      pts = [(i, j, k ) for i in x for j in x for k in x]
      return np.array(pts)
   
   gp = dx*gauss_points_unit_cell(2)
   id = 0
   for i in range(n_grid):
      for j in range(n_grid):
         for k in range(n_grid):
            xx = dx *np.array([i,j,k])
            for p in range(8):
               ls,f = Assymetric_holes(xx+gp[p],Input)
               if ls >0.5:
                  points_coords[id] = xx + gp[p]
                  flag[id] =f
                  id+=1
   return points_coords[:id],flag[:id]
                     

def get_s_vm(sigma):
   s_vm = np.sqrt(0.5*((sigma[:,0,0]-sigma[:,1,1])**2 + \
                       (sigma[:,1,1]-sigma[:,2,2])**2 + \
                       (sigma[:,2,2]-sigma[:,0,0])**2 + \
                        6.0*(sigma[:,0,1]**2 + sigma[:,1,2]**2 + sigma[:,0,2]**2)))
   return s_vm                  
               
def polar_to_cartesian(r, theta):
   x = r * np.cos(theta)
   y = r * np.sin(theta)
   z = r * 0.0
   xx = np.stack((x,y,z),axis=1)
   return xx

#Export .vtk files for Paraview
#--------------------------------------------------------
def GaussPoints_export(Particles:np.ndarray,Val:np.ndarray,inc:int,fname:str,name:str):
   
   points = vtk.vtkPoints()
   polydata = vtk.vtkPolyData()
   polydata.SetPoints(points)
   for field_name in Val:
      num_components = Val[field_name].shape[1] if len(Val[field_name].shape) > 1 else 1
      values_array = vtk.vtkDoubleArray()
      values_array.SetNumberOfComponents(num_components)
      values_array.SetName(field_name)
      for val in Val[field_name]:
         if num_components == 1:
               values_array.InsertNextValue(val)
         else:
               values_array.InsertNextTuple(val)
      polydata.GetPointData().AddArray(values_array)
      
   for Id, particle in enumerate(Particles):
      points.InsertNextPoint(particle[0], particle[1],  particle[2])

   writer = vtk.vtkPolyDataWriter()
   fname = f'./{fname}/{name}_particle_data{inc}.vtk'
   writer.SetFileName(fname)
   writer.SetInputData(polydata)
   writer.Write()

def save_error_state(field:MPM_state,x_true,i_frame:int,folder:str):
   '''
   Particle theta-displacement error
   '''
   x = polar_to_cartesian(100*np.ones_like(field.theta.numpy()),field.theta.numpy())
   field_data = {'error':np.sum(np.abs(x_true.numpy()-field.x_new.numpy()),axis=1)}
   GaussPoints_export(x,field_data,i_frame,fname=folder,name='points')

def save_state(field:MPM_state,x0,i_frame:int,folder:str,index=0):
   
   field_data = {
      'stress_xx':field.stress[index].numpy()[:,0,0],
      'stress_yy':field.stress[index].numpy()[:,1,1],
      'stress_zz':field.stress[index].numpy()[:,2,2],
      'stress_xy':field.stress[index].numpy()[:,0,1],
      'stress_xz':field.stress[index].numpy()[:,0,2],
      'stress_yz':field.stress[index].numpy()[:,1,2],
      'stress_s_vm':get_s_vm(field.stress[index].numpy()),
      'theta':field.theta[index].numpy(),
      'velocity':field.v[index].numpy(),
      'displacement':field.x[index].numpy() - x0,
      'flag':field.flag.numpy(),
      }
   GaussPoints_export(field.x[index].numpy(),field_data,i_frame,fname=folder,name='mat_points')
#--------------------------------------------------------

class MLP(torch.nn.Module):
   def __init__(self,hidden_size):
      super().__init__()
      self.layers = torch.nn.ModuleList()
      self.layers_num = len(hidden_size)-1
      for i in range(self.layers_num):
         self.layers.append(torch.nn.Linear(hidden_size[i],hidden_size[i+1]))
         if i != self.layers_num - 1:
            self.layers.append(torch.nn.ELU())      
   def forward(self, x):
      for layer in self.layers:
         x = layer(x)
      return x
   
def find_YF(wp_mlp,Hardening,scale_inp,scale_out,num):
   
   theta = np.linspace(0,2*np.pi,num)
   if Hardening:
      dim = 3
      pl_lvls = 5
      rhoNN = np.zeros(theta.shape[0]*pl_lvls)
   else:
      dim = 2
      rhoNN = np.zeros(theta.shape[0])

   model = MLP(hidden_size=[dim,64,64,1])
   scale_x = torch.tensor(scale_inp.numpy())
   scale_y = torch.tensor(scale_out)

   model.layers[0].weight.data = torch.tensor(wp_mlp.w1.numpy())
   model.layers[0].bias.data = torch.tensor(wp_mlp.b1.numpy()[:,0])
   model.layers[2].weight.data = torch.tensor(wp_mlp.w2.numpy())
   model.layers[2].bias.data = torch.tensor(wp_mlp.b2.numpy()[:,0])
   model.layers[4].weight.data = torch.tensor(wp_mlp.w3.numpy())
   model.layers[4].bias.data = torch.tensor(wp_mlp.b3.numpy()[:,0])
   
   def NR_solve(fun,x):
      eps = 1e-3
      while abs(fun(x[0],x[1])) > 1e-3:
         df_dx  = (fun(x[0]+eps,x[1]) - fun(x[0],x[1]))/(eps)
         x[0] = x[0] - fun(x[0],x[1])/df_dx
      return x[0]
   
   def NR_solve_hard(fun,x):
      eps = 1e-3
      while abs(fun(x[0],x[1],x[2])) > 1e-3:
         df_dx  = (fun(x[0]+eps,x[1],x[2]) - fun(x[0],x[1],x[2]))/(eps)
         x[0] = x[0] - fun(x[0],x[1],x[2])/df_dx
      return x[0]
   
   def YF_NN(rho,theta):
      data = np.hstack((rho,np.sin(theta*3)))
      data = torch.tensor(data,dtype=torch.float32)
      with torch.no_grad():
         out = model(data*scale_x)*scale_y
      return out.numpy()
   
   def YF_NN_hard(rho,theta,e_pl):
      data = np.hstack((rho,np.sin(theta*3),e_pl))
      data = torch.tensor(data,dtype=torch.float32)
      with torch.no_grad():
         out = model(data*scale_x)*scale_y
      return out.numpy()

   if not Hardening:
      for i in range(theta.shape[0]):
         x = np.array([0.0,theta[i]])
         outNN = NR_solve(YF_NN,x)
         rhoNN[i] = outNN
   else:
      pl_lvls = 5
      for j in range(pl_lvls):
         epl = j*0.02/pl_lvls
         for i in range(theta.shape[0]):
            x = np.array([0.0,theta[i],epl])
            outNN = NR_solve_hard(YF_NN_hard,x)
            rhoNN[i+j*theta.shape[0]] = outNN
   return rhoNN
   

      
      