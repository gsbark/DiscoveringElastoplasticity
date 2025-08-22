import numpy as np 
import torch
import vtk
from NN_train.train import MLP
from Utils.wp_MPM import MPM_state

def get_s_vm(sigma):
   s_vm = np.sqrt(1/2*((sigma[:,0,0]-sigma[:,1,1])**2 + \
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
      #'stress_s_vm':get_s_vm(field.stress[index].numpy()),
      # 'theta':field.theta[index].numpy(),
      #'velocity':field.v[index].numpy(),
      'displacement':field.x[index].numpy() - x0,
      #'flag':field.flag.numpy(),
      }
   GaussPoints_export(field.x[index].numpy(),field_data,i_frame,fname=folder,name='mat_points')
#--------------------------------------------------------

def find_YF(wp_mlp,w_norm,scale_inp,scale_out,num):
   
   model = MLP(hidden_size=[2,64,64,1])
   scale_x = torch.tensor(scale_inp.numpy())
   scale_y = torch.tensor(scale_out.numpy())

   if w_norm: 
      w1 = wp_mlp.w1_scaled.numpy()
      w2 = wp_mlp.w2_scaled.numpy()
      w3 = wp_mlp.w3_scaled.numpy()
   else:
      w1 = wp_mlp.w1.numpy()
      w2 = wp_mlp.w2.numpy()
      w3 = wp_mlp.w3.numpy()

   model.layers[0].weight.data = torch.tensor(w1)
   model.layers[0].bias.data = torch.tensor(wp_mlp.b1.numpy())
   model.layers[2].weight.data = torch.tensor(w2)
   model.layers[2].bias.data = torch.tensor(wp_mlp.b2.numpy())
   model.layers[4].weight.data = torch.tensor(w3)
   model.layers[4].bias.data = torch.tensor(wp_mlp.b3.numpy())
   
   theta = np.linspace(0,2*np.pi,num)
   rhoNN = np.zeros(theta.shape[0])

   def NR_solve(fun,x):
      eps = 1e-3
      while abs(fun(x[0],x[1])) > 1e-3:
         df_dx  = (fun(x[0]+eps,x[1]) - fun(x[0],x[1]))/(eps)
         x[0] = x[0] - fun(x[0],x[1])/df_dx
      return x[0]
   
   def YF_NN(rho,theta):
      data = np.hstack((rho,np.sin(theta*3)))
      data = torch.tensor(data,dtype=torch.float32)
      with torch.no_grad():
         out = model(data*scale_x)*scale_y
      return out.numpy()

   for i in range(theta.shape[0]):
      x = np.array([0.0,theta[i]])
      outNN = NR_solve(YF_NN,x)
      rhoNN[i] = outNN
   return rhoNN
