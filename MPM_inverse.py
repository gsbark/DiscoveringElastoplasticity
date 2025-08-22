import shutil
import os 
import torch 
import numpy as np 
import warp as wp 
from torch.utils.tensorboard import SummaryWriter
from warp.optim import Adam
from tqdm import tqdm
from time import perf_counter
from Utils.material_laws import stress_update33_NN
from Utils.geometry import geometry
from Utils.export_utils import *
from Utils.wp_funcs import *
from Utils.wp_MPM import *
from Utils.wp_MPM_funcs import *

class MPM_program_inv:
    def __init__(self,MPM_config,Material,Analysis_config,Geometry,device,w_norm,epochs):
        
        self.device = device
        self.num_particles = MPM_config['particles']
     
        self.n_grid = MPM_config['n_grid']
        self.is_w_norm = w_norm

        self.dataset_x = np.load('./Output/dataset/positions.npy')
    
    
        self.steps = self.dataset_x.shape[0]
        self.acc_every = 10
        self.epochs = epochs
        self.init_lr = 1e-5
        self.min_lr = 1e-8

        self.MPM_var = MPM_vars()
        self.States = MPM_state()

        self.States.initialize(self.acc_every+1,self.n_grid,self.num_particles,requires_grad=True,device=device)
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
                                                   [0.5 + a, 0.5 + a, 0.5 + a]],dtype=np.float32)

        self.gp = wp.from_numpy(gauss_points,device=device,dtype=wp.vec3f)
       
        path = './NN_train/Circle'
        checkpoint = torch.load(path+'/last_model.pt',weights_only=True)
        torch_weights = checkpoint['model_state_dict']

        scale_x = torch.load(path+'/scale_x.pt',weights_only=True).numpy()
        scale_y = torch.load(path+'/scale_y.pt',weights_only=True)

        x1 = 1.0/float(scale_x[0])
        x2 = 1.0/float(scale_x[1])
        y1 = float(scale_y)
        
        self.wp_scale_inp = wp.array([x1,x2],dtype=wp.float32,requires_grad=True,device=device)
        self.wp_scale_out = wp.array([y1],dtype=wp.float32,requires_grad=True,device=device)

        self.mlp_inp1 = wp.zeros((self.acc_every,2,self.num_particles),dtype=wp.float32,requires_grad=True,device=device)
        self.mlp_inp2 = wp.zeros((self.acc_every,2,self.num_particles),dtype=wp.float32,requires_grad=True,device=device)
        self.wp_mlp = wp_MLP(torch_weights,self.num_particles,self.acc_every,device=device,w_norm=w_norm)        
        self.loss_arr = wp.zeros((1), dtype=wp.float32,requires_grad=True,device=device)

        #Custom Gradients arrays to accumulate gradients 
        self.grad_w1 = wp.zeros_like(self.wp_mlp.w1.grad.flatten(),device=device)
        self.grad_b1 = wp.zeros_like(self.wp_mlp.b1.grad.flatten(),device=device)
        self.grad_w2 = wp.zeros_like(self.wp_mlp.w2.grad.flatten(),device=device)
        self.grad_b2 = wp.zeros_like(self.wp_mlp.b2.grad.flatten(),device=device)
        self.grad_w3 = wp.zeros_like(self.wp_mlp.w3.grad.flatten(),device=device)
        self.grad_b3 = wp.zeros_like(self.wp_mlp.b3.grad.flatten(),device=device)

        self.optimizer = Adam([self.wp_mlp.w1.flatten(),self.wp_mlp.b1.flatten(),
                               self.wp_mlp.w2.flatten(),self.wp_mlp.b2.flatten(),
                               self.wp_mlp.w3.flatten(),self.wp_mlp.b3.flatten()],lr=0.00005)
        
        dataset_steps = self.steps//self.acc_every
        displacements = self.dataset_x[::self.acc_every][1:]-self.dataset_x[::self.acc_every][:-1]
        
        self.dx_true = wp.zeros(shape=(dataset_steps,self.num_particles),dtype=wp.vec3f,requires_grad=True,device=device)
        self.dx_true.assign(wp.array(displacements,dtype=wp.vec3f))
        self.substep_x0 = wp.zeros(shape=(self.num_particles),dtype=wp.vec3f,requires_grad=True,device=device)

        shutil.rmtree('./Output/Saved_model',ignore_errors=True)
        os.makedirs('./Output/Saved_model',exist_ok=True)

        shutil.rmtree('./Output/Paraview_NN',ignore_errors=True)
        os.makedirs('./Output/Paraview_NN',exist_ok=True)

        shutil.rmtree('./Output/error',ignore_errors=True)
        os.makedirs('./Output/error',exist_ok=True)

        shutil.rmtree('./runs',ignore_errors=True)
        
        #YF_plot
        self.out_points = 100
        self.output_rhoNN = np.zeros((self.epochs,self.out_points))
        self.best_loss = np.inf

    def reset_grads(self):
        self.grad_w1.zero_()
        self.grad_b1.zero_()

        self.grad_w2.zero_()
        self.grad_b2.zero_()

        self.grad_w3.zero_()
        self.grad_b3.zero_()
    
    def reset_geo(self):
        self.particles_id.zero_()
        self.States.reset_grid()
        self.States.reset_particles()
        self.reset_grads()

        wp.launch(
            kernel=geometry,
            dim=[1],
            inputs=[self.States,self.MPM_var,self.particles_id,self.gp,self.n_grid]
            )
        if not hasattr(self, 'x0'): self.x0 = np.copy(self.States.x[0].numpy())
        assert self.num_particles == self.particles_id.numpy()[0], self.particles_id.numpy()[0]
           
    def Gradient_step(self,grads):
        self.optimizer.step([*grads])

    def Add_grads(self,total_grads,grads):
        total_grads = [*total_grads]
        gradients = [*grads]
        for grad,acc_grad in zip(gradients,total_grads):
            wp.launch(
                kernel=add,
                dim=grad.shape,
                inputs=[acc_grad,grad]
                )
            
    def run_NN(self,mlp_inp,out1,out2,out3,out_true,step,get_jacobian):

        wp.launch(
            kernel=run_mlp,
            dim=[self.num_particles],
            inputs=[self.wp_mlp.w1,self.wp_mlp.b1,self.wp_mlp.w2,self.wp_mlp.b2,
                    self.wp_mlp.w3,self.wp_mlp.b3,mlp_inp,out1,out2,out3]
                    )
        wp.launch(
            kernel=scale_f,
            dim=[self.num_particles],
            inputs=[out3,self.wp_scale_out,out_true]
            )
        
        if get_jacobian:
            wp.launch(kernel=Get_z,
                      dim=self.wp_mlp.z1[step].shape,
                      inputs=[out1,self.wp_mlp.w1_copy,self.wp_mlp.z1[step]]
                      )
            wp.launch(kernel=Get_z,
                      dim=self.wp_mlp.z2[step].shape,
                      inputs=[out2,self.wp_mlp.w2_copy,self.wp_mlp.z2[step]]
                      )
            wp.batched_matmul(self.wp_mlp.z2[step],self.wp_mlp.z1[step],self.wp_mlp.dummy1[step],self.wp_mlp.z2z1[step]) 
            wp.launch(
                kernel=copy_kernel,
                dim=self.wp_mlp.w3_exp[step].shape,
                inputs=[self.wp_mlp.w3_exp[step],self.wp_mlp.w3_copy]
                )
            wp.batched_matmul(self.wp_mlp.w3_exp[step],self.wp_mlp.z2z1[step],self.wp_mlp.dummy2[step],self.wp_mlp.df_dx[step]) 
            wp.launch(
                kernel=scale_df,
                dim=self.wp_mlp.df_dx[step].shape,
                inputs=[self.wp_mlp.df_dx[step],self.wp_scale_inp,self.wp_scale_out,self.wp_mlp.df_dx_true[step]]
                )

    def step_NN(self,step,m_param):

        wp.launch(
            kernel=Get_NN_input,
            dim=[self.num_particles],
            inputs=[self.States.stress[step],self.States.theta[step],
                    self.wp_scale_inp,self.mlp_inp1[step],self.mlp_inp2[step]]
                    )
        self.run_NN(self.mlp_inp1[step],self.wp_mlp.out1[step],self.wp_mlp.out2[step],
                    self.wp_mlp.out3[step],self.wp_mlp.f_val[step],step,run_back=True)
        self.run_NN(self.mlp_inp2[step],self.wp_mlp.out11[step],self.wp_mlp.out22[step],
                    self.wp_mlp.out33[step],self.wp_mlp.f_max[step],step,run_back=False)
        wp.launch(
            kernel=self.Stress_update,
            dim=[self.num_particles],
            inputs=[self.MPM_var,self.States.stress[step],self.States.log_e[step],self.States.L[step],
                    self.wp_mlp.f_val[step],self.wp_mlp.f_max[step],self.wp_mlp.df_dx_true[step],
                    self.States.stress[step+1],self.States.log_e[step+1],m_param]
                    )
        wp.launch(
            kernel=p2g,
            dim=[self.num_particles],
            inputs=[self.MPM_var,self.States.x[step],self.States.v[step],self.States.F[step],self.States.L[step],
                    self.States.stress[step+1],self.States.grid_m[step],self.States.grid_mv[step]]
                    )
        wp.launch(
            kernel=grid_update,
            dim=[self.n_grid,self.n_grid,self.n_grid],
            inputs=[self.MPM_var,self.States.grid_m[step],
                    self.States.grid_mv[step],self.States.grid_v[step]]
                    )
        wp.launch(
            kernel=g2p,
            dim=[self.num_particles],
            inputs=[self.MPM_var,self.States.x[step],self.States.F[step], self.States.grid_v[step],
                    self.States.x[step+1],self.States.v[step+1],self.States.F[step+1],self.States.L[step+1]]
                    )
        
    def train(self):
        '''
        Accumulate gradients and update NN paramters
        '''
        writer = SummaryWriter(comment='test')
        time_start = perf_counter()
        iter_loss = {'train_loss':0.0,'stress_loss':0.0}
        for epoch in range(self.epochs):
            step = 0 
            # self.optimizer.lr = self.get_lr(epoch) #Update lr
            param_m = self.get_param_m(epoch)
            iter_loss['train_loss'] = 0.0
            iter_loss['stress_loss'] = 0.0
            self.reset_geo()
            with wp.ScopedTimer('Sim'):
                for istep in range(self.steps//self.acc_every):
                    tape = wp.Tape()
                    if step>0:
                        self.States.reset_substeps(self.acc_every+1)
                        self.wp_mlp.reset()
                    self.substep_x0.assign(self.States.x[0])
                    with tape:
                        for substep in range(self.acc_every):
                            step = istep*self.acc_every + substep
                            self.step_NN(substep,param_m)
                        wp.launch(
                            kernel=loss_fun,
                            dim=[self.num_particles],
                            inputs=[self.States.x[substep+1],self.substep_x0,self.dx_true[istep],self.loss_arr,self.States.flag,0.01]
                            )
                    tape.backward(loss=self.loss_arr)
                    iter_loss['train_loss'] +=self.loss_arr.numpy()
                    self.loss_arr.zero_()
                    self.Add_grads([self.grad_w1,self.grad_b1,self.grad_w2,self.grad_b2,self.grad_w3,self.grad_b3],
                                   [self.wp_mlp.w1.grad.flatten(),self.wp_mlp.b1.grad.flatten(),
                                    self.wp_mlp.w2.grad.flatten(),self.wp_mlp.b2.grad.flatten(),
                                    self.wp_mlp.w3.grad.flatten(),self.wp_mlp.b3.grad.flatten()])
                    tape.reset()
                
                self.Gradient_step([self.grad_w1,self.grad_b1,self.grad_w2,self.grad_b2,self.grad_w3,self.grad_b3])
                
                self.wp_mlp.w1_copy.assign(self.wp_mlp.w1)
                self.wp_mlp.w2_copy.assign(self.wp_mlp.w2)
                self.wp_mlp.w3_copy.assign(self.wp_mlp.w3)

            self.output_rhoNN[epoch] = find_YF(self.wp_mlp,self.is_w_norm,self.wp_scale_inp,self.wp_scale_out,self.out_points)      
            print('Epoch:',epoch,' Displacement loss:',iter_loss['train_loss'], 'Stress_loss:', iter_loss['stress_loss'])
            np.save('./Output/out_rhoNN.npy',self.output_rhoNN[:epoch])

            #Write to tensorboard
            writer.add_scalar(tag='Loss/Displacement_loss',scalar_value = iter_loss['train_loss'],
                              global_step = epoch,walltime=(perf_counter() - time_start)/60)
            writer.add_scalar(tag='Loss/Stress_loss',scalar_value = iter_loss['stress_loss'],
                              global_step = epoch,walltime=(perf_counter() - time_start)/60)
            writer.flush()
            
            if iter_loss['train_loss']<self.best_loss:
                self.best_loss=iter_loss['train_loss']
                last_epoch = epoch
                self.save_model(epoch,iter_loss['train_loss'])
        
        writer.close()
        print(f'Exiting...\n best loss:{self.best_loss}\n epoch:{last_epoch}')
    
    def save_model(self,epoch,loss):
        print(f'---saving---\n epoch:{epoch}, loss:{loss}')
        os.makedirs('./Saved_model/',exist_ok=True)
        for attr_name, attr_value in self.wp_mlp.__dict__.items():
            if 'b' in attr_name or 'w' in attr_name:
                np.save(f'./Output/Saved_model/{attr_name}.npy',attr_value.numpy())

    def get_lr(self,epoch):
        '''cosine annealing'''
        return self.min_lr + 0.5 * (self.init_lr - self.min_lr) * \
               (1 + np.cos(np.pi * epoch / self.epochs))
    
    def get_param_m(self,epoch):
        self.min_val_m  = 3.0
        self.max_val_m  = 8.0 
        '''cosine scheduler'''
        return self.min_val_m + 0.5 * (self.max_val_m - self.min_val_m) * \
               (1 - np.cos(np.pi * epoch / self.epochs))

    @staticmethod
    @wp.kernel
    def Stress_update(MPM:MPM_vars,
                      stress:wp.array(dtype=wp.mat33f),
                      log_e:wp.array(dtype=wp.mat33f),
                      L:wp.array(dtype=wp.mat33f),
                      f_val:wp.array(dtype=wp.float32),
                      f_max:wp.array(dtype=wp.float32),
                      dF_dinp:wp.array2d(dtype=wp.float32),
                      new_stress:wp.array(dtype=wp.mat33f),
                      new_log_e:wp.array(dtype=wp.mat33f),
                      m_param:float):
      
      tid = wp.tid()
      tau,e = stress_update33_NN(L[tid],log_e[tid],stress[tid],
                                 f_val[tid],f_max[tid],dF_dinp[tid,0],dF_dinp[tid,1],
                                 MPM.mu,MPM.lamda,MPM.dt,m_param)
      new_stress[tid] = tau
      new_log_e[tid] = e
    

    

    

  


    
       

    

        
