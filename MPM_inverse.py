import shutil
import os 
import torch 
import numpy as np 
import warp as wp 
from torch.utils.tensorboard import SummaryWriter
from warp.optim import Adam
from time import perf_counter
from src.material_laws import ST_update_NN,ST_update_NN_hardening
from src.utils import *
from src.wp_helpers import *
from src.wp_MPM import *
from src.wp_NN import *


device = wp.get_device()
if device.is_cpu:
    NUM_THREADS = 1
else:
    NUM_THREADS = 32

TILESIZE = wp.constant(8)
DIM_IN = wp.constant(2)
DIM_HID = wp.constant(64)
DIM_OUT = wp.constant(1)

class MPM_program_inv:
    def __init__(self,MPM_config,Material,Analysis_config,Geometry,Optimization,device,epochs):
        
        self.device = device
        self.num_particles = MPM_config['particles']
     
        self.n_grid = MPM_config['n_grid']
        self.isHardening = Analysis_config['Hardening']
        self.MPM_var = MPM_vars()
        self.States = MPM_state()

        if self.isHardening:
            input_path = './Input/3D_input'
            self.output_path = './Output/3D_output'
            dim = 3
        else:
            input_path = './Input/2D_input'
            self.output_path = './Output/2D_output'
            dim = 2

        self.dataset_x = np.load(input_path+'/positions.npy')
        self.steps = self.dataset_x.shape[0]
        self.acc_every = Optimization['acc_every']
        self.init_lr = Optimization['lr']
        self.epochs = epochs
        
        self.States.initialize(self.acc_every+1,self.n_grid,self.num_particles,requires_grad=True,device=device)
        self.MPM_var.initialize(Geometry,MPM_config,Analysis_config,Material)
        
        self.mat_points_pos,self.flag_var = make_mat_points(self.MPM_var)
        self.States.flag.assign(self.flag_var)
       
        checkpoint = torch.load(input_path+'/last_model.pt',weights_only=True)
        torch_weights = checkpoint['model_state_dict']

        scale_x = torch.load(input_path+'/scale_x.pt',weights_only=True).numpy()
        scale_y = torch.load(input_path+'/scale_y.pt',weights_only=True)

        self.wp_scale_inp = wp.array([1/i for i in scale_x],dtype=wp.float32,requires_grad=True,device=device)
        self.wp_scale_out = scale_y 

        self.mlp_inp1 = wp.zeros((self.acc_every,dim,self.num_particles),dtype=wp.float32,requires_grad=True,device=device)
        self.mlp_inp2 = wp.zeros((self.acc_every,dim,self.num_particles),dtype=wp.float32,requires_grad=True,device=device)
        self.wp_mlp = wp_MLP(torch_weights,self.num_particles,self.acc_every,device=device)        
        self.loss_arr = wp.zeros((1), dtype=wp.float32,requires_grad=True,device=device)

        #Custom Gradients arrays to accumulate gradients 
        self.grad_w1 = wp.zeros_like(self.wp_mlp.w1.grad.flatten(),device=device)
        self.grad_b1 = wp.zeros_like(self.wp_mlp.b1.grad.flatten(),device=device)
        self.grad_w2 = wp.zeros_like(self.wp_mlp.w2.grad.flatten(),device=device)
        self.grad_b2 = wp.zeros_like(self.wp_mlp.b2.grad.flatten(),device=device)
        self.grad_w3 = wp.zeros_like(self.wp_mlp.w3.grad.flatten(),device=device)
        self.grad_b3 = wp.zeros_like(self.wp_mlp.b3.grad.flatten(),device=device)

        self.optimizer = Adam(
            [self.wp_mlp.w1.flatten(),self.wp_mlp.b1.flatten(),
            self.wp_mlp.w2.flatten(),self.wp_mlp.b2.flatten(),
            self.wp_mlp.w3.flatten(),self.wp_mlp.b3.flatten()],
            lr=self.init_lr
            )
        
        dataset_steps = self.steps//self.acc_every
        displacements = self.dataset_x[::self.acc_every][1:]
        
        self.dx_curr = wp.zeros(shape=(dataset_steps,self.num_particles),dtype=wp.vec3f,requires_grad=True,device=device)
        self.dx_curr.assign(wp.array(displacements,dtype=wp.vec3f))

        self.dx_prev = wp.zeros(shape=(dataset_steps,self.num_particles),dtype=wp.vec3f,requires_grad=True,device=device)
        self.dx_prev.assign(wp.array(self.dataset_x[::self.acc_every][:-1],dtype=wp.vec3f))
        
        self.substep_x0 = wp.zeros(shape=(self.num_particles),dtype=wp.vec3f,requires_grad=True,device=device)

        # Create output dir
        shutil.rmtree(self.output_path,ignore_errors=True)
        os.makedirs(self.output_path,exist_ok=True)

        shutil.rmtree('./runs',ignore_errors=True)
        
        #YF_plot
        self.out_points = 100
        if self.isHardening:self.output_rhoNN = np.zeros((self.epochs,5*self.out_points))
        else:self.output_rhoNN = np.zeros((self.epochs,self.out_points))
        self.best_loss = np.inf
    
    def reset_grads(self):
        self.grad_w1.zero_()
        self.grad_b1.zero_()

        self.grad_w2.zero_()
        self.grad_b2.zero_()

        self.grad_w3.zero_()
        self.grad_b3.zero_()
    
    def reset_geo(self):
        
        self.States.reset_grid()
        self.States.reset_particles()
        self.reset_grads()
        self.wp_mlp.reset()

        self.States.x[0].assign(self.mat_points_pos)
        self.States.F[0].fill_(wp.diag(wp.vec3f(wp.float32(1.0),wp.float32(1.0),wp.float32(1.0))))
    
        if not hasattr(self, 'x0'): self.x0 = np.copy(self.States.x[0].numpy())
        assert self.num_particles == self.mat_points_pos.shape[0]
           
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
    
        wp.launch_tiled(
            kernel=run_mlp,
            dim=[self.num_particles//TILESIZE],
            inputs=[
                mlp_inp,
                self.wp_mlp.w1,
                self.wp_mlp.b1,
                out1,
                self.wp_mlp.w2,
                self.wp_mlp.b2,
                out2,
                self.wp_mlp.w3,
                self.wp_mlp.b3,
                out3
                ],
            block_dim=NUM_THREADS,
            device=device
            )
        
        wp.launch(
            kernel=mult_scalar,
            dim=[self.num_particles],
            inputs=[
                out3,
                self.wp_scale_out,
                out_true
                ],
            device=device
            )
        if get_jacobian:
        
            wp.launch(
                kernel=Get_z,
                dim=self.wp_mlp.z1[step].shape,
                inputs=[
                    out1,
                    self.wp_mlp.w1_copy,
                    self.wp_mlp.z1[step]
                    ],
                device=device
                )
            wp.launch(
                kernel=Get_z,
                dim=self.wp_mlp.z2[step].shape,
                inputs=[
                    out2,
                    self.wp_mlp.w2_copy,
                    self.wp_mlp.z2[step]
                    ],
                device=device
                )
            
            wp.launch_tiled(
                tile_grouped_gemm, 
                dim=[self.num_particles],
                inputs=[
                    self.wp_mlp.z2[step],
                    self.wp_mlp.z1[step],
                    self.wp_mlp.z2z1[step],
                    self.wp_mlp.w3_copy,
                    self.wp_mlp.df_dx[step]
                    ], 
                block_dim=NUM_THREADS, 
                device=device
                )
            
            wp.launch(
                kernel=scale_df,
                dim=self.wp_mlp.df_dx[step].shape,
                inputs=[
                    self.wp_mlp.df_dx[step],
                    self.wp_scale_inp,
                    self.wp_scale_out,
                    self.wp_mlp.df_dx_true[step]
                    ],
                device=device 
                )
           
    def step_NN(self,step,m_param):
        
        if self.isHardening:
            wp.launch(
                kernel=Get_NN_input_3d,
                dim=[self.num_particles],
                inputs=[
                    self.States.stress[step],
                    self.States.pl_mult[step],
                    self.States.theta[step],
                    self.wp_scale_inp,
                    self.mlp_inp1[step],
                    self.mlp_inp2[step]
                    ],
                device=device
                )
        else:
            wp.launch(
                kernel=Get_NN_input_2d,
                dim=[self.num_particles],
                inputs=[
                    self.States.stress[step],
                    self.States.theta[step],
                    self.wp_scale_inp,
                    self.mlp_inp1[step],
                    self.mlp_inp2[step]
                    ],
                device=device
                )
        self.run_NN(
            self.mlp_inp1[step],
            self.wp_mlp.out1[step],
            self.wp_mlp.out2[step],
            self.wp_mlp.out3[step],
            self.wp_mlp.f_val[step],
            step,
            get_jacobian=True
            )
        self.run_NN(
            self.mlp_inp2[step],
            self.wp_mlp.out11[step],
            self.wp_mlp.out22[step],
            self.wp_mlp.out33[step],
            self.wp_mlp.f_max[step],
            step,
            get_jacobian=False
            )
        
        if self.isHardening:
            wp.launch(
                kernel=self.Stress_update_hardening,
                dim=[self.num_particles],
                inputs=[
                    self.MPM_var,
                    self.States.stress[step],
                    self.States.log_e[step],
                    self.States.pl_mult[step],
                    self.States.L[step],
                    self.wp_mlp.f_val[step],
                    self.wp_mlp.f_max[step],
                    self.wp_mlp.df_dx_true[step],
                    self.States.stress[step+1],
                    self.States.log_e[step+1],
                    self.States.pl_mult[step+1],
                    m_param
                    ],
                device=device
                )
        else:
            wp.launch(
                kernel=self.Stress_update,
                dim=[self.num_particles],
                inputs=[
                    self.MPM_var,
                    self.States.stress[step],
                    self.States.log_e[step],
                    self.States.L[step],
                    self.wp_mlp.f_val[step],
                    self.wp_mlp.f_max[step],
                    self.wp_mlp.df_dx_true[step],
                    self.States.stress[step+1],
                    self.States.log_e[step+1],
                    m_param
                    ],
                device=device
                )
        wp.launch(
            kernel=p2g,
            dim=[self.num_particles],
            inputs=[
                self.MPM_var,
                self.States.x[step],
                self.States.v[step],
                self.States.F[step],
                self.States.L[step],
                self.States.stress[step+1],
                self.States.grid_m[step],
                self.States.grid_mv[step]
                ],
            device=device
            )
        
        wp.launch(
            kernel=grid_update,
            dim=[self.n_grid,self.n_grid,self.n_grid],
            inputs=[
                self.MPM_var,
                self.States.grid_m[step],
                self.States.grid_mv[step],
                self.States.grid_v[step]
                ],
            device=device
            )
        
        wp.launch(
            kernel=g2p,
            dim=[self.num_particles],
            inputs=[
                self.MPM_var,
                self.States.x[step],
                self.States.F[step],
                self.States.grid_v[step],
                self.States.x[step+1],
                self.States.v[step+1],
                self.States.F[step+1],
                self.States.L[step+1]
                ],
            device=device
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
            # self.optimizer.lr = self.get_lr(epoch)    # Update lr
            param_m = self.get_param_m(epoch)           # Update smoothing param
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
                            inputs=[
                                self.States.x[substep+1],
                                self.substep_x0,
                                self.dx_curr[istep],
                                self.dx_prev[istep],
                                self.loss_arr,
                                self.States.flag,
                                0.1,
                                ]
                                )
                    
                    tape.backward(loss=self.loss_arr)
                    iter_loss['train_loss'] +=self.loss_arr.numpy()
                    self.loss_arr.zero_()
                    
                    self.Add_grads([self.grad_w1,self.grad_b1,self.grad_w2,self.grad_b2,self.grad_w3,self.grad_b3],
                                   [self.wp_mlp.w1.grad.flatten(),self.wp_mlp.b1.grad.flatten(),
                                    self.wp_mlp.w2.grad.flatten(),self.wp_mlp.b2.grad.flatten(),
                                    self.wp_mlp.w3.grad.flatten(),self.wp_mlp.b3.grad.flatten()]
                                    )
                    
                    tape.reset()
            
                self.Gradient_step([self.grad_w1,self.grad_b1,self.grad_w2,self.grad_b2,self.grad_w3,self.grad_b3])
                
                self.wp_mlp.w1_copy.assign(self.wp_mlp.w1)
                self.wp_mlp.w2_copy.assign(self.wp_mlp.w2)
                self.wp_mlp.w3_copy.assign(self.wp_mlp.w3)
                
            self.output_rhoNN[epoch] = find_YF(self.wp_mlp,self.isHardening,self.wp_scale_inp,self.wp_scale_out,self.out_points)      
            print('Epoch:',epoch,' Displacement loss:',iter_loss['train_loss'], 'Stress_loss:', iter_loss['stress_loss'])
        
            np.save(self.output_path + '/out_rhoNN.npy',self.output_rhoNN[:epoch])

            #Write to tensorboard
            # writer.add_scalar(tag='Loss/Displacement_loss',scalar_value = iter_loss['train_loss'],
            #                   global_step = epoch,walltime=(perf_counter() - time_start)/60)
            # writer.add_scalar(tag='Loss/Stress_loss',scalar_value = iter_loss['stress_loss'],
            #                   global_step = epoch,walltime=(perf_counter() - time_start)/60)
            # writer.flush()
            
            # if iter_loss['train_loss']<self.best_loss:
            #     self.best_loss=iter_loss['train_loss']
            #     last_epoch = epoch
            #     self.save_model(epoch,iter_loss['train_loss'])
        
        writer.close()
        #print(f'Exiting...\n best loss:{self.best_loss}\n epoch:{last_epoch}')
    
    def save_model(self,epoch,loss):
        print(f'---saving---\n epoch:{epoch}, loss:{loss}')
        os.makedirs('./Saved_model/',exist_ok=True)
        for attr_name, attr_value in self.wp_mlp.__dict__.items():
            if 'b' in attr_name or 'w' in attr_name:
                np.save(f'./Output/Saved_model/{attr_name}.npy',attr_value.numpy())

    def get_lr(self,epoch):
        '''cosine annealing'''
        self.min_lr = 1e-8
        return self.min_lr + 0.5 * (self.init_lr - self.min_lr) * \
               (1 + np.cos(np.pi * epoch / self.epochs))
    
    def get_param_m(self,epoch):
        self.min_val_m  = 8.0
        self.max_val_m  = 8.0 
        '''cosine scheduler'''
        return self.min_val_m + 0.5 * (self.max_val_m - self.min_val_m) * \
               (1 - np.cos(np.pi * epoch / self.epochs))

    @staticmethod
    @wp.kernel
    def Stress_update(
        MPM:MPM_vars,
        stress:wp.array(dtype=wp.mat33f),
        log_e:wp.array(dtype=wp.mat33f),
        L:wp.array(dtype=wp.mat33f),
        f_val:wp.array(dtype=wp.float32),
        f_max:wp.array(dtype=wp.float32),
        dF_dinp:wp.array2d(dtype=wp.float32),
        new_stress:wp.array(dtype=wp.mat33f),
        new_log_e:wp.array(dtype=wp.mat33f),
        m_param:wp.float32
        ):
      
        tid = wp.tid()
        tau,e = ST_update_NN(
            L[tid],
            log_e[tid],
            stress[tid],
            f_val[tid],
            f_max[tid],
            dF_dinp[tid,0],
            dF_dinp[tid,1],
            MPM.mu,
            MPM.lamda,
            MPM.dt,
            m_param
            )
        
        new_stress[tid] = tau
        new_log_e[tid] = e

    @staticmethod
    @wp.kernel
    def Stress_update_hardening(
        MPM:MPM_vars,
        stress:wp.array(dtype=wp.mat33f),
        log_e:wp.array(dtype=wp.mat33f),
        e_pl:wp.array(dtype=wp.float32),
        L:wp.array(dtype=wp.mat33f),
        f_val:wp.array(dtype=wp.float32),
        f_max:wp.array(dtype=wp.float32),
        dF_dinp:wp.array2d(dtype=wp.float32),
        new_stress:wp.array(dtype=wp.mat33f),
        new_log_e:wp.array(dtype=wp.mat33f),
        new_e_pl:wp.array(dtype=wp.float32),
        m_param:wp.float32
        ):
      
        tid = wp.tid()
        tau,e,new_pl = ST_update_NN_hardening(
            L[tid],
            log_e[tid],
            stress[tid],
            e_pl[tid],
            f_val[tid],
            f_max[tid],
            dF_dinp[tid,0],
            dF_dinp[tid,1],
            dF_dinp[tid,2],
            MPM.mu,
            MPM.lamda,
            MPM.dt,
            m_param
            )
        
        new_stress[tid] = tau
        new_log_e[tid] = e
        new_e_pl[tid] = new_pl

@wp.kernel
def run_mlp(
    inputs: wp.array2d(dtype=wp.float32),       # (DIM_IN, batch_size)
    weights: wp.array2d(dtype=wp.float32),      # (DIM_HID1, DIM_IN)
    bias: wp.array2d(dtype=wp.float32),         # (DIM_HID1, 1)
    outputs: wp.array2d(dtype=wp.float32),      # (DIM_HID1, batch_size)
    weights2: wp.array2d(dtype=wp.float32),     # (DIM_HID1, DIM_HID1)
    bias2: wp.array2d(dtype=wp.float32),        # (DIM_HID1, 1)
    outputs2: wp.array2d(dtype=wp.float32),     # (DIM_HID1, batch_size)
    weights3: wp.array2d(dtype=wp.float32),     # (DIM_OUT, DIM_HID1)
    bias3: wp.array2d(dtype=wp.float32),        # (DIM_OUT, 1)
    outputs3: wp.array2d(dtype=wp.float32),     # (DIM_OUT, batch_size)
    ):
    
    batch = wp.tid()

    x1 = wp.tile_load(inputs, shape=(DIM_IN, TILESIZE), offset=(0, batch*TILESIZE))
    for i in range(0,DIM_HID//TILESIZE):
        w1 = wp.tile_load(weights, shape=(TILESIZE, DIM_IN),offset=(i*TILESIZE, 0))
        b1 = wp.tile_load(bias, shape=(TILESIZE, 1), offset=(i*TILESIZE, 0))
        z1 = wp.tile_matmul(w1, x1) + wp.tile_broadcast(b1, shape=(TILESIZE, TILESIZE))
        z1 = wp.tile_map(ELU, z1)
        wp.tile_store(outputs, z1, offset=(i*TILESIZE, batch*TILESIZE))

    x2 =  wp.tile_load(outputs, shape=(DIM_HID, TILESIZE), offset=(0, batch*TILESIZE))
    for j in range(0,DIM_HID//TILESIZE):
        w2 = wp.tile_load(weights2, shape=(TILESIZE, DIM_HID),offset=(j*TILESIZE, 0))
        b2 = wp.tile_load(bias2, shape=(TILESIZE, 1), offset=(j*TILESIZE, 0))
        z2 = wp.tile_matmul(w2, x2) + wp.tile_broadcast(b2, shape=(TILESIZE, TILESIZE))
        z2 = wp.tile_map(ELU, z2)
        wp.tile_store(outputs2, z2, offset=(j*TILESIZE, batch*TILESIZE))
    
    x3 = wp.tile_load(outputs2, shape=(DIM_HID, TILESIZE), offset=(0, batch*TILESIZE))
    w3 = wp.tile_load(weights3, shape=(DIM_OUT, DIM_HID))
    b3 = wp.tile_load(bias3, shape=(DIM_OUT, 1))
    z3 = wp.tile_matmul(w3, x3) + wp.tile_broadcast(b3, shape=(DIM_OUT, TILESIZE))
    wp.tile_store(outputs3, z3, offset=(0, batch*TILESIZE))

@wp.kernel
def tile_grouped_gemm(
    A: wp.array3d(dtype=float),
    B: wp.array3d(dtype=float),
    C: wp.array3d(dtype=float),
    D:wp.array2d(dtype=float),
    F:wp.array3d(dtype=float)
    ):

    batch = wp.tid()

    J = A[batch].shape[0]
    K = A[batch].shape[1]

    for i in range(0,int(J / TILESIZE)):
        sum = wp.tile_zeros(shape=(TILESIZE, DIM_IN), dtype=A.dtype)
        for k in range(0, int(K / TILESIZE)):
            a = wp.tile_load(A[batch], shape=(TILESIZE, TILESIZE), offset=(i * TILESIZE, k * TILESIZE))
            b = wp.tile_load(B[batch], shape=(TILESIZE, DIM_IN), offset=(k * TILESIZE, 0))
            wp.tile_matmul(a, b, sum)
        wp.tile_store(C[batch], sum, offset=(i * TILESIZE, 0))
    
    sum2 = wp.tile_zeros(shape=(1,DIM_IN), dtype=A.dtype)
    for j in range(0, DIM_HID//TILESIZE):
        c = wp.tile_load(D, shape=(1, TILESIZE), offset=(0, j * TILESIZE))
        d = wp.tile_load(C[batch], shape=(TILESIZE, DIM_IN), offset=(j * TILESIZE, 0))
        wp.tile_matmul(c, d, sum2)
    wp.tile_store(F[batch], sum2)
