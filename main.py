from MPM_inverse import MPM_program_inv
from MPM_forward import MPM_program
import warp as wp 
import yaml

wp.init()
if wp.is_cuda_available():
   device = 'cuda'
else:
   device = 'cpu'

with open('input.yaml', 'r') as file:
   Inp_file = yaml.safe_load(file)

MPM_config = MPM_program_inv(**Inp_file,device=device,epochs=1000)
MPM_config.train()


