from MPM_forward import MPM_program
from MPM_inverse import MPM_program_inv
import warp as wp 
import yaml


wp.init()
# wp.config.verbose=True
# wp.config.verify_autograd_array_access = True

if wp.is_cuda_available():
   device = 'cuda'
else:
   device = 'cpu'

with open('input.yaml', 'r') as file:
   Inp_file = yaml.safe_load(file)

#case='Forward'
case='Inverse'


if case=='Forward':
   MPM_config = MPM_program(**Inp_file,device=device)
   MPM_config.run_forward() 
elif case=='Inverse':
   MPM_config = MPM_program_inv(**Inp_file,device=device,w_norm=False,epochs=1000)
   MPM_config.train()


