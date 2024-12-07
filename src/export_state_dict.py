from models import GPT2
from core import Config
import os
import torch

config = Config()
config.override("ckpt_path", "../outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C/checkpoints/step=192000.ckpt")

#get the parent directory of the ckpt_path
parent_dir = os.path.dirname(config.ckpt_path)
#get the parent directory of the parent directory
output_dir = os.path.dirname(parent_dir)
# instantiate gpt2 model
model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
        n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)


# Load the trained checkpoint
checkpoint_path = config.ckpt_path
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Load the state dictionary into the model
model.load_state_dict(checkpoint['state_dict'])

# Export the state dictionary
#get the parent directory of the ckpt_path
parent_dir = os.path.dirname(config.ckpt_path)
os.makedirs(parent_dir + "/state_dicts", exist_ok=True)
export_path = parent_dir + "/state_dicts/state_dict.pth"
torch.save(model.state_dict(), export_path)

print(f"Model state dictionary exported to {export_path}")