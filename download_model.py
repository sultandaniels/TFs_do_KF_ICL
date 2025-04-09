import os
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

repo_id = "sultan-daniels/TFs_do_KF_ICL_ortho_med_GPT2_checkpoints"
filename = "step=42000.ckpt"
custom_path = "/scratch/users/dhruvgautam/models"  
checkpoint_path = "/scratch/users/dhruvgautam/models/models--sultan-daniels--TFs_do_KF_ICL_ortho_med_GPT2_checkpoints/snapshots/824c3034ec025999d7bc2923335142b19152ab71/post_emerge.ckpt"

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found at {checkpoint_path}. Downloading from Hugging Face...")
    downloaded_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=custom_path
    )
    checkpoint_path = downloaded_file_path
    print(f"Downloaded to: {checkpoint_path}")
else:
    print(f"Checkpoint already exists at: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location="cpu")

safetensor_path = checkpoint_path.replace("post_emerge.ckpt", "model.safetensors")
checkpoint_tensors = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
save_file(checkpoint_tensors, safetensor_path)
print(f"Checkpoint converted and saved to {safetensor_path}")

# from huggingface_hub import hf_hub_download

# # Define the repository ID and the filename
# repo_id = 'sultan-daniels/TFs_do_KF_ICL_ortho_med_GPT2_experiment'
# filename = 'data/val_ortho_ident_C_state_dim_5.pkl'

# # Download the file
# custom_path = "/data/dhruv_gautam/models" 
# file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=custom_path)

# print(f'File downloaded to: {file_path}')