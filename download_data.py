import os
from huggingface_hub import hf_hub_download

save_path = "/scratch/users/dhruvgautam/TFs_do_KF_ICL/identity_data/val_ident_ident_C_state_dim_5.pkl"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

hf_hub_download(
    repo_id="sultan-daniels/TFs_do_KF_ICL_ident_med_GPT2_experiment",
    filename="data/val_ident_ident_C_state_dim_5.pkl",
    local_dir=os.path.dirname(save_path),
    local_dir_use_symlinks=False
)

print(f"File saved to: {save_path}")