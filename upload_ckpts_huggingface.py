import os
from huggingface_hub import HfApi, HfFolder

# Read the token from an environment variable
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if hf_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable is not set")

# Save the token
HfFolder.save_token(hf_token)
api = HfApi()

api.upload_large_folder(
    repo_id="sultan-daniels/TFs_do_KF_ICL_ident_med_GPT2_experiment",
    repo_type="model",
    folder_path="/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250124_052617.8dd0f8_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"
)